from __future__ import annotations
import abc, warnings, numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple, Set
from sklearn.metrics import (
    classification_report, precision_recall_curve, average_precision_score,
    jaccard_score, f1_score
)
from evaluate.DatasetModel import DatasetModel
from evaluate.COCOAnnotations import COCOAnnotation


def _safe_float(x: Union[float, int, str, np.floating, np.ndarray, None], default: float = 0.0) -> float:
    try:
        if x is None: return default
        v = float(x); return v if np.isfinite(v) else default
    except Exception:
        return default

def _safe_box(bb: Optional[Union[List[float], Tuple[float, float, float, float]]]) -> Optional[List[float]]:
    if not isinstance(bb, (list, tuple)) or len(bb) != 4: return None
    x, y, w, h = (_safe_float(bb[0]), _safe_float(bb[1]), _safe_float(bb[2]), _safe_float(bb[3]))
    return [x, y, w, h]

def _xywh_to_xyxy(b: List[float]) -> List[float]:
    x, y, w, h = b; return [x, y, x + w, y + h]

def _iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1); ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1); inter = iw * ih
    if inter <= 0: return 0.0
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / max(ua + ub - inter, 1e-12))
@dataclass
class Stats:
    categories: List[str] = field(default_factory=list)
    gt_counts: List[int] = field(default_factory=list)
    pred_counts: List[int] = field(default_factory=list)
    per_class_ap: List[float] = field(default_factory=list)
    per_class_ap_avg: List[float] = field(default_factory=list)
    map: Optional[float] = None
    map_50: Optional[float] = None
    map_75: Optional[float] = None
    ap_iou_macro: List[float] = field(default_factory=list)
    ap_iou_micro: List[float] = field(default_factory=list)
    iou_thresholds: List[float] = field(default_factory=list)
    pr_macro: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    pr_micro: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    pr_curves_per_threshold: Dict[float, Dict[str, Dict[str, List[float]]]] = field(default_factory=dict)
    doc: str = ""
    formula: Optional[str] = None
@dataclass
class ClsReportStats:
    num_files: int = 0
    dict: Dict[str, dict] = field(default_factory=dict)
    text: str = ""
    labels: List[str] = field(default_factory=list)
    doc: str = ""
@dataclass
class MetricOutputModel:
    metric_name: str
    score: float
    stats: Union[Stats, ClsReportStats]
class Metric(abc.ABC):
    name: str = "metric"
    @abc.abstractmethod
    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs) -> MetricOutputModel: ...
class MeanAveragePrecision(Metric):
    name = "mAP"
    def __init__(self, iou_thresholds: Optional[List[float]] = None) -> None:
        self.iou_thresholds = iou_thresholds or np.arange(0.5, 0.95 + 1e-9, 0.05).tolist()

    def _group_input(self, gt: DatasetModel, prediction: List[COCOAnnotation]):
        gt_by_file: Dict[str, List[COCOAnnotation]] = {}
        pr_by_file: Dict[str, List[COCOAnnotation]] = {}
        for a in (gt.data_points or []):
            fn = getattr(a, 'file_name', None)
            if fn is not None: gt_by_file.setdefault(str(fn), []).append(a)
        for a in (prediction or []):
            fn = getattr(a, 'file_name', None)
            if fn is not None: pr_by_file.setdefault(str(fn), []).append(a)
        files = sorted(set(gt_by_file) | set(pr_by_file))
        labs: List[str] = []
        for arr in (gt.data_points or []):
            lab = getattr(arr, 'label', None)
            if lab is not None: labs.append(str(lab))
        for arr in (prediction or []):
            lab = getattr(arr, 'label', None)
            if lab is not None: labs.append(str(lab))
        uniq_labels = sorted({str(label) for label in labs})
        return gt_by_file, pr_by_file, files, uniq_labels

    def _prepare_boxes(self, gt_by_file, pr_by_file, files, uniq_labels):
        gt_boxes_by_label_file: Dict[str, Dict[str, List[List[float]]]] = {lab: {} for lab in uniq_labels}
        for fn in files:
            for a in gt_by_file.get(fn, []):
                lab, bb = str(getattr(a, 'label', '')), getattr(a, 'bbox', None)
                sb = _safe_box(bb)
                if lab in gt_boxes_by_label_file and sb is not None:
                    gt_boxes_by_label_file[lab].setdefault(fn, []).append(_xywh_to_xyxy(sb))
        preds_by_label: Dict[str, List[Tuple[str, List[float], float]]] = {lab: [] for lab in uniq_labels}
        for fn in files:
            for p in pr_by_file.get(fn, []):
                lab, bb = str(getattr(p, 'label', '')), getattr(p, 'bbox', None)
                sc = _safe_float(getattr(p, 'score', getattr(p, 'confidence', 1.0)))
                sb = _safe_box(bb)
                if lab in preds_by_label and sb is not None:
                    preds_by_label[lab].append((fn, _xywh_to_xyxy(sb), sc))
        for lab in uniq_labels:
            preds_by_label[lab].sort(key=lambda d: d[2], reverse=True)
        return gt_boxes_by_label_file, preds_by_label

    def _ap_pr(self, y_true: List[int], y_scores: List[float]) -> Tuple[float, List[float], List[float]]:
        if not y_true: return 0.0, [1.0], [0.0]
        try:
            yt = [1 if int(v) == 1 else 0 for v in y_true]
            ys = [_safe_float(v) for v in y_scores]
            pos = int(np.sum(yt)); neg = len(yt) - pos
            if pos == 0: return 0.0, [1.0], [0.0]
            if neg == 0: return 1.0, [1.0], [1.0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                ap = _safe_float(average_precision_score(yt, ys))
                p, r, _ = precision_recall_curve(yt, ys)
            return ap, [float(x) for x in p], [float(x) for x in r]
        except Exception:
            return 0.0, [1.0], [0.0]

    def _match_make_targets(self, preds_list: List[Tuple[str, List[float], float]], gt_per_file: Dict[str, List[List[float]]], thr: float):
        matched: Dict[str, Set[int]] = {fn: set() for fn in gt_per_file.keys()}
        y_true: List[int] = []; y_scores: List[float] = []
        for fn, box_pred, score in preds_list:
            gts = gt_per_file.get(fn, [])
            best_iou, best_idx = 0.0, None
            for gi, gbb in enumerate(gts):
                if gi in matched.get(fn, set()): continue
                v = _iou(box_pred, gbb)
                if v > best_iou: best_iou, best_idx = v, gi
            if best_iou >= _safe_float(thr) and best_idx is not None:
                y_true.append(1); y_scores.append(float(score))
                matched.setdefault(fn, set()).add(best_idx)
            else:
                y_true.append(0); y_scores.append(float(score))
        return y_true, y_scores

    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs) -> MetricOutputModel:
        gt_by_file, pr_by_file, files, labels = self._group_input(gt, prediction)
        if not files:
            return MetricOutputModel(self.name, 0.0, Stats(doc="empty GT or predictions"))

        cats = [l for l in labels]
        gt_boxes_by_label_file, preds_by_label = self._prepare_boxes(gt_by_file, pr_by_file, files, labels)

        ap_per_label_per_thr = {lab: [0.0] * len(self.iou_thresholds) for lab in labels}
        pr_curves_per_thr: Dict[float, Dict[str, Dict[str, List[float]]]] = {}
        y_store: Dict[float, Dict[str, Tuple[List[int], List[float]]]] = {}

        for ti, thr in enumerate(self.iou_thresholds):
            pr_curves_per_thr[thr] = {}; y_store[thr] = {}
            for lab in labels:
                preds_list = preds_by_label.get(lab, [])
                gt_per_file = gt_boxes_by_label_file.get(lab, {}) or {}
                yt, ys = self._match_make_targets(preds_list, gt_per_file, float(thr))
                y_store[thr][lab] = (yt, ys)
                ap, p, r = self._ap_pr(yt, ys)
                ap_per_label_per_thr[lab][ti] = _safe_float(ap)
                pr_curves_per_thr[thr][lab] = {'precision': p, 'recall': r}
        recall_grid = np.linspace(0.0, 1.0, 101)
        pr_micro: Dict[str, Dict[str, List[float]]] = {}
        pr_macro: Dict[str, Dict[str, List[float]]] = {}
        ap_iou_micro: List[float] = []; macro_map_iou: List[float] = []
        for thr in self.iou_thresholds:
            key = f"{_safe_float(thr):.2f}"
            all_yt: List[int] = []; all_ys: List[float] = []
            for _, (yt, ys) in (y_store.get(thr, {}) or {}).items():
                if yt: all_yt += yt; all_ys += ys
            if all_yt:
                ap_mi, p_mi, r_mi = self._ap_pr(all_yt, all_ys)
            else:
                ap_mi, p_mi, r_mi = 0.0, [1.0], [0.0]
            ap_iou_micro.append(_safe_float(ap_mi))
            pr_micro[key] = {'precision': [float(x) for x in p_mi], 'recall': [float(x) for x in r_mi]}

            stack: List[np.ndarray] = []
            for lab, prc in (pr_curves_per_thr.get(thr, {}) or {}).items():
                r_arr = np.asarray(prc.get('recall') or []); p_arr = np.asarray(prc.get('precision') or [])
                if r_arr.size > 1 and p_arr.size > 1:
                    o = np.argsort(r_arr)
                    stack.append(np.interp(recall_grid, r_arr[o], p_arr[o], left=p_arr[o][0], right=p_arr[o][-1]))
            p_ma = np.mean(np.stack(stack, 0), 0) if stack else np.zeros_like(recall_grid)
            pr_macro[key] = {'precision': [float(x) for x in p_ma], 'recall': [float(x) for x in recall_grid]}

        for i in range(len(self.iou_thresholds)):
            vals = [ap_per_label_per_thr[lab][i] for lab in labels] if labels else []
            macro_map_iou.append(float(np.mean(vals)) if vals else 0.0)

        overall = float(np.mean(macro_map_iou)) if macro_map_iou else 0.0
        def _pick(t: float) -> float:
            for i, thr in enumerate(self.iou_thresholds):
                if abs(_safe_float(thr) - _safe_float(t)) < 1e-6:
                    return float(macro_map_iou[i]) if i < len(macro_map_iou) else 0.0
            return 0.0
        map50, map75 = _pick(0.5), _pick(0.75)
        gt_counts = [sum(len(v) for v in (gt_boxes_by_label_file.get(l, {}) or {}).values()) for l in labels]
        pred_counts = [len(preds_by_label.get(l, []) or []) for l in labels]
        stats = Stats(
            categories=cats,
            gt_counts=gt_counts,
            pred_counts=pred_counts,
            per_class_ap=[float(np.mean(ap_per_label_per_thr[l])) if ap_per_label_per_thr[l] else 0.0 for l in labels],
            per_class_ap_avg=[float(np.mean(ap_per_label_per_thr[l])) if ap_per_label_per_thr[l] else 0.0 for l in labels],
            map=overall, map_50=map50, map_75=map75,
            ap_iou_macro=[float(x) for x in macro_map_iou],
            ap_iou_micro=[float(x) for x in ap_iou_micro],
            iou_thresholds=[float(x) for x in self.iou_thresholds],
            pr_macro=pr_macro, pr_micro=pr_micro,
            pr_curves_per_threshold=pr_curves_per_thr,
            doc="", formula=""
        )
        return MetricOutputModel(self.name, overall, stats)
class MeanIntersectionOverUnion(Metric):
    name = "mIoU"
    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs) -> MetricOutputModel:
        y_true, y_pred = self._prepare_masks(gt, prediction)
        s = float(jaccard_score(y_true, y_pred, average="binary"))
        return MetricOutputModel(self.name, s, Stats(doc="IoU = |X∩Y|/|X∪Y|"))

    def _prepare_masks(self, gt: DatasetModel, predictions: List[COCOAnnotation]) -> Tuple[np.ndarray, np.ndarray]:
        gt_by_file: Dict[str, List[COCOAnnotation]] = {}; pr_by_file: Dict[str, List[COCOAnnotation]] = {}
        for a in (gt.data_points or []):
            fn = str(getattr(a, 'file_name', None) or 'f'); gt_by_file.setdefault(fn, []).append(a)
        for a in (predictions or []):
            fn = str(getattr(a, 'file_name', None) or 'f'); pr_by_file.setdefault(fn, []).append(a)
        files = sorted(set(gt_by_file) | set(pr_by_file))
        H = int(getattr(gt, 'image_height', 0) or 0); W = int(getattr(gt, 'image_width', 0) or 0)
        if H <= 0 or W <= 0:
            for arr in (gt.data_points or []) + (predictions or []):
                m = getattr(arr, 'mask', None)
                if isinstance(m, np.ndarray) and m.ndim >= 2:
                    H, W = int(m.shape[0]), int(m.shape[1]); break
        if H <= 0 or W <= 0: return np.zeros((1,), np.uint8), np.zeros((1,), np.uint8)
        g_all: List[np.ndarray] = []; p_all: List[np.ndarray] = []
        for fn in files:
            g = np.zeros((H, W), np.uint8); p = np.zeros((H, W), np.uint8)
            for a in gt_by_file.get(fn, []):
                m = getattr(a, 'mask', None)
                if isinstance(m, np.ndarray) and m.shape[:2] == (H, W): g |= (m > 0).astype(np.uint8)
            for a in pr_by_file.get(fn, []):
                m = getattr(a, 'mask', None)
                if isinstance(m, np.ndarray) and m.shape[:2] == (H, W): p |= (m > 0).astype(np.uint8)
            g_all.append(g.reshape(-1)); p_all.append(p.reshape(-1))
        y_true = np.concatenate(g_all, 0) if g_all else np.zeros((1,), np.uint8)
        y_pred = np.concatenate(p_all, 0) if p_all else np.zeros((1,), np.uint8)
        return y_true, y_pred
class DiceCoefficient(Metric):
    name = "dice"
    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs) -> MetricOutputModel:
        y_true, y_pred = MeanIntersectionOverUnion()._prepare_masks(gt, prediction)
        s = float(f1_score(y_true, y_pred, average="binary"))
        return MetricOutputModel(self.name, s, Stats(doc="Dice = 2PR/(P+R)"))
class ClassificationReportMetric(Metric):
    name = "classification_report"
    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs) -> MetricOutputModel:
        gt_by_file: Dict[str, List[Union[int, str]]] = {}; pred_by_file: Dict[str, List[Union[int, str]]] = {}
        for a in (gt.data_points or []):
            fn = getattr(a, 'file_name', None); lab = getattr(a, 'label', None)
            if fn is None or lab is None: continue
            gt_by_file.setdefault(str(fn), []).append(lab)
        for a in (prediction or []):
            fn = getattr(a, 'file_name', None); lab = getattr(a, 'label', None)
            if fn is None or lab is None: continue
            pred_by_file.setdefault(str(fn), []).append(lab)
        files = sorted(set(gt_by_file) | set(pred_by_file))
        if not files:
            return MetricOutputModel(self.name, 0.0, ClsReportStats(doc="no files to compare"))

        def maj(labels: List[Union[int, str]]) -> str:
            if not labels: return "none"
            c: Dict[str, int] = {}
            for lb in labels:
                s = str(lb); c[s] = c.get(s, 0) + 1
            return max(c.items(), key=lambda x: x[1])[0]

        y_t: List[str] = []; y_p: List[str] = []
        for fn in files:
            g = maj(gt_by_file.get(fn, [])); p = maj(pred_by_file.get(fn, []))
            if g == "none" or p == "none": continue
            y_t.append(g); y_p.append(p)
        if not y_t or not y_p:
            return MetricOutputModel(self.name, 0.0, ClsReportStats(doc="no valid labels after filtering"))
        rep_dict = classification_report(y_t, y_p, output_dict=True, zero_division=0)
        rep_text = classification_report(y_t, y_p, output_dict=False, zero_division=0)
        acc = float(sum(1 for a, b in zip(y_t, y_p) if a == b)) / float(len(y_t))
        stats = ClsReportStats(num_files=len(files), dict=rep_dict, text=rep_text, labels=sorted(list(set(y_t) | set(y_p))), doc="")
        return MetricOutputModel(self.name, acc, stats)
