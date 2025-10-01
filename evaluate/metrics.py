import abc
import warnings
import numpy as np
import torch
from torchvision.ops import box_iou
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union, Tuple, Set
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score
from torchmetrics import JaccardIndex, F1Score
from evaluate.DatasetModel import DatasetModel
from evaluate.COCOAnnotations import COCOAnnotation

class PredItem(TypedDict):
    file: str
    box: List[float]
    score: float
def _safe_float(x: Optional[Union[float, int, str, np.floating, np.ndarray]], default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)  
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default
def _safe_box(bb: Optional[Union[List[float], Tuple[float, float, float, float]]]) -> Optional[List[float]]:
    if not isinstance(bb, (list, tuple)) or len(bb) != 4:
        return None
    x = _safe_float(bb[0])
    y = _safe_float(bb[1])
    w = _safe_float(bb[2])
    h = _safe_float(bb[3])
    return [x, y, w, h]
@dataclass
class MetricOutputModel:
    metric_name: str
    score: float
    stats: Dict[str, Any]


class Metric(abc.ABC):
    name: str = "metric"
    
    @abc.abstractmethod
    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs) -> MetricOutputModel:
        pass

class MeanAveragePrecision(Metric):
    name = "mAP"

    def __init__(self, iou_thresholds: Optional[List[float]] = None) -> None:
        self.iou_thresholds = iou_thresholds or np.arange(0.5, 0.95 + 1e-9, 0.05).tolist()

    @staticmethod
    def _xywh_to_xyxy(b: List[float]) -> List[float]:
        x, y, w, h = b
        return [x, y, x + w, y + h]

    @staticmethod
    def _ap_pr(y_true: List[int], y_scores: List[float]) -> Tuple[float, List[float], List[float]]:
        # sanitize inputs to avoid None and non-finite values
        if not y_true:
            return 0.0, [1.0], [0.0]
        try:
            y_true_san = [1 if int(v) == 1 else 0 for v in y_true]
            y_scores_san = [_safe_float(v) for v in y_scores]
            pos = int(np.sum(y_true_san))
            neg = len(y_true_san) - pos
            # Handle degenerate cases explicitly to avoid sklearn warnings
            if pos == 0:
                return 0.0, [1.0], [0.0]
            if neg == 0:
                return 1.0, [1.0], [1.0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                ap = _safe_float(average_precision_score(y_true_san, y_scores_san))
                p, r, _ = precision_recall_curve(y_true_san, y_scores_san)
            return ap, [_safe_float(x) for x in p], [_safe_float(x) for x in r]
        except Exception:
            return 0.0, [1.0], [0.0]

    def _group_input(self, gt: DatasetModel, prediction: List[COCOAnnotation]) -> Tuple[Dict[str, List[COCOAnnotation]], Dict[str, List[COCOAnnotation]], List[str], List[str]]:
        gt_by_file: Dict[str, List[COCOAnnotation]] = {}
        pr_by_file: Dict[str, List[COCOAnnotation]] = {}
        for a in (gt.data_points or []):
            fn = getattr(a, 'file_name', None)
            if fn is not None:
                gt_by_file.setdefault(str(fn), []).append(a)
        for a in (prediction or []):
            fn = getattr(a, 'file_name', None)
            if fn is not None:
                pr_by_file.setdefault(str(fn), []).append(a)
        files = sorted(set(gt_by_file) | set(pr_by_file))
        labels: List[str] = []
        for arr in (gt.data_points or []):
            lab = getattr(arr, 'label', None)
            if lab is not None:
                labels.append(str(lab))
        for arr in (prediction or []):
            lab = getattr(arr, 'label', None)
            if lab is not None:
                labels.append(str(lab))
        uniq_labels = sorted({str(label) for label in labels})
        return gt_by_file, pr_by_file, files, uniq_labels

    def _prepare_boxes(self, gt_by_file: Dict[str, List[COCOAnnotation]], pr_by_file: Dict[str, List[COCOAnnotation]], files: List[str], uniq_labels: List[str]) -> Tuple[Dict[str, Dict[str, List[List[float]]]], Dict[str, List[PredItem]]]:
        gt_boxes_by_label_file: Dict[str, Dict[str, List[List[float]]]] = {lab: {} for lab in uniq_labels}
        for fn in files:
            for a in gt_by_file.get(fn, []):
                lab, bb = str(getattr(a, 'label', '')), getattr(a, 'bbox', None)
                safe_bb = _safe_box(bb)
                if lab in gt_boxes_by_label_file and safe_bb is not None:
                    gt_boxes_by_label_file[lab].setdefault(fn, []).append(
                        self._xywh_to_xyxy([safe_bb[0], safe_bb[1], safe_bb[2], safe_bb[3]])
                    )
        preds_by_label: Dict[str, List[PredItem]] = {lab: [] for lab in uniq_labels}
        for fn in files:
            for p in pr_by_file.get(fn, []):
                lab, bb = str(getattr(p, 'label', '')), getattr(p, 'bbox', None)
                score = _safe_float(getattr(p, 'score', getattr(p, 'confidence', 1.0)))
                safe_bb = _safe_box(bb)
                if lab in preds_by_label and safe_bb is not None:
                    preds_by_label[lab].append({
                        'file': fn,
                        'box': self._xywh_to_xyxy([safe_bb[0], safe_bb[1], safe_bb[2], safe_bb[3]]),
                        'score': score,
                    })
        for lab in uniq_labels:
            preds_by_label[lab].sort(key=lambda d: d['score'], reverse=True)
        return gt_boxes_by_label_file, preds_by_label

    def _match_make_targets(self, preds_list: List[PredItem], gt_per_file: Dict[str, List[List[float]]], thr: float) -> Tuple[List[int], List[float]]:
        matched_by_file: Dict[str, Set[int]] = {fn: set() for fn in gt_per_file.keys()}
        y_true: List[int] = []
        y_scores: List[float] = []
        for p in preds_list:
            fn, box_pred, score = p['file'], p['box'], _safe_float(p['score'])
            gts = gt_per_file.get(fn, [])
            best_iou, best_idx = 0.0, None
            if gts:
                pred_t = torch.tensor([box_pred], dtype=torch.float32)
                gts_t = torch.tensor(gts, dtype=torch.float32)
                ious = box_iou(pred_t, gts_t).squeeze(0).tolist()
                for gi, iou_val in enumerate(ious):
                    if gi in matched_by_file.get(fn, set()):  # уже сматчен
                        continue
                    if iou_val > best_iou:
                        best_iou, best_idx = iou_val, gi
            if best_iou >= _safe_float(thr) and best_idx is not None:
                y_true.append(1)
                y_scores.append(score)
                matched_by_file.setdefault(fn, set()).add(best_idx)
            else:
                y_true.append(0)
                y_scores.append(score)
        return y_true, y_scores

    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs) -> MetricOutputModel:
        gt_by_file, pr_by_file, files, uniq_labels = self._group_input(gt, prediction)
        if not files:
            return MetricOutputModel(metric_name=self.name, score=0.0, stats={"error": "empty GT or predictions"})

        categories = [label for label in uniq_labels]
        gt_boxes_by_label_file, preds_by_label = self._prepare_boxes(gt_by_file, pr_by_file, files, uniq_labels)

        ap_per_label_per_thr = {lab: [0.0] * len(self.iou_thresholds) for lab in uniq_labels}
        pr_curves_per_thr: Dict[float, Dict[str, Dict[str, List[float]]]] = {}
        y_store: Dict[float, Dict[str, Tuple[List[int], List[float]]]] = {}

        for t_idx, thr in enumerate(self.iou_thresholds):
            pr_curves_per_thr[thr] = {}
            y_store[thr] = {}
            for lab in uniq_labels:
                preds_list = preds_by_label.get(lab, [])
                gt_per_file = gt_boxes_by_label_file.get(lab, {}) or {}
                yt, ys = self._match_make_targets(preds_list, gt_per_file, thr)
                y_store[thr][lab] = (yt, ys)
                ap, p, r = self._ap_pr(yt, ys)
                ap_per_label_per_thr[lab][t_idx] = _safe_float(ap)
                pr_curves_per_thr[thr][lab] = {'precision': p, 'recall': r}
        recall_grid = np.linspace(0.0, 1.0, 101)
        pr_micro, pr_macro = {}, {}
        ap_iou_micro: List[float] = []
        for thr in self.iou_thresholds:
            thr_key = f"{_safe_float(thr):.2f}"
            all_y_true: List[int] = []
            all_y_scores: List[float] = []
            for lab, (yt, ys) in (y_store.get(thr, {}) or {}).items():
                if yt:
                    all_y_true.extend(yt)
                    all_y_scores.extend(ys)
            if all_y_true:
                ap_micro_val, p_micro, r_micro = self._ap_pr(all_y_true, all_y_scores)
            else:
                ap_micro_val, p_micro, r_micro = 0.0, [1.0], [0.0]
            ap_iou_micro.append(_safe_float(ap_micro_val))
            pr_micro[thr_key] = {'precision': [_safe_float(x) for x in p_micro], 'recall': [_safe_float(x) for x in r_micro]}

            # макро: усредняем precision после интерполяции на общую сетку recall
            macro_stack: List[np.ndarray] = []
            for lab, pr in (pr_curves_per_thr.get(thr, {}) or {}).items():
                r_arr = np.asarray(pr.get('recall') or [])
                p_arr = np.asarray(pr.get('precision') or [])
                if r_arr.size > 1 and p_arr.size > 1:
                    order = np.argsort(r_arr)
                    p_interp = np.interp(recall_grid, r_arr[order], p_arr[order],
                                         left=p_arr[order][0], right=p_arr[order][-1])
                    macro_stack.append(p_interp)
            p_macro = np.mean(np.stack(macro_stack, axis=0), axis=0) if macro_stack else np.zeros_like(recall_grid)
            pr_macro[thr_key] = {'precision': [_safe_float(x) for x in p_macro],
                                 'recall': [_safe_float(x) for x in recall_grid]}
        per_class_ap_avg: List[float] = []
        for lab in uniq_labels:
            ap_values = ap_per_label_per_thr[lab]
            if ap_values and len(ap_values) > 0:
                mean_val = np.mean(ap_values)
                if np.isfinite(mean_val):
                    per_class_ap_avg.append(_safe_float(mean_val))
                else:
                    per_class_ap_avg.append(0.0)
            else:
                per_class_ap_avg.append(0.0)
        
        macro_map_iou: List[float] = []
        for i in range(len(self.iou_thresholds)):
            if uniq_labels:
                ap_values = [ap_per_label_per_thr[lab][i] for lab in uniq_labels if i < len(ap_per_label_per_thr[lab])]
                if ap_values:
                    mean_val = np.mean(ap_values)
                    if np.isfinite(mean_val):
                        macro_map_iou.append(_safe_float(mean_val))
                    else:
                        macro_map_iou.append(0.0)
                else:
                    macro_map_iou.append(0.0)
            else:
                macro_map_iou.append(0.0)
        if macro_map_iou:
            overall_mean = np.mean(macro_map_iou)
            overall_map = _safe_float(overall_mean) if np.isfinite(overall_mean) else 0.0
        else:
            overall_map = 0.0
        def _thr_val(target: float) -> float:
            for i, thr in enumerate(self.iou_thresholds):
                if abs(_safe_float(thr) - _safe_float(target)) < 1e-6:
                    if i < len(macro_map_iou) and macro_map_iou[i] is not None:
                        val = macro_map_iou[i]
                        return _safe_float(val) if np.isfinite(_safe_float(val)) else 0.0
                    return 0.0
            return 0.0
        map50, map75 = _thr_val(0.5), _thr_val(0.75)
        iou_thresholds_list = [_safe_float(t) for t in self.iou_thresholds]

        gt_counts = [sum(len(v) for v in (gt_boxes_by_label_file.get(lab, {}) or {}).values()) for lab in uniq_labels]
        pred_counts = [len(preds_by_label.get(lab, []) or []) for lab in uniq_labels]
        stats = {
            'categories': categories,
            'gt_counts': gt_counts,
            'pred_counts': pred_counts,
            'map_per_class': per_class_ap_avg,
            'per_class_ap': per_class_ap_avg, 
            'ap_iou_macro': macro_map_iou,
            'ap_iou_micro': ap_iou_micro,
            'iou_thresholds': iou_thresholds_list,
            'map': overall_map,
            'map_50': map50, 'map_75': map75,
            'mAP@0.5': map50, 'mAP@0.75': map75,
            'pr_macro': pr_macro,
            'pr_micro': pr_micro,
            'pr_curves_per_threshold': pr_curves_per_thr,
            'doc': (self.__class__.__doc__ or '').strip(),
        }
        return MetricOutputModel(metric_name=self.name, score=overall_map, stats=stats)

class MeanIntersectionOverUnion(Metric):
    name = "mIoU"
    
    def __init__(self, iou_threshold: float = 0.5, use_torchmetrics: bool = True):
        self.iou_threshold = iou_threshold
        self.use_torchmetrics = True

    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs) -> MetricOutputModel:
        y_true, y_pred = self._prepare_masks(gt, prediction)
        tm = JaccardIndex(task="binary")
        score = float(tm(torch.from_numpy(y_pred).int(), torch.from_numpy(y_true).int()).item())
        return MetricOutputModel(
            metric_name=self.name,
            score=score,
            stats={
                'doc': (self.__class__.__doc__ or '').strip(),
                'formula': 'IoU = area(intersection) / area(union); mIoU = mean IoU over classes'
            }
        )

    def _prepare_masks(self, gt: DatasetModel, predictions: List[COCOAnnotation]) -> Tuple[np.ndarray, np.ndarray]:
        gt_by_file: Dict[str, List[COCOAnnotation]] = {}
        pr_by_file: Dict[str, List[COCOAnnotation]] = {}
        for ann in (gt.data_points or []):
            fn = str(getattr(ann, 'file_name', None) or 'unknown.jpg')
            gt_by_file.setdefault(fn, []).append(ann)
        for ann in (predictions or []):
            fn = str(getattr(ann, 'file_name', None) or 'unknown.jpg')
            pr_by_file.setdefault(fn, []).append(ann)

        files = sorted(set(gt_by_file.keys()) | set(pr_by_file.keys()))
        gt_all: List[np.ndarray] = []
        pr_all: List[np.ndarray] = []
        for fn in files:
            H = W = 0
            src = gt_by_file.get(fn) or pr_by_file.get(fn) or []
            if src:
                H = int(getattr(src[0], 'height', getattr(src[0], 'image_size', (0, 0))[1] if hasattr(src[0], 'image_size') else 0) or 0)
                W = int(getattr(src[0], 'width', getattr(src[0], 'image_size', (0, 0))[0] if hasattr(src[0], 'image_size') else 0) or 0)
            if H <= 0 or W <= 0:
                continue
            gt_mask = np.zeros((H, W), dtype=np.uint8)
            pr_mask = np.zeros((H, W), dtype=np.uint8)
            for a in gt_by_file.get(fn, []):
                m = getattr(a, 'mask', None)
                if isinstance(m, np.ndarray) and m.shape[:2] == (H, W):
                    gt_mask |= (m > 0).astype(np.uint8)
            for a in pr_by_file.get(fn, []):
                m = getattr(a, 'mask', None)
                if isinstance(m, np.ndarray) and m.shape[:2] == (H, W):
                    pr_mask |= (m > 0).astype(np.uint8)
            gt_all.append(gt_mask)
            pr_all.append(pr_mask)

        if not gt_all:
            return np.zeros((1,), dtype=np.uint8), np.zeros((1,), dtype=np.uint8)
        gt_stack = np.concatenate([m.ravel() for m in gt_all], axis=0)
        pr_stack = np.concatenate([m.ravel() for m in pr_all], axis=0)
        return gt_stack, pr_stack

class DiceCoefficient(Metric):
    name = "dice"
    
    def __init__(self, use_torchmetrics: bool = True):
        self.use_torchmetrics = True
        self.metric = F1Score(task="binary")

    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs) -> MetricOutputModel:
        try:
            gt_binary, pred_binary = self._prepare_binary_data(gt, prediction)
            dice_score = float(self.metric(torch.from_numpy(pred_binary).int(), torch.from_numpy(gt_binary).int()).item())
            return MetricOutputModel(
                metric_name=self.name,
                score=dice_score,
                stats={
                    'doc': (self.__class__.__doc__ or '').strip(),
                    'formula': 'Dice = 2TP / (2TP + FP + FN)'
                }
            )
        except Exception as e:
            return MetricOutputModel(metric_name=self.name, score=0.0, stats={'error': str(e), 'fallback': True})
    
    def _prepare_binary_data(self, gt: DatasetModel, predictions: List[COCOAnnotation]) -> Tuple[np.ndarray, np.ndarray]:
        gt_by_file: Dict[str, List[COCOAnnotation]] = {}
        pr_by_file: Dict[str, List[COCOAnnotation]] = {}
        for ann in (gt.data_points or []):
            fn = str(getattr(ann, 'file_name', None) or 'unknown.jpg')
            gt_by_file.setdefault(fn, []).append(ann)
        for ann in (predictions or []):
            fn = str(getattr(ann, 'file_name', None) or 'unknown.jpg')
            pr_by_file.setdefault(fn, []).append(ann)
        files = sorted(set(gt_by_file.keys()) | set(pr_by_file.keys()))
        gt_all: List[np.ndarray] = []
        pr_all: List[np.ndarray] = []
        for fn in files:
            H = W = 0
            src = gt_by_file.get(fn) or pr_by_file.get(fn) or []
            if src:
                H = int(getattr(src[0], 'height', getattr(src[0], 'image_size', (0, 0))[1] if hasattr(src[0], 'image_size') else 0) or 0)
                W = int(getattr(src[0], 'width', getattr(src[0], 'image_size', (0, 0))[0] if hasattr(src[0], 'image_size') else 0) or 0)
            if H <= 0 or W <= 0:
                continue
            gt_mask = np.zeros((H, W), dtype=np.uint8)
            pr_mask = np.zeros((H, W), dtype=np.uint8)
            for a in gt_by_file.get(fn, []):
                m = getattr(a, 'mask', None)
                if isinstance(m, np.ndarray) and m.shape[:2] == (H, W):
                    gt_mask |= (m > 0).astype(np.uint8)
            for a in pr_by_file.get(fn, []):
                m = getattr(a, 'mask', None)
                if isinstance(m, np.ndarray) and m.shape[:2] == (H, W):
                    pr_mask |= (m > 0).astype(np.uint8)
            gt_all.append(gt_mask)
            pr_all.append(pr_mask)

        if not gt_all:
            return np.zeros((1,), dtype=np.uint8), np.zeros((1,), dtype=np.uint8)
        gt_stack = np.concatenate([m.ravel() for m in gt_all], axis=0)
        pr_stack = np.concatenate([m.ravel() for m in pr_all], axis=0)
        return gt_stack, pr_stack

class ClassificationReportMetric(Metric):
    name = "classification_report"
    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs) -> MetricOutputModel:
        try:
            gt_by_file: Dict[str, List[Union[int, str]]] = {}
            pred_by_file: Dict[str, List[Union[int, str]]] = {}
            for ann in (gt.data_points or []):
                fn = getattr(ann, 'file_name', None)
                lab = getattr(ann, 'label', None)
                if fn is None or lab is None:
                    continue
                gt_by_file.setdefault(str(fn), []).append(lab)
            for ann in (prediction or []):
                fn = getattr(ann, 'file_name', None)
                lab = getattr(ann, 'label', None)
                if fn is None or lab is None:
                    continue
                pred_by_file.setdefault(str(fn), []).append(lab)
            files = sorted(set(gt_by_file.keys()) | set(pred_by_file.keys()))
            if not files:
                return MetricOutputModel(metric_name=self.name, score=0.0, stats={"error": "no files to compare", "dict": {}, "text": ""})
            def majority_label(labels: List[Union[int, str]]) -> str:
                if not labels:
                    return "none"
                counts: Dict[str, int] = {}
                for label in labels:
                    s = str(label)
                    counts[s] = counts.get(s, 0) + 1
                return max(counts.items(), key=lambda x: x[1])[0]

            y_true: List[str] = []
            y_pred: List[str] = []
            for fn in files:
                gt_labels = gt_by_file.get(fn, [])
                pr_labels = pred_by_file.get(fn, [])
                if not gt_labels and not pr_labels:
                    continue
                gt_lab = majority_label(gt_labels)
                pr_lab = majority_label(pr_labels)
                if gt_lab == "none" or pr_lab == "none":
                    continue
                    
                y_true.append(gt_lab)
                y_pred.append(pr_lab)
            if not y_true or not y_pred:
                return MetricOutputModel(metric_name=self.name, score=0.0, stats={"error": "no valid labels after filtering", "dict": {}, "text": ""})

            rep_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            rep_text = classification_report(y_true, y_pred, output_dict=False, zero_division=0)
            correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
            acc = float(correct) / float(len(y_true)) if y_true else 0.0
            stats = {
                "num_files": len(files),
                "dict": rep_dict,
                "text": rep_text,
                "labels": sorted(list(set(y_true) | set(y_pred))),
                'doc': (self.__class__.__doc__ or '').strip(),
            }
            return MetricOutputModel(metric_name=self.name, score=float(acc), stats=stats)
        except Exception as e:
            return MetricOutputModel(metric_name=self.name, score=0.0, stats={"error": str(e), "fallback": True})