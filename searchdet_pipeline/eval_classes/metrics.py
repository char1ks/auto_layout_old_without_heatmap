import abc, numpy as np, warnings, tempfile, json, os
import torch
from torchvision.ops import box_iou
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union, Tuple
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score
from torchmetrics import JaccardIndex, F1Score

from searchdet_pipeline.eval_classes.DatasetModel import DatasetModel
from searchdet_pipeline.eval_classes.COCOAnnotations import COCOAnnotation


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

    def __init__(self, iou_thresholds: Optional[List[float]] = None):
        self.iou_thresholds = iou_thresholds or np.arange(0.5, 0.95 + 1e-9, 0.05).tolist()

    @staticmethod
    def _calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
        b1 = torch.tensor([MeanAveragePrecision._xywh_to_xyxy(bbox1)], dtype=torch.float32)
        b2 = torch.tensor([MeanAveragePrecision._xywh_to_xyxy(bbox2)], dtype=torch.float32)
        return float(box_iou(b1, b2)[0, 0].item())

    @staticmethod
    def _xywh_to_xyxy(b: List[float]) -> List[float]:
        x, y, w, h = b
        return [x, y, x + w, y + h]

    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs) -> MetricOutputModel:
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

        files = sorted(set(gt_by_file.keys()) | set(pr_by_file.keys()))
        if not files:
            return MetricOutputModel(metric_name=self.name, score=0.0, stats={"error": "empty GT or predictions"})

        labels_set: List[Union[str, int]] = []
        for a in (gt.data_points or []):
            if getattr(a, 'label', None) is not None:
                labels_set.append(a.label)
        for a in (prediction or []):
            if getattr(a, 'label', None) is not None:
                labels_set.append(a.label)
        uniq_labels = sorted({str(l) for l in labels_set})
        categories = [l for l in uniq_labels]

        gt_boxes_by_label_file: Dict[str, Dict[str, List[List[float]]]] = {lab: {} for lab in uniq_labels}
        for fn in files:
            for a in gt_by_file.get(fn, []):
                lab = str(getattr(a, 'label', ''))
                bb = getattr(a, 'bbox', None)
                if lab in gt_boxes_by_label_file and isinstance(bb, (list, tuple)) and len(bb) == 4:
                    gt_boxes_by_label_file[lab].setdefault(fn, []).append(
                        self._xywh_to_xyxy([float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])])
                    )

        preds_by_label: Dict[str, List[Dict[str, Any]]] = {lab: [] for lab in uniq_labels}
        for fn in files:
            for p in pr_by_file.get(fn, []):
                lab = str(getattr(p, 'label', ''))
                bb = getattr(p, 'bbox', None)
                score = float(getattr(p, 'score', getattr(p, 'confidence', 1.0)))
                if lab in preds_by_label and isinstance(bb, (list, tuple)) and len(bb) == 4:
                    preds_by_label[lab].append({
                        'file': fn,
                        'box': self._xywh_to_xyxy([float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]),
                        'score': score,
                    })

        for lab in uniq_labels:
            preds_by_label[lab].sort(key=lambda d: d['score'], reverse=True)

        ap_per_label_per_thr: Dict[str, List[float]] = {lab: [0.0] * len(self.iou_thresholds) for lab in uniq_labels}
        pr_curves_per_thr: Dict[float, Dict[str, Dict[str, List[float]]]] = {}
        y_store: Dict[float, Dict[str, Tuple[List[int], List[float]]]] = {}

        for t_idx, thr in enumerate(self.iou_thresholds):
            pr_curves_per_thr[thr] = {}
            y_store[thr] = {}
            for lab in uniq_labels:
                gt_per_file = gt_boxes_by_label_file.get(lab, {})
                n_gt = sum(len(lst) for lst in gt_per_file.values())
                preds_list = preds_by_label.get(lab, [])

                matched_by_file: Dict[str, set] = {fn: set() for fn in gt_per_file.keys()}

                y_true: List[int] = []
                y_scores: List[float] = []

                for p in preds_list:
                    fn = p['file']
                    box_pred = p['box']
                    score = float(p['score'])
                    gts = gt_per_file.get(fn, [])

                    best_iou = 0.0
                    best_idx = None
                    if gts:
                        pred_t = torch.tensor([box_pred], dtype=torch.float32)
                        gts_t = torch.tensor(gts, dtype=torch.float32)
                        ious = box_iou(pred_t, gts_t).squeeze(0).tolist()
                        for gi, iou_val in enumerate(ious):
                            if gi in matched_by_file.get(fn, set()):
                                continue
                            if iou_val > best_iou:
                                best_iou = iou_val
                                best_idx = gi

                    if best_iou >= float(thr) and best_idx is not None:
                        y_true.append(1)
                        y_scores.append(score)
                        matched_by_file.setdefault(fn, set()).add(best_idx)
                    else:
                        y_true.append(0)
                        y_scores.append(score)

                ap = 0.0
                prec_list: List[float] = [1.0]
                rec_list: List[float] = [0.0]
                if len(y_true) > 0:
                    try:
                        ap = float(average_precision_score(y_true, y_scores))
                        precision, recall, _ = precision_recall_curve(y_true, y_scores)
                        prec_list = [float(x) for x in precision]
                        rec_list = [float(x) for x in recall]
                    except Exception:
                        ap = 0.0
                        prec_list = [1.0]
                        rec_list = [0.0]
                if n_gt > 0 and len(y_true) == 0:
                    ap = 0.0

                ap_per_label_per_thr[lab][t_idx] = max(0.0, min(1.0, ap))
                pr_curves_per_thr[thr][lab] = {
                    'precision': prec_list,
                    'recall': rec_list,
                }
                y_store[thr][lab] = (y_true.copy(), y_scores.copy())
        iou_thresholds_list = [float(t) for t in self.iou_thresholds]
        ap_iou_micro: List[float] = []
        pr_macro: Dict[str, Dict[str, List[float]]] = {}
        pr_micro: Dict[str, Dict[str, List[float]]] = {}
        recall_grid = np.linspace(0.0, 1.0, 101)
        for thr in self.iou_thresholds:
            thr_key = f"{float(thr):.2f}"
            all_y_true: List[int] = []
            all_y_scores: List[float] = []
            for lab, pair in (y_store.get(thr, {}) or {}).items():
                yt, ys = pair
                if yt:
                    all_y_true.extend(list(yt))
                    all_y_scores.extend(list(ys))
            if len(all_y_true) > 0:
                try:
                    ap_micro_val = float(average_precision_score(all_y_true, all_y_scores))
                    p_micro, r_micro, _ = precision_recall_curve(all_y_true, all_y_scores)
                except Exception:
                    ap_micro_val = 0.0
                    p_micro, r_micro = np.array([1.0, 1.0]), np.array([0.0, 1.0])
            else:
                ap_micro_val = 0.0
                p_micro, r_micro = np.array([1.0, 1.0]), np.array([0.0, 1.0])
            ap_iou_micro.append(max(0.0, min(1.0, ap_micro_val)))
            pr_micro[thr_key] = {
                'precision': [float(x) for x in p_micro],
                'recall': [float(x) for x in r_micro],
            }
            macro_stack: List[np.ndarray] = []
            for lab, pr in (pr_curves_per_thr.get(thr, {}) or {}).items():
                r = np.array(pr.get('recall') or [])
                p = np.array(pr.get('precision') or [])
                if r.size > 1 and p.size > 1:
                    order = np.argsort(r)
                    r_sorted = r[order]
                    p_sorted = p[order]
                    p_interp = np.interp(recall_grid, r_sorted, p_sorted, left=p_sorted[0], right=p_sorted[-1])
                    macro_stack.append(p_interp)
            if macro_stack:
                p_macro = np.mean(np.stack(macro_stack, axis=0), axis=0)
            else:
                p_macro = np.zeros_like(recall_grid)
            pr_macro[thr_key] = {
                'precision': [float(x) for x in p_macro],
                'recall': [float(x) for x in recall_grid],
            }

        macro_map_iou = []
        for t_idx, thr in enumerate(self.iou_thresholds):
            vals = [ap_per_label_per_thr[lab][t_idx] for lab in uniq_labels]
            macro_map_iou.append(float(np.mean(vals)) if len(vals) > 0 else 0.0)
        overall_map = float(np.mean(macro_map_iou)) if len(macro_map_iou) > 0 else 0.0
        def _get_thr_value(target: float) -> float:
            for t_idx, thr in enumerate(self.iou_thresholds):
                if abs(float(thr) - target) < 1e-6:
                    return macro_map_iou[t_idx]
            return 0.0
        map50 = _get_thr_value(0.5)
        map75 = _get_thr_value(0.75)
        per_class_ap_avg = [float(np.mean(ap_per_label_per_thr[lab])) for lab in uniq_labels]

        stats = {
            'categories': categories,
            'map_per_class': per_class_ap_avg,
            'per_class_ap': per_class_ap_avg,
            'ap_iou_macro': macro_map_iou,
            'ap_iou_micro': ap_iou_micro,
            'iou_thresholds': iou_thresholds_list,
            'map': overall_map,
            'map_50': map50,
            'map_75': map75,
            'mAP@0.5': map50,
            'mAP@0.75': map75,
            'pr_macro': pr_macro,
            'pr_micro': pr_micro,
            'pr_curves_per_threshold': pr_curves_per_thr,
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
        return MetricOutputModel(metric_name=self.name, score=score, stats={})

    def _prepare_masks(self, gt: DatasetModel, predictions: List[COCOAnnotation]):
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
            return MetricOutputModel(metric_name=self.name, score=dice_score, stats={})
        except Exception as e:
            return MetricOutputModel(metric_name=self.name, score=0.0, stats={'error': str(e), 'fallback': True})
    
    def _prepare_binary_data(self, gt: DatasetModel, predictions: List[COCOAnnotation]):
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
                for l in labels:
                    s = str(l)
                    counts[s] = counts.get(s, 0) + 1
                return max(counts.items(), key=lambda x: x[1])[0]

            y_true: List[str] = []
            y_pred: List[str] = []
            for fn in files:
                gt_lab = majority_label(gt_by_file.get(fn, []))
                pr_lab = majority_label(pred_by_file.get(fn, []))
                y_true.append(gt_lab)
                y_pred.append(pr_lab)

            rep_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            rep_text = classification_report(y_true, y_pred, output_dict=False, zero_division=0)
            correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
            acc = float(correct) / float(len(y_true)) if y_true else 0.0

            stats = {
                "num_files": len(files),
                "dict": rep_dict,
                "text": rep_text,
                "labels": sorted(list(set(y_true) | set(y_pred))),
            }
            return MetricOutputModel(metric_name=self.name, score=float(acc), stats=stats)
        except Exception as e:
            return MetricOutputModel(metric_name=self.name, score=0.0, stats={"error": str(e), "fallback": True})