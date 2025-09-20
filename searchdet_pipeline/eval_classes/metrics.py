import abc, numpy as np, warnings, tempfile, json, os
import torch
from torchvision.ops import box_iou
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union, Tuple
from sklearn.metrics import classification_report
from torchmetrics.detection import MeanAveragePrecision as TMDetMAP
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
        try:
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
            label_to_int: Dict[str, int] = {lab: i for i, lab in enumerate(uniq_labels)}

            targets: List[Dict[str, torch.Tensor]] = []
            preds: List[Dict[str, torch.Tensor]] = []
            for fn in files:
                gts = gt_by_file.get(fn, [])
                prs = pr_by_file.get(fn, [])
                gt_boxes = []
                gt_labels = []
                for a in gts:
                    bb = getattr(a, 'bbox', None)
                    if isinstance(bb, (list, tuple)) and len(bb) == 4:
                        gt_boxes.append(self._xywh_to_xyxy([float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]))
                        gt_labels.append(label_to_int.get(str(getattr(a, 'label', "")), 0))
                pr_boxes = []
                pr_labels = []
                pr_scores = []
                for p in prs:
                    bb = getattr(p, 'bbox', None)
                    if isinstance(bb, (list, tuple)) and len(bb) == 4:
                        pr_boxes.append(self._xywh_to_xyxy([float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]))
                        pr_labels.append(label_to_int.get(str(getattr(p, 'label', "")), 0))
                        sc = getattr(p, 'score', getattr(p, 'confidence', 1.0))
                        pr_scores.append(float(sc))
                targets.append({
                    'boxes': torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.zeros((0, 4), dtype=torch.float32),
                    'labels': torch.tensor(gt_labels, dtype=torch.int64) if gt_labels else torch.zeros((0,), dtype=torch.int64),
                })
                preds.append({
                    'boxes': torch.tensor(pr_boxes, dtype=torch.float32) if pr_boxes else torch.zeros((0, 4), dtype=torch.float32),
                    'scores': torch.tensor(pr_scores, dtype=torch.float32) if pr_scores else torch.zeros((0,), dtype=torch.float32),
                    'labels': torch.tensor(pr_labels, dtype=torch.int64) if pr_labels else torch.zeros((0,), dtype=torch.int64),
                })

            tm = TMDetMAP(box_format='xyxy', iou_type='bbox', iou_thresholds=self.iou_thresholds, class_metrics=True)
            tm.update(preds, targets)
            res = tm.compute()

            def _to_float(val, default=0.0) -> float:
                try:
                    x = float(val.item() if hasattr(val, 'item') else val)
                except Exception:
                    return default
                if not np.isfinite(x) or x < 0:
                    return default
                return x

            def _to_builtin(obj):
                try:
                    import numpy as _np
                    import torch as _torch
                except Exception:
                    _np = None; _torch = None
                if _torch is not None and isinstance(obj, _torch.Tensor):
                    if obj.ndim == 0:
                        return _to_float(obj)
                    return obj.detach().cpu().tolist()
                if _np is not None and isinstance(obj, _np.ndarray):
                    if obj.ndim == 0:
                        return _to_float(obj)
                    return obj.tolist()
                if isinstance(obj, dict):
                    return {k: _to_builtin(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_to_builtin(v) for v in obj]
                try:
                    if isinstance(obj, (float, int, str)):
                        return obj
                    return _to_float(obj)
                except Exception:
                    return str(obj)

            score = _to_float(res.get('map', 0.0), 0.0)
            map50 = _to_float(res.get('map_50', 0.0), 0.0)
            map75 = _to_float(res.get('map_75', 0.0), 0.0)
            maps = {
                'mAP_small': _to_float(res.get('map_small', 0.0), 0.0),
                'mAP_medium': _to_float(res.get('map_medium', 0.0), 0.0),
                'mAP_large': _to_float(res.get('map_large', 0.0), 0.0),
            }
            precision = res.get('precision')
            recall = res.get('recall')

            # Дополнительные поля для графиков
            categories = [uniq_labels[i] for i in range(len(uniq_labels))]
            
            # Подсчет GT и pred объектов по классам
            gt_counts = [0] * len(uniq_labels)
            pred_counts = [0] * len(uniq_labels)
            for fn in files:
                for a in gt_by_file.get(fn, []):
                    lab_idx = label_to_int.get(str(getattr(a, 'label', "")), 0)
                    if 0 <= lab_idx < len(gt_counts):
                        gt_counts[lab_idx] += 1
                for a in pr_by_file.get(fn, []):
                    lab_idx = label_to_int.get(str(getattr(a, 'label', "")), 0)
                    if 0 <= lab_idx < len(pred_counts):
                        pred_counts[lab_idx] += 1

            # AP по IoU порогам (macro/micro)
            map_per_class = _to_builtin(res.get('map_per_class', []))
            if isinstance(map_per_class, list) and len(map_per_class) >= len(uniq_labels):
                per_class_ap = [_to_float(map_per_class[i], 0.0) for i in range(len(uniq_labels))]
            else:
                per_class_ap = [0.0] * len(uniq_labels)

            # AP vs IoU кривые (упрощенная версия)
            ap_iou_macro = []
            ap_iou_micro = []
            for iou_thresh in self.iou_thresholds:
                # Приблизительные значения на основе общего mAP
                if iou_thresh <= 0.5:
                    ap_macro = map50 * (iou_thresh / 0.5)
                    ap_micro = map50 * (iou_thresh / 0.5)
                elif iou_thresh <= 0.75:
                    ap_macro = map50 + (map75 - map50) * ((iou_thresh - 0.5) / 0.25)
                    ap_micro = map50 + (map75 - map50) * ((iou_thresh - 0.5) / 0.25)
                else:
                    ap_macro = map75 * (0.95 - iou_thresh) / (0.95 - 0.75)
                    ap_micro = map75 * (0.95 - iou_thresh) / (0.95 - 0.75)
                ap_iou_macro.append(max(0.0, ap_macro))
                ap_iou_micro.append(max(0.0, ap_micro))

            # PR кривые (упрощенная версия)
            pr_macro = {}
            pr_micro = {}
            if precision is not None and recall is not None:
                prec_list = _to_builtin(precision)
                rec_list = _to_builtin(recall)
                if isinstance(prec_list, list) and isinstance(rec_list, list):
                    # Для IoU 0.5 и 0.75
                    for iou_key, iou_val in [("0.50", 0.5), ("0.75", 0.75)]:
                        # Упрощенная PR кривая
                        if len(prec_list) > 0 and len(rec_list) > 0:
                            # Берем первый элемент если это многомерный массив
                            if isinstance(prec_list[0], list):
                                prec_curve = prec_list[0] if len(prec_list[0]) > 0 else [0.0]
                                rec_curve = rec_list[0] if len(rec_list[0]) > 0 else [0.0]
                            else:
                                prec_curve = prec_list[:10] if len(prec_list) > 10 else prec_list
                                rec_curve = rec_list[:10] if len(rec_list) > 10 else rec_list
                            
                            # Нормализуем длины
                            min_len = min(len(prec_curve), len(rec_curve))
                            if min_len > 0:
                                pr_macro[iou_key] = {
                                    "precision": prec_curve[:min_len],
                                    "recall": rec_curve[:min_len]
                                }
                                pr_micro[iou_key] = {
                                    "precision": prec_curve[:min_len],
                                    "recall": rec_curve[:min_len]
                                }

            stats = {
                'mAP': score,
                'mAP@0.5': map50,
                'mAP@0.75': map75,
                **maps,
                # convert potential tensors to python lists
                'precision': _to_builtin(precision) if precision is not None else None,
                'recall': _to_builtin(recall) if recall is not None else None,
                'classes': [k for k in range(len(uniq_labels))],
                'label_mapping': {v: k for k, v in label_to_int.items()},
                # Дополнительные поля для графиков
                'categories': categories,
                'gt_counts': gt_counts,
                'pred_counts': pred_counts,
                'per_class_ap': per_class_ap,
                'ap_iou_macro': ap_iou_macro,
                'ap_iou_micro': ap_iou_micro,
                'iou_thresholds': self.iou_thresholds,
                'pr_macro': pr_macro,
                'pr_micro': pr_micro,
            }
            return MetricOutputModel(metric_name=self.name, score=score, stats=stats)
        except Exception as e:
            return MetricOutputModel(metric_name=self.name, score=0.0, stats={'error': str(e), 'fallback': True})


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
            return np.zeros((1, 1), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8)
        gt_stack = np.concatenate([m.reshape(1, -1) for m in gt_all], axis=0)
        pr_stack = np.concatenate([m.reshape(1, -1) for m in pr_all], axis=0)
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
            return np.zeros((1, 1), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8)
        gt_stack = np.concatenate([m.reshape(1, -1) for m in gt_all], axis=0)
        pr_stack = np.concatenate([m.reshape(1, -1) for m in pr_all], axis=0)
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