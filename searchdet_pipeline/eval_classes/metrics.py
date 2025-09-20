#!/usr/bin/env python3
import abc, numpy as np, warnings, tempfile, json, os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
from sklearn.metrics import jaccard_score, f1_score, classification_report
import torch, torchmetrics

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
        """Вычисляет IoU между двумя bounding box'ами в формате [x, y, w, h]"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Координаты углов
        x1_max, y1_max = x1 + w1, y1 + h1
        x2_max, y2_max = x2 + w2, y2 + h2
        
        # Пересечение
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1_max, x2_max)
        inter_y2 = min(y1_max, y2_max)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    def _prepare_gt_data(self, gt: DatasetModel) -> Dict:
        """Подготавливает GT данные для расчета mAP"""
        images = {}
        annotations = []
        categories = set()
        
        for dp in gt.data_points:
            file_name = getattr(dp, 'file_name', None)
            if not file_name:
                continue
                
            # Добавляем изображение
            if file_name not in images:
                images[file_name] = {
                    'file_name': file_name,
                    'width': getattr(dp, 'width', 500),
                    'height': getattr(dp, 'height', 400)
                }
            
            # Добавляем аннотации
            for ann in getattr(dp, 'annotations', []):
                bbox = self._bbox_from_ann(ann)
                if bbox and len(bbox) == 4:
                    label = getattr(ann, 'label', getattr(dp, 'label', None))
                    if label:
                        categories.add(label)
                        annotations.append({
                            'file_name': file_name,
                            'bbox': bbox,
                            'label': label,
                            'area': bbox[2] * bbox[3]
                        })
        
        return {
            'images': list(images.values()),
            'annotations': annotations,
            'categories': sorted(list(categories))
        }

    def _prepare_prediction_data(self, predictions: List[COCOAnnotation], gt_data: Dict) -> List[Dict]:
        """Подготавливает данные предсказаний для расчета mAP"""
        pred_data = []
        gt_files = {img['file_name'] for img in gt_data['images']}
        gt_categories = set(gt_data['categories'])
        
        for pred in predictions:
            file_name = getattr(pred, 'file_name', None)
            label = getattr(pred, 'label', None)
            
            # Проверяем, что файл и класс есть в GT
            if file_name in gt_files and label in gt_categories:
                bbox = self._bbox_from_ann(pred)
                if bbox and len(bbox) == 4:
                    score = getattr(pred, 'score', getattr(pred, 'confidence', 1.0))
                    pred_data.append({
                        'file_name': file_name,
                        'bbox': bbox,
                        'label': label,
                        'score': float(score)
                    })
        
        return pred_data

    def _calculate_map(self, gt_data: Dict, pred_data: List[Dict]) -> Tuple[float, Dict]:
        """Вычисляет mAP и возвращает детальную статистику"""
        categories = gt_data['categories']
        iou_thresholds = self.iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
        # Группируем данные по классам
        gt_by_class = {}
        pred_by_class = {}
        
        for cat in categories:
            gt_by_class[cat] = [ann for ann in gt_data['annotations'] if ann['label'] == cat]
            pred_by_class[cat] = [pred for pred in pred_data if pred['label'] == cat]
        
        # Вычисляем AP для каждого класса и IoU порога
        ap_results = {}
        per_class_ap = []
        
        for cat in categories:
            gt_anns = gt_by_class[cat]
            pred_anns = sorted(pred_by_class[cat], key=lambda x: x['score'], reverse=True)
            
            if not gt_anns:
                per_class_ap.append(0.0)
                continue
            
            # Вычисляем AP для разных IoU порогов
            ap_scores = []
            for iou_thresh in iou_thresholds:
                ap = self._calculate_ap_for_class(gt_anns, pred_anns, iou_thresh)
                ap_scores.append(ap)
            
            class_ap = np.mean(ap_scores)
            per_class_ap.append(class_ap)
            ap_results[cat] = {
                'ap': class_ap,
                'ap_per_iou': ap_scores,
                'gt_count': len(gt_anns),
                'pred_count': len(pred_anns)
            }
        
        # Общий mAP
        map_score = np.mean(per_class_ap) if per_class_ap else 0.0
        
        # Статистика
        stats = {
            'mAP': map_score,
            'mAP@0.5': self._calculate_map_at_iou(gt_data, pred_data, 0.5),
            'mAP@0.75': self._calculate_map_at_iou(gt_data, pred_data, 0.75),
            'per_class_ap': per_class_ap,
            'categories': categories,
            'ap_results': ap_results,
            'iou_thresholds': iou_thresholds,
            'description': (self.__doc__ or '').strip()
        }
        
        return map_score, stats

    def _calculate_ap_for_class(self, gt_anns: List[Dict], pred_anns: List[Dict], iou_thresh: float) -> float:
        """Вычисляет AP для одного класса при заданном IoU пороге"""
        if not gt_anns or not pred_anns:
            return 0.0
        
        # Группируем GT по файлам
        gt_by_file = {}
        for gt in gt_anns:
            file_name = gt['file_name']
            if file_name not in gt_by_file:
                gt_by_file[file_name] = []
            gt_by_file[file_name].append(gt)
        
        # Отслеживаем использованные GT
        gt_matched = {i: False for i, _ in enumerate(gt_anns)}
        
        tp = []
        fp = []
        
        for pred in pred_anns:
            file_name = pred['file_name']
            pred_bbox = pred['bbox']
            
            best_iou = 0.0
            best_gt_idx = -1
            
            # Ищем лучшее совпадение среди GT этого файла
            if file_name in gt_by_file:
                for gt in gt_by_file[file_name]:
                    gt_idx = gt_anns.index(gt)
                    if not gt_matched[gt_idx]:
                        iou = self._calculate_iou(pred_bbox, gt['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
            
            # Определяем TP или FP
            if best_iou >= iou_thresh and best_gt_idx >= 0:
                tp.append(1)
                fp.append(0)
                gt_matched[best_gt_idx] = True
            else:
                tp.append(0)
                fp.append(1)
        
        if not tp:
            return 0.0
        
        # Вычисляем precision и recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(gt_anns)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Вычисляем AP (площадь под PR кривой)
        return self._compute_ap(recalls, precisions)

    def _compute_ap(self, recalls: np.ndarray, precisions: np.ndarray) -> float:
        """Вычисляет AP как площадь под PR кривой"""
        # Добавляем точки (0,1) и (1,0) для корректного вычисления
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))
        
        # Делаем precision монотонно убывающей
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
        # Находим точки, где recall изменяется
        i = np.where(mrec[1:] != mrec[:-1])[0]
        
        # Вычисляем площадь
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return float(ap)

    def _calculate_map_at_iou(self, gt_data: Dict, pred_data: List[Dict], iou_thresh: float) -> float:
        """Вычисляет mAP при конкретном IoU пороге"""
        categories = gt_data['categories']
        ap_scores = []
        
        for cat in categories:
            gt_anns = [ann for ann in gt_data['annotations'] if ann['label'] == cat]
            pred_anns = [pred for pred in pred_data if pred['label'] == cat]
            pred_anns = sorted(pred_anns, key=lambda x: x['score'], reverse=True)
            
            if gt_anns:
                ap = self._calculate_ap_for_class(gt_anns, pred_anns, iou_thresh)
                ap_scores.append(ap)
        
        return np.mean(ap_scores) if ap_scores else 0.0
    
    @staticmethod
    def _norm_file(fn: Any) -> str:
        return str(fn or 'unknown.jpg')
    @staticmethod
    def _norm_label(label: Any) -> Optional[str]:
        return str(label) if label is not None else None
    @staticmethod
    def _bbox_from_ann(a: Any) -> Optional[List[float]]:
        # Try to get image size context
        img_w = int(getattr(a, 'width', getattr(a, 'image_size', (0, 0))[0] if hasattr(a, 'image_size') else 0) or 0)
        img_h = int(getattr(a, 'height', getattr(a, 'image_size', (0, 0))[1] if hasattr(a, 'image_size') else 0) or 0)

        # Prefer mask-derived bbox when mask is available and non-empty
        mask = getattr(a, 'mask', None)
        if isinstance(mask, np.ndarray) and mask.size > 0 and mask.max() > 0:
            ys, xs = np.where(mask > 0)
            if ys.size > 0 and xs.size > 0:
                y0, x0 = int(ys.min()), int(xs.min())
                y1, x1 = int(ys.max()), int(xs.max())
                x, y = float(x0), float(y0)
                # +1 to include boundary pixels (consistent with detector bbox from mask)
                w, h = float(max(0, x1 - x0 + 1)), float(max(0, y1 - y0 + 1))
                # Clamp to image bounds if known
                if img_w > 0 and img_h > 0:
                    # Handle normalized coordinates in [0,1]
                    if (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                        x, y, w, h = x * img_w, y * img_h, w * img_w, h * img_h
                    x = max(0.0, min(x, img_w - 1.0))
                    y = max(0.0, min(y, img_h - 1.0))
                    w = max(0.0, min(w, img_w - x))
                    h = max(0.0, min(h, img_h - y))
                return [x, y, w, h]

        # Otherwise, use bbox field with sanity checks and conversions
        bbox = getattr(a, 'bbox', None) or []
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x, y, w, h = [float(v) for v in bbox]
            if img_w > 0 and img_h > 0:
                if (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    x, y, w, h = x * img_w, y * img_h, w * img_w, h * img_h
                x = max(0.0, min(x, img_w - 1.0))
                y = max(0.0, min(y, img_h - 1.0))
                w = max(0.0, min(w, img_w - x))
                h = max(0.0, min(h, img_h - y))
            else:
                w = max(0.0, w)
                h = max(0.0, h)
            return [x, y, w, h]
        if img_w > 0 and img_h > 0:
            return [0.0, 0.0, float(img_w), float(img_h)]
        return None
    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs) -> MetricOutputModel:
        """Чистая реализация mAP без pycocotools и DEBUG-печатей"""
        try:
            # Подготовка данных
            gt_data = self._prepare_gt_data(gt)
            pred_data = self._prepare_prediction_data(prediction, gt_data)
            
            if not gt_data['annotations'] or not pred_data:
                return MetricOutputModel(
                    metric_name=self.name,
                    score=0.0,
                    stats={'error': 'empty GT or predictions', 'fallback': True, 'description': (self.__doc__ or '').strip()}
                )
            
            # Вычисление mAP
            map_score, stats = self._calculate_map(gt_data, pred_data)
            
            return MetricOutputModel(
                metric_name=self.name,
                score=map_score,
                stats=stats
            )
        except Exception as e:
            return MetricOutputModel(
                metric_name=self.name,
                score=0.0,
                stats={'error': str(e), 'fallback': True, 'description': (self.__doc__ or '').strip()}
            )





class MeanIntersectionOverUnion(Metric):
    """Средний IoU между бинарными масками GT и предсказаний (foreground vs background)."""
    name = "mIoU"
    
    def __init__(self, iou_threshold: float = 0.5, use_torchmetrics: bool = True):
        self.iou_threshold = iou_threshold
        self.use_torchmetrics = use_torchmetrics

    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs) -> MetricOutputModel:
        if self.use_torchmetrics:
            try:
                tm = torchmetrics.JaccardIndex(task="binary")
                y_true, y_pred = self._prepare_masks(gt, prediction)
                score = float(tm(torch.tensor(y_pred), torch.tensor(y_true)).item())
            except Exception:
                y_true, y_pred = self._prepare_masks(gt, prediction)
                score = float(jaccard_score(y_true.flatten(), y_pred.flatten()))
        else:
            y_true, y_pred = self._prepare_masks(gt, prediction)
            score = float(jaccard_score(y_true.flatten(), y_pred.flatten()))
        return MetricOutputModel(metric_name=self.name, score=score, stats={
            "description": (self.__doc__ or "").strip(),
        })

    def _prepare_masks(self, gt: DatasetModel, predictions: List[COCOAnnotation]):
        # Build per-file masks (union of all instance masks) and concatenate across files
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
    """Коэффициент Дайса между бинарными масками GT и предсказаний (эквивалент F1 для пикселей)."""
    name = "dice"
    
    def __init__(self, use_torchmetrics: bool = True):
        self.use_torchmetrics = use_torchmetrics

    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs) -> MetricOutputModel:
        try:
            gt_binary, pred_binary = self._prepare_binary_data(gt, prediction)
            
            if self.use_torchmetrics:
                gt_tensor = torch.tensor(gt_binary, dtype=torch.long)
                pred_tensor = torch.tensor(pred_binary, dtype=torch.long)
                dice_score = float(self.metric(pred_tensor, gt_tensor))
            else:
                dice_score = f1_score(gt_binary.flatten(), pred_binary.flatten(), average='macro')
            
            stats = {
                'num_gt': len(gt.data_points) if gt.data_points else 0,
                'num_predictions': len(prediction),
                'library_used': 'torchmetrics' if self.use_torchmetrics else 'sklearn',
                'description': (self.__doc__ or '').strip(),
            }
            
            return MetricOutputModel(
                metric_name=self.name,
                score=float(dice_score),
                stats=stats
            )
            
        except Exception as e:
            return MetricOutputModel(
                metric_name=self.name,
                score=0.0,
                stats={'error': str(e), 'fallback': True, 'description': (self.__doc__ or '').strip()}
            )
    
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
    """Классификационный отчёт sklearn по мажоритарным меткам на файл (precision, recall, f1, support)."""
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
                return MetricOutputModel(
                    metric_name=self.name,
                    score=0.0,
                    stats={"error": "no files to compare", "dict": {}, "text": "", 'description': (self.__doc__ or '').strip()}
                )
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
                'description': (self.__doc__ or '').strip(),
            }
            return MetricOutputModel(
                metric_name=self.name,
                score=float(acc),
                stats=stats
            )
        except Exception as e:
            return MetricOutputModel(
                metric_name=self.name,
                score=0.0,
                stats={"error": str(e), "fallback": True, 'description': (self.__doc__ or '').strip()}
            )
