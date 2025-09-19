#!/usr/bin/env python3
import abc, numpy as np, warnings, tempfile, json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
from sklearn.metrics import jaccard_score, f1_score
import torch, torchmetrics
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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
        self.iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05).tolist()
        # Mappings will be populated during GT conversion and reused for predictions
        self._file_to_image_id: Dict[str, int] = {}
        self._label_to_cat_id: Dict[Union[int, str], int] = {}
        self._categories: List[Dict[str, Union[int, str]]] = []
    
    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs) -> MetricOutputModel:
        try:
            gt_coco_format = self._convert_gt_to_coco(gt)
            pred_coco_format = self._convert_predictions_to_coco(prediction)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as gt_file:
                json.dump(gt_coco_format, gt_file)
                gt_file_path = gt_file.name
            coco_gt = COCO(gt_file_path)
            coco_dt = coco_gt.loadRes(pred_coco_format)
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.params.iouThrs = np.array(self.iou_thresholds)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            stats = {
                'mAP': coco_eval.stats[0],  
                'mAP@0.5': coco_eval.stats[1],
                'mAP@0.75': coco_eval.stats[2], 
                'mAP_small': coco_eval.stats[3], 
                'mAP_medium': coco_eval.stats[4],
                'mAP_large': coco_eval.stats[5],
                'iou_thresholds': self.iou_thresholds,
                'num_gt': len(gt.data_points) if gt.data_points else 0,
                'num_predictions': len(prediction)
            }
            
            return MetricOutputModel(
                metric_name=self.name,
                score=float(coco_eval.stats[0]),  
                stats=stats
            )
            
        except Exception as e:
            return MetricOutputModel(
                metric_name=self.name,
                score=0.0,
                stats={'error': str(e), 'fallback': True}
            )
    
    def _convert_gt_to_coco(self, gt: DatasetModel) -> Dict:
        images: List[Dict[str, Any]] = []
        annotations: List[Dict[str, Any]] = []

        self._file_to_image_id = {}
        self._label_to_cat_id = {}
        self._categories = []

        if not gt or not getattr(gt, 'data_points', None):
            return {
                'images': [],
                'annotations': [],
                'categories': []
            }
        unique_labels: List[Union[int, str]] = []
        file_names: List[str] = []
        for ann in gt.data_points:
            fn = getattr(ann, 'file_name', None) or 'unknown.jpg'
            if fn not in file_names:
                file_names.append(fn)
            label = getattr(ann, 'label', None)
            if label is not None and label not in unique_labels:
                unique_labels.append(label)

        for idx, fn in enumerate(file_names, start=1):
            width = 0
            height = 0
            for a in gt.data_points:
                if getattr(a, 'file_name', None) == fn:
                    width = int(getattr(a, 'width', getattr(a, 'image_size', (0, 0))[0] if hasattr(a, 'image_size') else 0) or 0)
                    height = int(getattr(a, 'height', getattr(a, 'image_size', (0, 0))[1] if hasattr(a, 'image_size') else 0) or 0)
                    break
            self._file_to_image_id[fn] = idx
            images.append({
                'id': idx,
                'file_name': fn,
                'width': int(width),
                'height': int(height)
            })

        for cid, lab in enumerate(unique_labels, start=1):
            self._label_to_cat_id[lab] = cid
            self._categories.append({'id': cid, 'name': str(lab)})

        ann_id = 1
        for a in gt.data_points:
            fn = getattr(a, 'file_name', None) or 'unknown.jpg'
            image_id = self._file_to_image_id.get(fn)
            if image_id is None:
                continue
            label = getattr(a, 'label', None)
            if label not in self._label_to_cat_id:
                continue
            cat_id = self._label_to_cat_id[label]

            bbox = getattr(a, 'bbox', None) or []
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                w = int(getattr(a, 'width', 0) or 0)
                h = int(getattr(a, 'height', 0) or 0)
                bbox = [0.0, 0.0, float(max(0, w)), float(max(0, h))]
            x, y, w, h = [float(b) for b in bbox]
            if w < 0: w = 0.0
            if h < 0: h = 0.0
            area = getattr(a, 'area', None)
            try:
                area_val = float(area) if area is not None else float(w * h)
            except Exception:
                area_val = float(w * h)

            annotations.append({
                'id': ann_id,
                'image_id': int(image_id),
                'category_id': int(cat_id),
                'bbox': [float(x), float(y), float(w), float(h)],
                'area': float(area_val),
                'iscrowd': 0
            })
            ann_id += 1

        return {
            'images': images,
            'annotations': annotations,
            'categories': self._categories
        }
    
    def _convert_predictions_to_coco(self, predictions: List[COCOAnnotation]) -> List[Dict]:
        results: List[Dict[str, Any]] = []
        if not predictions:
            return results

        file_to_image_id = getattr(self, '_file_to_image_id', {}) or {}
        label_to_cat_id = getattr(self, '_label_to_cat_id', {}) or {}

        for a in predictions:
            fn = getattr(a, 'file_name', None) or 'unknown.jpg'
            image_id = file_to_image_id.get(fn)
            if image_id is None:
                continue
            label = getattr(a, 'label', None)
            if label not in label_to_cat_id:
                continue
            cat_id = label_to_cat_id[label]
            bbox = getattr(a, 'bbox', None) or []
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                w = int(getattr(a, 'width', 0) or 0)
                h = int(getattr(a, 'height', 0) or 0)
                bbox = [0.0, 0.0, float(max(0, w)), float(max(0, h))]
            x, y, w, h = [float(b) for b in bbox]
            if w < 0: w = 0.0
            if h < 0: h = 0.0
            score = 1.0
            results.append({
                'image_id': int(image_id),
                'category_id': int(cat_id),
                'bbox': [float(x), float(y), float(w), float(h)],
                'score': float(score)
            })
        return results


class MeanIntersectionOverUnion(Metric):
    name = "mIoU"
    
    def __init__(self, iou_threshold: float = 0.5, use_torchmetrics: bool = True):
        self.iou_threshold = iou_threshold
        self.use_torchmetrics = use_torchmetrics
        
        if self.use_torchmetrics:
            self.metric = torchmetrics.JaccardIndex(task='multiclass', num_classes=2)
    
    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs) -> MetricOutputModel:
        try:
            gt_masks, pred_masks = self._prepare_masks(gt, prediction)
            
            if self.use_torchmetrics:
                gt_tensor = torch.tensor(gt_masks, dtype=torch.long)
                pred_tensor = torch.tensor(pred_masks, dtype=torch.long)
                iou_score = float(self.metric(pred_tensor, gt_tensor))
            else:
                iou_score = jaccard_score(gt_masks.flatten(), pred_masks.flatten(), average='macro')
            
            stats = {
                'iou_threshold': self.iou_threshold,
                'num_gt': len(gt.data_points) if gt.data_points else 0,
                'num_predictions': len(prediction),
                'library_used': 'torchmetrics' if self.use_torchmetrics else 'sklearn'
            }
            
            return MetricOutputModel(
                metric_name=self.name,
                score=float(iou_score),
                stats=stats
            )
            
        except Exception as e:
            return MetricOutputModel(
                metric_name=self.name,
                score=0.0,
                stats={'error': str(e), 'fallback': True}
            )
    
    def _prepare_masks(self, gt: DatasetModel, predictions: List[COCOAnnotation]):
        gt_masks = np.zeros((100, 100))  
        pred_masks = np.zeros((100, 100))  
        return gt_masks, pred_masks


class DiceCoefficient(Metric):
    name = "dice"
    
    def __init__(self, use_torchmetrics: bool = True):
        self.use_torchmetrics = use_torchmetrics
        
        if self.use_torchmetrics:
            self.metric = torchmetrics.Dice(num_classes=2)
    
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
                'library_used': 'torchmetrics' if self.use_torchmetrics else 'sklearn'
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
                stats={'error': str(e), 'fallback': True}
            )
    
    def _prepare_binary_data(self, gt: DatasetModel, predictions: List[COCOAnnotation]):
        gt_binary = np.zeros((100, 100))  
        pred_binary = np.zeros((100, 100))
        return gt_binary, pred_binary


class CombinedMetric(Metric):
    name = "combined"
    
    def __init__(self, 
                 primary_metric: str = "mAP",
                 weights: Optional[Dict[str, float]] = None,
                 include_metrics: Optional[List[str]] = None):
        self.primary_metric = primary_metric
        self.weights = weights or {"mAP": 0.5, "mIoU": 0.3, "dice": 0.2}
        self.include_metrics = include_metrics or ["mAP", "mIoU", "dice"]
        
        self.metrics = {}
        if "mAP" in self.include_metrics:
            try:
                self.metrics["mAP"] = MeanAveragePrecision()
            except ImportError:
                warnings.warn("mAP метрика недоступна")
        
        if "mIoU" in self.include_metrics:
            try:
                self.metrics["mIoU"] = MeanIntersectionOverUnion()
            except ImportError:
                warnings.warn("mIoU метрика недоступна")
        
        if "dice" in self.include_metrics:
            try:
                self.metrics["dice"] = DiceCoefficient()
            except ImportError:
                warnings.warn("Dice метрика недоступна")
    
    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs) -> MetricOutputModel:
        results = {}
        scores = {}
        
        for metric_name, metric in self.metrics.items():
            try:
                result = metric.compute(gt, prediction, **kwargs)
                results[metric_name] = result
                scores[metric_name] = result.score
            except Exception as e:
                warnings.warn(f"Ошибка при вычислении {metric_name}: {e}")
                scores[metric_name] = 0.0
        
        weighted_score = 0.0
        total_weight = 0.0
        for metric_name, weight in self.weights.items():
            if metric_name in scores:
                weighted_score += scores[metric_name] * weight
                total_weight += weight
        
        if total_weight > 0:
            weighted_score /= total_weight
        
        primary_score = scores.get(self.primary_metric, weighted_score)
        
        stats = {
            'individual_scores': scores,
            'weighted_score': weighted_score,
            'primary_metric': self.primary_metric,
            'weights': self.weights,
            'individual_results': {k: v.stats for k, v in results.items()},
            'num_gt': len(gt.data_points) if gt.data_points else 0,
            'num_predictions': len(prediction)
        }
        
        return MetricOutputModel(
            metric_name=self.name,
            score=float(primary_score),
            stats=stats
        )
