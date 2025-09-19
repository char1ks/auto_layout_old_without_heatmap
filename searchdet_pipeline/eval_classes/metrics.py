from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, List, Dict, Tuple, Iterable, Optional, DefaultDict

import numpy as np

from searchdet_pipeline.eval_classes.DatasetModel import DatasetModel
from searchdet_pipeline.eval_classes.COCOAnnotations import COCOAnnotation


@dataclass
class MetricOutputModel:
    metric_name: str
    score: float
    uid: Optional[str] = None
    stats: dict[str, Any] = field(default_factory=dict)


class Metric(abc.ABC):
    name: str = "metric"
    @abc.abstractmethod
    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs) -> MetricOutputModel:
        pass


class MeanAveragePrecision(Metric):
    name: str = "mean_average_precision"
    
    def __init__(self, iou_thresholds: Optional[List[float]] = None):
        if iou_thresholds is None:
            self.iou_thresholds = [t / 100 for t in range(50, 100, 5)]
        else:
            self.iou_thresholds = iou_thresholds
    
    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs) -> MetricOutputModel:
        average = kwargs.get("average", "macro")
        ap_per_class = self._map_per_class(gt.data_points, prediction, self.iou_thresholds)
        ap50_per_class = self._map_per_class(gt.data_points, prediction, [0.5])
        ap75_per_class = self._map_per_class(gt.data_points, prediction, [0.75])
        map_score = (sum(ap_per_class.values()) / len(ap_per_class)) if ap_per_class else 0.0
        map_50 = (sum(ap50_per_class.values()) / len(ap50_per_class)) if ap50_per_class else 0.0
        map_75 = (sum(ap75_per_class.values()) / len(ap75_per_class)) if ap75_per_class else 0.0
        
        stats = {
            "mAP": map_score,
            "mAP@0.5": map_50,
            "mAP@0.75": map_75,
            "per_class_AP": ap_per_class,
            "per_class_AP50": ap50_per_class,
            "per_class_AP75": ap75_per_class,
            "iou_thresholds": self.iou_thresholds,
            "num_classes": len(ap_per_class),
            "num_gt": len(gt.data_points),
            "num_predictions": len(prediction)
        }
        
        return MetricOutputModel(
            metric_name=self.name,
            score=map_score,
            stats=stats
        )


class MeanIntersectionOverUnion(Metric):
    name: str = "mean_intersection_over_union"
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
    
    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs) -> MetricOutputModel:
        per_class_stats = self._compute_per_class_iou_stats(gt, prediction)
        all_ious = [iou for cls_stats in per_class_stats.values() for iou in cls_stats["ious"]]
        micro_mean_iou = (sum(all_ious) / len(all_ious)) if all_ious else 0.0
        classes_with_gt = [c for c, st in per_class_stats.items() if st["num_gt"] > 0]
        macro_mean_iou = (sum((st["mean_iou"] for c, st in per_class_stats.items() if st["num_gt"] > 0)) / len(classes_with_gt)) if classes_with_gt else 0.0
        stats = {
            "micro_mIoU": micro_mean_iou,
            "macro_mIoU": macro_mean_iou,
            "iou_threshold": self.iou_threshold,
            "per_class": {c: {"mean_iou": st["mean_iou"], "num_pairs": st["num_pairs"], "num_gt": st["num_gt"], "num_pred": st["num_pred"]} for c, st in per_class_stats.items()},
            "num_classes": len(per_class_stats),
            "num_gt": len(gt.data_points),
            "num_predictions": len(prediction)
        }
        
        return MetricOutputModel(
            metric_name=self.name,
            score=macro_mean_iou,
            stats=stats
        )


class DiceCoefficient(Metric):
    name: str = "dice_coefficient"
    
    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs) -> MetricOutputModel:
        per_class_stats = self._compute_per_class_iou_stats(gt, prediction)
        all_dices = [dice for cls_stats in per_class_stats.values() for dice in cls_stats["dices"]]
        micro_mean_dice = (sum(all_dices) / len(all_dices)) if all_dices else 0.0
        classes_with_gt = [c for c, st in per_class_stats.items() if st["num_gt"] > 0]
        macro_mean_dice = (sum((st["mean_dice"] for c, st in per_class_stats.items() if st["num_gt"] > 0)) / len(classes_with_gt)) if classes_with_gt else 0.0
        stats = {
            "micro_dice": micro_mean_dice,
            "macro_dice": macro_mean_dice,
            "per_class": {c: {"mean_dice": st["mean_dice"], "num_pairs": st["num_pairs"], "num_gt": st["num_gt"], "num_pred": st["num_pred"]} for c, st in per_class_stats.items()},
            "num_classes": len(per_class_stats),
            "num_gt": len(gt.data_points),
            "num_predictions": len(prediction)
        }
        
        return MetricOutputModel(
            metric_name=self.name,
            score=macro_mean_dice,
            stats=stats
        )


class CombinedMetric(Metric):
    name: str = "combined_metric"
    
    def __init__(self, primary_metric: str = "mAP", weights: Optional[Dict[str, float]] = None):
        self.primary_metric = primary_metric
        self.weights = weights or {}
        self.map_metric = MeanAveragePrecision()
        self.iou_metric = MeanIntersectionOverUnion()
        self.dice_metric = DiceCoefficient()
    
    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs) -> MetricOutputModel:
        map_result = self.map_metric.compute(gt, prediction, **kwargs)
        iou_result = self.iou_metric.compute(gt, prediction, **kwargs)
        dice_result = self.dice_metric.compute(gt, prediction, **kwargs)
        
        if self.weights:
            final_score = (
                self.weights.get("mAP", 0) * map_result.score +
                self.weights.get("mIoU", 0) * iou_result.score +
                self.weights.get("dice", 0) * dice_result.score
            )
        else:
            if self.primary_metric == "mIoU":
                final_score = iou_result.score
            elif self.primary_metric == "dice":
                final_score = dice_result.score
            else: 
                final_score = map_result.score
        
        combined_stats = {
            "primary_metric": self.primary_metric,
            "weights": self.weights,
            "mAP": {
                "score": map_result.score,
                "stats": map_result.stats
            },
            "mIoU": {
                "score": iou_result.score,
                "stats": iou_result.stats
            },
            "dice": {
                "score": dice_result.score,
                "stats": dice_result.stats
            }
        }
        
        return MetricOutputModel(
            metric_name=self.name,
            score=final_score,
            stats=combined_stats
        )

    def _mask_or_bbox_iou(self, g: COCOAnnotation, p: COCOAnnotation) -> float:
        gm = getattr(g, "mask", None)
        pm = getattr(p, "mask", None)
        if isinstance(gm, np.ndarray) and isinstance(pm, np.ndarray) and gm.size and pm.size and gm.shape == pm.shape:
            gmb = (gm.astype(bool))
            pmb = (pm.astype(bool))
            inter = np.logical_and(gmb, pmb).sum(dtype=np.int64)
            union = np.logical_or(gmb, pmb).sum(dtype=np.int64)
            return float(inter) / float(union) if union > 0 else 0.0
        box_a = getattr(g, "bbox", None)
        box_b = getattr(p, "bbox", None)
        if box_a is None or box_b is None:
            return 0.0
        
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        ax2, ay2 = ax + max(0.0, aw), ay + max(0.0, ah)
        bx2, by2 = bx + max(0.0, bw), by + max(0.0, bh)
        
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        
        area_a = max(0.0, (ax2 - ax)) * max(0.0, (ay2 - ay))
        area_b = max(0.0, (bx2 - bx)) * max(0.0, (by2 - by))
        union = area_a + area_b - inter
        return 0.0 if union <= 0.0 else inter / union

    def _map_per_class(self, gt_anns: List[COCOAnnotation], pred_anns: List[COCOAnnotation], iou_thresholds: List[float] = [t / 100 for t in range(50, 100, 5)], ) -> Dict[str, float]:
        gt_by_key: Dict[Tuple[Any, Any], List[COCOAnnotation]] = {}
        for g in gt_anns:
            fname = getattr(g, "file_name", None)
            label = getattr(g, "label", None)
            gt_by_key.setdefault((fname, label), []).append(g)
        categories = sorted({getattr(g, "label", None) for g in gt_anns})
        ap_by_class: Dict[str, float] = {}
        for cat in categories:
            aps_for_cat_per_thr: List[float] = []
            for thr in iou_thresholds:
                gt_used: Dict[Tuple[Any, int], bool] = {}
                gt_per_img: Dict[Any, List[COCOAnnotation]] = {}
                for (fname, c), lst in gt_by_key.items():
                    if c == cat:
                        gt_per_img[fname] = lst
                preds_cat = [p for p in pred_anns if getattr(p, "label", None) == cat]
                preds_cat.sort(key=lambda x: getattr(x, "score", 0.0), reverse=True)
                tps: List[int] = []
                fps: List[int] = []
                for p in preds_cat:
                    best_iou = 0.0
                    best_idx = -1
                    p_fname = getattr(p, "file_name", None)
                    gts = gt_per_img.get(p_fname, [])
                    for gi, g in enumerate(gts):
                        key = (p_fname, gi)
                        if gt_used.get(key, False):
                            continue
                        iou = self._mask_or_bbox_iou(g, p)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = gi
                    if best_iou >= thr and best_idx >= 0:
                        gt_used[(p_fname, best_idx)] = True
                        tps.append(1)
                        fps.append(0)
                    else:
                        tps.append(0)
                        fps.append(1)
                tp_cum, fp_cum = 0, 0
                precisions: List[float] = []
                recalls: List[float] = []
                total_gt = sum(len(v) for v in gt_per_img.values())
                if total_gt == 0:
                    aps_for_cat_per_thr.append(0.0)
                    continue
                for tp, fp in zip(tps, fps):
                    tp_cum += tp
                    fp_cum += fp
                    prec = tp_cum / max(1, (tp_cum + fp_cum))
                    rec = tp_cum / total_gt
                    precisions.append(prec)
                    recalls.append(rec)
                ap = 0.0
                for r_i in range(101):
                    r_target = r_i / 100.0
                    prec_at_r = 0.0
                    for pr, rc in zip(precisions, recalls):
                        if rc >= r_target and pr > prec_at_r:
                            prec_at_r = pr
                    ap += prec_at_r
                ap /= 101.0
                aps_for_cat_per_thr.append(ap)
            if aps_for_cat_per_thr:
                ap_by_class[cat] = sum(aps_for_cat_per_thr) / len(aps_for_cat_per_thr)
            else:
                ap_by_class[cat] = 0.0
        return ap_by_class

    def _compute_per_class_iou_stats(self, gt: DatasetModel, prediction: List[COCOAnnotation]) -> Dict[str, Dict[str, Any]]:
        gt_by_file_label: Dict[Tuple[str, Any], List[COCOAnnotation]] = {}
        pred_by_file_label: Dict[Tuple[str, Any], List[COCOAnnotation]] = {}
        categories = sorted({getattr(g, "label", None) for g in gt.data_points})

        for g in gt.data_points:
            fname = getattr(g, "file_name", "") or ""
            label = getattr(g, "label", None)
            gt_by_file_label.setdefault((fname, label), []).append(g)
        for p in prediction:
            fname = getattr(p, "file_name", "") or ""
            label = getattr(p, "label", None)
            pred_by_file_label.setdefault((fname, label), []).append(p)

        per_class: Dict[str, Dict[str, Any]] = {}
        for label in categories:
            ious: List[float] = []
            dices: List[float] = []
            num_pairs = 0
            num_gt = sum(len(v) for (f, l), v in gt_by_file_label.items() if l == label)
            num_pred = sum(len(v) for (f, l), v in pred_by_file_label.items() if l == label)
            files_with_label = sorted({f for (f, l) in set(list(gt_by_file_label.keys()) + list(pred_by_file_label.keys())) if l == label})
            for fname in files_with_label:
                gts = gt_by_file_label.get((fname, label), [])
                preds = pred_by_file_label.get((fname, label), [])
                if not gts or not preds:
                    continue
                used: set[int] = set()
                preds_sorted = sorted(preds, key=lambda x: getattr(x, "score", 0.0), reverse=True)
                for p in preds_sorted:
                    best_iou = 0.0
                    best_idx = -1
                    for gi, g in enumerate(gts):
                        if gi in used:
                            continue
                        iou_val = self._mask_or_bbox_iou(g, p)
                        if iou_val > best_iou:
                            best_iou = iou_val
                            best_idx = gi
                    if best_idx >= 0:
                        used.add(best_idx)
                        ious.append(best_iou)
                        dice = (2 * best_iou / (1 + best_iou)) if best_iou > 0.0 else 0.0
                        dices.append(dice)
                        num_pairs += 1

            mean_iou = (sum(ious) / len(ious)) if ious else 0.0
            mean_dice = (sum(dices) / len(dices)) if dices else 0.0
            per_class[str(label)] = {
                "ious": ious,
                "dices": dices,
                "mean_iou": mean_iou,
                "mean_dice": mean_dice,
                "num_pairs": num_pairs,
                "num_gt": num_gt,
                "num_pred": num_pred,
            }
        return per_class
