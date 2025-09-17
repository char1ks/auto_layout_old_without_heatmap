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
    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], average: str = "micro") -> MetricOutputModel:
        gt_by_file: Dict[str, List[COCOAnnotation]] = {}
        for ann in gt.data_points:
            fname = getattr(ann, "file_name", "") or ""
            gt_by_file.setdefault(fname, []).append(ann)

        pred_by_file: Dict[str, List[COCOAnnotation]] = {}
        for ann in prediction:
            fname = getattr(ann, "file_name", "") or ""
            pred_by_file.setdefault(fname, []).append(ann)

        per_class_stats = self._compute_per_class_iou_stats(gt, prediction)

        all_ious = [iou for cls_stats in per_class_stats.values() for iou in cls_stats["ious"]]
        all_dices = [di for cls_stats in per_class_stats.values() for di in cls_stats["dices"]]
        micro_mean_iou = (sum(all_ious) / len(all_ious)) if all_ious else 0.0
        micro_mean_dice = (sum(all_dices) / len(all_dices)) if all_dices else 0.0
        classes_with_gt = [c for c, st in per_class_stats.items() if st["num_gt"] > 0]
        macro_mean_iou = (sum((st["mean_iou"] for c, st in per_class_stats.items() if st["num_gt"] > 0)) / len(classes_with_gt)) if classes_with_gt else 0.0
        macro_mean_dice = (sum((st["mean_dice"] for c, st in per_class_stats.items() if st["num_gt"] > 0)) / len(classes_with_gt)) if classes_with_gt else 0.0

        ap_per_class = self._map_per_class(gt.data_points, prediction)
        ap50_per_class = self._map_per_class(gt.data_points, prediction, [0.5])
        ap75_per_class = self._map_per_class(gt.data_points, prediction, [0.75])
        map_macro = (sum(ap_per_class.values()) / len(ap_per_class)) if ap_per_class else 0.0
        map_50_macro = (sum(ap50_per_class.values()) / len(ap50_per_class)) if ap50_per_class else 0.0
        map_75_macro = (sum(ap75_per_class.values()) / len(ap75_per_class)) if ap75_per_class else 0.0

        total_gt_by_class: Dict[str, int] = {c: st["num_gt"] for c, st in per_class_stats.items()}
        total_gt_all = sum(total_gt_by_class.values())
        if total_gt_all > 0:
            map_micro = sum(ap_per_class.get(c, 0.0) * total_gt_by_class.get(c, 0) for c in per_class_stats.keys()) / total_gt_all
            map50_micro = sum(ap50_per_class.get(c, 0.0) * total_gt_by_class.get(c, 0) for c in per_class_stats.keys()) / total_gt_all
            map75_micro = sum(ap75_per_class.get(c, 0.0) * total_gt_by_class.get(c, 0) for c in per_class_stats.keys()) / total_gt_all
        else:
            map_micro = map50_micro = map75_micro = 0.0

        pairs: List[dict] = []
        matched_images = 0
        for fname, gt_list in gt_by_file.items():
            preds = pred_by_file.get(fname, [])
            if not preds:
                continue
            matched_images += 1
            gt_by_label: Dict[str, List[COCOAnnotation]] = {}
            for g in gt_list:
                gt_by_label.setdefault(getattr(g, "label", None), []).append(g)
            pred_by_label: Dict[str, List[COCOAnnotation]] = {}
            for p in preds:
                pred_by_label.setdefault(getattr(p, "label", None), []).append(p)
            for label, preds_of_label in pred_by_label.items():
                if label not in gt_by_label:
                    continue
                used: set[int] = set()
                preds_sorted = sorted(preds_of_label, key=lambda x: getattr(x, "score", 0.0), reverse=True)
                gts = gt_by_label[label]
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
                        dice = (2 * best_iou / (1 + best_iou)) if best_iou > 0.0 else 0.0
                        pairs.append({
                            "file_name": fname,
                            "label": label,
                            "gt_uid": getattr(gts[best_idx], "uid", None),
                            "pred_uid": getattr(p, "uid", None),
                            "iou": best_iou,
                            "dice": dice,
                        })

        avg_type = (average or "micro").lower().strip()
        if avg_type == "macro":
            final_score = macro_mean_iou
            metric_name = "combined_macro"
        else:
            final_score = micro_mean_iou
            metric_name = "combined_micro"

        stats = {
            "mean_iou": micro_mean_iou, 
            "mAP": map_macro,           
            "mAP50": map_50_macro,
            "mAP75": map_75_macro,
            "jaccard": micro_mean_iou,
            "dice": micro_mean_dice,
            "mean_iou_micro": micro_mean_iou,
            "mean_iou_macro": macro_mean_iou,
            "dice_micro": micro_mean_dice,
            "dice_macro": macro_mean_dice,
            "mAP_macro": map_macro,
            "mAP50_macro": map_50_macro,
            "mAP75_macro": map_75_macro,
            "mAP_micro": map_micro,
            "mAP50_micro": map50_micro,
            "mAP75_micro": map75_micro,
            "num_images_matched": matched_images,
            "map_iou_thresholds": [t / 100 for t in range(50, 100, 5)],
            "per_class": {},
            "pairs": pairs,
        }


        for label, st in per_class_stats.items():
            stats["per_class"][label] = {
                "mean_iou": st["mean_iou"],
                "mean_dice": st["mean_dice"],
                "num_pairs": st["num_pairs"],
                "num_gt": st["num_gt"],
                "num_pred": st["num_pred"],
                "AP": ap_per_class.get(label, 0.0),
                "AP50": ap50_per_class.get(label, 0.0),
                "AP75": ap75_per_class.get(label, 0.0),
            }

        return MetricOutputModel(metric_name=metric_name, score=final_score, stats=stats)

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
        # Собираем файлы и классы
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
