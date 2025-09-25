from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict, Counter

import numpy as np
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    precision_recall_curve,
)

# Local types expected by the rest of the pipeline
from DatasetModel import DatasetModel
from COCOAnnotations import COCOAnnotation


@dataclass
class MetricOutputModel:
    metric_name: str
    score: float
    stats: Dict[str, Any]


class Metric:
    name: str = "metric"
    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs) -> MetricOutputModel:
        raise NotImplementedError


def _xywh_to_xyxy(b: Union[List[float], Tuple[float, float, float, float]]) -> np.ndarray:
    x, y, w, h = map(float, b)
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (xa2 - xa1)) * max(0.0, (ya2 - ya1))
    area_b = max(0.0, (xb2 - xb1)) * max(0.0, (yb2 - yb1))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _group_by_file(items: List[COCOAnnotation]) -> Dict[str, List[COCOAnnotation]]:
    byf: Dict[str, List[COCOAnnotation]] = defaultdict(list)
    for a in items or []:
        fn = getattr(a, "file_name", None)
        if fn is not None:
            byf[str(fn)].append(a)
    return byf


class MeanAveragePrecision(Metric):
    name = "mAP"

    def __init__(self, iou_thresh: float = 0.5) -> None:
        self.iou_thresh = float(iou_thresh)

    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs) -> MetricOutputModel:
        gt_by_file = _group_by_file(gt.data_points or [])
        pr_by_file = _group_by_file(prediction or [])

        labels = sorted({str(getattr(a, "label", "")) for a in (gt.data_points or []) if getattr(a, "label", None) is not None})
        if not labels:
            return MetricOutputModel(self.name, 0.0, {"error": "no labels in GT"})

        per_class_ap: List[float] = []
        ap_details: Dict[str, Dict[str, Any]] = {}

        for lab in labels:
            y_true_all: List[int] = []
            y_score_all: List[float] = []

            for fn, gts in gt_by_file.items():
                g = [a for a in gts if str(getattr(a, "label", "")) == lab and getattr(a, "bbox", None) is not None]
                p = [a for a in pr_by_file.get(fn, []) if str(getattr(a, "label", "")) == lab and getattr(a, "bbox", None) is not None]
                g_boxes = [_xywh_to_xyxy(getattr(a, "bbox")) for a in g]
                p_boxes = [_xywh_to_xyxy(getattr(a, "bbox")) for a in p]
                p_scores = [float(getattr(a, "score", getattr(a, "confidence", 1.0))) for a in p]

                used_gt = set()
                order = np.argsort(-np.array(p_scores)) if p_scores else np.array([], dtype=int)
                for idx in order:
                    pb = p_boxes[idx]
                    score = p_scores[idx]
                    best_iou, best_j = 0.0, -1
                    for j, gb in enumerate(g_boxes):
                        if j in used_gt:
                            continue
                        iou = _iou(pb, gb)
                        if iou > best_iou:
                            best_iou, best_j = iou, j
                    if best_iou >= self.iou_thresh and best_j >= 0:
                        used_gt.add(best_j)
                        y_true_all.append(1)
                    else:
                        y_true_all.append(0)
                    y_score_all.append(score)

                for _ in range(len(g_boxes) - len(used_gt)):
                    y_true_all.append(1)
                    y_score_all.append(0.0)

            if not y_true_all:
                ap = 0.0
                pr_curve = {"precision": [], "recall": [], "thresholds": []}
            else:
                ap = float(average_precision_score(y_true_all, y_score_all))
                prec, rec, thr = precision_recall_curve(y_true_all, y_score_all)
                pr_curve = {
                    "precision": prec.tolist(),
                    "recall": rec.tolist(),
                    "thresholds": thr.tolist() if thr is not None else [],
                }
            per_class_ap.append(ap)
            ap_details[lab] = {"AP": ap, "curve": pr_curve}

        mAP = float(np.mean(per_class_ap)) if per_class_ap else 0.0
        stats = {
            "categories": labels,
            "per_class_ap": per_class_ap,
            "mAP@{:.2f}".format(self.iou_thresh): mAP,
            "details": ap_details,
        }
        return MetricOutputModel(self.name, mAP, stats)


class MeanIntersectionOverUnion(Metric):
    name = "mIoU"

    def __init__(self, iou_thresh: float = 0.5) -> None:
        self.iou_thresh = float(iou_thresh)

    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs) -> MetricOutputModel:
        gt_by_file = _group_by_file(gt.data_points or [])
        pr_by_file = _group_by_file(prediction or [])

        ious: List[float] = []
        for fn, gts in gt_by_file.items():
            p = pr_by_file.get(fn, [])
            g_boxes = [_xywh_to_xyxy(getattr(a, "bbox")) for a in gts if getattr(a, "bbox", None) is not None]
            p_boxes = [_xywh_to_xyxy(getattr(a, "bbox")) for a in p if getattr(a, "bbox", None) is not None]
            used_gt = set()
            for pb in p_boxes:
                best_iou, best_j = 0.0, -1
                for j, gb in enumerate(g_boxes):
                    if j in used_gt:
                        continue
                    iou = _iou(pb, gb)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_j >= 0:
                    used_gt.add(best_j)
                    ious.append(best_iou)

        score = float(np.mean(ious)) if ious else 0.0
        return MetricOutputModel(self.name, score, {"matched_pairs": len(ious), "iou_thresh": self.iou_thresh})


class DiceCoefficient(Metric):
    name = "dice"

    def __init__(self, iou_thresh: float = 0.5) -> None:
        self.iou_thresh = float(iou_thresh)

    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs) -> MetricOutputModel:
        gt_by_file = _group_by_file(gt.data_points or [])
        pr_by_file = _group_by_file(prediction or [])

        tp = fp = fn = 0
        for fnm, gts in gt_by_file.items():
            p = pr_by_file.get(fnm, [])
            g_boxes = [_xywh_to_xyxy(getattr(a, "bbox")) for a in gts if getattr(a, "bbox", None) is not None]
            p_boxes = [_xywh_to_xyxy(getattr(a, "bbox")) for a in p if getattr(a, "bbox", None) is not None]

            matched = set()
            for pb in p_boxes:
                ok = False
                for j, gb in enumerate(g_boxes):
                    if j in matched:
                        continue
                    if _iou(pb, gb) >= self.iou_thresh:
                        matched.add(j)
                        ok = True
                        break
                if ok:
                    tp += 1
                else:
                    fp += 1
            fn += max(0, len(g_boxes) - len(matched))

        dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
        return MetricOutputModel(self.name, float(dice), {"tp": tp, "fp": fp, "fn": fn, "iou_thresh": self.iou_thresh})


class ClassificationReportMetric(Metric):
    name = "classification_report"

    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs) -> MetricOutputModel:
        gt_by_file = _group_by_file(gt.data_points or [])
        pr_by_file = _group_by_file(prediction or [])

        y_true: List[str] = []
        y_pred: List[str] = []
        for fn in sorted(set(gt_by_file) | set(pr_by_file)):
            gt_labels = [str(getattr(a, "label", "")) for a in gt_by_file.get(fn, []) if getattr(a, "label", None) is not None]
            pr_items = pr_by_file.get(fn, [])
            if gt_labels:
                true_lab = Counter(gt_labels).most_common(1)[0][0]
                y_true.append(true_lab)
                if pr_items:
                    lab, sc = max(((str(getattr(a, "label", "")), float(getattr(a, "score", getattr(a, "confidence", 1.0)))) for a in pr_items), key=lambda t: t[1])
                    y_pred.append(lab)
                else:
                    y_pred.append("__none__")

        if not y_true:
            return MetricOutputModel(self.name, 0.0, {"error": "no ground truth labels"})

        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        report_text = classification_report(y_true, y_pred, output_dict=False, zero_division=0)
        acc = float(np.mean([a == b for a, b in zip(y_true, y_pred)])) if y_true else 0.0
        stats = {"dict": report_dict, "text": report_text, "labels": sorted(set(y_true) | set(y_pred))}
        return MetricOutputModel(self.name, acc, stats)
