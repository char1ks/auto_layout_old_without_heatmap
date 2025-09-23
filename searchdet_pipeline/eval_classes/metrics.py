from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    f1_score,
    jaccard_score,
    precision_recall_curve,
    average_precision_score,
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision as TMDetMAP

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
    def compute(
        self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs: Any
    ) -> MetricOutputModel:  # pragma: no cover - interface
        ...


# ----------------------------
# Helpers
# ----------------------------
def _to_xyxy_xywh(box: List[float]) -> Tuple[List[float], List[float]]:
    """Accepts an xywh box; returns (xyxy, xywh) with floats."""
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h], [x, y, w, h]


def _group_by_file(
    anns: List[COCOAnnotation],
) -> Dict[str, List[COCOAnnotation]]:
    out: Dict[str, List[COCOAnnotation]] = {}
    for a in anns or []:
        fn = getattr(a, "file_name", None)
        if fn is None:
            # fall back to uid if missing
            fn = getattr(a, "uid", None)
        if fn is None:
            continue
        out.setdefault(str(fn), []).append(a)
    return out


def _as_bool_mask(m: Any, expected_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Safely coerce to boolean HxW numpy mask."""
    if m is None:
        return None  # type: ignore[return-value]
    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()
    m = np.asarray(m)
    if m.ndim == 3:
        # (H, W, C) -> single channel by argmax/any
        if m.shape[-1] == 1:
            m = m[..., 0]
        else:
            m = (m > 0).any(-1)
    if expected_hw is not None and (m.shape[0], m.shape[1]) != expected_hw:
        # naive resize via padding/cropping to keep code compact
        h, w = expected_hw
        m = m[:h, :w]
        if m.shape[0] < h or m.shape[1] < w:
            pad_h = max(0, h - m.shape[0])
            pad_w = max(0, w - m.shape[1])
            m = np.pad(m, ((0, pad_h), (0, pad_w)), mode="constant")
    return m.astype(bool)


# ----------------------------
# Detection mAP (COCO-style IoU matching via torchmetrics)
# ----------------------------
class MeanAveragePrecision(Metric):
    name = "mAP"

    def __init__(self, iou_thresholds: Optional[List[float]] = None) -> None:
        # Default COCO thresholds: 0.50:0.95
        self.iou_thresholds = (
            iou_thresholds
            if iou_thresholds is not None
            else list(np.arange(0.50, 0.95 + 1e-9, 0.05))
        )

    @staticmethod
    def _build_tm_item(anns: List[COCOAnnotation]) -> Dict[str, torch.Tensor]:
        boxes: List[List[float]] = []
        labels: List[int] = []
        scores: List[float] = []
        for a in anns or []:
            bb = getattr(a, "bbox", None)
            if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                continue
            xyxy, _ = _to_xyxy_xywh(list(bb))
            boxes.append(xyxy)
            lbl = getattr(a, "label", 0)
            if isinstance(lbl, str):
                try:
                    lbl = int(lbl)
                except Exception:
                    # hash stable to int range
                    lbl = abs(hash(lbl)) % (2**31)
            labels.append(int(lbl))
            sc = getattr(a, "score", getattr(a, "confidence", None))
            if sc is None:
                sc = 1.0
            scores.append(float(sc))
        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32)
            if boxes
            else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
            if labels
            else torch.zeros((0,), dtype=torch.int64),
        }
        if scores:
            res["scores"] = torch.tensor(scores, dtype=torch.float32)
        return res

    def compute(
        self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs: Any
    ) -> MetricOutputModel:
        gt_by_file = _group_by_file(getattr(gt, "data_points", []) or [])
        pr_by_file = _group_by_file(prediction or [])

        # Align files present in either set
        files = sorted(set(gt_by_file.keys()) | set(pr_by_file.keys()))

        targets: List[Dict[str, torch.Tensor]] = []
        preds: List[Dict[str, torch.Tensor]] = []

        for f in files:
            targets.append(self._build_tm_item(gt_by_file.get(f, [])))
            preds.append(self._build_tm_item(pr_by_file.get(f, [])))

        metric = TMDetMAP(iou_type="bbox")
        metric.iou_thresholds = torch.tensor(self.iou_thresholds, dtype=torch.float32)
        metric.update(preds=preds, target=targets)
        out = metric.compute()

        # torchmetrics returns tensors; convert
        def _t2f(x: Any) -> Any:
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().item() if x.ndim == 0 else x.detach().cpu().tolist()
            if isinstance(x, dict):
                return {k: _t2f(v) for k, v in x.items()}
            return x

        stats = {k: _t2f(v) for k, v in out.items()}
        score = float(stats.get("map", 0.0) or 0.0)
        return MetricOutputModel(metric_name=self.name, score=score, stats=stats)

class MeanIntersectionOverUnion(Metric):
    name = "mIoU"

    def compute(
        self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs: Any
    ) -> MetricOutputModel:
        gt_by_file = _group_by_file(getattr(gt, "data_points", []) or [])
        pr_by_file = _group_by_file(prediction or [])

        files = sorted(set(gt_by_file.keys()) | set(pr_by_file.keys()))
        per_image_iou: List[float] = []

        for f in files:
            gms: List[np.ndarray] = []
            h = w = None
            for g in gt_by_file.get(f, []):
                h = getattr(g, "height", None) or (getattr(g, "image_size", (None, None))[1] if getattr(g, "image_size", None) else None)
                w = getattr(g, "width", None) or (getattr(g, "image_size", (None, None))[0] if getattr(g, "image_size", None) else None)
                gm = _as_bool_mask(getattr(g, "mask", None))
                if gm is not None:
                    gms.append(gm)
            if gms:
                gm = np.logical_or.reduce(gms)
            else:
                # Empty mask
                if h is None or w is None:
                    continue
                gm = np.zeros((h, w), dtype=bool)

            # Union all predicted masks
            pms: List[np.ndarray] = []
            for p in pr_by_file.get(f, []):
                pm = _as_bool_mask(getattr(p, "mask", None), expected_hw=gm.shape)
                if pm is not None:
                    pms.append(pm)
            pm = np.logical_or.reduce(pms) if pms else np.zeros_like(gm, dtype=bool)

            # sklearn operates on 1D arrays
            y_true = gm.reshape(-1).astype(int)
            y_pred = pm.reshape(-1).astype(int)
            iou = jaccard_score(y_true, y_pred, average="binary", zero_division=0)
            per_image_iou.append(float(iou))

        score = float(np.mean(per_image_iou)) if per_image_iou else 0.0
        stats = {"per_image": per_image_iou, "num_images": len(per_image_iou)}
        return MetricOutputModel(metric_name=self.name, score=score, stats=stats)


# ----------------------------
# Segmentation Dice via sklearn (F1 for binary masks)
# ----------------------------
class DiceCoefficient(Metric):
    name = "Dice"

    def compute(
        self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs: Any
    ) -> MetricOutputModel:
        gt_by_file = _group_by_file(getattr(gt, "data_points", []) or [])
        pr_by_file = _group_by_file(prediction or [])
        files = sorted(set(gt_by_file.keys()) | set(pr_by_file.keys()))
        per_image_dice: List[float] = []

        for f in files:
            gms: List[np.ndarray] = []
            h = w = None
            for g in gt_by_file.get(f, []):
                h = getattr(g, "height", None) or (getattr(g, "image_size", (None, None))[1] if getattr(g, "image_size", None) else None)
                w = getattr(g, "width", None) or (getattr(g, "image_size", (None, None))[0] if getattr(g, "image_size", None) else None)
                gm = _as_bool_mask(getattr(g, "mask", None))
                if gm is not None:
                    gms.append(gm)
            if gms:
                gm = np.logical_or.reduce(gms)
            else:
                if h is None or w is None:
                    continue
                gm = np.zeros((h, w), dtype=bool)

            pms: List[np.ndarray] = []
            for p in pr_by_file.get(f, []):
                pm = _as_bool_mask(getattr(p, "mask", None), expected_hw=gm.shape)
                if pm is not None:
                    pms.append(pm)
            pm = np.logical_or.reduce(pms) if pms else np.zeros_like(gm, dtype=bool)

            y_true = gm.reshape(-1).astype(int)
            y_pred = pm.reshape(-1).astype(int)
            dice = f1_score(y_true, y_pred, average="binary", zero_division=0)
            per_image_dice.append(float(dice))

        score = float(np.mean(per_image_dice)) if per_image_dice else 0.0
        stats = {"per_image": per_image_dice, "num_images": len(per_image_dice)}
        return MetricOutputModel(metric_name=self.name, score=score, stats=stats)


# ----------------------------
# Optional: simple classification report per-image using top-1 label
# ----------------------------
class ClassificationReportMetric(Metric):
    """Build a per-image classification report by taking the most confident
    predicted label in each image and comparing with the most frequent GT label.
    Uses sklearn.classification_report + precision_recall_curve."""

    name = "classification_report"

    def compute(
        self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs: Any
    ) -> MetricOutputModel:
        gt_by_file = _group_by_file(getattr(gt, "data_points", []) or [])
        pr_by_file = _group_by_file(prediction or [])

        files = sorted(set(gt_by_file.keys()) | set(pr_by_file.keys()))
        if not files:
            return MetricOutputModel(self.name, 0.0, {"error": "no data"})

        # map arbitrary labels to ints for sklearn
        labels_set: List[Any] = []
        for f in files:
            for a in gt_by_file.get(f, []):
                labels_set.append(getattr(a, "label", 0))
            for p in pr_by_file.get(f, []):
                labels_set.append(getattr(p, "label", 0))
        uniq = sorted({str(x) for x in labels_set})
        l2i = {k: i for i, k in enumerate(uniq)}

        y_true: List[int] = []
        y_pred: List[int] = []
        y_scores: List[float] = []

        for f in files:
            # GT = most frequent label in this image
            g_labels = [l2i[str(getattr(a, "label", 0))] for a in gt_by_file.get(f, [])]
            if not g_labels:
                continue
            gt_label = int(np.bincount(np.array(g_labels)).argmin() if len(set(g_labels)) == 0 else np.bincount(np.array(g_labels)).argmax())

            # Prediction = most confident label in this image
            best = None
            for p in pr_by_file.get(f, []):
                sc = getattr(p, "score", getattr(p, "confidence", 1.0))
                best = max(best, (float(sc), l2i[str(getattr(p, "label", 0))])) if best else (float(sc), l2i[str(getattr(p, "label", 0))])
            if best is None:
                # treat as background: predict GT to avoid dropping the sample
                pred_label = gt_label
                score = 0.0
            else:
                score, pred_label = best

            y_true.append(gt_label)
            y_pred.append(pred_label)
            # positive score only meaningful for binary; still store for PR curve across 1-vs-rest of the GT label
            y_scores.append(float(score))

        # Build sklearn reports
        rep_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        rep_text = classification_report(y_true, y_pred, output_dict=False, zero_division=0)
        acc = float(np.mean(np.array(y_true) == np.array(y_pred))) if y_true else 0.0

        # Binary PR curve using the majority GT class as positive (fallback)
        pr_stats: Dict[str, Any] = {}
        if len(set(y_true)) >= 2:
            pos_label = max(set(y_true), key=y_true.count)
            y_true_bin = [1 if t == pos_label else 0 for t in y_true]
            try:
                precision, recall, thresholds = precision_recall_curve(y_true_bin, y_scores)
                ap = average_precision_score(y_true_bin, y_scores)
                pr_stats = {
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                    "thresholds": thresholds.tolist() if thresholds is not None else [],
                    "average_precision": float(ap),
                    "positive_class": pos_label,
                }
            except Exception as e:
                pr_stats = {"error": str(e)}

        stats = {
            "report_dict": rep_dict,
            "report_text": rep_text,
            "labels": uniq,
            "pr_curve": pr_stats,
            "num_samples": len(y_true),
        }
        return MetricOutputModel(metric_name=self.name, score=float(acc), stats=stats)
