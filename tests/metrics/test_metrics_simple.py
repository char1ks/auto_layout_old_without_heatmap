import sys
from pathlib import Path as _P
EVAL_CLASSES_PATH = _P(__file__).resolve().parents[2] / "evaluate"
sys.path.insert(0, str(EVAL_CLASSES_PATH))

from metrics import MeanAveragePrecision, MeanIntersectionOverUnion, DiceCoefficient, ClassificationReportMetric
from COCOAnnotations import COCOAnnotation
from DatasetModel import DatasetModel

from .conftest import (
    build_gt_dataset,
    build_predictions,
    build_perfect_predictions_from_gt,
    build_empty_predictions,
    build_empty_gt,
)

def test_map_perfect_predictions():
    gt = build_gt_dataset()
    preds = build_perfect_predictions_from_gt()
    metric = MeanAveragePrecision(iou_thresholds=[0.5])
    out = metric.compute(gt, preds)
    assert out.metric_name == "mAP"
    assert out.score >= 0.95, f"Expected high mAP, got {out.score}"
    assert len(out.stats.get("categories", [])) >= 1


def test_map_empty_predictions():
    gt = build_gt_dataset()
    preds = build_empty_predictions()
    metric = MeanAveragePrecision(iou_thresholds=[0.5])
    out = metric.compute(gt, preds)
    assert out.metric_name == "mAP"
    assert out.score == 0.0


def test_miou_perfect_masks():
    gt = build_gt_dataset()
    preds = build_perfect_predictions_from_gt()
    metric = MeanIntersectionOverUnion()
    out = metric.compute(gt, preds)
    assert out.metric_name == "mIoU"
    assert abs(out.score - 1.0) < 1e-6


def test_miou_no_predictions():
    gt = build_gt_dataset()
    preds = build_empty_predictions()
    metric = MeanIntersectionOverUnion()
    out = metric.compute(gt, preds)
    assert out.metric_name == "mIoU"
    assert out.score == 0.0


def test_dice_perfect_masks():
    gt = build_gt_dataset()
    preds = build_perfect_predictions_from_gt()
    metric = DiceCoefficient()
    out = metric.compute(gt, preds)
    assert out.metric_name == "dice"
    assert abs(out.score - 1.0) < 1e-6


def test_dice_no_predictions():
    gt = build_gt_dataset()
    preds = build_empty_predictions()
    metric = DiceCoefficient()
    out = metric.compute(gt, preds)
    assert out.metric_name == "dice"
    assert out.score == 0.0


def test_classification_report_majority_labels():
    gt = build_gt_dataset()
    preds = build_predictions()
    metric = ClassificationReportMetric()
    out = metric.compute(gt, preds)
    assert out.metric_name == "classification_report"
    assert isinstance(out.stats.get("dict", {}), dict)
    assert out.score >= 0.0


def test_classification_report_empty():
    gt = build_empty_gt()
    preds = build_empty_predictions()
    metric = ClassificationReportMetric()
    out = metric.compute(gt, preds)
    assert out.metric_name == "classification_report"
    assert out.score == 0.0