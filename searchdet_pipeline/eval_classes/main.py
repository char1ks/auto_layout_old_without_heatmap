from __future__ import annotations

from pathlib import Path
import sys

from searchdet_pipeline.eval_classes.Example_datasets.ArchiveVOCDataset import ArchiveVOCDataset
from searchdet_pipeline.eval_classes.DatasetPoint import DatasetPoint
from searchdet_pipeline.eval_classes.metrics import (
    Metric, MeanAveragePrecision, MeanIntersectionOverUnion, DiceCoefficient
)
from searchdet_pipeline.core.detector import SearchDetDetector
from searchdet_pipeline.core.config import get_preset_config
from searchdet_pipeline.eval_classes.Tracer import Tracer


def main() -> int:
    args = sys.argv[1:]
    dataset_dir = Path("archive")
    ann_dir: Path | None = None
    img_dir: Path | None = None

    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--ann", "--ann-dir", "--annotations") and i + 1 < len(args):
            ann_dir = Path(args[i + 1])
            i += 2
            continue
        if arg in ("--img", "--img-dir", "--images") and i + 1 < len(args):
            img_dir = Path(args[i + 1])
            i += 2
            continue
        if not arg.startswith("-") and str(dataset_dir) == "archive":
            dataset_dir = Path(arg)
            i += 1
            continue
        i += 1

    if not dataset_dir.exists():
        return 1

    positive_dir = "examples/positive"
    negative_dir = None

    dataset = ArchiveVOCDataset.from_path(dataset_dir, ann_dir=ann_dir, img_dir=img_dir)

    config = get_preset_config("balanced")
    detector = SearchDetDetector(config=config)
    reporter = Tracer(to_stdout=True, trace_file="context_trace.jsonl")
    metrics_list = [
        MeanAveragePrecision(),
        MeanIntersectionOverUnion(),
        DiceCoefficient()
    ]
    dp = DatasetPoint(dataset=dataset, detector=detector, metrics=metrics_list, reporter=reporter)
    image_root = img_dir if (img_dir is not None and img_dir.exists()) else None
    preds, metrics = dp.run(
        positive_dir=positive_dir,
        negative_dir=negative_dir,
        image_root=image_root,
        dump_report=True,
        report_output_dir=Path.cwd(),
    )
    print("РЕЗУЛЬТАТЫ МЕТРИК")
    if metrics:
        print(f"Получено {len(metrics)} метрик:")
        print("-" * 50)
        for metric_result in metrics:
            print(f"\n{metric_result.metric_name.upper()}: {metric_result.score:.4f}")
            
            if hasattr(metric_result, 'stats') and metric_result.stats:
                stats = metric_result.stats
                if metric_result.metric_name == "mAP":
                    print(f"  mAP (micro): {stats.get('mAP_micro', 0):.4f}")
                    print(f"  mAP50 (micro): {stats.get('mAP50_micro', 0):.4f}")
                    print(f"  mAP75 (micro): {stats.get('mAP75_micro', 0):.4f}")
                    print(f"  Совпавших изображений: {stats.get('num_images_matched', 0)}")
                elif metric_result.metric_name == "mIoU":
                    print(f"  Micro IoU: {stats.get('mean_iou_micro', 0):.4f}")
                    print(f"  Macro IoU: {stats.get('mean_iou_macro', 0):.4f}")
                elif metric_result.metric_name == "dice":
                    print(f"  Micro Dice: {stats.get('dice_micro', 0):.4f}")
                    print(f"  Macro Dice: {stats.get('dice_macro', 0):.4f}")
                per_class = stats.get('per_class', {})
                if per_class:
                    print(f"  Статистика по классам:")
                    for class_name, class_stats in per_class.items():
                        print(f"    {class_name}: GT={class_stats.get('num_gt', 0)}, "
                              f"Pred={class_stats.get('num_pred', 0)}, "
                              f"Score={class_stats.get(metric_result.metric_name.lower(), 0):.4f}")
    else:
        print("Метрики не получены")
    
    
    return 0


if __name__ == "__main__":
    sys.exit(main())