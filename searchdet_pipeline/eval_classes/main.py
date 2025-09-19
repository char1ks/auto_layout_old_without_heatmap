from __future__ import annotations

from pathlib import Path
import sys

from searchdet_pipeline.eval_classes.Example_datasets.ArchiveVOCDataset import ArchiveVOCDataset
from searchdet_pipeline.eval_classes.Dataset_Point import Dataset_Point
from searchdet_pipeline.eval_classes.metrics import Metric
from searchdet_pipeline.core.detector import SearchDetDetector
from searchdet_pipeline.core.config import get_preset_config
from searchdet_pipeline.eval_classes.ContextReporter import ContextReporter


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
    reporter = ContextReporter(to_stdout=True, trace_file="context_trace.jsonl")
    dp = Dataset_Point(dataset=dataset, detector=detector, reporter=reporter)

    image_root = img_dir if (img_dir is not None and img_dir.exists()) else None
    preds, metrics = dp.run(
        positive_dir=positive_dir,
        negative_dir=negative_dir,
        image_root=image_root,
    )
    
    print("РЕЗУЛЬТАТЫ МЕТРИК")
    
    if metrics:
        print(f"Основная метрика: {metrics.metric_name} = {metrics.score:.4f}")
        
        if hasattr(metrics, 'stats') and metrics.stats:
            stats = metrics.stats
            print(f"Micro IoU: {stats.get('mean_iou_micro', 0):.4f}")
            print(f"Macro IoU: {stats.get('mean_iou_macro', 0):.4f}")
            print(f"Micro Dice: {stats.get('dice_micro', 0):.4f}")
            print(f"Macro Dice: {stats.get('dice_macro', 0):.4f}")
            print(f"mAP (micro): {stats.get('mAP_micro', 0):.4f}")
            print(f"mAP50 (micro): {stats.get('mAP50_micro', 0):.4f}")
            print(f"mAP75 (micro): {stats.get('mAP75_micro', 0):.4f}")
            print(f"Совпавших изображений: {stats.get('num_images_matched', 0)}")
            per_class = stats.get('per_class', {})
            if per_class:
                print("\nСтатистика по классам:")
                for class_name, class_stats in per_class.items():
                    print(f"  {class_name}:")
                    print(f"    GT: {class_stats.get('num_gt', 0)}, Pred: {class_stats.get('num_pred', 0)}, Pairs: {class_stats.get('num_pairs', 0)}")
                    print(f"    IoU: {class_stats.get('mean_iou', 0):.4f}, Dice: {class_stats.get('mean_dice', 0):.4f}")
                    print(f"    AP: {class_stats.get('AP', 0):.4f}, AP50: {class_stats.get('AP50', 0):.4f}, AP75: {class_stats.get('AP75', 0):.4f}")
    else:
        print("Метрики не получены")
    
    
    return 0


if __name__ == "__main__":
    sys.exit(main())