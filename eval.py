from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from searchdet_pipeline.core.detector import SearchDetDetector
from searchdet_pipeline.eval_classes.ChickenDataset import ChickenDataset
from searchdet_pipeline.eval_classes.Dataset_without_changes import SimpleDataset
from searchdet_pipeline.eval_classes.DatasetModel import DatasetModel
from searchdet_pipeline.eval_classes.COCOAnnotations import COCOAnnotation
from searchdet_pipeline.eval_classes.metrics import Metric


def load_dataset(json_path: Path, dataset_type: str) -> Tuple[DatasetModel, Dict[str, List[COCOAnnotation]]]:
    if dataset_type == "chicken":
        ds = ChickenDataset.from_path(json_path)
    else:
        ds = SimpleDataset.from_path(json_path)
    gt: DatasetModel = ds

    gt_by_file: Dict[str, List[COCOAnnotation]] = {}
    for ann in gt.data_points:
        fname = getattr(ann, "file_name", "") or ""
        gt_by_file.setdefault(fname, []).append(ann)
    return gt, gt_by_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка детектора на COCO-подобном датасете c мерами IoU/mAP")
    parser.add_argument("--json", type=str, required=True, help="Путь к COCO JSON аннотациям")
    parser.add_argument("--dataset_type", type=str, default="simple", choices=["simple", "chicken"], help="Тип датасета из eval_classes")
    parser.add_argument("--positive_dir", type=str, required=True, help="Директория с позитивными образцами (по классам)")
    parser.add_argument("--negative_dir", type=str, default=None, help="Директория с негативными образцами (опционально)")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число изображений для быстрой проверки")
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        print("coco file not found:", json_path)
        return
    gt, gt_by_file = load_dataset(json_path, args.dataset_type)
    detector = SearchDetDetector()
    pos_by_class, neg_imgs = detector.read_reference_images(
        positive_dir=args.positive_dir, negative_dir=args.negative_dir
    )
    detector.set_references(pos_by_class, neg_imgs)
    predictions: List[COCOAnnotation] = []
    timing_total = []

    all_files = list(gt_by_file.keys())
    if args.limit is not None:
        all_files = all_files[: max(0, args.limit)]

    for i, fname in enumerate(all_files, start=1):
        anns = gt_by_file.get(fname, [])
        if not anns:
            continue
        image_np = anns[0].img
        if image_np is None or not isinstance(image_np, np.ndarray) or image_np.size == 0:
            continue
        preds = detector.detect(image_np, file_name=fname)
        predictions.extend(preds)

    metric = Metric()
    metric_out = metric.compute(gt=gt, prediction=predictions)
    print("РЕЗЫ")
    print(f"фоток обработано: {len(all_files)}")
    if timing_total:
        print(f"avg time на изображение: {np.mean(timing_total):.3f}с")
        print(f"sum time: {np.sum(timing_total):.3f}с")
    print(f"metric: {metric_out.metric_name}")
    print(f"Score (mean IoU): {metric_out.score:.4f}")
    stats = metric_out.stats or {}
    if stats:
        mean_iou = stats.get("mean_iou")
        map_score = stats.get("mAP")
        map_50 = stats.get("mAP50")
        map_75 = stats.get("mAP75")
        dice = stats.get("dice")
        if mean_iou is not None:
            print(f"Mean IoU (Jaccard): {mean_iou:.4f}")
        if dice is not None:
            print(f"Mean Dice: {dice:.4f}")
        if map_score is not None:
            print(f"mAP@[0.50:0.95]: {map_score:.4f}")
        if map_50 is not None:
            print(f"mAP@0.50: {map_50:.4f}")
        if map_75 is not None:
            print(f"mAP@0.75: {map_75:.4f}")


if __name__ == "__main__":
    main()