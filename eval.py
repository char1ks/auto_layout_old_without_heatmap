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


def detections_to_coco(found: List[dict],file_name: str,image_np: np.ndarray,) -> List[COCOAnnotation]:
    H, W = image_np.shape[:2]
    preds: List[COCOAnnotation] = []
    for f in found:
        mask_dict = f.get("mask", {})
        seg = mask_dict.get("segmentation")
        if seg is None:
            continue
        seg_np = np.array(seg).astype(bool)
        bbox = f.get("bbox", mask_dict.get("bbox", [0, 0, 0, 0]))
        area = int(seg_np.sum()) if seg_np.size > 0 else int(mask_dict.get("area", 0))
        label = f.get("class", "unknown")
        conf = float(f.get("confidence", mask_dict.get("confidence", 0.0)))

        ann = COCOAnnotation(
            img=image_np,
            mask=seg_np,
            label=label,
            width=W,
            height=H,
            area=float(area),
            bbox=[int(b) for b in bbox],
            image_resolution=(W, H),
            file_name=file_name,
        )
        setattr(ann, "score", conf)
        preds.append(ann)
    return preds


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

    # 2) Готовим детектор и референсы
    detector = SearchDetDetector()
    pos_by_class, neg_imgs = detector.read_reference_images(
        positive_dir=args.positive_dir, negative_dir=args.negative_dir
    )
    detector.set_references(pos_by_class, neg_imgs)

    # 3) Прогоняем детектор по каждому изображению один раз и конвертируем в COCOAnnotation для метрик
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

        result = detector.find_present_elements(image_np)
        found = result.get("found_elements", [])
        preds = detections_to_coco(found, file_name=fname, image_np=image_np)
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
        if mean_iou is not None:
            print(f"Mean IoU: {mean_iou:.4f}")
        if map_score is not None:
            print(f"mAP@[0.50:0.95]: {map_score:.4f}")


if __name__ == "__main__":
    main()