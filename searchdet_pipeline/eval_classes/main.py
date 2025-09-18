from __future__ import annotations

from pathlib import Path
import sys

from searchdet_pipeline.eval_classes.Example_datasets.ArchiveVOCDataset import ArchiveVOCDataset
from searchdet_pipeline.eval_classes.Dataset_Point import Dataset_Point
from searchdet_pipeline.eval_classes.metrics import Metric
from searchdet_pipeline.core.detector import SearchDetDetector
from searchdet_pipeline.core.config import get_preset_config


def main() -> int:
    # Параметры запуска: !python -m searchdet_pipeline.eval_classes.main archive --ann-dir archive/annotations/ --img-dir archive/images
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

    print(f"[i] Рабочая папка: {Path.cwd().resolve()}")
    print(f"[i] Указанный путь к датасету: {dataset_dir} -> {dataset_dir.resolve()}")
    if ann_dir is not None:
        print(f"[i] Папка аннотаций (--ann-dir): {ann_dir} -> {ann_dir.resolve()}")
    if img_dir is not None:
        print(f"[i] Папка изображений (--img-dir): {img_dir} -> {img_dir.resolve()}")

    if not dataset_dir.exists():
        print(f"[!] Папка с датасетом не найдена: {dataset_dir.resolve()}")
        return 1

    positive_dir = "examples/positive"
    negative_dir = None

    dataset = ArchiveVOCDataset.from_path(dataset_dir, ann_dir=ann_dir, img_dir=img_dir)

    print(f"🔍 Загружен датасет: {len(dataset.data_points)} аннотаций")

    for i, ann in enumerate(dataset.data_points[:3]):
        print(f"  📋 Аннотация {i+1}:")
        print(f"    - file_name: {ann.file_name}")
        print(f"    - label: {ann.label}")
        print(f"    - image_size: {ann.image_size}")
        print(f"    - img shape: {ann.img.shape if ann.img is not None else 'None'}")
        print(f"    - mask shape: {ann.mask.shape if ann.mask is not None else 'None'}")

    config = get_preset_config("balanced")
    detector = SearchDetDetector(config=config)
    dp = Dataset_Point(dataset=dataset, detector=detector, metric=Metric())

    image_root = img_dir if (img_dir is not None and img_dir.exists()) else None
    preds, metrics = dp.run(
        positive_dir=positive_dir,
        negative_dir=negative_dir,
        image_root=image_root,
    )
    print(f"Предсказаний: {len(preds)}")
    print(f"Метрика {metrics.metric_name}: {metrics.score:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())