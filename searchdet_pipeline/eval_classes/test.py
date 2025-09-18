from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Any

from searchdet_pipeline.eval_classes.Example_datasets.ArchiveVOCDataset import ArchiveVOCDataset


def main() -> int:
    dataset_dir = Path("archive")
    if not dataset_dir.exists():
        print(f"Папка с датасетом не найдена: {dataset_dir.resolve()}")
        return 1

    try:
        dataset = ArchiveVOCDataset.from_path(dataset_dir)
    except Exception as e:
        print(f"Ошибка при загрузке датасета из {dataset_dir}: {e}")
        return 2
    dm = dataset.data
    anns = list(dm.data_points)
    total_annotations = len(anns)
    file_names = sorted({getattr(a, "file_name", None) for a in anns if getattr(a, "file_name", None)})
    total_images = len(file_names)
    categories = sorted({str(getattr(a, "label", "")) for a in anns if getattr(a, "label", None) is not None})
    print("Информация о датасете")
    print(f"Имя датасета: {dataset.name}")
    print(f"Корневая папка: {dataset_dir.resolve()}")
    print(f"Всего изображений: {total_images}")
    print(f"Всего аннотаций: {total_annotations}")
    print(f"Категории ({len(categories)}): {categories}")

    per_class: Dict[Any, int] = {}
    for a in anns:
        label = getattr(a, "label", None)
        if label is None:
            continue
        per_class[label] = per_class.get(label, 0) + 1

    if per_class:
        print("Аннотаций по категориям:")
        for label in sorted(per_class, key=lambda x: str(x)):
            print(f"  - {label}: {per_class[label]}")
    sample_n = min(5, total_annotations)
    if sample_n > 0:
        print(f"\nПервые {sample_n} аннотаций:")
        for a in anns[:sample_n]:
            file_name = getattr(a, "file_name", None)
            label = getattr(a, "label", None)
            bbox = getattr(a, "bbox", None)
            width = getattr(a, "width", None)
            height = getattr(a, "height", None)
            print(f"- file: {file_name} | label: {label} | bbox: {bbox} | size: {width}x{height}")
    try:
        print("\nМетаданные датасета:")
        print(dm.meta)
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())