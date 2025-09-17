from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from searchdet_pipeline.eval_classes.Example_datasets.ChickenDataset import ChickenDataset
from searchdet_pipeline.eval_classes.Dataset import Dataset
from searchdet_pipeline.eval_classes.Example_datasets.Dataset_without_changes import SimpleDataset

def main() -> None:
    parser = argparse.ArgumentParser(description="Demo для ChickenDataset и SimpleDataset")
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="coco json path",
    )
    args = parser.parse_args()
    default_json = Path(__file__).resolve().parents[2] / "_annotations.coco.json"
    json_path = Path(args.json) if args.json else default_json
    if not json_path.exists():
        print("coco file not found")
        return
        Chicken
    chicken_ds = ChickenDataset.from_path(json_path)
    simple_ds = SimpleDataset.from_path(json_path)
    print("ВСЕ АННОТАЦИИ")
    print(chicken_ds.annotations_len())
    print(simple_ds.annotations_len())
    print("КАТЕГОРИИ ФИЛЬТР")
    print(len(chicken_ds.filter_by_category("Chicken")))
    print(len(simple_ds.filter_by_category("Chicken")))
    print("МЕТА-ДАННЫЕ")
    print(chicken_ds.meta.get("total_annotations"))
    print(chicken_ds.meta.get("dataset_type"))
if __name__ == "__main__":
    main()