from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from .ChickenDataset import ChickenDataset
from .Dataset import Dataset
from .Dataset_without_changes import SimpleDataset

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
    chicken_ds = ChickenDataset.from_path(json_path)
    simple_ds = SimpleDataset.from_path(json_path)
if __name__ == "__main__":
    main()