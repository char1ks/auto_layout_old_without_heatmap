from __future__ import annotations

from pathlib import Path
import sys

from searchdet_pipeline.eval_classes.Example_datasets.ChickenDataset import ChickenDataset
from searchdet_pipeline.eval_classes.Dataset_Point import Dataset_Point
from searchdet_pipeline.eval_classes.metrics import Metric
from searchdet_pipeline.core.detector import SearchDetDetector
from searchdet_pipeline.core.config import get_preset_config


def main() -> int:
    json = Path("path/to/.json")
    positive_dir = Path("path/to/positive_dir")
    negative_dir = None  
    dataset = ChickenDataset.from_path(json)
    config = get_preset_config("balanced")
    detector = SearchDetDetector(config=config)

    dp = Dataset_Point(dataset=dataset, detector=detector, metric=Metric())
    preds, metrics = dp.run(positive_dir=positive_dir, negative_dir=negative_dir)

    print(f"Предсказаний: {len(preds)}")
    print(f"Метрика {metrics.metric_name}: {metrics.score:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())