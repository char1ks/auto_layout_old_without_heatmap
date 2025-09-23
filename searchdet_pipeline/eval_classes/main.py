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
    dataset_dir = Path("sheep-detection")
    ann_dir: Path | None = None
    img_dir: Path | None = None
    positive_dir: str | None = "examples/positive"
    negative_dir: str | None = None
    print("1212313")
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
        if arg in ("--positive", "--positive-dir", "-p") and i + 1 < len(args):
            positive_dir = args[i + 1]
            i += 2
            continue
        if arg in ("--negative", "--negative-dir", "-n") and i + 1 < len(args):
            negative_dir = args[i + 1]
            i += 2
            continue
        if not arg.startswith("-") and str(dataset_dir) == "sheep-detection":
            dataset_dir = Path(arg)
            i += 1
            continue
        i += 1

    dataset = ArchiveVOCDataset.from_path(dataset_dir)
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
        positive_dir=positive_dir if positive_dir is not None else "examples/positive",
        negative_dir=negative_dir,
        image_root=image_root,
        dump_report=True,
        report_output_dir=Path.cwd(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())