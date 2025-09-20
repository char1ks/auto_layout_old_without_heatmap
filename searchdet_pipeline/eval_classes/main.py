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
    # Отчёт и консольное резюме выводит ReportGenerator; отдельный вывод метрик здесь не требуется.
    return 0


if __name__ == "__main__":
    sys.exit(main())