from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
from dataclasses import asdict, is_dataclass
from evaluation.pipeline import Pipeline
from evaluation.metrics import (
    Metric,
    MetricOutputModel,
    MeanAveragePrecision,
    MeanIntersectionOverUnion,
    DiceCoefficient,
    ClassificationReportMetric,
)
from evaluation.report_generator import ReportGenerator
from evaluation.tracer import Tracer
from evaluation.log_utils import get_logger
from evaluation.dataset import Dataset
from flashbone.core.detection.base import DetectorBase
logger = get_logger(__name__)

_METRIC_BY_NAME: Dict[str, type[Metric]] = {
    "mAP": MeanAveragePrecision,
    "MeanAveragePrecision": MeanAveragePrecision,
    "mIoU": MeanIntersectionOverUnion,
    "MeanIntersectionOverUnion": MeanIntersectionOverUnion,
    "dice": DiceCoefficient,
    "DiceCoefficient": DiceCoefficient,
    "clf_report": ClassificationReportMetric,
    "ClassificationReportMetric": ClassificationReportMetric,
}

class Eval:
    def __init__(
        self,
        dataset: Dataset,
        detector: DetectorBase,
        metrics: Optional[List[Union[str, Metric]]] = None,
    ) -> None:
        self.dataset = dataset
        self.detector = detector
        self.metrics: List[Metric] = self._normalize_metrics(metrics)

    def _normalize_metrics(self, metrics: Optional[List[Union[str, Metric]]]) -> List[Metric]:
        if not metrics:
            metrics = ["mAP", "mIoU", "dice", "clf_report"]
        result: List[Metric] = []
        for m in metrics:
            if isinstance(m, Metric):
                result.append(m)
                continue
            if isinstance(m, str):
                cls = _METRIC_BY_NAME.get(m)
                if cls:
                    result.append(cls())
            if isinstance(m, type) and issubclass(m, Metric):
                result.append(m())
                continue

            logger.warning(f"Unsupported metric spec '{m}'; skipping.")
        return result

    def run(
        self,
        positive_dir: Union[str, Path],
        negative_dir: Optional[Union[str, Path]] = None,
        image_root: Optional[Union[str, Path]] = None,
        average: str = "micro",
        output_dir: Optional[Union[str, Path]] = None,
        dump_report: bool = True,
        enable_profile: bool = False,
        progress: Optional[Any] = None,
        **kwargs: Any,
    ) -> tuple[List[Any], List[MetricOutputModel], Path]:
        out_dir = Path(output_dir) if output_dir else (Path.cwd() / "results_cache")
        out_dir.mkdir(parents=True, exist_ok=True)

        tracer = Tracer(to_stdout=True, trace_file=str(out_dir / "context_trace.jsonl"))

        pipeline = Pipeline(dataset=self.dataset, detector=self.detector, metrics=self.metrics, reporter=tracer)
        pipeline.set_references(positive_dir, negative_dir)

        progress_cb = progress or (lambda i, t, f: logger.info(f"[{i}/{t}] {f or ''}"))
        image_root = Path(image_root) if image_root is not None else None

        if enable_profile:
            with tracer.profile(
                sort="cumtime",
                top_k=50,
                dump_path=str(out_dir / "profiler.prof"),
                flamegraph_path=str(out_dir / "flamegraph.svg"),
            ):
                preds = pipeline.detect_all(image_root=image_root, progress=progress_cb, **kwargs)
                metrics = pipeline.evaluate(preds, average=average)
        else:
            preds = pipeline.detect_all(image_root=image_root, progress=progress_cb, **kwargs)
            metrics = pipeline.evaluate(preds, average=average)

        ReportGenerator().generate_report(pipeline._contexts, metrics, dump_report=dump_report, output_dir=out_dir)
        self._save_artifacts(preds, metrics, out_dir)
        return preds, metrics, out_dir

    def _save_artifacts(self, preds: List[Any], metrics: List[MetricOutputModel], out_dir: Path) -> None:

        preds_path = out_dir / "predictions.jsonl"
        with preds_path.open("w", encoding="utf-8") as f:
            for pd in preds:
                if not hasattr(pd, "to_dict"):
                    continue
                rec = pd.to_dict()
                serializable = {
                    "file_name": rec.get("file_name"),
                    "label": rec.get("label"),
                    "mask": rec.get("mask"),
                    "bbox": rec.get("bbox"),
                    "score": (rec.get("score") if "score" in rec else rec.get("confidence")),
                    "width": rec.get("width"),
                    "height": rec.get("height"),
                }
                f.write(json.dumps(serializable, ensure_ascii=False) + "\n")

        metrics_path = out_dir / "metrics.json"
        serializable: List[Dict[str, Any]] = []
        for m in metrics:
            stats_obj = m.stats
            if isinstance(stats_obj, dict):
                stats_ser = stats_obj
            elif is_dataclass(stats_obj):
                stats_ser = asdict(stats_obj)
            else:
                stats_ser = str(stats_obj)
            serializable.append({"metric_name": m.metric_name, "score": m.score, "stats": stats_ser})
        metrics_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")