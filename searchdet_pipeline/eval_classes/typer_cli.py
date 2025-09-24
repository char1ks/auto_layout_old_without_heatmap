from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from searchdet_pipeline.eval_classes.Example_datasets.ArchiveVOCDataset import ArchiveVOCDataset
from searchdet_pipeline.eval_classes.DatasetPoint import DatasetPoint
from searchdet_pipeline.eval_classes.metrics import (
    MetricOutputModel, MeanAveragePrecision, MeanIntersectionOverUnion, DiceCoefficient, ClassificationReportMetric
)
from searchdet_pipeline.core.detector import SearchDetDetector
from searchdet_pipeline.core.config import get_preset_config
from searchdet_pipeline.eval_classes.Tracer import Tracer

App = typer.Typer()
console = Console()


class EvalCLI:
    def __init__(
        self,
        dataset_dir: Path,
        ann_dir: Optional[Path] = None,
        img_dir: Optional[Path] = None,
        positive_dir: Union[str, Path] = "examples/positive",
        negative_dir: Optional[Union[str, Path]] = None,
        average: str = "micro",
        output_dir: Optional[Path] = None,
        save_predictions: bool = True,
        save_metrics: bool = True,
        run_name: Optional[str] = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.ann_dir = Path(ann_dir) if ann_dir is not None else None
        self.img_dir = Path(img_dir) if img_dir is not None else None
        self.positive_dir = str(positive_dir)
        self.negative_dir = str(negative_dir) if negative_dir is not None else None
        self.average = average
        self.output_dir = output_dir
        self.save_predictions = save_predictions
        self.save_metrics = save_metrics
        self.run_name = run_name

    def _ensure_results_dir(self) -> Path:
        base = self.output_dir or (Path.cwd() / "results_cache")
        base.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = self.run_name or f"{self.dataset_dir.name}_{stamp}"
        run_dir = base / name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _pretty_print(self, metrics: List[MetricOutputModel], out_dir: Path) -> None:
        console.print(Panel.fit(f"Results saved to: [bold green]{out_dir}[/]", title="Output"))
        by_name: Dict[str, MetricOutputModel] = {m.metric_name: m for m in metrics}
        if "mAP" in by_name:
            m = by_name["mAP"]
            stats: Dict[str, Any] = m.stats or {}
            t = Table(title="Mean Average Precision (mAP)")
            t.add_column("Metric")
            t.add_column("Value", justify="right")
            t.add_row("mAP", f"{m.score:.4f}")
            if isinstance(stats.get("mAP@0.5"), (int, float)):
                t.add_row("mAP@0.5", f"{stats['mAP@0.5']:.4f}")
            if isinstance(stats.get("mAP@0.75"), (int, float)):
                t.add_row("mAP@0.75", f"{stats['mAP@0.75']:.4f}")
            console.print(t)
            if isinstance(stats.get("categories"), list) and isinstance(stats.get("per_class_ap"), list):
                cats = stats["categories"]
                aps = stats["per_class_ap"]
                tbl = Table(title="Per-class AP")
                tbl.add_column("Class")
                tbl.add_column("AP", justify="right")
                for c, a in zip(cats, aps):
                    try:
                        val = float(a)
                    except Exception:
                        val = 0.0
                    tbl.add_row(str(c), f"{val:.4f}")
                console.print(tbl)
        if "mIoU" in by_name:
            m = by_name["mIoU"]
            console.print(Panel.fit(f"mIoU: [bold]{m.score:.4f}[/]"))
        if "dice" in by_name:
            m = by_name["dice"]
            console.print(Panel.fit(f"Dice (F1): [bold]{m.score:.4f}[/]"))

    def _save_artifacts(self, preds, metrics: List[MetricOutputModel], out_dir: Path) -> None:
        if self.save_predictions:
            preds_path = out_dir / "predictions.jsonl"
            with preds_path.open("w", encoding="utf-8") as f:
                for p in preds:
                    rec = {
                        "file_name": getattr(p, "file_name", None),
                        "label": getattr(p, "label", None),
                        "bbox": getattr(p, "bbox", None),
                        "score": getattr(p, "score", getattr(p, "confidence", None)),
                        "width": getattr(p, "width", None),
                        "height": getattr(p, "height", None),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if self.save_metrics:
            metrics_path = out_dir / "metrics.json"
            serializable = [
                {
                    "metric_name": m.metric_name,
                    "score": m.score,
                    "stats": m.stats,
                }
                for m in metrics
            ]
            metrics_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")

    def run(self) -> int:
        out_dir = self._ensure_results_dir()
        config = get_preset_config("balanced")
        detector = SearchDetDetector(config=config)
        reporter = Tracer(to_stdout=True, trace_file=str(out_dir / "context_trace.jsonl"))
        dataset = ArchiveVOCDataset.from_path(self.dataset_dir, ann_dir=self.ann_dir, img_dir=self.img_dir)
        metrics_list = [MeanAveragePrecision(), MeanIntersectionOverUnion(), DiceCoefficient(), ClassificationReportMetric()]
        dp = DatasetPoint(dataset=dataset, detector=detector, metrics=metrics_list, reporter=reporter)
        image_root = self.img_dir if (self.img_dir is not None and self.img_dir.exists()) else None
        preds, metrics = dp.run(
            positive_dir=self.positive_dir,
            negative_dir=self.negative_dir,
            image_root=image_root,
            dump_report=True,
            report_output_dir=out_dir,
        )
        self._save_artifacts(preds, metrics, out_dir)
        self._pretty_print(metrics, out_dir)
        console.print(Panel.fit("Evaluation completed", style="bold green"))
        return 0


# Use the single App instance defined above
@App.command("eval")
def eval_command(
    dataset_dir: Path = typer.Argument(..., exists=True, readable=True, help="Path to dataset root (VOC-like)"),
    ann_dir: Optional[Path] = typer.Option(None, help="Path to annotations directory (optional)", exists=False),
    img_dir: Optional[Path] = typer.Option(None, help="Path to images directory (optional)", exists=False),
    positive_dir: Optional[Path] = typer.Option("examples/positive", help="Directory with positive reference images"),
    negative_dir: Optional[Path] = typer.Option(None, help="Directory with negative reference images"),
    average: str = typer.Option("micro", help="Averaging for evaluation (micro/macro)"),
    output_dir: Optional[Path] = typer.Option(None, help="Base output dir; a timestamped run subfolder will be created"),
    run_name: Optional[str] = typer.Option(None, help="Optional custom run subfolder name"),
    save_predictions: bool = typer.Option(True, help="Save predictions.jsonl"),
    save_metrics: bool = typer.Option(True, help="Save metrics.json"),
) -> int:
    evl = EvalCLI(
        dataset_dir=dataset_dir,
        ann_dir=ann_dir,
        img_dir=img_dir,
        positive_dir=str(positive_dir) if positive_dir is not None else "examples/positive",
        negative_dir=str(negative_dir) if negative_dir is not None else None,
        average=average,
        output_dir=output_dir,
        save_predictions=save_predictions,
        save_metrics=save_metrics,
        run_name=run_name,
    )
    return evl.run()


def main() -> None:
    App()


if __name__ == "__main__":
    main()