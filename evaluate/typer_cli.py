from __future__ import annotations

import json
import importlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
EVAL_CLASSES_PATH = Path(__file__).parent
sys.path.insert(0, str(EVAL_CLASSES_PATH))
SEARCHDET_PIPELINE_PATH = Path(__file__).parent.parent / "searchdet_pipeline"
sys.path.insert(0, str(SEARCHDET_PIPELINE_PATH.parent))

from Example_datasets.ArchiveVOCDataset import ArchiveVOCDataset
from DatasetPoint import DatasetPoint
from metrics import (
    MetricOutputModel,
    MeanAveragePrecision,
    MeanIntersectionOverUnion,
    DiceCoefficient,
    ClassificationReportMetric,
)
from searchdet_pipeline.core.detector import SearchDetDetector
from searchdet_pipeline.core.config import get_preset_config
from Tracer import Tracer

App = typer.Typer()
console = Console()

from contextlib import contextmanager
import inspect
import builtins

@contextmanager
def suppress_print_from(prefixes: list[str]):
    orig_print = builtins.print
    def filtered_print(*args, **kwargs):
        try:
            st = inspect.stack()
            for rec in st[1:]:
                module = inspect.getmodule(rec[0])
                name = getattr(module, "__name__", None)
                if name and any(name.startswith(p) for p in prefixes):
                    return
                if name:
                    break
        except Exception:
            pass
        return orig_print(*args, **kwargs)
    builtins.print = filtered_print
    try:
        yield
    finally:
        builtins.print = orig_print


def _load_obj(dotted: str) -> Any:
    module_path, _, obj_name = dotted.replace(":", ".").rpartition(".")
    if not module_path:
        raise ValueError(f"Bad dotted path: {dotted!r}")
    mod = importlib.import_module(module_path)
    return getattr(mod, obj_name)


def _read_config_file(path: Path) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    suffix = Path(path).suffix.lower()
    if suffix in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("PyYAML is required: pip install pyyaml") from e
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError("YAML root must be a mapping")
        return data
    elif suffix == ".json":
        data = json.loads(text) or {}
        if not isinstance(data, dict):
            raise ValueError("JSON root must be an object")
        return data
    else:
        raise ValueError(f"Unsupported config extension {suffix!r}")


def _resolve_metrics(metric_specs: Optional[List[str]]) -> List[Any]:
    registry = {
        "mAP": MeanAveragePrecision,
        "MeanAveragePrecision": MeanAveragePrecision,
        "mIoU": MeanIntersectionOverUnion,
        "MeanIntersectionOverUnion": MeanIntersectionOverUnion,
        "dice": DiceCoefficient,
        "DiceCoefficient": DiceCoefficient,
        "clf_report": ClassificationReportMetric,
        "ClassificationReportMetric": ClassificationReportMetric,
    }
    if not metric_specs:
        return [MeanAveragePrecision(), MeanIntersectionOverUnion(), DiceCoefficient(), ClassificationReportMetric()]
    out = []
    for name in metric_specs:
        if name in registry:
            out.append(registry[name]())
        else:
            cls = _load_obj(name)
            out.append(cls())
    return out


class EvalCLI:
    def __init__(self, config_path: Path) -> None:
        self.config_path = Path(config_path)
        self.positive_dir: Union[str, Path] = "examples/positive"
        self.negative_dir: Optional[Union[str, Path]] = None
        self.output_dir: Optional[Path] = None
        self.run_name: Optional[str] = None
        self.save_predictions: bool = True
        self.save_metrics: bool = True

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
                {"metric_name": m.metric_name, "score": m.score, "stats": m.stats}
                for m in metrics
            ]
            metrics_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_from_config(self) -> tuple[Any, Any, List[Any]]:
        cfg = _read_config_file(self.config_path)
        ds_spec = (cfg.get("dataset") or {})
        det_spec = (cfg.get("detector") or {})
        run_spec = (cfg.get("run") or {})
        metrics_spec = cfg.get("metrics")
        self.positive_dir = run_spec.get("positive_dir", self.positive_dir)
        self.negative_dir = run_spec.get("negative_dir", self.negative_dir)
        ds_root = ds_spec.get("root") or ds_spec.get("dataset_dir") or ds_spec.get("path")
        if not ds_root:
            raise ValueError("dataset.root is required in config")
        DatasetClass = _load_obj(ds_spec.get("cls")) if ds_spec.get("cls") else ArchiveVOCDataset
        DetectorClass = _load_obj(det_spec.get("cls")) if det_spec.get("cls") else SearchDetDetector
        dataset = DatasetClass.from_path(
            Path(ds_root),
            ann_dir=Path(ds_spec["ann_dir"]) if ds_spec.get("ann_dir") else None,
            img_dir=Path(ds_spec["img_dir"]) if ds_spec.get("img_dir") else None,
            **(ds_spec.get("kwargs") or {}),
        )
        config = get_preset_config("balanced")
        detector = DetectorClass(config=config, **(det_spec.get("kwargs") or {}))
        metrics_list = _resolve_metrics(metrics_spec)
        return dataset, detector, metrics_list

    def run(self) -> int:
        out_dir = Path.cwd() / "results_cache"
        out_dir.mkdir(parents=True, exist_ok=True)
        dataset, detector, metrics_list = self._build_from_config()
        reporter = Tracer(to_stdout=True, trace_file=str(out_dir / "context_trace.jsonl"))
        with suppress_print_from(["searchdet_pipeline"]):
            preds, metrics = DatasetPoint(dataset=dataset, detector=detector, metrics=metrics_list, reporter=reporter).run(
                positive_dir=self.positive_dir,
                negative_dir=self.negative_dir,
                image_root=Path(dataset.root) if hasattr(dataset, "root") else None,
                dump_report=True,
                report_output_dir=out_dir,
            )
        self._save_artifacts(preds, metrics, out_dir)
        self._pretty_print(metrics, out_dir)
        console.print(Panel.fit("Evaluation completed", style="bold green"))
        return 0


@App.command("eval")
def eval_command(config: Path = typer.Argument(..., exists=True, readable=True, help="path_to_yaml_config")) -> int:
    return EvalCLI(config_path=config).run()


def main() -> None:
    App()


if __name__ == "__main__":
    main()
