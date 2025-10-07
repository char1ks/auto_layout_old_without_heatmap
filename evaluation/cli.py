from __future__ import annotations
import json
import importlib
import sys
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import inspect
from dataclasses import asdict, is_dataclass
import builtins
from contextlib import contextmanager
import contextlib
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import yaml 
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from evaluation.log_utils import setup_logging, get_logger
setup_logging()
from evaluation.example_datasets.voc_dataset import VocDataset
from evaluation.pipeline import Pipeline
from evaluation.metrics import (
    MetricOutputModel,
    MeanAveragePrecision,
    MeanIntersectionOverUnion,
    DiceCoefficient,
    ClassificationReportMetric,
)
from evaluation.tracer import Tracer
from evaluation.report_generator import ReportGenerator
from flashbone.core.detection.searchdet_detector import SearchDetDetector
from ultralytics import FastSAM
from flashbone.core.encoding.dinov3.image import DinoV3EncoderGaz
from flashbone.core.segmentation import SamSegmenter, SegmenterConfig
from flashbone.core.heatmap_generation import HeatmapGenerator
from flashbone.core.classification.mask_classifier_knn import MaskClassifierKNN
from flashbone.core.image_resizing import ImageResizer, ResizeContext
App = typer.Typer()
console = Console()
logger = get_logger(__name__)

@contextmanager
def suppress_print_from(prefixes: list[str]):
    orig_print = builtins.print
    def filtered_print(*args, **kwargs):
        try:
            st = inspect.stack()
            for rec in st[1:]:
                module = inspect.getmodule(rec[0])
                name = module.__name__ if module is not None else None
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
    return vars(mod)[obj_name]


def _read_config_file(path: Path) -> Dict[str, Any]:
    try:
        text = Path(path).read_text(encoding="utf-8")
        suffix = Path(path).suffix.lower()
        
        if suffix in {".yml", ".yaml"}:
            data = yaml.safe_load(text)
            return data if isinstance(data, dict) else {}
        elif suffix == ".json":
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        else:
            return {}
    except Exception:
        return {}


 


def _odict(obj: Any) -> Dict[str, Any]:
    return obj.__dict__ if "__dict__" in dir(obj) else {}


class SortedClassifier:
    def __init__(self, classifier):
        self.classifier = classifier
    
    def __getattr__(self, name):
        return getattr(self.classifier, name)
    
    def predict(self, request):
        predictions = self.classifier.predict(request)
        if predictions:
            predictions.sort(key=lambda x: x.score, reverse=True)
        return predictions


class EvalCLI:
    def __init__(self, config_path: Path) -> None:
        self.config_path = Path(config_path)
        self.positive_dir: Union[str, Path] = "examples/positive"
        self.negative_dir: Optional[Union[str, Path]] = None
        self.output_dir: Optional[Path] = None
        self.run_name: Optional[str] = None
        self.save_predictions: bool = True
        self.save_metrics: bool = True
        self.enable_profile: bool = True

    def _pretty_print(self, metrics: List[MetricOutputModel], out_dir: Path) -> None:
        console.print(Panel.fit(f"Results saved to: [bold green]{out_dir}[/]", title="Output"))
        by_name: Dict[str, MetricOutputModel] = {m.metric_name: m for m in metrics}
        if "mAP" in by_name:
            m = by_name["mAP"]
            stats_obj = m.stats
            stats: Dict[str, Any]
            if isinstance(stats_obj, dict):
                stats = stats_obj
            elif is_dataclass(stats_obj):
                stats = asdict(stats_obj).get("data", {})
            else:
                stats = {}
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
                    val = float(a) if isinstance(a, (int, float)) or (isinstance(a, str) and a.replace(".", "", 1).isdigit()) else 0.0
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
                    pd = _odict(p)
                    rec = {
                        "file_name": pd.get("file_name"),
                        "label": pd.get("label") ,
                        "bbox": pd.get("bbox"),
                        "score": (pd["score"] if "score" in pd else pd.get("confidence")),
                        "width": pd.get("width"),
                        "height": pd.get("height"),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if self.save_metrics:
            metrics_path = out_dir / "metrics.json"
            serializable = []
            for m in metrics:
                stats_obj = m.stats
                stats_ser: Union[Dict[str, Any], str]
                if isinstance(stats_obj, dict):
                    stats_ser = stats_obj
                elif is_dataclass(stats_obj):
                    stats_ser = asdict(stats_obj)
                else:
                    stats_ser = str(stats_obj)
                serializable.append({"metric_name": m.metric_name, "score": m.score, "stats": stats_ser})
            metrics_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    def _build_from_config(self) -> tuple[Any, Any, List[Any]]:
        cfg = _read_config_file(self.config_path); ds = cfg.get("dataset") or {}; det = cfg.get("detector") or {}; run = cfg.get("run") or {}
        self.positive_dir = run.get("positive_dir", self.positive_dir); self.negative_dir = run.get("negative_dir", self.negative_dir); self.enable_profile = bool(run.get("enable_profile", True))
        ds_root = ds.get("root") or ds.get("dataset_dir") or ds.get("path")
        
        DatasetClass = _load_obj(ds["cls"]) if isinstance(ds.get("cls"), str) else VocDataset
        DetectorClass = _load_obj(det["cls"]) if isinstance(det.get("cls"), str) else SearchDetDetector
        dataset = DatasetClass.from_path(Path(ds_root), ann_dir=Path(ds["ann_dir"]) if ds.get("ann_dir") else None, img_dir=Path(ds["img_dir"]) if ds.get("img_dir") else None, **(ds.get("kwargs") or {}))
        if DetectorClass is SearchDetDetector:
            kwargs = det.get("kwargs") or {}
            encoder = kwargs.get("encoder") or DinoV3EncoderGaz(); heatmap_generator = kwargs.get("heatmap_generator") or HeatmapGenerator(dino_fe=encoder, use_cosine_similarity_for_heatmap=True, threshold_cosine=0.3)
            sam_model = kwargs.get("sam_model") or FastSAM("FastSAM-x.pt"); seg_cfg = kwargs.get("segmenter_config") or SegmenterConfig(min_mask_area=200, confidence_threshold=0.5, iou_threshold=0.8, mask_threshold=0.5)
            segmenter = kwargs.get("segmenter") or SamSegmenter(sam_model=sam_model, config=seg_cfg)
            base_classifier = kwargs.get("classifier") or MaskClassifierKNN(encoder=encoder, d=1024); classifier = SortedClassifier(base_classifier)
            image_resizer = kwargs.get("image_resizer") or ImageResizer(max_side=1024)
            _orig_resize = image_resizer.resize; 
            image_resizer.resize = lambda img: (lambda o,c: (o, ResizeContext(scale=float(c.get("scale",1.0)), orig_shape=tuple(c.get("orig_shape", o.shape[:2])))) if isinstance(c, dict) else (o, c)) (*_orig_resize(img))
            detector = SearchDetDetector(segmenter=segmenter, classifier=classifier, heatmap_generator=heatmap_generator, image_resizer=image_resizer)
        else:
            detector = DetectorClass(**(det.get("kwargs") or {}))
        return dataset, detector, [(({"mAP": MeanAveragePrecision, "MeanAveragePrecision": MeanAveragePrecision, "mIoU": MeanIntersectionOverUnion, "MeanIntersectionOverUnion": MeanIntersectionOverUnion, "dice": DiceCoefficient, "DiceCoefficient": DiceCoefficient, "clf_report": ClassificationReportMetric, "ClassificationReportMetric": ClassificationReportMetric}).get(n) or _load_obj(n))() for n in (cfg.get("metrics") or ["mAP","mIoU","dice","clf_report"])]

    def run(self) -> int:
        out_dir = Path.cwd() / "results_cache"; out_dir.mkdir(parents=True, exist_ok=True)
        dataset, detector, metrics_list = self._build_from_config()
        try:
            dp = getattr(getattr(dataset, "data", None), "data_points", None)
            if isinstance(dp, list) and len(dp) > 100:
                dataset.data.data_points = dp[:100]; logger.info(f"Dataset truncated to 100 images (from {len(dp)}).")
        except Exception:
            pass
        tracer = Tracer(to_stdout=True, trace_file=str(out_dir / "context_trace.jsonl"))
        progress = lambda i, t, f: logger.info(f"[{i}/{t}] {f or ''}")
        with suppress_print_from(["searchdet_pipeline"]):
            ctx = tracer.profile(sort="cumtime", top_k=50, dump_path=str(out_dir / "profiler.prof"), flamegraph_path=str(out_dir / "flamegraph.svg")) if self.enable_profile else contextlib.nullcontext()
            with ctx:
                image_root = Path(getattr(dataset, "root", "")) if getattr(dataset, "root", None) else None
                pipeline = Pipeline(dataset=dataset, detector=detector, metrics=metrics_list, reporter=tracer)
                pipeline.set_references(self.positive_dir, self.negative_dir)
                preds = pipeline.detect_all(image_root=image_root, progress=progress)
                metrics = pipeline.evaluate(preds, average="micro")
                ReportGenerator().generate_report(pipeline._contexts, metrics, dump_report=True, output_dir=out_dir)
        self._save_artifacts(preds, metrics, out_dir); self._pretty_print(metrics, out_dir)
        console.print(Panel.fit("Evaluation completed", style="bold green")); return 0


@App.command("eval")
def eval_command(config: Path = typer.Argument(..., exists=True, readable=True, help="path_to_yaml_config")) -> int:
    return EvalCLI(config_path=config).run()

if __name__ == "__main__":
    App()