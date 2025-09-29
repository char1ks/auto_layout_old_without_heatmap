from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any, Union, Callable
import sys

import numpy as np
EVAL_CLASSES_PATH = Path(__file__).parent
sys.path.insert(0, str(EVAL_CLASSES_PATH))
from DatasetModel import DatasetModel
from Dataset import Dataset
from DetectorBase import DetectorBase
from metrics import (
    Metric, MetricOutputModel, MeanAveragePrecision, 
    MeanIntersectionOverUnion, DiceCoefficient
)
from COCOAnnotations import COCOAnnotation
from Tracer import Tracer
from ReportGenerator import ReportGenerator
from Context import Context


class DatasetPoint:
    def __init__(self,dataset: Union[Dataset, DatasetModel],detector: DetectorBase,metrics: Optional[List[Metric]] = None, reporter: Optional[Tracer] = None,) -> None:
        if isinstance(dataset, Dataset):
            self.dataset_model: DatasetModel = dataset.data
        elif isinstance(dataset, DatasetModel):
            self.dataset_model = dataset
        else:
            raise TypeError("dataset must be Dataset or DatasetModel")

        self.detector: DetectorBase = detector
        self.metrics: List[Metric] = metrics or []
        self.reporter: Optional[Tracer] = reporter
        self._predictions: List[COCOAnnotation] = []
        self._last_metrics: Optional[List[MetricOutputModel]] = None
        self._contexts: List[Context] = []
        
    def set_references(self,positive_dir: Union[str, Path],negative_dir: Optional[Union[str, Path]] = None,) -> None:
        pos_by_class, neg_imgs = self.detector.read_reference_images(positive_dir, negative_dir)
        self.detector.set_references(pos_by_class, neg_imgs)

    def detect_all(self,image_root: Optional[Union[str, Path]] = None,progress: Optional[Callable[[int, int, str | None], None]] = None,*args,**kwargs,) -> List[COCOAnnotation]:
        image_root = Path(image_root) if image_root is not None else None
        file_names: List[str] = sorted({str(getattr(ann, "file_name")) for ann in self.dataset_model.data_points if getattr(ann, "file_name", None) is not None})
        self._contexts = []
        predictions: List[COCOAnnotation] = []
        total = len(file_names)
        for idx, fname in enumerate(file_names, start=1):
            image_np: Optional[np.ndarray] = None
            for ann in self.dataset_model.data_points:
                if getattr(ann, "file_name", None) == fname and isinstance(getattr(ann, "img", None), np.ndarray):
                    image_np = getattr(ann, "img")
                    break
            if image_np is None:
                if image_root is None:
                    if progress:
                        progress(idx, total, fname)
                    continue
                img_path = (image_root / fname) if fname is not None else None
                if img_path is None:
                    if progress:
                        progress(idx, total, fname)
                    continue
                image_np = self.detector.read_input_img(img_path)
            def _cb(ctx: Context) -> None:
                try:
                    self._contexts.append(ctx)
                finally:
                    if self.reporter:
                        try:
                            self.reporter.emit(ctx)
                        except Exception:
                            pass
            det_anns = self.detector.detect(image_np, callback=_cb, file_name=fname, *args, **kwargs)
            predictions.extend(det_anns)
            if progress:
                progress(idx, total, fname)

        self._predictions = predictions
        return predictions

    def evaluate(self,predictions: Optional[List[COCOAnnotation]] = None,average: str = "micro",) -> List[MetricOutputModel]:
        preds = predictions if predictions is not None else self._predictions
        results = []
        for metric in (self.metrics or []):
            try:
                result = metric.compute(self.dataset_model, preds, average=average)
                results.append(result)
            except Exception as e:
                print(f"Ошибка при вычислении метрики {metric.name}: {e}")
        self._last_metrics = results
        return results

    def run(self,positive_dir: Union[str, Path],negative_dir: Optional[Union[str, Path]] = None,image_root: Optional[Union[str, Path]] = None,average: str = "micro",progress: Optional[Callable[[int, int, str | None], None]] = None,*args, dump_report: bool = False, report_output_dir: Optional[Union[str, Path]] = None, **kwargs,) -> Tuple[List[COCOAnnotation], List[MetricOutputModel]]:
        self.set_references(positive_dir, negative_dir)
        preds = self.detect_all(image_root=image_root, progress=progress, **kwargs)
        metrics = self.evaluate(preds, average=average)
        try:
            reporter = ReportGenerator()
            reporter.generate_report(self._contexts, metrics, dump_report=dump_report, output_dir=report_output_dir)
        except Exception as e:
            print(f"Report generation failed: {e}")
        return preds, metrics
    @property
    def predictions(self) -> List[COCOAnnotation]:
        return list(self._predictions)

    @property
    def last_metrics(self) -> Optional[List[MetricOutputModel]]:
        return self._last_metrics