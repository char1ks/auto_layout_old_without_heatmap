from __future__ import annotations

from pathlib import Path
from typing import  List, Optional, Tuple,Union, Callable
from evaluate.dataset_model import DatasetModel
from evaluate.dataset import Dataset
from evaluate.detector_base import DetectorBase
from evaluate.metrics import (
    Metric, MetricOutputModel
)
from evaluate.coco_annotation import CocoAnnotation
from evaluate.tracer import Tracer
from evaluate.report_generator import ReportGenerator
from evaluate.context import Context
from evaluate.logging import get_logger
import numpy as np

logger = get_logger(__name__)

class Pipeline:
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
        self._predictions: List[CocoAnnotation] = []
        self._last_metrics: Optional[List[MetricOutputModel]] = None
        self._contexts: List[Context] = []
        
    def set_references(self,positive_dir: Union[str, Path],negative_dir: Optional[Union[str, Path]] = None,) -> None:
        pos_by_class, neg_imgs = self.detector.read_reference_images(positive_dir, negative_dir)
        self.detector.set_references(pos_by_class, neg_imgs)

    def detect_all(self,image_root: Optional[Union[str, Path]] = None,progress: Optional[Callable[[int, int, str | None], None]] = None,*args,**kwargs,) -> List[CocoAnnotation]:
        image_root = Path(image_root) if image_root is not None else None
        file_names: List[str] = sorted({str(getattr(ann, "file_name")) for ann in self.dataset_model.data_points if getattr(ann, "file_name", None) is not None})
        self._contexts = []
        predictions: List[CocoAnnotation] = []
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
            # Передаем callback только через kwargs, чтобы mypy не считал дублирование
            _kwargs = dict(kwargs)
            _kwargs.pop("callback", None)
            _kwargs["callback"] = _cb
            det_anns = self.detector.detect(image_np, file_name=fname, *args, **_kwargs)
            predictions.extend(det_anns)
            if progress:
                progress(idx, total, fname)

        self._predictions = predictions
        return predictions

    def evaluate(self,predictions: Optional[List[CocoAnnotation]] = None,average: str = "micro",) -> List[MetricOutputModel]:
        preds = predictions if predictions is not None else self._predictions
        results = []
        for metric in (self.metrics or []):
            try:
                result = metric.compute(self.dataset_model, preds, average=average)
                results.append(result)
            except Exception as e:
                logger.error(f"Ошибка при вычислении метрики {metric.name}: {e}")
        self._last_metrics = results
        return results

    def run(self,positive_dir: Union[str, Path],negative_dir: Optional[Union[str, Path]] = None,image_root: Optional[Union[str, Path]] = None,average: str = "micro",progress: Optional[Callable[[int, int, str | None], None]] = None,*args, dump_report: bool = False, report_output_dir: Optional[Union[str, Path]] = None, **kwargs,) -> Tuple[List[CocoAnnotation], List[MetricOutputModel]]:
        self.set_references(positive_dir, negative_dir)
        preds = self.detect_all(image_root=image_root, progress=progress, **kwargs)
        metrics = self.evaluate(preds, average=average)
        try:
            reporter = ReportGenerator()
            reporter.generate_report(self._contexts, metrics, dump_report=dump_report, output_dir=report_output_dir)
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
        return preds, metrics