from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any, Union, Callable

import numpy as np

from searchdet_pipeline.eval_classes.DatasetModel import DatasetModel
from searchdet_pipeline.eval_classes.Dataset import Dataset
from searchdet_pipeline.eval_classes.detector_base import DetectorBase
from searchdet_pipeline.eval_classes.metrics import Metric, MetricOutputModel
from searchdet_pipeline.eval_classes.COCOAnnotations import COCOAnnotation
from searchdet_pipeline.eval_classes.ContextReporter import ContextReporter


class Dataset_Point:
    def __init__(self,dataset: Union[Dataset, DatasetModel],detector: DetectorBase,metric: Optional[Metric] = None, reporter: Optional[ContextReporter] = None,) -> None:
        if isinstance(dataset, Dataset):
            self.dataset_model: DatasetModel = dataset.data
        elif isinstance(dataset, DatasetModel):
            self.dataset_model = dataset
        else:
            raise TypeError("dataset must be Dataset or DatasetModel")

        self.detector: DetectorBase = detector
        self.metric: Metric = metric if metric is not None else Metric()
        self.reporter: Optional[ContextReporter] = reporter

        self._predictions: List[COCOAnnotation] = []
        self._last_metrics: Optional[MetricOutputModel] = None
    def set_references(self,positive_dir: Union[str, Path],negative_dir: Optional[Union[str, Path]] = None,) -> None:
        pos_by_class, neg_imgs = self.detector.read_reference_images(positive_dir, negative_dir)
        self.detector.set_references(pos_by_class, neg_imgs)

    def detect_all(self,image_root: Optional[Union[str, Path]] = None,progress: Optional[Callable[[int, int, str | None], None]] = None,*args,**kwargs,) -> List[COCOAnnotation]:
        image_root = Path(image_root) if image_root is not None else None
        file_names: List[str] = sorted({getattr(ann, "file_name", None) for ann in self.dataset_model.data_points if getattr(ann, "file_name", None)})

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
                if img_path is None or not img_path.exists():
                    if progress:
                        progress(idx, total, fname)
                    continue
                image_np = self.detector.read_input_img(img_path)
            cb = self.reporter.emit if self.reporter else None
            det_anns = self.detector.detect(image_np, callback=cb, file_name=fname, *args, **kwargs)
            predictions.extend(det_anns)
            if progress:
                progress(idx, total, fname)

        self._predictions = predictions
        return predictions

    def evaluate(self,predictions: Optional[List[COCOAnnotation]] = None,average: str = "micro",) -> MetricOutputModel:
        preds = predictions if predictions is not None else self._predictions
        result = self.metric.compute(self.dataset_model, preds, average=average)
        self._last_metrics = result
        return result

    def run(self,positive_dir: Union[str, Path],negative_dir: Optional[Union[str, Path]] = None,image_root: Optional[Union[str, Path]] = None,average: str = "micro",progress: Optional[Callable[[int, int, str | None], None]] = None,*args,**kwargs,) -> Tuple[List[COCOAnnotation], MetricOutputModel]:
        self.set_references(positive_dir, negative_dir)
        
        preds = self.detect_all(image_root=image_root, progress=progress, *args, **kwargs)
        metrics = self.evaluate(preds, average=average)
        return preds, metrics
    @property
    def predictions(self) -> List[COCOAnnotation]:
        return list(self._predictions)

    @property
    def last_metrics(self) -> Optional[MetricOutputModel]:
        return self._last_metrics