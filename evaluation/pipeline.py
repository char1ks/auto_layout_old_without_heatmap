from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union, Callable, Any, Dict
import numpy as np
from PIL import Image, ImageDraw
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))
from evaluation.dataset_model import DatasetModel
from evaluation.dataset import Dataset
from evaluation.metrics import Metric, MetricOutputModel
from evaluation.coco_annotation import CocoAnnotation
from evaluation.tracer import Tracer
from evaluation.report_generator import ReportGenerator
from evaluation.context import Context
from evaluation.log_utils import get_logger
from flashbone.core.detection.base import DetectorBase, DetectionResult
logger = get_logger(__name__)

class Pipeline:
    def __init__(self,dataset: Union[Dataset, DatasetModel],detector: DetectorBase,metrics: Optional[List[Metric]] = None,reporter: Optional[Tracer] = None,) -> None:
        if isinstance(dataset, Dataset):
            self.dataset_model: DatasetModel = dataset.data
        elif isinstance(dataset, DatasetModel):
            self.dataset_model = dataset
        self.detector: DetectorBase = detector
        self.metrics: List[Metric] = metrics or []
        self.reporter: Optional[Tracer] = reporter
        self._predictions: List[CocoAnnotation] = []
        self._last_metrics: Optional[List[MetricOutputModel]] = None
        self._contexts: List[Context] = []
    def _read_reference_images(self, positive_dir: Union[str, Path], negative_dir: Optional[Union[str, Path]] = None) -> Tuple[dict[str, List[Image.Image]], List[Image.Image]]:
        pos_by_class: dict[str, List[Image.Image]] = {}
        neg_imgs: List[Image.Image] = []

        pdir = Path(positive_dir)
        if pdir.exists():
            for sub in sorted([p for p in pdir.iterdir() if p.is_dir()]):
                cls_name = sub.name
                imgs: List[Image.Image] = []
                for img_path in sorted(sub.glob("*")):
                    try:
                        with Image.open(img_path) as im:
                            imgs.append(im.convert("RGB"))
                    except Exception:
                        continue
                if imgs:
                    pos_by_class[cls_name] = imgs

        if negative_dir is not None:
            npath = Path(negative_dir)
            if npath.exists():
                for img_path in sorted(npath.glob("*")):
                    try:
                        with Image.open(img_path) as im:
                            neg_imgs.append(im.convert("RGB"))
                    except Exception:
                        continue

        return pos_by_class, neg_imgs

    def _read_input_img(self, img_path: Path) -> np.ndarray:
        try:
            return np.array(Image.open(img_path).convert("RGB"))
        except Exception:
            return np.zeros((1, 1, 3), dtype=np.uint8)

    def _dets_to_coco(self, dets: Union[List[DetectionResult], List[Dict[str, Any]], Dict[str, Any], DetectionResult], image_np: np.ndarray, file_name: str) -> List[CocoAnnotation]:
        H, W = image_np.shape[:2]
        if not isinstance(dets, list):
            dets = [dets]  

        def as_dict(d: Union[DetectionResult, Dict[str, Any]]) -> Dict[str, Any]:
            if isinstance(d, dict):
                return d
            keys = ("polygons", "bbox", "score", "area", "class_id")
            return {k: getattr(d, k, None) for k in keys}

        def to_float(x: Any, default: Optional[float] = None) -> Optional[float]:
            try:
                return float(x)
            except Exception:
                return default

        anns: List[CocoAnnotation] = []
        for d in dets:  
            dd = as_dict(d)
            polygons = dd.get("polygons") or []
            bbox = dd.get("bbox")
            bbox = [float(v) for v in bbox] if isinstance(bbox, list) and len(bbox) == 4 else [0.0, 0.0, float(W), float(H)]

            mask_np = np.zeros((H, W), np.uint8)
            if polygons:
                pil = Image.fromarray(mask_np, "L")
                draw = ImageDraw.Draw(pil)
                for poly in (p for p in polygons if isinstance(p, list) and len(p) >= 3):
                    xy = [(int(pt[0]), int(pt[1])) for pt in poly if isinstance(pt, (list, tuple)) and len(pt) >= 2]
                    if len(xy) >= 3:
                        draw.polygon(xy, outline=1, fill=1)
                mask_np = np.array(pil, np.uint8)
            if not mask_np.any():
                x, y, w, h = bbox
                x1, y1 = int(max(0, np.floor(x))), int(max(0, np.floor(y)))
                x2 = int(min(W, np.ceil(x + w)))
                y2 = int(min(H, np.ceil(y + h)))
                if x2 > x1 and y2 > y1:
                    mask_np[y1:y2, x1:x2] = 1

            area = to_float(dd.get("area"), None) or float(bbox[2] * bbox[3])
            class_val = dd.get("class_id")
            label = str(int(class_val)) if isinstance(class_val, (int, float, np.integer)) else str(class_val)
            score = to_float(dd.get("score"), None)

            anns.append(CocoAnnotation(img=image_np, mask=mask_np, label=label, image_size=(int(W), int(H)), width=int(W), height=int(H), area=area, file_name=file_name, bbox=bbox, score=score))
        return anns

    def set_references(self,positive_dir: Union[str, Path],negative_dir: Optional[Union[str, Path]] = None,) -> None:
        pos_by_class, neg_imgs = self._read_reference_images(positive_dir, negative_dir)
        self.detector.set_references(pos_by_class, neg_imgs)

    def detect_all(self,image_root: Optional[Union[str, Path]] = None,progress: Optional[Callable[[int, int, str | None], None]] = None,*args,**kwargs,) -> List[CocoAnnotation]:
        image_root = Path(image_root) if image_root is not None else None

        file_names: List[str] = sorted({
            str(ann.file_name)
            for ann in self.dataset_model.data_points
            if ann.file_name is not None
        })

        self._contexts = []
        predictions: List[CocoAnnotation] = []
        total = len(file_names)

        for idx, fname in enumerate(file_names, start=1):
            logger.info(f"[{idx}/{total}] start {fname}")
            image_array: Optional[np.ndarray] = None

            for ann in self.dataset_model.data_points:
                if ann.file_name == fname and isinstance(ann.img, np.ndarray):
                    image_array = ann.img
                    break

            if image_array is None:
                if image_root is None:
                    if progress:
                        progress(idx, total, fname)
                    logger.info(f"[{idx}/{total}] skipped (no image in dataset) {fname}")
                    continue
                img_path = (image_root / fname) if fname is not None else None
                if img_path is None:
                    if progress:
                        progress(idx, total, fname)
                    logger.info(f"[{idx}/{total}] skipped (bad path) {fname}")
                    continue
                image_array = self._read_input_img(img_path)

            def _cb(ctx: Context) -> None:
                self._contexts.append(ctx)
                if self.reporter:
                    self.reporter.emit(ctx)

            _kwargs = dict(kwargs)
            _kwargs.pop("callback", None)
            _kwargs["callback"] = _cb

            det_results = self.detector.detect(image_array, *args, **_kwargs)
            detection_annotations = self._dets_to_coco(det_results, image_array, file_name=fname)
            predictions.extend(detection_annotations)

            if progress:
                progress(idx, total, fname)
            logger.info(f"[{idx}/{total}] done {fname} preds={len(detection_annotations)}")

        self._predictions = predictions
        return predictions

    def evaluate(self,predictions: Optional[List[CocoAnnotation]] = None,average: str = "micro",) -> List[MetricOutputModel]:
        preds = predictions if predictions is not None else self._predictions
        results: List[MetricOutputModel] = []
        for metric in (self.metrics or []):
            result = metric.compute(self.dataset_model, preds, average=average)
            results.append(result)
        self._last_metrics = results
        return results

    def run(self,positive_dir: Union[str, Path],negative_dir: Optional[Union[str, Path]] = None,image_root: Optional[Union[str, Path]] = None,average: str = "micro",progress: Optional[Callable[[int, int, str | None], None]] = None,*args,dump_report: bool = False,report_output_dir: Optional[Union[str, Path]] = None,**kwargs,) -> Tuple[List[CocoAnnotation], List[MetricOutputModel]]:
        self.set_references(positive_dir, negative_dir)
        preds = self.detect_all(image_root=image_root, progress=progress, **kwargs)
        metrics = self.evaluate(preds, average=average)

        reporter = ReportGenerator()
        reporter.generate_report(self._contexts, metrics, dump_report=dump_report, output_dir=report_output_dir)

        return preds, metrics
