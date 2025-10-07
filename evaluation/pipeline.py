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
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}; pos_by_class, neg_imgs = {}, []
        p = Path(positive_dir)
        for sub in sorted([d for d in p.iterdir() if d.is_dir()]) if p.exists() else []:
            imgs = [Image.open(f).convert("RGB") for f in sorted(sub.rglob("*")) if f.is_file() and f.suffix.lower() in exts]
            if imgs: logger.info(f"[refs] class '{sub.name}': {len(imgs)} images"); pos_by_class[sub.name] = imgs
        if negative_dir:
            n = Path(negative_dir)
            neg_imgs = [Image.open(f).convert("RGB") for f in sorted(n.rglob("*")) if f.is_file() and f.suffix.lower() in exts]
            logger.info(f"[refs] negatives: {len(neg_imgs)} images")
        return pos_by_class, neg_imgs

    def _read_input_img(self, img_path: Path) -> np.ndarray:
        try:
            return np.array(Image.open(img_path).convert("RGB"))
        except Exception:
            return np.zeros((1, 1, 3), dtype=np.uint8)

    def _dets_to_coco(self, dets: Union[List[DetectionResult], List[Dict[str, Any]], Dict[str, Any], DetectionResult], image_np: np.ndarray, file_name: str) -> List[CocoAnnotation]:
        H, W = image_np.shape[:2]
        anns: List[CocoAnnotation] = []
        for d in dets:
            dd = d if isinstance(d, dict) else {
                "polygons": getattr(d, "polygons", None),
                "bbox": getattr(d, "bbox", None),
                "score": getattr(d, "score", None),
                "area": getattr(d, "area", None),
                "class_id": getattr(d, "class_id", None),
            }
            bbox = dd.get("bbox")
            bbox = [float(v) for v in bbox] if isinstance(bbox, list) and len(bbox) == 4 else [0.0, 0.0, float(W), float(H)]

            mask_np = np.zeros((H, W), np.uint8)
            polygons = dd.get("polygons") or []
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

            area_raw = dd.get("area")
            area = float(area_raw) if isinstance(area_raw, (int, float, np.integer, np.floating)) else float(bbox[2] * bbox[3])
            score_raw = dd.get("score")
            score = float(score_raw) if isinstance(score_raw, (int, float, np.integer, np.floating)) else None
            lr = dd.get("class_id")
            ls = str(lr).strip()
            label = ({'1': 'dragon fruit', '2': 'pineapple', '3': 'snake fruit'}.get(ls, ls) if ls.isdigit() else ls)

            anns.append(CocoAnnotation(img=image_np, mask=mask_np, label=label, image_size=(int(W), int(H)), width=int(W), height=int(H), area=area, file_name=file_name, bbox=bbox, score=score))
        return anns

    def set_references(self,positive_dir: Union[str, Path],negative_dir: Optional[Union[str, Path]] = None,) -> None:
        pos_by_class, neg_imgs = self._read_reference_images(positive_dir, negative_dir)
        details = ", ".join([f"{cid}: {len(imgs)}" for cid, imgs in pos_by_class.items()])
        logger.info(f"Индексируем эталоны: {details}")
        self.detector.set_references(pos_by_class, neg_imgs)

    def detect_all(self,image_root: Optional[Union[str, Path]] = None,progress: Optional[Callable[[int, int, str | None], None]] = None,*args,**kwargs,) -> List[CocoAnnotation]:
        image_root = Path(image_root) if image_root is not None else None
        file_names: List[str] = sorted({str(a.file_name) for a in self.dataset_model.data_points if a.file_name is not None})

        self._contexts = []
        predictions: List[CocoAnnotation] = []
        total = len(file_names)

        def _cb(ctx: Context) -> None:
            self._contexts.append(ctx)
            if self.reporter:
                self.reporter.emit(ctx)

        def _img_for(fname: str) -> Optional[np.ndarray]:
            img = next((a.img for a in self.dataset_model.data_points if a.file_name == fname and isinstance(a.img, np.ndarray)), None)
            if img is not None:
                return img
            return self._read_input_img((image_root / fname)) if image_root is not None and fname is not None else None

        for idx, fname in enumerate(file_names, start=1):
            logger.info(f"[{idx}/{total}] start {fname}")
            image_array = _img_for(fname)
            if image_array is None:
                if progress:
                    progress(idx, total, fname)
                logger.info(f"[{idx}/{total}] skipped {fname}")
                continue
            det_results = self.detector.detect(image_array, *args, callback=_cb, **kwargs)
            anns = self._dets_to_coco(det_results, image_array, file_name=fname)
            predictions.extend(anns)
            if progress:
                progress(idx, total, fname)
            logger.info(f"[{idx}/{total}] done {fname} preds={len(anns)}")

        self._predictions = predictions
        return predictions

    def evaluate(self,predictions: Optional[List[CocoAnnotation]] = None,average: str = "micro",) -> List[MetricOutputModel]:
        raw_preds = predictions if predictions is not None else self._predictions
        preds: List[CocoAnnotation] = list(raw_preds or [])
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
