import abc
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from PIL import Image
from .Context import Context
from .COCOAnnotations import COCOAnnotation

class DetectorBase(abc.ABC):
    def __init__(self, name: str = None):
        self.detector_name = name or self.__class__.__name__
    @classmethod
    def read_input_img(cls, image_path: str | Path) -> np.ndarray:
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for read_input_img; install opencv-python.")
        img_bgr = cv2.imread(str(image_path))
        image_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return image_np

    @abc.abstractmethod
    def read_reference_images(
        self,
        positive_dir: Union[str, Path],
        negative_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[Dict[str, List[Image.Image]], List[Image.Image]]:
        pass

    @abc.abstractmethod
    def set_references(
        self,
        pos_by_class: Dict[str, List[Image.Image]],
        neg_imgs: List[Image.Image],
    ) -> None:
        pass

    @abc.abstractmethod
    def find_present_elements(self, image_np: np.ndarray, context: Context, *args, **kwargs) -> Dict[str, Any]:
        pass
    
    def detect(self,image_np: np.ndarray, callback: Optional[Callable[[Context], None]] = None,*args, **kwargs) -> List[COCOAnnotation]:
        context = Context(
            detector_name=self.detector_name,
            image_shape=image_np.shape if image_np is not None else None
        )
        context.extra['original_image'] = image_np
        file_name = kwargs.get('file_name')
        if file_name is not None:
            context.extra['file_name'] = file_name
        try:
            results = self.find_present_elements(image_np, context, *args, **kwargs)
            annotations = self._convert_to_coco_annotations(results, context)
            context.finish(success=True)
            context.metrics['num_detections'] = len(annotations)
            return annotations
        except Exception as e:
            context.finish(success=False, error=str(e))
            raise
        finally:
            if callback:
                callback(context)
    
    def _convert_to_coco_annotations(self, results: Dict[str, Any], context: Context) -> List[COCOAnnotation]:
        annotations: List[COCOAnnotation] = []
        detections = []
        if isinstance(results, dict):
            if 'detections' in results and isinstance(results['detections'], list):
                detections = results['detections']
            elif 'found_elements' in results and isinstance(results['found_elements'], list):
                detections = results['found_elements']
        elif isinstance(results, list):
            detections = results

        img_np = context.extra.get('original_image')
        H = W = None
        if isinstance(img_np, np.ndarray) and img_np.size > 0:
            H, W = img_np.shape[:2]
        file_name = context.extra.get('file_name')
        for det in detections:
            det = det or {}
            mask_dict = det.get('mask') if isinstance(det.get('mask'), dict) else {}
            seg = mask_dict.get('segmentation') if isinstance(mask_dict, dict) else None
            if seg is None:
                seg = det.get('segmentation')
            seg_np = None
            if seg is not None:
                try:
                    seg_np = np.array(seg).astype(bool)
                except Exception:
                    seg_np = None
            bbox = det.get('bbox') or (mask_dict.get('bbox') if isinstance(mask_dict, dict) else None) or [0, 0, 0, 0]
            label = det.get('label') or det.get('class') or 'unknown'
            conf = det.get('confidence', None)
            if conf is None and isinstance(mask_dict, dict):
                conf = mask_dict.get('confidence', None)
            if conf is None:
                conf = det.get('score', 0.0)
            try:
                conf = float(conf)
            except Exception:
                conf = 0.0
            if H is None or W is None:
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    W = W or int(bbox[2])
                    H = H or int(bbox[3])
            area = 0
            if isinstance(seg_np, np.ndarray):
                area = int(seg_np.sum())
            elif isinstance(mask_dict, dict):
                try:
                    area = int(mask_dict.get('area', 0))
                except Exception:
                    area = 0
            ann = COCOAnnotation(
                img=img_np,
                mask=seg_np if isinstance(seg_np, np.ndarray) else None,
                label=label,
                width=W,
                height=H,
                area=float(area),
                bbox=[int(b) for b in bbox] if isinstance(bbox, (list, tuple)) else [0, 0, 0, 0],
                image_resolution=(W, H) if (W is not None and H is not None) else None,
                file_name=file_name,
            )
            setattr(ann, 'score', conf)
            annotations.append(ann)
        
        return annotations