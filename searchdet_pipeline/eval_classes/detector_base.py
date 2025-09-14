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
        annotations = []
        if 'detections' in results:
            for detection in results['detections']:
                annotation = COCOAnnotation(
                    img=context.extra.get('original_image'),
                    bbox=detection.get('bbox', [0, 0, 0, 0]),
                    label=detection.get('label', 'unknown'),
                    confidence=detection.get('confidence', 0.0)
                )
                annotations.append(annotation)
        
        return annotations