import abc
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from PIL import Image

class DetectorBase(abc.ABC):
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
        """Stores references (by class, plus optional negatives)."""
        pass

    @abc.abstractmethod
    def find_present_elements(self, image_np: np.ndarray, *args, **kwargs) -> Dict[str, Any]:
        """Find present elements in the given image."""
        pass