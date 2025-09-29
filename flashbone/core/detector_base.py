import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import cv2
from PIL import Image


@dataclass
class DetectionResult:
    mask: np.ndarray
    bbox: list[int]
    area: float
    score: float 
    class_id: int


class DetectorBase(abc.ABC):
    @classmethod
    def read_input_img(cls, image_path: str | Path) -> np.ndarray:
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for read_input_img; install opencv-python.")
        img_bgr = cv2.imread(str(image_path))
        image_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return image_np

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
        """Runs detection on a single image (RGB ndarray)."""
        pass


class MockDetector(DetectorBase):
    """
    Mock implementation of DetectorBase.

    - set_references: remembers class IDs present in the provided pos_by_class keys.
    - find_present_elements: creates 1–3 random detections with:
        segmentation: 2D list (mask_size x mask_size) of 0/1
        bbox: [x, y, w, h] in pixel coords
        area: sum(seg)  (i.e., count of 1s)
        confidence: float in [0.5, 0.99]
        class: class id (int parsed from key if possible)
    """

    def __init__(self, mask_size: int = 32, seed: int = 1337):
        self.mask_size = int(mask_size)
        self.rng = np.random.default_rng(seed)
        self.known_classes: List[str] = []
        self.neg_count: int = 0

    # ---------- helpers ----------
    def _rand_bbox(self, H: int, W: int) -> List[int]:
        # generate bbox within image
        bw = int(self.rng.integers(max(10, W // 20), max(20, W // 3)))
        bh = int(self.rng.integers(max(10, H // 20), max(20, H // 3)))
        x = int(self.rng.integers(0, max(1, W - bw)))
        y = int(self.rng.integers(0, max(1, H - bh)))
        return [x, y, bw, bh]

    def _bbox_mask(self, H: int, W: int, bbox_xywh: List[int]) -> List[List[int]]:
        gx = gy = self.mask_size
        x, y, bw, bh = bbox_xywh
        x2, y2 = x + bw, y + bh
        # map grid centers into image coords and rasterize bbox
        xs = (np.arange(gx) + 0.5) * (W / gx)
        ys = (np.arange(gy) + 0.5) * (H / gy)
        xv, yv = np.meshgrid(xs, ys)
        mask = ((x <= xv) & (xv < x2) & (y <= yv) & (yv < y2)).astype(np.uint8)
        return mask.tolist()

    # ---------- base API ----------
    def read_reference_images(
        self,
        positive_dir: Union[str, Path],
        negative_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[Dict[str, List[Image.Image]], List[Image.Image]]:
        pos_by_class: Dict[str, List[Image.Image]] = {}
        neg_imgs: List[Image.Image] = []

        positive_dir = Path(positive_dir)
        if positive_dir.exists():
            # Expect subfolders named by class id
            for sub in sorted([p for p in positive_dir.iterdir() if p.is_dir()]):
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

    def set_references(
        self,
        pos_by_class: Dict[str, List[Image.Image]],
        neg_imgs: List[Image.Image],
    ) -> None:
        # store known class names; ignore actual image content for the mock
        self.known_classes = sorted(list(pos_by_class.keys()))
        self.neg_count = len(neg_imgs)

    def find_present_elements(self, image_np: np.ndarray, output_dir: str = "output") -> Dict[str, Any]:
        if image_np.ndim != 3 or image_np.shape[2] != 3:
            raise ValueError("image_np must be an RGB image ndarray of shape (H, W, 3).")
        H, W = image_np.shape[:2]

        num = int(self.rng.integers(1, 4))  # 1..3 detections
        results: List[Dict[str, Any]] = []
        for _ in range(num):
            bbox = self._rand_bbox(H, W)
            mask_2d = self._bbox_mask(H, W, bbox)
            area = int(np.sum(mask_2d))
            conf = float(np.round(self.rng.uniform(0.5, 0.99), 4))
            # choose class
            if self.known_classes:
                cls_key = str(self.rng.choice(self.known_classes))
                try:
                    cls_val = int(cls_key)
                except ValueError:
                    # hash non-int class strings into a small stable int
                    cls_val = int(abs(hash(cls_key)) % 1000)
            else:
                cls_val = int(self.rng.integers(0, 10))

            results.append(
                {
                    "segmentation": mask_2d,
                    "bbox": [int(b) for b in bbox],
                    "area": area,
                    "confidence": conf,
                    "class": cls_val,
                }
            )

        return {
            "found_elements": len(results),
            "masks": results,
        }
