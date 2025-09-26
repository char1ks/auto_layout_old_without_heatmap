import os
from typing import List ,Tuple ,Optional ,Union ,Dict ,Any

import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import FastSAM, SAM
from ultralytics.models.sam.predict import SAM2Predictor
from ultralytics.engine.results import Masks
from ultralytics.engine.model import Model


# TODO: (@gas) move model instantiation to detector
def load_fastsam_model():
    # model = FastSAM('FastSAM-s.pt')
    model = FastSAM('FastSAM-x.pt')
    print ("✅ FastSAM модель загружена и закэширована")
    return model


def load_sam_model():
    model = SAM("sam2.1_t.pt")
    print ("✅ SAMv2 модель загружена и закэширована")
    return model


def load_sam_predictor():
    model = SAM2Predictor(overrides={"model": "sam2.1_t.pt", "imgsz": 1024})
    # model.setup_model() 
    # model = SAM2Predictor(overrides={"model": "sam2.1_s.pt"})
    print ("✅ SAMv2 модель загружена и закэширована")
    return model


def sample_points_with_value(
    arr: np.ndarray,
    value: float,
    n: int,
    *,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    match_nan: bool = False,
    replace: bool = False,
    seed: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """
    Return N random (x, y) image coordinates from a 2D array where arr[y, x] == value.

    - Exact match by default; use atol/rtol for float tolerance via np.isclose.
    - Set match_nan=True and value=np.nan to sample NaN positions.
    - If replace=False, raises if fewer than N matches exist.

    Returns: list of (x, y) integer tuples.
    """
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError("arr must be 2D")

    if match_nan and np.isnan(value):
        mask = np.isnan(a)
    elif atol is not None or rtol is not None:
        mask = np.isclose(a, value, atol=atol or 0.0, rtol=rtol or 0.0, equal_nan=False)
    else:
        mask = (a == value)

    flat_idx = np.flatnonzero(mask)
    if flat_idx.size == 0:
        return []

    if not replace and n > flat_idx.size:
        raise ValueError(f"Requested n={n} but only {flat_idx.size} matching points exist.")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(flat_idx, size=n, replace=replace)
    rows, cols = np.unravel_index(chosen, a.shape)  # rows=y, cols=x

    # Convert to list of (x, y) ints
    return [(int(x), int(y)) for y, x in zip(rows, cols)]


def merge_masks_with_heatmap_np(
    fastsam_masks: List[np.ndarray],
    heatmap: np.ndarray,
    min_overlap_ratio: float = 0.5,
)->List[np.ndarray]:
    filtered_masks = []
    for i, mask in enumerate(fastsam_masks):
        binary_full = mask > 0.5
        binary_hot = heatmap > 0.5 # TODO: (@gas) since it should be -1,1; if not - change.
        merged_mask = binary_full * binary_hot

        mask_area = np.sum(binary_full)
        overlap_area = np.sum(merged_mask)

        if mask_area > 0:
            overlap_ratio = overlap_area / mask_area
            if overlap_ratio >= min_overlap_ratio:
                filtered_masks.append(binary_full)
    return filtered_masks


class SamSegmenter:
    def __init__(self, sam_model: Model):
        self._sam_model = sam_model
        # TODO: (@gas) move to config or args
        self._min_mask_area = 200
        self._confidence_threshold = 0.5
        self._iou_threshold = 0.8
        self._mask_threshold = 0.5
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _generate_sam_masks_np(self, img: Image.Image, query_points: Optional[List[Tuple[int, int]]] = None, query_labels: List[int] = None) -> List[np.ndarray]:
        """
        Args:
            cropped_image (PIL.Image): Image crop to segment.
    
        Returns:
            List[np.ndarray]: List of binary masks (H x W, dtype=uint8) with values {0,1}.
        """
        image_np = np.array(img)
            
        if query_points is not None and query_points:
            pts = np.asarray(query_points, dtype=np.int32)
            # NOTE: (@gas) 0 stands for background and 1 stands for object
            if query_labels is None or not query_labels:
                query_labels = np.ones(len(pts), dtype=np.int32) 
            results = self._sam_model(
                image_np,
                points=pts, 
                labels=query_labels,
                device=self._device,
                retina_masks=True,
                imgsz=1024,
                conf=self._confidence_threshold,
                iou=self._iou_threshold,
                verbose=False,
            )
        else: 
            results = self._sam_model(
                image_np,
                device=self._device,
                retina_masks=True,
                imgsz=1024,
                conf=self._confidence_threshold,
                iou=self._iou_threshold,
                verbose=False,
            )
    
        if (
            len(results) == 0
            or not hasattr(results[0], "masks")
            or results[0].masks is None
        ):
            return []
    
        mask_data = results[0].masks.data  # torch.Tensor [N, H, W]
        result_masks: List[np.ndarray] = []
    
        num_masks = len(mask_data)
        for i in range(num_masks):
            mask = mask_data[i].detach().cpu().numpy()
            mask_bin = (mask > self._mask_threshold).astype(np.uint8)
            if mask_bin.sum() >= self._min_mask_area:
                result_masks.append(mask_bin)
        return result_masks

    def segment(
        self,
        image: Image.Image,
        heatmap: Optional[torch.Tensor] = None,
        min_overlap_ratio: float = 0.8,
    ):
        if heatmap is None :
            heatmap = torch.rand(image.size[1]//8, image.size[0]//8)
        thrsh = 0.5
        heatmap_mask = heatmap > thrsh
        heatmap = np.where(heatmap_mask, heatmap, 0)
        points = sample_points_with_value(heatmap, value=0.0, n=5, seed=42)
        # NOTE: (@gas) pass background points as 0's
        sam_masks = self._generate_sam_masks_np(image, points, [0]*len(points))
        merged_masks = merge_masks_with_heatmap_np(
            sam_masks, heatmap, min_overlap_ratio=min_overlap_ratio,
        )
        # TODO: (@gas) output satruct should look like: `md = {'segmentation': seg, 'bbox': bbox, 'area': int(seg.sum()), 'confidence': 0.9, 'class': class_name}`  
        return merged_masks
