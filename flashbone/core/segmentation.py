from dataclasses import dataclass

import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import FastSAM
from ultralytics.engine.model import Model


def sample_points_with_value(
    arr: np.ndarray,
    value: float,
    n: int,
    *,
    atol: float | None = None,
    rtol: float | None = None,
    match_nan: bool = False,
    replace: bool = False,
    seed: int | None = None,
) -> list[tuple[int, int]]:
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


@dataclass
class SegmenterConfig:
    min_mask_area: int = 200
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.8
    mask_threshold: float = 0.5


class SamSegmenter:
    def __init__(self, sam_model: Model, config: SegmenterConfig = SegmenterConfig()) -> None:
        self._sam_model = sam_model
        self._config = config
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _generate_sam_masks_np(self, img: Image.Image, query_points: list[tuple[int, int]] | None = None, query_labels: list[int] = None) -> list[np.ndarray]:
        """
        Args:
            cropped_image (PIL.Image): Image crop to segment.
    
        Returns:
            list[np.ndarray]: List of binary masks (H x W, dtype=uint8) with values {0,1}.
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
                conf=self._config.confidence_threshold,
                iou=self._config.iou_threshold,
                verbose=False,
            )
        else: 
            results = self._sam_model(
                image_np,
                device=self._device,
                retina_masks=True,
                imgsz=1024,
                conf=self._config.confidence_threshold,
                iou=self._config.iou_threshold,
                verbose=False,
            )
    
        if (
            len(results) == 0
            or not hasattr(results[0], "masks")
            or results[0].masks is None
        ):
            return []
    
        mask_data = results[0].masks.data  # torch.Tensor [N, H, W]
        result_masks: list[np.ndarray] = []
    
        num_masks = len(mask_data)
        for i in range(num_masks):
            mask = mask_data[i].detach().cpu().numpy()
            mask_bin = (mask > self._config.mask_threshold).astype(np.uint8)
            if mask_bin.sum() >= self._config.min_mask_area:
                result_masks.append(mask_bin)
        return result_masks

    def _merge_masks_with_heatmap_np(
        self,
        masks: list[np.ndarray],
        heatmap: np.ndarray,
        min_overlap_ratio: float = 0.5,
    ) -> list[np.ndarray]:
        filtered_masks = []
        for i, mask in enumerate(masks):
            binary_full = mask > 0
            binary_hot = heatmap > 0
            merged_mask = binary_full * binary_hot
    
            mask_area = np.sum(binary_full)
            overlap_area = np.sum(merged_mask)
    
            if mask_area > 0:
                overlap_ratio = overlap_area / mask_area
                if overlap_ratio >= min_overlap_ratio:
                    filtered_masks.append(binary_full)
        return filtered_masks

    def _merge_masks(self, masks: list[np.ndarray], min_overlap_ratio: float = 0.5) -> list[np.ndarray]:
        bm = []
        areas = []
        for m in masks:
            b = (m > 0)
            a = int(b.sum())
            if a == 0:
                continue
            bm.append(b)
            areas.append(a)
        n = len(bm)
        if n == 0:
            return []
    
        parent = list(range(n))
    
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
    
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx
    
        for i in range(n):
            for j in range(i + 1, n):
                inter = np.sum(bm[i] & bm[j])
                if inter == 0:
                    continue
                if inter / areas[i] >= min_overlap_ratio or inter / areas[j] >= min_overlap_ratio:
                    union(i, j)
    
        groups = {}
        for i in range(n):
            r = find(i)
            groups.setdefault(r, []).append(i)
    
        out = []
        for idxs in groups.values():
            acc = np.zeros_like(bm[0], dtype=bool)
            for k in idxs:
                acc |= bm[k]
            out.append(acc)
        return out

    def segment(
        self,
        image: Image.Image,
        heatmap: np.ndarray | None = None,
        min_overlap_ratio: float = 0.6,
    ) -> list[np.ndarray]:
        if heatmap is not None:
            # TODO: (@gas) with points, it could be false negatives
            # points = sample_points_with_value(heatmap, value=0.0, n=5, seed=42)
            # # NOTE: (@gas) pass background points as 0's
            # masks = self._generate_sam_masks_np(image, points, [0]*len(points))

            masks = self._generate_sam_masks_np(image)

            # # DEBUG
            # for i, m in enumerate(masks):
            #     cv2.imwrite(f".local/detector_debug_sam_{i}.png", (m*255).astype(np.uint8))

            masks = self._merge_masks_with_heatmap_np(
                masks, heatmap, min_overlap_ratio=min_overlap_ratio,
            )
        else:
            masks = self._generate_sam_masks_np(image)
        masks = self._merge_masks(masks, min_overlap_ratio=min_overlap_ratio)
        return masks


if __name__=="__main__":
    import time 
    import cv2

    from flashbone.core.encoding import DinoV3EncoderGaz
    from flashbone.core.heatmap_generation import HeatmapGenerator, crop_by_mask

    img_pil_ex = Image.open(".local/example.jpg").convert("RGB")

    img_pil_left = Image.open(".local/image_left.jpg").convert("RGB")
    mask_left = Image.open(".local/image_left_fg.png")
    mask_left = mask_left.split()[-1]
    img_pil_right = Image.open(".local/image_right.jpg").convert("RGB")

    train_image_pos = crop_by_mask(img_pil_left, mask_left)
    # sky crop
    train_image_neg_1 = img_pil_left.crop((0, 0, 150, 150)) 
    # grass crop
    train_image_neg_2 = img_pil_left.crop((img_pil_left.width-150, img_pil_left.height-150, img_pil_left.width, img_pil_left.height))

    # ---
    sam_model = FastSAM('FastSAM-x.pt')

    model = DinoV3EncoderGaz()
    # warmup
    _ = model.encode([img_pil_ex])

    heatmap_generator = HeatmapGenerator(
        dino_fe=model, 
        use_cosine_similarity_for_heatmap=False,
        threshold_dotp=10,
    )
    heatmap_generator.init_pooled_features_train(
        positive_images=[train_image_pos], negative_images=[train_image_neg_1, train_image_neg_2])

    sam = SamSegmenter(sam_model=sam_model)
    # warmup
    _ = sam.segment(img_pil_ex)
    # ---

    heatmap, heatmap_resized = heatmap_generator.generate_heatmap(img_pil_right)
    heatmap_resized = heatmap_generator.apply_threshold(heatmap_resized)
    heatmap_np = heatmap_resized.cpu().numpy()

    start = time.perf_counter()
    masks = sam.segment(img_pil_right, heatmap=heatmap_np)
    end = time.perf_counter()
    print(f"{int((end-start)*1000)} ms.") 
    print("Masks no.:", len(masks))

    w, h = img_pil_right.size
    mask_ = np.zeros((h, w))
    for m in masks:
        mask_ += m
    mask_ = np.clip(mask_, 0, 1)

    cv2.imwrite(".local/masks_merged_debug_gaz.png", (mask_*255).astype(np.uint8))
