import os
from typing import List ,Tuple ,Optional ,Union ,Dict ,Any

import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import FastSAM, SAM
from ultralytics.models.sam.predict import SAM2Predictor
from ultralytics.engine.results import Masks

from .heatmap_generator import HeatmapGenerator ,crop_heatmap_region ,merge_masks_with_heatmap_np ,save_crop_debug_info ,visualize_crop_region, merge_overlapping_masks_np


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

class FastSAMHeatmapProcessor:
    def __init__ (
        self, 
        fastsam_model = None, 
        embedding_extractor: EmbeddingExtractor = None, 
        decision_threshold: float = 0.5
    ) -> None:
        self.fastsam_model = fastsam_model
        self.embedding_extractor = embedding_extractor

        self.decision_threshold = decision_threshold
        # self.max_masks_per_crop = 15
        self.min_mask_area = 200
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.8

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _cleanup_gpu_memory(self):
        try:
            import torch
            if torch .cuda .is_available ():
                torch .cuda .empty_cache ()
                torch .cuda .synchronize ()
                if hasattr (torch .cuda ,'reset_peak_memory_stats'):
                    torch .cuda .reset_peak_memory_stats ()
        except Exception :
            pass

    def process_image(
        self, 
        image: Image.Image,
        heatmap: Optional[torch.Tensor] = None,
        min_overlap_ratio: float = 0.8,
    ) -> List[torch.Tensor]:
        # try:
        if heatmap is None :
            print ("⚠️ Heatmap не предоставлена, генерируем заглушку...")
            heatmap = torch.rand(image.size[1]//8, image.size[0]//8)

        print (f"🔍 Применение FastSAM к изображению...")
        thrsh = 0.5
        heatmap_mask = heatmap > thrsh
        heatmap = np.where(heatmap_mask, heatmap, 0)

        # # NOTE: (@gas) for debug only
        # debug_path = ".local/debug"
        # os.makedirs(debug_path, exist_ok=True)
        # cv2.imwrite(os.path.join(debug_path, f"heatmap_thrsh_{thrsh}.png"), heatmap*255)
        # # 

        points = sample_points_with_value(heatmap, value=0.0, n=5, seed=42)

        # fastsam_masks = self._generate_sam_masks_np(image, heatmap)
        # fastsam_masks = self._generate_sam_masks_np(image)

        # NOTE: (@gas) pass background points as 0's
        fastsam_masks = self._generate_fastsam_masks_np(image, points, [0]*len(points))

        # # NOTE: (@gas) debug
        # debug_path = ".local/debug"
        # os.makedirs(debug_path, exist_ok=True)
        # mask_debug = np.zeros(image.size[::-1], dtype=np.uint8)
        # for idx, mask in enumerate(fastsam_masks):
        #     mask_debug += mask
        #     cv2.imwrite(os.path.join(debug_path, f"sam_mask_{idx}.png"), mask*255)
        # mask_debug = np.clip(mask_debug, 0, 1)
        # cv2.imwrite(os.path.join(debug_path, f"sam_mask_unite.png"), mask_debug*255)
        # # 

        print (f"🔗 Мердж {len(fastsam_masks)} FastSAM масок с горячей зоной...")
        merged_masks = merge_masks_with_heatmap_np(
            fastsam_masks, heatmap, min_overlap_ratio=min_overlap_ratio,
        )

        if len(merged_masks) > 1:
            print (f"   🔗 Объединение {len(merged_masks)} масок по IoU...")
            merged_masks = merge_overlapping_masks_np(merged_masks, iou_threshold=0.3, verbose=True)
            print (f"   📊 Результат: {len(fastsam_masks)} -> {len(merged_masks)} масок")

        if merged_masks:
            print(f"📊 Применение скоринга к {len(merged_masks)} финальным маскам...")
            scoring_decisions, scored_masks = self.score_fastsam_masks(
                image=image,
                masks=merged_masks,
            )

            if scored_masks:
                print(f"✅ Финальный результат: {len(scored_masks)} масок прошли скоринг")
                return scored_masks
            else:
                print("⚠️ Ни одна маска не прошла скоринг")
                return []
        else:
            print(f"✅ Финальный результат: {len(merged_masks)} масок после мерджа (без скоринга)")
            return merged_masks

        # except Exception as e:
        #     print(f"❌ Ошибка в process_image: {e}")
        #     return []

    def _generate_sam_masks_np(self, img: Image.Image, roi_mask: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Generate plain FastSAM masks for a cropped image, returned as NumPy arrays.
    
        Args:
            cropped_image (PIL.Image): Image crop to segment.
    
        Returns:
            List[np.ndarray]: List of binary masks (H x W, dtype=uint8) with values {0,1}.
        """
        # try:
        image_np = np.array(img)

        self.fastsam_model.set_image(image_np)

        if roi_mask is not None: 
            lowres = cv2.resize((roi_mask > 0).astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)[np.newaxis, ...]
            # lowres = cv2.resize(roi_mask.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)[np.newaxis, ...]
            masks, scores = self.fastsam_model.inference_features(
                self.fastsam_model.features, src_shape=image_np.shape[:2], masks=lowres, multimask_output=True)
            masks_np = masks.detach().cpu().numpy()
        else: 
            results = self.fastsam_model(
                crop_n_layers=1,                 # ↑ to 1–2 to enable multi-scale crops
                points_stride=32,                # grid density for point sampling
                points_batch_size=64,            # batching for speed
                conf_thres=0.85,                 # quality filter (lower -> more masks)
                stability_score_thresh=0.95,     # stability filter (lower -> more masks)
                crop_nms_thresh=0.7,             # NMS between crop results
                multimask_output=True,
            )
            masks = results[0].masks
            masks_np = masks.cpu().numpy()

        result_masks: List[np.ndarray] = []
        for i, mask in enumerate(masks_np):
            mask_bin = (mask.data.reshape(mask.data.shape[1:3]) > 0.5).astype(int)
            if mask_bin.sum() >= self.min_mask_area:
                result_masks.append(mask_bin)
    
        print(f"✅ Generated {len(result_masks)} FastSAM masks")
        self.fastsam_model.reset_image()
        return result_masks
    
        # except Exception as e:
        #     print(f"⚠️ Error generating FastSAM masks: {e}")
        #     return []

    def _generate_fastsam_masks_np(self, img: Image.Image, query_points: Optional[List[Tuple[int, int]]] = None, query_labels: List[int] = None) -> List[np.ndarray]:
        """
        Generate plain FastSAM masks for a cropped image, returned as NumPy arrays.
    
        Args:
            cropped_image (PIL.Image): Image crop to segment.
    
        Returns:
            List[np.ndarray]: List of binary masks (H x W, dtype=uint8) with values {0,1}.
        """
        if self.fastsam_model is None:
            return []
    
        try:
            image_np = np.array(img)
            
            if query_points is not None and query_points:
                pts = np.asarray(query_points, dtype=np.int32)
                # NOTE: (@gas) 0 stands for background and 1 stands for object
                if query_labels is None or not query_labels:
                    query_labels = np.ones(len(pts), dtype=np.int32) 
                results = self.fastsam_model(
                    image_np,
                    points=pts, 
                    labels=query_labels,
                    device=self.device,
                    retina_masks=True,
                    imgsz=1024,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    verbose=False,
                )
            else: 
                results = self.fastsam_model(
                    image_np,
                    device=self.device,
                    retina_masks=True,
                    imgsz=1024,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
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
                mask_bin = (mask > 0.5).astype(np.uint8)
                if mask_bin.sum() >= self.min_mask_area:
                    result_masks.append(mask_bin)
    
            print(f"✅ Generated {len(result_masks)} FastSAM masks")
            return result_masks
    
        except Exception as e:
            print(f"⚠️ Error generating FastSAM masks: {e}")
            return []

    def _generate_heatmap_masks_np(
        self, heatmap: np.ndarray, threshold: float = 0.4, crop: bool = False
    ) -> List[np.ndarray]:
        binary_mask = (heatmap > threshold).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for contour in contours:
            mask = np.zeros_like(heatmap, dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 1)
            if np.sum(mask) >= self.min_mask_area:
                region = heatmap * mask
                if crop:
                    x, y, w, h = cv2.boundingRect(contour)
                    region = region[y:y+h, x:x+w]
                regions.append(region)
        return regions

    def score_fastsam_masks(
        self,
        image: Image.Image,
        masks: List[np.ndarray],
    ) -> Tuple[List[Dict[str,Any]], List[np.ndarray]]:
        if not masks :
            print ("   ⚠️ Нет масок для обработки")
            return [],[]

        try:
            masks_dicts =[]
            for i, mask in enumerate(masks):
                mask_np = (mask>0.5).astype(bool)
                mask_dict = {
                    'segmentation': mask_np, 
                    'area': int(np.sum(mask_np)), 
                    'bbox': self._mask_to_bbox(mask_np),
                    'predicted_iou': 0.8, 
                    'stability_score': 0.8,
                    'crop_box': [0, 0, mask_np.shape[1], mask_np.shape[0]],
                }
                masks_dicts.append(mask_dict)

            print (f"   🔍 Извлечение эмбеддингов для {len(masks_dicts)} FastSAM масок...")
            # TODO: (@gas) generate in the same way with dino fe, as pos images features being extracted
            masks_vecs = self.embedding_extractor.extract_mask_embeddings(image, masks_dicts)

            if masks_vecs.shape[0] == 0:
                print ("   ❌ Не удалось извлечь эмбеддинги для масок")
                return [], []

            print (f"   📊 Применение скоринга к {masks_vecs.shape[0]} маскам...")
            decisions = self.score_multiclass_v2(image, masks_vecs=masks_vecs)

            accepted_masks = []
            accepted_decisions = []

            for i, decision in enumerate(decisions):
                if decision.get('accepted', False):
                    accepted_masks.append(masks[i])
                    accepted_decisions.append(decision)
                    print(f"   ✅ Маска {i}: класс={decision.get('class')}, "
                    f"pos={decision.get('pos', 0):.3f}, "
                    f"diff={decision.get('diff', 0):.3f}")
                else :
                    print(f"   ❌ Маска {i}: отклонена, "
                    f"pos={decision.get('pos', 0):.3f}, "
                    f"diff={decision.get('diff', 0):.3f}")

            print(f"   📈 Скоринг завершен: {len(accepted_masks)}/{len(masks)} масок прошли скоринг")
            return accepted_decisions ,accepted_masks

        except Exception as e:
            print(f"   ⚠️ Ошибка при скоринге FastSAM масок: {e}")
            return [], []

    # TODO: (@gas) debug - create a new function, for cosine sim. measure.
    def score_multiclass_v2(
        self,
        image: Image.Image, 
        masks_vecs: np.ndarray,
    ):
        decisions =[]
        for m in range (M ):
            accepted = (best_pos [m ] >= self.decision_threshold) and (diff[m] >= self.decision_threshold)
            decisions.append({
                'class': best_cls[m],
                'pos': float(best_pos[m]),
                'neg_raw': float(neg_raw[m]),
                'neg': float(neg[m]),
                'diff': float(diff[m]),
                'accepted': bool(accepted),
            })
        return decisions

    def _mask_to_bbox (self ,mask :np .ndarray )->List [int ]:

        if not np .any (mask ):
            return [0 ,0 ,0 ,0 ]

        rows =np .any (mask ,axis =1 )
        cols =np .any (mask ,axis =0 )

        y_min ,y_max =np .where (rows )[0 ][[0 ,-1 ]]
        x_min ,x_max =np .where (cols )[0 ][[0 ,-1 ]]

        return [int (x_min ),int (y_min ),int (x_max -x_min +1 ),int (y_max -y_min +1 )]
