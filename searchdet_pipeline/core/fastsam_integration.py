import torch
import numpy as np
from PIL import Image
from typing import List ,Tuple ,Optional ,Union ,Dict ,Any
import cv2
from .heatmap_generator import HeatmapGenerator ,crop_heatmap_region ,merge_masks_with_heatmap_np ,save_crop_debug_info ,visualize_crop_region, merge_overlapping_masks_np
from .scoring import ScoreCalculator ,score_multiclass
from .embeddings import EmbeddingExtractor

from ultralytics.models.fastsam import FastSAMPredictor

class FastSAMHeatmapProcessor :
    def __init__ (self ,heatmap_generator :HeatmapGenerator ,fastsam_model =None ,
    embedding_extractor :EmbeddingExtractor =None ,
    score_calculator :ScoreCalculator =None ,debug_mode :bool =False ):

        self.heatmap_generator =heatmap_generator
        self.fastsam_model =fastsam_model
        self.embedding_extractor =embedding_extractor
        self.score_calculator =score_calculator
        self.debug_mode =debug_mode
        self._model_loaded =False

        self.max_masks_per_crop =15
        self.min_mask_area =200
        self.confidence_threshold =0.5
        self.iou_threshold =0.8

        self._model_cache ={}
        self._embedding_cache ={}
        self._heatmap_cache ={}
        self.cache_size_limit =10

        self.fast_mode =True
        self.skip_small_crops =True
        self.min_crop_size =64
        self.max_processing_time =0.08

        self._load_fastsam_model()

    def _load_fastsam_model(self):

        if self ._model_loaded :
            return

        model_key ="fastsam_default"
        if model_key in self ._model_cache :
            self .fastsam_model =self ._model_cache [model_key ]
            self ._model_loaded =True
            print ("📦 FastSAM модель загружена из кэша")
            return

        try:
            from ultralytics import FastSAM

            if self .fastsam_model is None :

                print ("📦 Загрузка FastSAM модели по умолчанию...")
                self .fastsam_model =FastSAM ('FastSAM-s.pt')

            self ._model_cache [model_key ]=self .fastsam_model
            self ._cleanup_cache ()

            self ._model_loaded =True
            print ("✅ FastSAM модель загружена и закэширована")

        except ImportError :
            print ("⚠️ FastSAM не установлен, используем fallback")
            self .fastsam_model =None
        except Exception as e :
            print (f"❌ Ошибка загрузки FastSAM: {e}")
            self .fastsam_model =None

    def _cleanup_cache (self ):
        for cache in [self ._model_cache ,self ._embedding_cache ,self ._heatmap_cache ]:
            if len (cache )>self .cache_size_limit :
                keys_to_remove =list (cache .keys ())[:-self .cache_size_limit ]
                for key in keys_to_remove :
                    del cache [key ]

    def _get_cache_key (self ,image_size ,threshold ,padding ):

        return f"{image_size[0]}x{image_size[1]}_{threshold}_{padding}"

    def _cleanup_gpu_memory (self ):

        try :
            import torch
            if torch .cuda .is_available ():
                torch .cuda .empty_cache ()
                torch .cuda .synchronize ()

                if hasattr (torch .cuda ,'reset_peak_memory_stats'):
                    torch .cuda .reset_peak_memory_stats ()
        except Exception :
            pass

    def _sample_anchor_points(
        self,
        regions: List[np.ndarray],
        points_per_region: int,
        crop: bool = True,
        orig_shape: Optional[Tuple[int, int]] = None,
        min_dist: int = 8,
        region_offsets: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Tuple[int, int, float]]:
        """
        Select spatially diverse high-score points from heatmap-valued region masks.
    
        Args:
            regions: Output from _generate_heatmap_masks_np(..., crop=<flag>).
                     - If crop=False: each region is full-size (same HxW as original), values retained inside mask, 0 elsewhere.
                     - If crop=True: each region is a tight crop (HxW of that contour), values retained inside crop, 0 elsewhere.
            points_per_region: Max number of points to sample per region.
            crop: Must match the flag used to produce `regions`.
            orig_shape: Optional (H, W). If provided with crop=True, used only for validation/sanity.
            min_dist: Minimum pixel separation enforced between points in the SAME region.
            region_offsets: Optional list of (x_offset, y_offset) for each cropped region.
                            Only meaningful if crop=True and you want ABSOLUTE coordinates.
                            Offsets should be the (x, y) top-left of each crop in the original image space.
    
        Returns:
            List of (x, y, score) tuples. If:
              - crop=False -> (x, y) are absolute coordinates in the original image.
              - crop=True and region_offsets is provided -> absolute coordinates.
              - crop=True and region_offsets is None -> coordinates are LOCAL to each crop
                and returned as (x, y) within that crop.
        """
        if crop and orig_shape is not None:
            if len(orig_shape) != 2 or not all(isinstance(v, int) for v in orig_shape):
                raise ValueError("orig_shape must be (H, W) with integers.")
    
        if crop and region_offsets is not None and len(region_offsets) != len(regions):
            raise ValueError("region_offsets length must match number of regions when provided.")
    
        # Precompute a circular (disk) mask for suppression
        r = int(max(1, min_dist))
        yy, xx = np.ogrid[-r:r+1, -r:r+1]
        disk = (xx*xx + yy*yy) <= (r*r)
    
        results: List[Tuple[int, int, float]] = []
    
        for i, region in enumerate(regions):
            if region.size == 0:
                continue
    
            # We will pick greedily by score while zeroing a disk around each chosen point
            # Work on a copy to avoid mutating caller data
            scores = region.astype(np.float32).copy()
    
            # Mask out zeros as unselectable by setting to -inf (so they never win)
            scores[scores <= 0] = -np.inf
    
            selected_local: List[Tuple[int, int, float]] = []
    
            # Greedy selection loop
            for _ in range(points_per_region):
                # Find current maximum
                flat_idx = np.argmax(scores)
                max_val = scores.flat[flat_idx]
                if not np.isfinite(max_val):
                    break  # no more valid points
    
                y, x = np.unravel_index(flat_idx, scores.shape)
                selected_local.append((x, y, float(max_val)))
    
                # Suppress a disk around (y, x)
                y0, x0 = y - r, x - r
                y1, x1 = y + r + 1, x + r + 1
    
                # Clip to region bounds
                ry0, rx0 = max(0, y0), max(0, x0)
                ry1, rx1 = min(scores.shape[0], y1), min(scores.shape[1], x1)
    
                # Corresponding slice in the disk
                dy0, dx0 = ry0 - y0, rx0 - x0
                dy1, dx1 = dy0 + (ry1 - ry0), dx0 + (rx1 - rx0)
    
                # Apply suppression
                sub = scores[ry0:ry1, rx0:rx1]
                sub[disk[dy0:dy1, dx0:dx1]] = -np.inf
    
            # Map local (within-region) to absolute (image) coords if possible/desired
            if crop and region_offsets is not None:
                x_off, y_off = region_offsets[i]
                for (lx, ly, s) in selected_local:
                    results.append((lx + x_off, ly + y_off, s))
            else:
                # Either full-size (already absolute), or cropped (stay local)
                for (lx, ly, s) in selected_local:
                    results.append((lx, ly, s))
    
        # Done: we collected up to points_per_region per input region.
        # If you want the global top-N regardless of region, you can sort here.
        # The user asked "output N points with the highest score"; interpret N as
        # points_per_region * len(regions).
        # If you instead want a strict cap N (independent of regions), change below.
        results.sort(key=lambda t: t[2], reverse=True)
        # NOTE: (@gas) omit scores
        results = [(p[0], p[1]) for p in results]
    
        return results

    def process_image(
        self, 
        image: Image.Image,
        pos_by_class: Dict[str, np.ndarray],
        heatmap: Optional[torch.Tensor] = None,
        neg_imgs: np.ndarray = None,
        min_overlap_ratio: float = 0.8,
        skip_scoring_for_hotspot_masks: bool = False,
    ) -> List[torch.Tensor]:
        try:
            if heatmap is None :
                print ("⚠️ Heatmap не предоставлена, генерируем заглушку...")
                heatmap = torch.rand(image.size[1]//8, image.size[0]//8)

            # TODO: (@gas) try to extract masks from heatmap (needed for points extraction)
            heatmap_masks = self._generate_heatmap_masks_np(heatmap, threshold=0.4, crop=False)

            print (f"🔍 Применение FastSAM к изображению...")
            # points = self._sample_anchor_points(
            #     regions=heatmap_masks,
            #     points_per_region=3,
            #     crop=False,             # must match how regions were generated
            #     orig_shape=heatmap.shape,
            #     min_dist=10,
            # )
            # fastsam_masks = self._generate_fastsam_masks_with_points_np(image, points)
            fastsam_masks = self._generate_fastsam_masks_np(image)

            print (f"🔗 Мердж {len(fastsam_masks)} FastSAM масок с горячей зоной...")
            merged_masks = merge_masks_with_heatmap_np(
                fastsam_masks, heatmap, min_overlap_ratio=min_overlap_ratio,
            )

            if len(merged_masks) > 1:
                print (f"   🔗 Объединение {len(merged_masks)} масок по IoU...")
                merged_masks = merge_overlapping_masks_np(merged_masks, iou_threshold=0.3, verbose=True)
                print (f"   📊 Результат: {len(fastsam_masks)} -> {len(merged_masks)} масок")

            if merged_masks and pos_by_class is not None:
                if skip_scoring_for_hotspot_masks :
                    print(f"🎯 Пропуск скоринга для {len(merged_masks)} масок из горячих зон")

                    scoring_decisions =[]
                    for i in range(len(merged_masks)):
                        scoring_decisions.append({'accepted':True ,'class':'hotspot_mask','pos':1.0 ,'neg':0.0 ,'diff':1.0 ,'mask_index':i})
                    scored_masks = merged_masks
                    print(f"✅ Все {len(scored_masks)} масок из горячих зон приняты без скоринга")
                    return scored_masks
                else:
                    print(f"📊 Применение скоринга к {len(merged_masks)} финальным маскам...")
                    scoring_decisions, scored_masks = self.score_fastsam_masks(
                        image=image,
                        masks=merged_masks,
                        pos_by_class=pos_by_class,
                        neg_imgs=neg_imgs,
                        skip_scoring=False,
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

        except Exception as e:
            print(f"❌ Ошибка в process_image: {e}")
            return []

    def _generate_fastsam_masks_with_points_np(
        self,
        cropped_image: Image.Image,
        query_points: Optional[List[Tuple[int, int]]] = None,  # (x, y) in cropped image coords
    ) -> List[np.ndarray]:
        """
        Generate FastSAM masks guided by query points.
    
        Args:
            cropped_image (Image.Image): PIL image of the crop to segment.
            query_points (list of (x,y), optional): Pixel coordinates in cropped image space.
                                                    If None, falls back to unguided "everything".
    
        Returns:
            List[np.ndarray]: Binary masks (H x W, dtype=uint8) with values {0,1}.
        """
        if self.fastsam_model is None:
            return [np.array(m) for m in self._generate_fallback_masks(cropped_image)]
    
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            image_np = np.array(cropped_image)
    
            # Run once to get embeddings/candidates
            results = self.fastsam_model(
                image_np,
                device=device,
                retina_masks=True,
                imgsz=1024,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )
    
            if len(results) == 0 or not hasattr(results[0], "masks") or results[0].masks is None:
                return [np.array(m) for m in self._generate_fallback_masks(cropped_image)]
    
            result_masks: List[np.ndarray] = []
    
            # === If query points provided, try FastSAMPrompt ===
            if query_points and len(query_points) > 0:
                try: 
                    # TODO: (@gas) adopt for that new class instead of FastSAMPrompt
                    prompt = FastSAMPredictor(image_np, results, device=device)
    
                    pointlabel = [1] * len(query_points)  # all positive
                    ann = prompt.point_prompt(points=query_points, pointlabel=pointlabel)
    
                    def _to_numpy(m) -> np.ndarray:
                        if isinstance(m, torch.Tensor):
                            m = m.detach().cpu().numpy()
                        return (m > 0).astype(np.uint8)
    
                    extracted: List[np.ndarray] = []
                    if ann is not None:
                        if isinstance(ann, (list, tuple)):
                            extracted = [_to_numpy(m) for m in ann]
                        elif hasattr(ann, "masks") and getattr(ann, "masks") is not None:
                            am = ann.masks.data if hasattr(ann.masks, "data") else ann.masks
                            am = torch.as_tensor(am).cpu().numpy()
                            extracted = [(am[i] > 0.5).astype(np.uint8) for i in range(am.shape[0])]
                        elif isinstance(ann, np.ndarray):
                            if ann.ndim == 2:
                                extracted = [_to_numpy(ann)]
                            else:
                                extracted = [(ann[i] > 0).astype(np.uint8) for i in range(ann.shape[0])]
    
                    # Filter & cap
                    for m in extracted:
                        if m.sum() >= self.min_mask_area:
                            result_masks.append(m)
                            if len(result_masks) >= self.max_masks_per_crop:
                                break
    
                    if len(result_masks) > 0:
                        print(f"✅ Generated {len(result_masks)} FastSAM masks (query-guided)")
                        return result_masks
    
                except Exception as e:
                    print(f"⚠️ Query-point path failed, falling back: {e}")
    
            # === Fallback: keep top-N masks from "everything" ===
            mask_data = results[0].masks.data
            mask_data = (mask_data > 0.5).to(torch.uint8).cpu().numpy()
    
            for i in range(min(len(mask_data), self.max_masks_per_crop)):
                m = mask_data[i]
                if m.sum() >= self.min_mask_area:
                    result_masks.append(m)
    
            print(f"✅ Generated {len(result_masks)} FastSAM masks (fallback)")
            return result_masks
    
        except Exception as e:
            print(f"⚠️ FastSAM mask generation error: {e}")
            return [np.array(m) for m in self._generate_fallback_masks(cropped_image)]

    def _generate_fastsam_masks (self ,cropped_image :Image .Image )->List [torch .Tensor ]:
        if self .fastsam_model is None :

            return self ._generate_fallback_masks (cropped_image )

        try :
            image_np =np .array (cropped_image )
            results =self .fastsam_model (image_np ,device ='cuda'if torch .cuda .is_available ()else 'cpu',retina_masks =True ,imgsz =1024 ,conf =self .confidence_threshold ,iou =self .iou_threshold ,verbose =False)
            if len (results )==0 or not hasattr (results [0 ],'masks')or results [0 ].masks is None :
                return self ._generate_fallback_masks (cropped_image )
            mask_data =results [0 ].masks .data
            result_masks =[]
            num_masks =min (len (mask_data ),self .max_masks_per_crop )
            for i in range (num_masks ):
                mask =mask_data [i ].cpu ()
                if torch .sum (mask >0.5 )>=self .min_mask_area :
                    result_masks .append (mask )
                if len (result_masks )>=self .max_masks_per_crop :
                    break
            print (f"✅ Сгенерировано {len(result_masks)} FastSAM масок")
            return result_masks

        except Exception as e :
            print (f"Ошибка при генерации FastSAM масок: {e}")
            return self ._generate_fallback_masks (cropped_image )

    def _generate_fastsam_masks_np(self, cropped_image: Image.Image) -> List[np.ndarray]:
        """
        Generate plain FastSAM masks for a cropped image, returned as NumPy arrays.
    
        Args:
            cropped_image (PIL.Image): Image crop to segment.
    
        Returns:
            List[np.ndarray]: List of binary masks (H x W, dtype=uint8) with values {0,1}.
        """
        if self.fastsam_model is None:
            # Fallback already expected to return list of np.ndarray
            return [np.array(m) for m in self._generate_fallback_masks(cropped_image)]
    
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            image_np = np.array(cropped_image)
    
            results = self.fastsam_model(
                image_np,
                device=device,
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
                return [np.array(m) for m in self._generate_fallback_masks(cropped_image)]
    
            mask_data = results[0].masks.data  # torch.Tensor [N, H, W]
            result_masks: List[np.ndarray] = []
    
            num_masks = min(len(mask_data), self.max_masks_per_crop)
            for i in range(num_masks):
                mask = mask_data[i].detach().cpu().numpy()
                mask_bin = (mask > 0.5).astype(np.uint8)
                if mask_bin.sum() >= self.min_mask_area:
                    result_masks.append(mask_bin)
                if len(result_masks) >= self.max_masks_per_crop:
                    break
    
            print(f"✅ Generated {len(result_masks)} FastSAM masks")
            return result_masks
    
        except Exception as e:
            print(f"⚠️ Error generating FastSAM masks: {e}")
            return [np.array(m) for m in self._generate_fallback_masks(cropped_image)]

    def _calculate_mask_iou (self ,mask1 :torch .Tensor ,mask2 :torch .Tensor )->float :

        mask1_bin =(mask1 >0.5 ).float ()
        mask2_bin =(mask2 >0.5 ).float ()

        intersection =torch .sum (mask1_bin *mask2_bin )
        union =torch .sum (mask1_bin )+torch .sum (mask2_bin )-intersection

        if union ==0 :
            return 0.0

        return (intersection /union ).item ()

    def _calculate_heatmap_mask_overlap (self ,fastsam_mask :torch .Tensor ,heatmap :np .ndarray ,threshold :float =0.5 )->float :

        mask_np =(fastsam_mask .cpu ().numpy ()>0.5 ).astype (np .uint8 )

        if heatmap .shape !=mask_np .shape :
            heatmap_resized =cv2 .resize (heatmap ,(mask_np .shape [1 ],mask_np .shape [0 ]))
        else :
            heatmap_resized =heatmap

        heatmap_bin =(heatmap_resized >threshold ).astype (np .uint8 )

        intersection =np .sum (mask_np *heatmap_bin )
        mask_area =np .sum (mask_np )

        if mask_area ==0 :
            return 0.0

        return intersection /mask_area

    def _generate_heatmap_masks (self ,heatmap :np .ndarray ,threshold :float )->List [torch .Tensor ]:

        binary_mask =(heatmap >threshold ).astype (np .uint8 )

        contours ,_ =cv2 .findContours (binary_mask ,cv2 .RETR_EXTERNAL ,cv2 .CHAIN_APPROX_SIMPLE )

        masks =[]
        for contour in contours :

            mask =np .zeros_like (binary_mask ,dtype =np .uint8 )
            cv2 .fillPoly (mask ,[contour ],1 )

            if np .sum (mask )>=self .min_mask_area :
                masks .append (torch .from_numpy (mask .astype (np .float32 )))

        return masks

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

    def _generate_fallback_masks (self ,cropped_image :Image .Image )->List [torch .Tensor ]:

        image_np =np .array (cropped_image )

        gray =cv2 .cvtColor (image_np ,cv2 .COLOR_RGB2GRAY )

        binary =cv2 .adaptiveThreshold (
        gray ,255 ,cv2 .ADAPTIVE_THRESH_GAUSSIAN_C ,cv2 .THRESH_BINARY ,11 ,2)
        contours ,_ =cv2 .findContours (binary ,cv2 .RETR_EXTERNAL ,cv2 .CHAIN_APPROX_SIMPLE )

        masks =[]
        h ,w =gray .shape

        for contour in contours :

            if cv2 .contourArea (contour )>=self .min_mask_area :

                mask =np .zeros ((h ,w ),dtype =np .uint8 )
                cv2 .fillPoly (mask ,[contour ],255 )

                mask_tensor =torch .from_numpy (mask /255.0 ).float ()
                masks .append (mask_tensor )

        return masks

    def set_performance_params (self ,max_masks :int =50 ,min_area :int =100 ,confidence :float =0.4 ,iou :float =0.9 ):
        self .max_masks_per_crop =max_masks
        self .min_mask_area =min_area
        self .confidence_threshold =confidence
        self .iou_threshold =iou

    def visualize_results (self ,image :Image .Image ,masks :List [torch .Tensor ],heatmap :torch .Tensor ,bbox :Tuple [int ,int ,int ,int ],save_path :Optional [str ]=None )->Image .Image :

        import matplotlib .pyplot as plt
        import matplotlib .patches as patches
        from matplotlib .colors import ListedColormap

        fig ,axes =plt .subplots (1 ,3 ,figsize =(15 ,5 ))

        axes [0 ].imshow (image )
        axes [0 ].set_title ('Исходное изображение + Кроп область')
        x1 ,y1 ,x2 ,y2 =bbox
        rect =patches .Rectangle ((x1 ,y1 ),x2 -x1 ,y2 -y1 ,
        linewidth =2 ,edgecolor ='red',facecolor ='none')
        axes [0 ].add_patch (rect )
        axes [0 ].axis ('off')

        axes [1 ].imshow (image ,alpha =0.7 )
        heatmap_resized =torch .nn .functional .interpolate (
        heatmap .unsqueeze (0 ).unsqueeze (0 ),
        size =(image .size [1 ],image .size [0 ]),
        mode ='bilinear'
        ).squeeze ()
        axes [1 ].imshow (heatmap_resized .cpu ().numpy (),alpha =0.5 ,cmap ='hot')
        axes [1 ].set_title ('Тепловая карта')
        axes [1 ].axis ('off')

        axes [2 ].imshow (image ,alpha =0.7 )
        if masks :
            combined_mask =torch .zeros_like (masks [0 ])
            for i ,mask in enumerate (masks ):
                combined_mask +=mask *(i +1 )

            colors =plt .cm .Set3 (np .linspace (0 ,1 ,len (masks )+1 ))
            cmap =ListedColormap (colors )

            axes [2 ].imshow (combined_mask .cpu ().numpy (),alpha =0.6 ,cmap =cmap )
        axes [2 ].set_title (f'FastSAM маски ({len(masks)} шт.)')
        axes [2 ].axis ('off')

        plt .tight_layout ()

        if save_path :
            plt .savefig (save_path ,dpi =150 ,bbox_inches ='tight')

        fig .canvas .draw ()
        buf =np .frombuffer (fig .canvas .tostring_rgb (),dtype =np .uint8 )
        buf =buf .reshape (fig .canvas .get_width_height ()[::-1 ]+(3 ,))
        plt .close (fig )

        return Image .fromarray (buf )

    def score_fastsam_masks (self ,
    image :Image .Image ,
    masks :List [torch .Tensor ],
    pos_by_class :Dict [str ,np .ndarray ],
    neg_imgs :np .ndarray =None ,
    min_overlap_ratio :float =0.8 ,
    skip_scoring :bool =False )->Tuple [List [Dict [str ,Any ]],List [torch .Tensor ]]:

        if not masks :
            print ("   ⚠️ Нет масок для обработки")
            return [],[]

        if skip_scoring :
            print (f"   🎯 Пропуск скоринга для {len(masks)} масок из горячих зон - принимаем все")
            decisions =[]
            for i ,mask in enumerate (masks ):
                decision ={'accepted':True ,'class':'hotspot_mask','pos':1.0 ,'neg':0.0 ,'diff':1.0 ,'mask_index':i}
                decisions .append (decision )
                print (f"   ✅ Маска {i}: принята без скоринга (горячая зона)")

            print (f"   📈 Все {len(masks)} масок из горячих зон приняты без скоринга")
            return decisions ,masks

        if self .embedding_extractor is None or self .score_calculator is None :
            print ("   ⚠️ Нет компонентов для скоринга")
            return [],[]

        try :

            mask_dicts =[]
            for i ,mask in enumerate (masks ):

                mask_np =(mask .cpu ().numpy ()>0.5 ).astype (bool )

                mask_dict ={'segmentation':mask_np ,'area':int (np .sum (mask_np )),'bbox':self ._mask_to_bbox (mask_np ),'predicted_iou':0.8 ,'stability_score':0.8 ,'crop_box':[0 ,0 ,mask_np .shape [1 ],mask_np .shape [0 ]]}
                mask_dicts .append (mask_dict )

            print (f"   🔍 Извлечение эмбеддингов для {len(mask_dicts)} FastSAM масок...")
            mask_vecs =self .embedding_extractor .extract_mask_embeddings (image ,mask_dicts )

            if mask_vecs .shape [0 ]==0 :
                print ("   ❌ Не удалось извлечь эмбеддинги для масок")
                return [],[]

            def _is_numeric_ndarray (x ):
                return isinstance (x ,np .ndarray )and np .issubdtype (x .dtype ,np .number )and x .size >=0

            def _looks_like_vec_list (x ):
                return isinstance (x ,list )and len (x )>0 and isinstance (x [0 ],np .ndarray )and np .issubdtype (x [0 ].dtype ,np .number )

            pos_is_embeddings =True
            for cls ,Q in (pos_by_class or {}).items ():
                if _is_numeric_ndarray (Q )or _looks_like_vec_list (Q ):
                    continue

                pos_is_embeddings =False
                break

            q_pos_ready :Dict [str ,np .ndarray ]
            q_neg_ready :Optional [np .ndarray ]

            if not pos_is_embeddings :
                print ("   🔧 q_pos содержит изображения, кодируем в эмбеддинги через EmbeddingExtractor...")
                q_pos_ready ,q_neg_ready =self .embedding_extractor .build_queries_multiclass (pos_by_class ,neg_imgs or [])
            else :

                q_pos_ready ={}
                for cls ,Q in (pos_by_class or {}).items ():
                    if isinstance (Q ,np .ndarray ):
                        arr =Q .astype (np .float32 )
                        if arr .ndim ==1 :
                            arr =arr [None ,:]
                        q_pos_ready [cls ]=arr
                    elif _looks_like_vec_list (Q ):
                        try :
                            arr =np .vstack ([v .reshape (1 ,-1 )if v .ndim ==1 else v for v in Q ]).astype (np .float32 )
                        except Exception :

                            print (f"   ⚠️ Класс '{cls}': не удалось собрать массив из списка, кодируем через энкодер")
                            arr =self .embedding_extractor ._encode_pil_list (Q )
                            arr =self .embedding_extractor ._filter_bad (arr ,cls_name =cls ,kind ="q_pos")
                        q_pos_ready [cls ]=arr
                    else :

                        print (f"   ⚠️ Класс '{cls}': непонятный тип, кодируем через энкодер")
                        arr =self .embedding_extractor ._encode_pil_list (Q if isinstance (Q ,list )else [Q ])
                        arr =self .embedding_extractor ._filter_bad (arr ,cls_name =cls ,kind ="q_pos")
                        q_pos_ready [cls ]=arr

                if isinstance (neg_imgs ,np .ndarray ):
                    q_neg_ready =neg_imgs .astype (np .float32 )
                    if q_neg_ready .ndim ==1 :
                        q_neg_ready =q_neg_ready [None ,:]
                else :
                    q_neg_ready =None

            print (f"   📊 Применение скоринга к {mask_vecs.shape[0]} маскам...")
            decisions ,debug_info =score_multiclass (mask_vecs =mask_vecs ,q_pos =q_pos_ready ,q_neg =q_neg_ready ,min_pos_score =self .score_calculator .min_pos_score ,decision_threshold =self .score_calculator .decision_threshold ,verbose =True)

            accepted_masks =[]
            accepted_decisions =[]

            for i ,decision in enumerate (decisions ):
                if decision .get ('accepted',False ):
                    accepted_masks .append (masks [i ])
                    accepted_decisions .append (decision )
                    print (f"   ✅ Маска {i}: класс={decision.get('class')}, "
                    f"pos={decision.get('pos', 0):.3f}, "
                    f"diff={decision.get('diff', 0):.3f}")
                else :
                    print (f"   ❌ Маска {i}: отклонена, "
                    f"pos={decision.get('pos', 0):.3f}, "
                    f"diff={decision.get('diff', 0):.3f}")

            print (f"   📈 Скоринг завершен: {len(accepted_masks)}/{len(masks)} масок прошли скоринг")
            return accepted_decisions ,accepted_masks

        except Exception as e :
            print (f"   ⚠️ Ошибка при скоринге FastSAM масок: {e}")
            return [],[]

    # def _save_overlay_details_masks (self ,overlay_masks :List [Tuple [int ,torch .Tensor ,float ]],
    # input_image :Image .Image ,crop_coords :Tuple [int ,int ,int ,int ]):

    #     try :
    #         import os
    #         import json
    #         from datetime import datetime

    #         overlay_dir ="overlay_details"
    #         os .makedirs (overlay_dir ,exist_ok =True )

    #         timestamp =datetime .now ().strftime ("%Y%m%d_%H%M%S")
    #         overlay_file =os .path .join (overlay_dir ,f"fastsam_overlay_{timestamp}.json")

    #         overlay_data ={"timestamp":timestamp ,"image_size":{"width":input_image .width ,"height":input_image .height },"crop_coords":{"x1":crop_coords [0 ],"y1":crop_coords [1 ],"x2":crop_coords [2 ],"y2":crop_coords [3 ]},"fastsam_masks":[]}

    #         for mask_idx ,mask_tensor ,overlap_ratio in overlay_masks :
    #             mask_np =mask_tensor .cpu ().numpy ()
    #             mask_coords =np .where (mask_np >0.5 )
    #             mask_info ={"mask_index":mask_idx ,"overlap_ratio":float (overlap_ratio ),"mask_area":int (np .sum (mask_np >0.5 )),"bbox":self ._mask_to_bbox (mask_np >0.5 ),"coordinates_count":len (mask_coords [0 ])}
    #             overlay_data ["fastsam_masks"].append (mask_info )
    #         with open (overlay_file ,'w',encoding ='utf-8')as f :
    #             json .dump (overlay_data ,f ,indent =2 ,ensure_ascii =False )

    #         print (f"💾 Сохранено {len(overlay_masks)} FastSAM масок в overlay_details файл: {overlay_file}")

    #     except Exception as e :
    #         print (f"❌ Ошибка при сохранении overlay_details: {e}")

    def _mask_to_bbox (self ,mask :np .ndarray )->List [int ]:

        if not np .any (mask ):
            return [0 ,0 ,0 ,0 ]

        rows =np .any (mask ,axis =1 )
        cols =np .any (mask ,axis =0 )

        y_min ,y_max =np .where (rows )[0 ][[0 ,-1 ]]
        x_min ,x_max =np .where (cols )[0 ][[0 ,-1 ]]

        return [int (x_min ),int (y_min ),int (x_max -x_min +1 ),int (y_max -y_min +1 )]
