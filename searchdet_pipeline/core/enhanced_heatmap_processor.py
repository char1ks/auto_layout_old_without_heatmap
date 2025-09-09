import numpy as np
import torch
import cv2
from PIL import Image
from typing import List ,Dict ,Any ,Tuple ,Optional
from sklearn .cluster import DBSCAN
from scipy import ndimage
from skimage .measure import label ,regionprops
from .dinov3_encoder import DinoV3Encoder
from .heatmap_generator import HeatmapGenerator ,DinoV3FeatureExtractor
from .binning_processor import BinningProcessor

class EnhancedHeatmapProcessor :

    def __init__ (self ,dinov3_encoder :DinoV3Encoder ,
    hot_zone_threshold :float =0.45 ,
    pixel_similarity_threshold :float =0.8 ,
    min_zone_area :int =4 ,
    connectivity :int =8 ,
    resize_size :int =512 ):

        self .dinov3_encoder =dinov3_encoder

        self .heatmap_generator =HeatmapGenerator (dinov3_encoder ,attention_pool_examples =True ,enable_expansion =True ,resize_size =resize_size ,crop_images =False)
        self .feature_extractor =DinoV3FeatureExtractor (dinov3_encoder ,resize_images =True ,crop_images =False ,resize_size =resize_size)
        self .binning_processor =BinningProcessor (dinov3_encoder ,concept_threshold =3 )

        self .hot_zone_threshold =hot_zone_threshold
        self .pixel_similarity_threshold =pixel_similarity_threshold
        self .min_zone_area =min_zone_area
        self .connectivity =connectivity

    def extract_hot_zones (self ,heatmap :torch .Tensor )->np .ndarray :

        if isinstance (heatmap ,torch .Tensor ):
            hm =heatmap .detach ().cpu ().numpy ()
        else :
            hm =np .asarray (heatmap )

        print (f"[Бинаризация] Исходная heatmap: min={hm.min():.4f}, max={hm.max():.4f}, mean={hm.mean():.4f}")

        hm_normalized =(hm -hm .min ())/(hm .max ()-hm .min ()+1e-8 )

        pixel_values =hm_normalized .flatten ()
        pixel_mean =np .mean (pixel_values )
        pixel_std =np .std (pixel_values )
        pixel_median =np .median (pixel_values )

        q25 =np .quantile (pixel_values ,0.25 )
        q50 =np .quantile (pixel_values ,0.50 )
        q75 =np .quantile (pixel_values ,0.75 )
        q80 =np .quantile (pixel_values ,0.80 )
        q85 =np .quantile (pixel_values ,0.85 )
        q90 =np .quantile (pixel_values ,0.90 )
        q95 =np .quantile (pixel_values ,0.95 )
        q99 =np .quantile (pixel_values ,0.99 )

        print (f"[Статистика пикселей] Mean: {pixel_mean:.4f}, Std: {pixel_std:.4f}, Median: {pixel_median:.4f}")
        print (f"[Квантили] Q25: {q25:.4f}, Q50: {q50:.4f}, Q75: {q75:.4f}, Q80: {q80:.4f}, Q85: {q85:.4f}, Q90: {q90:.4f}, Q95: {q95:.4f}, Q99: {q99:.4f}")

        threshold_adaptive =min (pixel_mean +1.5 *pixel_std ,q85 )
        threshold_conservative =max (self .hot_zone_threshold ,q75 )

        final_threshold =min (threshold_adaptive ,threshold_conservative )

        if final_threshold >q80 :
            final_threshold =q80
            print (f"[Трешхолд] Ограничен до Q80: {final_threshold:.4f}")

        print (f"[Трешхолды] Adaptive: {threshold_adaptive:.4f}, Conservative: {threshold_conservative:.4f}, Final: {final_threshold:.4f}")

        hot_zones_binary =(hm_normalized >=final_threshold ).astype (np .uint8 )

        hot_pixels_count =np .sum (hot_zones_binary )
        total_pixels =hot_zones_binary .size
        hot_pixels_ratio =hot_pixels_count /total_pixels

        print (f"[Результат бинаризации] Горячих пикселей: {hot_pixels_count}/{total_pixels} ({hot_pixels_ratio:.2%})")

        import cv2
        num_labels ,labels =cv2 .connectedComponents (hot_zones_binary ,connectivity =8 )
        filtered_zones =np .zeros_like (hot_zones_binary ,dtype =np .uint8 )

        valid_components =0
        for label_id in range (1 ,num_labels ):
            component_mask =(labels ==label_id )
            component_area =np .sum (component_mask )

            if component_area >=self .min_zone_area :
                filtered_zones [component_mask ]=1
                valid_components +=1

        print (f"[Фильтрация] Найдено {num_labels-1} компонент, оставлено {valid_components} (площадь >= {self.min_zone_area})")

        kernel_small =cv2 .getStructuringElement (cv2 .MORPH_ELLIPSE ,(3 ,3 ))
        kernel_medium =cv2 .getStructuringElement (cv2 .MORPH_ELLIPSE ,(5 ,5 ))

        cleaned_zones =cv2 .morphologyEx (filtered_zones ,cv2 .MORPH_OPEN ,kernel_small ,iterations =1 )

        cleaned_zones =cv2 .morphologyEx (cleaned_zones ,cv2 .MORPH_CLOSE ,kernel_medium ,iterations =2 )

        cleaned_zones =cv2 .morphologyEx (cleaned_zones ,cv2 .MORPH_OPEN ,kernel_small ,iterations =1 )

        final_hot_pixels =np .sum (cleaned_zones )
        print (f"[Финальный результат] Горячих пикселей после очистки: {final_hot_pixels} ({final_hot_pixels/total_pixels:.2%})")

        return cleaned_zones

    def get_patch_features (self ,image :Image .Image ,patch_size :int =14 )->np .ndarray :

        with torch .no_grad ():
            _ ,patch_features =self .feature_extractor ([image ])

        patch_features =patch_features .squeeze (0 ).cpu ().numpy ()

        num_patches =patch_features .shape [0 ]
        feature_dim =patch_features .shape [1 ]

        grid_size =int (np .sqrt (num_patches ))

        if grid_size *grid_size !=num_patches :

            for h in range (1 ,int (np .sqrt (num_patches ))+1 ):
                if num_patches %h ==0 :
                    patch_h =h
                    patch_w =num_patches //h
            if 'patch_h'not in locals ():

                patch_h =patch_w =grid_size
                patch_features =patch_features [:patch_h *patch_w ]
        else :
            patch_h =patch_w =grid_size

        patch_features_reshaped =patch_features .reshape (patch_h ,patch_w ,feature_dim )

        return patch_features_reshaped

    def _build_prototypes (self ,images_pos :List [Image .Image ],images_neg :List [Image .Image ]=None ,
    k_pos :int =5 ,k_neg :int =5 )->Tuple [np .ndarray ,np .ndarray ]:

        def _collect_patches (imgs :List [Image .Image ])->np .ndarray :

            bank =[]
            for img in imgs :
                feats =self .get_patch_features (img ,patch_size =14 )
                bank .append (feats .reshape (-1 ,feats .shape [-1 ]))
            if not bank :
                return np .empty ((0 ,self .dinov3_encoder .feature_dim ),dtype =np .float32 )
            bank =np .concatenate (bank ,axis =0 )

            bank /=(np .linalg .norm (bank ,axis =1 ,keepdims =True )+1e-8 )
            return bank .astype (np .float32 )

        def _fps_select (X :np .ndarray ,k :int )->np .ndarray :

            if X .shape [0 ]==0 :
                return X
            if X .shape [0 ]<=k :
                return X

            idx =[np .random .randint (0 ,X .shape [0 ])]
            for _ in range (1 ,k ):
                d =1 -(X @X [idx [-1 ]][None ,:].T ).squeeze (1 )
                for j in idx :
                    d =np .minimum (d ,1 -(X @X [j ][None ,:].T ).squeeze (1 ))
                idx .append (int (np .argmax (d )))
            return X [idx ]

        P =_collect_patches (images_pos )
        N =_collect_patches (images_neg )if images_neg else np .empty ((0 ,P .shape [1 ]if P .shape [0 ]>0 else self .dinov3_encoder .feature_dim ),dtype =np .float32 )

        Pk =_fps_select (P ,k_pos )if P .shape [0 ]>0 else P
        Nk =_fps_select (N ,k_neg )if N .shape [0 ]>0 else N

        return Pk ,Nk

    def _contrastive_gating (self ,pixel_features :np .ndarray ,Pk :np .ndarray ,Nk :np .ndarray ,
    topk_pos :int =3 ,topk_neg :int =1 ,alpha :float =0.5 ,
    margin :float =0.05 ,beta :float =10.0 )->np .ndarray :

        H ,W ,D =pixel_features .shape
        X =pixel_features .reshape (-1 ,D )

        X /=(np .linalg .norm (X ,axis =1 ,keepdims =True )+1e-8 )

        def _topk_mean_cos (A :np .ndarray ,B :np .ndarray ,k :int ,default :float =0.0 )->np .ndarray :

            if B .shape [0 ]==0 :
                return np .full ((A .shape [0 ],),default ,dtype =np .float32 )
            S =(A @B .T )
            if k ==1 :
                return np .max (S ,axis =1 )
            k =min (k ,S .shape [1 ])
            part =np .partition (S ,-k ,axis =1 )[:,-k :]
            return np .mean (part ,axis =1 )

        s_pos =_topk_mean_cos (X ,Pk ,topk_pos ,default =0.0 )
        s_neg =_topk_mean_cos (X ,Nk ,topk_neg ,default =0.0 )
        s_neg =np .maximum (s_neg ,0.0 )

        z =s_pos -alpha *s_neg -margin
        gating =1.0 /(1.0 +np .exp (-beta *z ))
        return gating .reshape (H ,W ).astype (np .float32 )

    def get_pixel_features (self ,image :np .ndarray ,patch_size :int =14 )->np .ndarray :

        if isinstance (image ,np .ndarray ):
            pil_image =Image .fromarray (image )
        else :
            pil_image =image

        return self .get_patch_features (pil_image ,patch_size )

    def perform_pixel_binning (self ,hot_zones :np .ndarray ,
    pixel_features :np .ndarray ,
    reference_features :Tuple [np .ndarray ,np .ndarray ])->np .ndarray :

        Pk ,Nk =reference_features

        if hot_zones .shape !=pixel_features .shape [:2 ]:
            hot_zones_resized =cv2 .resize (hot_zones .astype (np .uint8 ),
            (pixel_features .shape [1 ],pixel_features .shape [0 ]),
            interpolation =cv2 .INTER_NEAREST )
        else :
            hot_zones_resized =hot_zones

        gating =self ._contrastive_gating (pixel_features ,Pk ,Nk ,
        topk_pos =3 ,topk_neg =1 ,alpha =0.25 ,margin =0.02 ,beta =5.0 )

        gated =(gating >0.3 ).astype (np .uint8 )*(hot_zones_resized >0 ).astype (np .uint8 )

        kernel_small =cv2 .getStructuringElement (cv2 .MORPH_ELLIPSE ,(3 ,3 ))
        kernel_medium =cv2 .getStructuringElement (cv2 .MORPH_ELLIPSE ,(5 ,5 ))
        kernel_large =cv2 .getStructuringElement (cv2 .MORPH_ELLIPSE ,(7 ,7 ))

        gated =cv2 .morphologyEx (gated ,cv2 .MORPH_OPEN ,kernel_small ,iterations =1 )

        gated =cv2 .morphologyEx (gated ,cv2 .MORPH_CLOSE ,kernel_small ,iterations =1 )

        if np .sum (gated )>200 :
            gated =cv2 .morphologyEx (gated ,cv2 .MORPH_CLOSE ,kernel_medium ,iterations =1 )

        return gated

    def split_zones_by_connectivity (self ,zones_mask :np .ndarray )->List [np .ndarray ]:

        connectivity_2d =2 if self .connectivity ==8 else 1
        labeled_zones =label (zones_mask ,connectivity =connectivity_2d )

        regions =regionprops (labeled_zones )

        individual_masks =[]

        for region in regions :

            if region .area <self .min_zone_area :
                continue

            mask =(labeled_zones ==region .label ).astype (np .uint8 )
            individual_masks .append (mask )

        return individual_masks

    def convert_masks_to_sam_format (self ,masks :List [np .ndarray ],
    original_size :Tuple [int ,int ])->List [Dict [str ,Any ]]:

        sam_masks =[]

        for mask in masks :

            if mask .shape !=original_size :
                mask_resized =cv2 .resize (mask ,(original_size [1 ],original_size [0 ]),
                interpolation =cv2 .INTER_NEAREST )
            else :
                mask_resized =mask

            if mask_resized .sum ()>0 :
                kernel =cv2 .getStructuringElement (cv2 .MORPH_ELLIPSE ,(3 ,3 ))
                mask_resized =cv2 .morphologyEx (mask_resized .astype (np .uint8 ),cv2 .MORPH_CLOSE ,kernel ,iterations =1 )
                mask_resized =cv2 .morphologyEx (mask_resized ,cv2 .MORPH_OPEN ,kernel ,iterations =1 )

            segmentation =mask_resized .astype (bool )

            area =int (segmentation .sum ())

            if area ==0 :
                continue

            ys ,xs =np .where (segmentation )
            x_min ,x_max =int (xs .min ()),int (xs .max ())
            y_min ,y_max =int (ys .min ()),int (ys .max ())
            bbox_w =x_max -x_min +1
            bbox_h =y_max -y_min +1
            bbox =[x_min ,y_min ,bbox_w ,bbox_h ]

            cx =x_min +bbox_w //2
            cy =y_min +bbox_h //2

            sam_mask ={'segmentation':segmentation ,'area':area ,'bbox':bbox ,'predicted_iou':1.0 ,'point_coords':[[cx ,cy ]],'stability_score':1.0 ,'crop_box':[0 ,0 ,original_size [1 ],original_size [0 ]]}
            sam_masks .append (sam_mask )
        return sam_masks

    def process_enhanced_heatmap (self ,input_image :Image .Image ,positive_images :List [Image .Image ],negative_images :List [Image .Image ]=None ,save_cleaned_heatmap :bool =True ,output_dir :str =None )->Tuple [torch .Tensor ,List [Dict [str ,Any ]]]:

        if negative_images is None :
            negative_images =[]

        print ("[EnhancedHeatmapProcessor] Генерация исходной heatmap...")

        original_heatmap =self .heatmap_generator .generate_heatmap (
        input_image ,positive_images ,negative_images
        )

        print ("[EnhancedHeatmapProcessor] Извлечение горячих зон...")

        hot_zones =self .extract_hot_zones (original_heatmap )

        print ("[EnhancedHeatmapProcessor] Извлечение патч-признаков...")

        pixel_features =self .get_patch_features (input_image ,patch_size =14 )

        print ("[EnhancedHeatmapProcessor] Построение прототипов...")

        Pk ,Nk =self ._build_prototypes (positive_images ,negative_images ,k_pos =5 ,k_neg =5 )

        print ("[EnhancedHeatmapProcessor] Выполнение пиксельного биннинга...")

        cleaned_zones =self .perform_pixel_binning (hot_zones ,pixel_features ,(Pk ,Nk ))

        print ("[EnhancedHeatmapProcessor] Разбиение зон по связности...")

        individual_masks =self .split_zones_by_connectivity (cleaned_zones )

        print (f"[EnhancedHeatmapProcessor] Найдено {len(individual_masks)} отдельных зон")

        original_size =(input_image .size [1 ],input_image .size [0 ])
        sam_masks =self .convert_masks_to_sam_format (individual_masks ,original_size )

        print (f"[EnhancedHeatmapProcessor] Создано {len(sam_masks)} масок в формате SAM")

        cleaned_heatmap =torch .from_numpy (cleaned_zones .astype (np .float32 ))

        if save_cleaned_heatmap and output_dir :
            self .save_cleaned_heatmap (cleaned_heatmap ,output_dir )

            try :
                import os
                os .makedirs (output_dir ,exist_ok =True )

                self .save_mask_image (hot_zones ,os .path .join (output_dir ,"threshold_mask.png"))

                orig_h ,orig_w =input_image .size [1 ],input_image .size [0 ]
                cleaned_resized =cleaned_zones
                if cleaned_resized .shape !=(orig_h ,orig_w ):
                    cleaned_resized =cv2 .resize (cleaned_resized .astype (np .uint8 ),(orig_w ,orig_h ),interpolation =cv2 .INTER_NEAREST )
                self .save_mask_image (cleaned_resized ,os .path .join (output_dir ,"cleaned_mask.png"))
                print (f"[EnhancedHeatmapProcessor] Бинарные маски сохранены в {output_dir}")
            except Exception as e :
                print (f"[EnhancedHeatmapProcessor] Ошибка при сохранении бинарных масок: {e}")

        return cleaned_heatmap ,sam_masks

    def save_cleaned_heatmap (self ,cleaned_heatmap :torch .Tensor ,output_dir :str )->None :

        import os

        heatmap_np =cleaned_heatmap .cpu ().numpy ()if isinstance (cleaned_heatmap ,torch .Tensor )else cleaned_heatmap

        heatmap_normalized =(heatmap_np *255 ).astype (np .uint8 )

        heatmap_colored =cv2 .applyColorMap (heatmap_normalized ,cv2 .COLORMAP_HOT )

        os .makedirs (output_dir ,exist_ok =True )
        save_path =os .path .join (output_dir ,"cleaned_heatmap.png")
        cv2 .imwrite (save_path ,heatmap_colored )

        print (f"[EnhancedHeatmapProcessor] Очищенная heatmap сохранена: {save_path}")

    def save_mask_image (self ,mask :np .ndarray ,output_path :str )->None :

        import os
        os .makedirs (os .path .dirname (output_path ),exist_ok =True )
        mask_uint8 =(mask .astype (np .uint8 )*255 )if mask .dtype !=np .uint8 else mask
        cv2 .imwrite (output_path ,mask_uint8 )
