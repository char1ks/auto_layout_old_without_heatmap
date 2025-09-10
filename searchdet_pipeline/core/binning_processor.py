import numpy as np
import torch
import cv2
from PIL import Image
from typing import List ,Dict ,Any ,Tuple ,Optional
from sklearn .metrics .pairwise import cosine_similarity
from sklearn .preprocessing import MinMaxScaler
import matplotlib .pyplot as plt
from .dinov3_encoder import DinoV3Encoder
from .heatmap_generator import DinoV3FeatureExtractor ,adjust_embedding

class BinningProcessor :

    def __init__ (self ,dinov3_encoder :DinoV3Encoder ,num_bins :int =7 ,concept_threshold :int =4 ,binning_mode :str ="quantile",soft :bool =True ,sigma_scale :float =0.5 ,p_high :float =92.0 ,p_low :float =80.0 ,min_area_ratio :float =1e-4 ,keep_largest :bool =False ):

        self .dinov3_encoder =dinov3_encoder
        self .feature_extractor =DinoV3FeatureExtractor (dinov3_encoder ,resize_images =True ,crop_images =False )
        self .num_bins =num_bins
        self .concept_threshold =concept_threshold
        self .binning_mode =binning_mode
        self .soft =soft
        self .sigma_scale =sigma_scale
        self .p_high =p_high
        self .p_low =p_low
        self .min_area_ratio =min_area_ratio
        self .keep_largest =keep_largest

    def get_dinov3_vector (self ,image :Image .Image )->np .ndarray :

        with torch .no_grad ():
            cls_token ,_ =self .feature_extractor ([image ])
        return cls_token .squeeze ().cpu ().numpy ()

    def extract_features_from_masks (self ,image :np .ndarray ,masks :List [Dict [str ,Any ]])->np .ndarray :

        features =[]
        for mask in masks :
            segmentation =mask ['segmentation']

            mask_image =np .full_like (image ,255 )

            seg_bool =segmentation .astype (bool )if segmentation .dtype !=bool else segmentation
            mask_image [seg_bool ]=image [seg_bool ]

            pil_image =Image .fromarray (mask_image )

            feature_vector =self .get_dinov3_vector (pil_image )
            features .append (feature_vector )

        return np .array (features )

    def extract_pixel_features_from_masks (self ,image :np .ndarray ,masks :List [Dict [str ,Any ]])->List [Tuple [int ,List [Tuple [Tuple [int ,int ],np .ndarray ]]]]:
        pil_image =Image .fromarray (image )
        with torch .no_grad ():
            cls_tokens ,patch_tokens =self .feature_extractor ([pil_image ])
        patch_features =patch_tokens [0 ].cpu ().numpy ()
        H ,W =image .shape [:2 ]
        patch_size =14
        num_patches_h =(H +patch_size -1 )//patch_size
        num_patches_w =(W +patch_size -1 )//patch_size
        mask_pixel_features =[]
        for mask_idx ,mask in enumerate (masks ):
            segmentation =mask ['segmentation']
            seg_bool =segmentation .astype (bool )if segmentation .dtype !=bool else segmentation
            pixel_features =[]
            y_coords ,x_coords =np .where (seg_bool )
            for y ,x in zip (y_coords ,x_coords ):
                patch_y =min (y //patch_size ,num_patches_h -1 )
                patch_x =min (x //patch_size ,num_patches_w -1 )
                patch_idx =patch_y *num_patches_w +patch_x
                if patch_idx <len (patch_features ):
                    feature_vector =patch_features [patch_idx ]
                    pixel_features .append (((int (x ),int (y )),feature_vector ))
            mask_pixel_features .append ((mask_idx ,pixel_features ))
        return mask_pixel_features

    def compute_adjusted_embeddings (self ,positive_images :List [Image .Image ],
    negative_images :List [Image .Image ]=None )->np .ndarray :
        if negative_images is None :
            negative_images =[]
        pos_embeddings =[]
        for img in positive_images :
            embedding =self .get_dinov3_vector (img )
            pos_embeddings .append (embedding )
        pos_embeddings =np .array (pos_embeddings )
        neg_embeddings =[]
        if negative_images :
            for img in negative_images :
                embedding =self .get_dinov3_vector (img )
                neg_embeddings .append (embedding )
            neg_embeddings =np .array (neg_embeddings )
        adjusted_embeddings =[]
        for pos_emb in pos_embeddings :
            pos_emb_tensor =torch .from_numpy (pos_emb ).to (self .dinov3_encoder .device )
            pos_embeddings_tensor =torch .from_numpy (pos_embeddings ).to (self .dinov3_encoder .device )
            if len (neg_embeddings )>0 :
                neg_embeddings_tensor =torch .from_numpy (neg_embeddings ).to (self .dinov3_encoder .device )
            else :
                neg_embeddings_tensor =torch .empty (0 ,device =self .dinov3_encoder .device )
            adjusted_emb =adjust_embedding (pos_emb_tensor ,pos_embeddings_tensor ,neg_embeddings_tensor )
            adjusted_embeddings .append (adjusted_emb .cpu ().numpy ())
        adjusted_embeddings =np .array (adjusted_embeddings ).astype ('float32')
        adjusted_embeddings =adjusted_embeddings /np .linalg .norm (adjusted_embeddings ,axis =1 ,keepdims =True )
        return adjusted_embeddings

    def compute_distances (self ,mask_embeddings :np .ndarray ,adjusted_embeddings :np .ndarray )->List [Tuple [int ,List [float ]]]:
        mask_distances =[]
        for i ,mask_emb in enumerate (mask_embeddings ):
            distances =[]
            for adj_emb in adjusted_embeddings :
                sim =cosine_similarity (mask_emb .reshape (1 ,-1 ),adj_emb .reshape (1 ,-1 ))[0 ][0 ]
                distance =1.0 -float (sim )
                distances .append (distance )
            mask_distances .append ((i ,distances ))
        return mask_distances

    def compute_pixel_distances (self ,mask_pixel_features :List [Tuple [int ,List [Tuple [Tuple [int ,int ],np .ndarray ]]]],
    adjusted_embeddings :np .ndarray )->List [Tuple [int ,List [Tuple [Tuple [int ,int ],List [float ]]]]]:
        mask_pixel_distances =[]
        for mask_idx ,pixel_features in mask_pixel_features :
            pixel_distances =[]
            for (x ,y ),pixel_feature in pixel_features :
                distances =[]
                for adj_emb in adjusted_embeddings :
                    sim =cosine_similarity (pixel_feature .reshape (1 ,-1 ),adj_emb .reshape (1 ,-1 ))[0 ][0 ]
                    distance =1.0 -float (sim )
                    distances .append (distance )
                pixel_distances .append (((x ,y ),distances ))
            mask_pixel_distances .append ((mask_idx ,pixel_distances ))
        return mask_pixel_distances

    def perform_binning (self ,mask_distances :List [Tuple [int ,List [float ]]])->Tuple [List [List [float ]],Dict [float ,int ]]:
        all_distances =[]
        for _ ,distances in mask_distances :
            all_distances .extend (distances )
        if len (all_distances )==0 :
            return [],{}
        distances_array =np .array (all_distances )
        print (f"[Биннинг] Всего дистанций: {len(all_distances)}")
        print (f"[Биннинг] Диапазон дистанций: [{distances_array.min():.4f}, {distances_array.max():.4f}]")
        print (f"[Биннинг] Среднее: {distances_array.mean():.4f}, Медиана: {np.median(distances_array):.4f}")
        if self .binning_mode =="quantile":
            quantiles =np .linspace (0 ,1 ,self .num_bins +1 )
            bin_edges =np .quantile (distances_array ,quantiles )
            bin_edges =np .unique (bin_edges )
        else :
            bin_edges =np .linspace (distances_array .min (),distances_array .max (),self .num_bins +1 )
            print (f"[Биннинг] Равномерные границы бинов: {bin_edges}")
        bins =[]
        distance_to_bin ={}
        for i in range (len (bin_edges )-1 ):
            if i ==len (bin_edges )-2 :
                mask =(distances_array >=bin_edges [i ])&(distances_array <=bin_edges [i +1 ])
            else :
                mask =(distances_array >=bin_edges [i ])&(distances_array <bin_edges [i +1 ])
            bin_distances =distances_array [mask ].tolist ()
            bins .append (bin_distances )
            for distance in bin_distances :
                distance_to_bin [float (distance )]=i
            print (f"[Биннинг] Бин {i}: [{bin_edges[i]:.4f}, {bin_edges[i+1]:.4f}] - {len(bin_distances)} дистанций")
        return bins ,distance_to_bin

    def detect_concepts (self ,mask_distances :List [Tuple [int ,List [float ]]],
    distance_to_bin :Dict [float ,int ])->List [int ]:
        selected_masks =[]
        print (f"[Детекция концептов] Анализируем {len(mask_distances)} масок")
        for mask_idx ,distances in mask_distances :
            if not distances :
                continue
            bins_for_mask =[distance_to_bin [distance ]for distance in distances if distance in distance_to_bin ]
            if not bins_for_mask :
                continue
            counts_in_first_bin =sum (1 for bin_idx in bins_for_mask if bin_idx ==0 )
            adaptive_threshold =min (self .concept_threshold ,len (distances ))
            print (f"[Детекция] Маска {mask_idx}: {counts_in_first_bin}/{len(distances)} дистанций в первом бине (порог: {adaptive_threshold})")
            if counts_in_first_bin >=adaptive_threshold :
                selected_masks .append (mask_idx )
                print (f"[Детекция] ✅ Маска {mask_idx} ВЫБРАНА (концепт обнаружен в первом бине)")
            else :
                print (f"[Детекция] ❌ Маска {mask_idx} отклонена (недостаточно дистанций в первом бине)")
        print (f"[Детекция концептов] Итого выбрано масок: {len(selected_masks)}/{len(mask_distances)}")
        return selected_masks

    def perform_pixel_binning (self ,mask_pixel_distances :List [Tuple [int ,List [Tuple [Tuple [int ,int ],List [float ]]]]])->Tuple [List [List [float ]],Dict [float ,int ]]:

        all_distances =[]
        for mask_idx ,pixel_distances in mask_pixel_distances :
            for (x ,y ),distances in pixel_distances :
                all_distances .extend (distances )

        if len (all_distances )==0 :
            return [],{}

        sorted_distances =sorted (all_distances )

        num_bins =min (self .num_bins ,len (sorted_distances ))
        np_bins =np .array_split (np .array (sorted_distances ,dtype =float ),num_bins )
        bins =[arr .tolist ()for arr in np_bins if arr .size >0 ]

        distance_to_bin ={}
        for bin_idx ,bin_distances in enumerate (bins ):
            for distance in bin_distances :
                distance_to_bin [float (distance )]=bin_idx

        return bins ,distance_to_bin

    def create_masks_from_pixels (self ,selected_pixels :List [Tuple [int ,List [Tuple [int ,int ]]]],original_masks :List [Dict [str ,Any ]],image_shape :Tuple [int ,int ])->List [Dict [str ,Any ]]:

        new_masks =[]

        for mask_idx ,pixel_coords in selected_pixels :
            if mask_idx >=len (original_masks ):
                continue

            new_mask =np .zeros (image_shape ,dtype =bool )

            for x ,y in pixel_coords :
                if 0 <=y <new_mask .shape [0 ]and 0 <=x <new_mask .shape [1 ]:
                    new_mask [y ,x ]=True

            if np .any (new_mask ):
                # Ограничиваем выбранные пиксели исходной маской, чтобы убрать выбросы вне контура
                try:
                    seg = original_masks[mask_idx].get('segmentation', None)
                    if seg is not None and seg.shape == new_mask.shape:
                        new_mask = np.logical_and(new_mask, seg.astype(bool))
                except Exception:
                    pass

                if not np.any(new_mask):
                    continue

                # 1) Пространственная когерентность: минимум 3 единицы в окне 3x3 (включая сам пиксель)
                nb_count = cv2.filter2D(new_mask.astype(np.uint8), ddepth=cv2.CV_16U, kernel=np.ones((3, 3), np.uint8))
                min_neighbors = 3
                new_mask = (nb_count >= min_neighbors)

                if not np.any(new_mask):
                    continue

                # 2) Морфология для очистки
                mask_u8 = (new_mask.astype(np.uint8)) * 255
                kernel = np.ones((3, 3), np.uint8)
                mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
                mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

                # 3) Агрессивная фильтрация компонент: поднять порог площади и оставить крупнейшую компоненту
                mask_u8 = self._postprocess_cc(mask_u8, min_area_ratio=max(self.min_area_ratio, 1e-3), keep_largest=True)
                new_mask = mask_u8 > 0

                if not np.any(new_mask):
                    continue

                ys ,xs =np .where (new_mask )
                x_min ,x_max =int (xs .min ()),int (xs .max ())
                y_min ,y_max =int (ys .min ()),int (ys .max ())
                bbox_w =x_max -x_min +1
                bbox_h =y_max -y_min +1
                bbox =[x_min ,y_min ,bbox_w ,bbox_h ]
                area =int (new_mask .sum ())
                cx =x_min +bbox_w //2
                cy =y_min +bbox_h //2

                mask_dict ={'segmentation':new_mask ,'area':area ,'bbox':bbox ,'predicted_iou':1.0 ,'point_coords':[[cx ,cy ]],'stability_score':1.0 ,'crop_box':[0 ,0 ,image_shape [1 ],image_shape [0 ]]}

                new_masks .append (mask_dict )

        return new_masks

    def detect_pixel_concepts (self ,mask_pixel_distances :List [Tuple [int ,List [Tuple [Tuple [int ,int ],List [float ]]]]],
    distance_to_bin :Dict [float ,int ])->List [Tuple [int ,List [Tuple [int ,int ]]]]:

        selected_pixel_masks =[]

        for mask_idx ,pixel_distances in mask_pixel_distances :
            concept_pixels =[]

            for (x ,y ),distances in pixel_distances :
                if not distances :
                    continue

                bins_for_pixel =[]
                for distance in distances :
                    if distance in distance_to_bin :
                        bins_for_pixel .append (distance_to_bin [distance ])

                # Учитываем только нулевой бин, чтобы уменьшить число лишних пикселей
                first_bin_count =sum (1 for bin_idx in bins_for_pixel if bin_idx ==0 )

                # Относительный порог: не менее 60% попаданий в BIN0, но не ниже concept_threshold
                import math
                required =max (self .concept_threshold ,int (math .ceil (0.6 *len (distances ))))

                if first_bin_count >=required :
                    concept_pixels .append ((x ,y ))

            if concept_pixels :
                selected_pixel_masks .append ((mask_idx ,concept_pixels ))

        return selected_pixel_masks

    def _fill_holes (self ,binary ):
        h ,w =binary .shape [:2 ]
        flood =binary .copy ()
        mask =np .zeros ((h +2 ,w +2 ),np .uint8 )
        cv2 .floodFill (flood ,mask ,(0 ,0 ),255 )
        flood_inv =cv2 .bitwise_not (flood )
        return cv2 .bitwise_or (binary ,flood_inv )

    def _extract_valid_contours (self ,binary ,min_area =50 ,min_pts =5 ,
    min_solidity =0.4 ,extent_range =(0.03 ,0.999 )):

        contours ,_ =cv2 .findContours (binary ,cv2 .RETR_EXTERNAL ,cv2 .CHAIN_APPROX_SIMPLE )
        out =[]

        print (f"[DEBUG] Найдено контуров: {len(contours)}")

        for i ,c in enumerate (contours or []):
            if c is None or len (c )<min_pts :
                print (f"[DEBUG] Контур {i}: слишком мало точек ({len(c) if c is not None else 0})")
                continue

            area =float (cv2 .contourArea (c ))
            if area <min_area :
                print (f"[DEBUG] Контур {i}: слишком маленькая площадь ({area:.1f} < {min_area})")
                continue

            hull =cv2 .convexHull (c )
            hull_area =float (cv2 .contourArea (hull ))+1e-6
            solidity =area /hull_area

            x ,y ,w ,h =cv2 .boundingRect (c )
            extent =area /float (w *h +1e-6 )

            print (f"[DEBUG] Контур {i}: area={area:.1f}, solidity={solidity:.3f}, extent={extent:.3f}, bbox=[{x},{y},{w},{h}]")

            if solidity <min_solidity :
                print (f"[DEBUG] Контур {i}: низкая solidity ({solidity:.3f} < {min_solidity})")
                continue

            if not (extent_range [0 ]<=extent <=extent_range [1 ]):
                print (f"[DEBUG] Контур {i}: extent вне диапазона ({extent:.3f} не в [{extent_range[0]}, {extent_range[1]}])")
                continue

            aspect_ratio =max (w ,h )/(min (w ,h )+1e-6 )
            if aspect_ratio >10 :
                print (f"[DEBUG] Контур {i}: слишком вытянутый (aspect_ratio={aspect_ratio:.1f})")
                continue

            print (f"[DEBUG] Контур {i}: ПРИНЯТ")
            out .append ((c ,{"area":area ,"bbox":[x ,y ,w ,h ],"solidity":solidity ,"extent":extent }))

        print (f"[DEBUG] Принято контуров: {len(out)}")
        return out

    def process_with_pixel_binning (self ,input_image :Image .Image ,positive_images :List [Image .Image ],negative_images :List [Image .Image ]=None ,masks :List [Dict [str ,Any ]]=None ,heatmap :torch .Tensor =None )->Tuple [List [Dict [str ,Any ]],List [Dict [str ,Any ]]]:

        if negative_images is None :
            negative_images =[]

        if masks is None :
            if heatmap is None :
                raise ValueError ("Необходимо предоставить либо маски, либо хитмапу")

            input_image_np =np .array (input_image )
            image_height ,image_width =input_image_np .shape [:2 ]
            masks =self .generate_masks_from_heatmap (heatmap =heatmap ,target_size =(image_height ,image_width ),threshold =self .config .heatmap_threshold ,min_area =self .config .heatmap_min_area ,min_solidity =self .config .heatmap_min_solidity ,extent_range =(self .config .heatmap_extent_min ,self .config .heatmap_extent_max ),max_masks =self .config .heatmap_max_masks)

        if not masks :
            return [],[]

        input_image_np =np .array (input_image )
        H ,W =input_image_np .shape [:2 ]

        adjusted_embeddings =self .compute_adjusted_embeddings ([input_image ]+positive_images ,negative_images )
        mask_pixel_features =self .extract_pixel_features_from_masks (input_image_np ,masks )
        mask_pixel_distances =self .compute_pixel_distances (mask_pixel_features ,adjusted_embeddings )
        pixel_bins ,pixel_distance_to_bin =self .perform_pixel_binning (mask_pixel_distances )
        selected_pixel_masks =self .detect_pixel_concepts (mask_pixel_distances ,pixel_distance_to_bin )
        new_masks =self .create_masks_from_pixels (selected_pixel_masks ,masks ,(H ,W ))
        return new_masks ,masks

    def generate_masks_from_heatmap (self ,heatmap :torch .Tensor ,threshold :float =0.3 ,min_area :int =50 ,target_size :Tuple [int ,int ]=None ,min_solidity :float =0.3 ,extent_range :Tuple [float ,float ]=(0.05 ,1.0 ),max_masks :int =5 ,guide_rgb :np .ndarray =None )->List [Dict [str ,Any ]]:
        if isinstance (heatmap ,torch .Tensor ):
            heatmap_np =heatmap .cpu ().numpy ()
        else :
            heatmap_np =heatmap
        print (f"[DEBUG] Исходный размер heatmap: {heatmap_np.shape}")
        H_orig ,W_orig =(None ,None )
        WORK_MAX_SIDE =384
        if target_size is not None :
            H_orig ,W_orig =target_size
            print (f"[DEBUG] Целевой размер (оригинал): {W_orig}x{H_orig}")
            if H_orig >=W_orig :
                H_work =WORK_MAX_SIDE
                W_work =max (1 ,int (round (WORK_MAX_SIDE *(W_orig /H_orig ))))
            else :
                W_work =WORK_MAX_SIDE
                H_work =max (1 ,int (round (WORK_MAX_SIDE *(H_orig /W_orig ))))
            print (f"[DEBUG] Рабочий размер для биннинга: {W_work}x{H_work} (max_side={WORK_MAX_SIDE})")

            Hm_up =cv2 .resize (heatmap_np ,(W_work ,H_work ),interpolation =cv2 .INTER_LINEAR )
            H ,W =H_work ,W_work
        else :
            H_hm ,W_hm =heatmap_np .shape [:2 ]
            if max (H_hm ,W_hm )>WORK_MAX_SIDE :
                if H_hm >=W_hm :
                    H_work =WORK_MAX_SIDE
                    W_work =max (1 ,int (round (WORK_MAX_SIDE *(W_hm /H_hm ))))
                else :
                    W_work =WORK_MAX_SIDE
                    H_work =max (1 ,int (round (WORK_MAX_SIDE *(H_hm /W_hm ))))
                print (f"[DEBUG] Уменьшаем heatmap до рабочего размера: {W_work}x{H_work}")
                Hm_up =cv2 .resize (heatmap_np ,(W_work ,H_work ),interpolation =cv2 .INTER_LINEAR )
                H ,W =H_work ,W_work
            else :
                H ,W =H_hm ,W_hm
                Hm_up =heatmap_np

        Hm_up =(Hm_up -Hm_up .min ())/(Hm_up .max ()-Hm_up .min ()+1e-8 )
        print (f"[DEBUG] Нормализованная heatmap: min={Hm_up.min():.3f}, max={Hm_up.max():.3f}")

        guide_for_filter =None
        if guide_rgb is not None :
            gh ,gw =guide_rgb .shape [:2 ]
            if (gh ,gw )!=(H ,W ):
                guide_for_filter =cv2 .resize (guide_rgb ,(W ,H ),interpolation =cv2 .INTER_AREA )
            else :
                guide_for_filter =guide_rgb
        Hm_filtered =self ._prefilter (Hm_up ,guide_for_filter )
        print (f"[DEBUG] После предфильтрации: min={Hm_filtered.min():.3f}, max={Hm_filtered.max():.3f}")
        edges =self ._compute_bins (Hm_filtered ,self .num_bins ,mode =self .binning_mode )
        print (f"[DEBUG] Границы бинов: {edges}")
        if self .soft :
            W_soft =self ._soft_assign (Hm_filtered ,edges ,self .sigma_scale )
            bin_centers =0.5 *(edges [:-1 ]+edges [1 :])
            Hm_processed =(W_soft *bin_centers [None ,None ,:]).sum (axis =-1 )
            print (f"[DEBUG] После мягкого биннинга: min={Hm_processed.min():.3f}, max={Hm_processed.max():.3f}")
        else :
            bin_indices =np .digitize (Hm_filtered ,edges )-1
            bin_indices =np .clip (bin_indices ,0 ,len (edges )-2 )
            bin_centers =0.5 *(edges [:-1 ]+edges [1 :])
            Hm_processed =bin_centers [bin_indices ]
            print (f"[DEBUG] После жёсткого биннинга: min={Hm_processed.min():.3f}, max={Hm_processed.max():.3f}")

        B =self ._hysteresis_mask (Hm_processed ,self .p_high ,self .p_low )
        print (f"[DEBUG] После hysteresis: активных пикселей {np.sum(B > 0)}")

        B =self ._postprocess_cc (B ,self .min_area_ratio ,self .keep_largest )
        print (f"[DEBUG] После пост-обработки: активных пикселей {np.sum(B > 0)}")

        adaptive_min_area =max (min_area ,(W *H )//10000 )
        print (f"[DEBUG] Адаптивная минимальная площадь: {adaptive_min_area}")

        conts =self ._extract_valid_contours (B ,min_area =adaptive_min_area ,min_pts =5 ,min_solidity =min_solidity ,extent_range =extent_range)

        print (f"[DEBUG] Найдено валидных контуров: {len(conts)}")

        masks =[]

        conts .sort (key =lambda x :x [1 ]["area"],reverse =True )

        if not conts :
            print ("[DEBUG] Не найдено валидных контуров")
            return []

        for i ,(c ,st )in enumerate (conts [:max_masks ]):
            print (f"[DEBUG] Обрабатываем контур {i+1}: area={st['area']:.1f}, bbox={st['bbox']}")

            seg =np .zeros ((H ,W ),dtype =np .uint8 )
            cv2 .drawContours (seg ,[c ],-1 ,255 ,thickness =-1 ,lineType =cv2 .LINE_AA )

            seg =cv2 .GaussianBlur (seg ,(3 ,3 ),0 )
            seg =(seg >127 ).astype (bool )

            if H_orig is not None and W_orig is not None :
                seg_resized =cv2 .resize (seg .astype (np .uint8 ),(W_orig ,H_orig ),interpolation =cv2 .INTER_NEAREST )>0
                seg_bool =seg_resized
                H_out ,W_out =H_orig ,W_orig
            else :
                seg_bool =seg
                H_out ,W_out =H ,W

            if not np .any (seg_bool ):
                print (f"[DEBUG] Контур {i+1} создал пустую маску, пропускаем")
                continue

            ys ,xs =np .where (seg_bool )
            x_min ,x_max =int (xs .min ()),int (xs .max ())
            y_min ,y_max =int (ys .min ()),int (ys .max ())
            w_final =x_max -x_min +1
            h_final =y_max -y_min +1
            bbox_final =[x_min ,y_min ,w_final ,h_final ]
            area_final =int (seg_bool .sum ())

            print (f"[DEBUG] Финальная маска {i+1}: area={area_final}, bbox={bbox_final}")

            masks .append ({"segmentation":seg_bool ,"area":area_final ,"bbox":bbox_final ,"predicted_iou":1.0 ,"point_coords":[[x_min +w_final //2 ,y_min +h_final //2 ]],"stability_score":1.0 ,"crop_box":[0 ,0 ,W_out ,H_out ],})

        print (f"[DEBUG] Создано финальных масок: {len(masks)}")
        return masks

    def process_with_binning (self ,input_image :Image .Image ,positive_images :List [Image .Image ],negative_images :List [Image .Image ]=None ,masks :List [Dict [str ,Any ]]=None ,heatmap :torch .Tensor =None ,use_pixel_level :bool =True )->Tuple [List [Dict [str ,Any ]],List [Dict [str ,Any ]]]:

        if negative_images is None :
            negative_images =[]

        if masks is None :
            if heatmap is None :
                raise ValueError ("Необходимо предоставить либо маски, либо хитмапу")

            input_image_np =np .array (input_image )
            image_height ,image_width =input_image_np .shape [:2 ]
            masks =self .generate_masks_from_heatmap (heatmap =heatmap ,target_size =(image_height ,image_width ),threshold =self .config .heatmap_threshold ,min_area =self .config .heatmap_min_area ,min_solidity =self .config .heatmap_min_solidity ,extent_range =(self .config .heatmap_extent_min ,self .config .heatmap_extent_max ),max_masks =self .config .heatmap_max_masks)

        if not masks :
            return [],[]

        if use_pixel_level :

            selected_masks ,all_masks =self .process_with_pixel_binning (input_image ,positive_images ,negative_images ,masks ,heatmap)
            return selected_masks ,all_masks
        else :

            input_image_np =np .array (input_image )
            H ,W =input_image_np .shape [:2 ]

            adjusted_masks =[]
            for mask in masks :
                seg =mask .get ('segmentation')
                if seg is None :
                    continue

                if seg .dtype ==bool :
                    seg_u8 =(seg .astype (np .uint8 ))*255
                else :
                    seg_u8 =(seg >0 ).astype (np .uint8 )*255

                if seg_u8 .shape [:2 ]!=(H ,W ):
                    seg_u8_resized =cv2 .resize (seg_u8 ,(W ,H ),interpolation =cv2 .INTER_NEAREST )
                else :
                    seg_u8_resized =seg_u8
                seg_bool =seg_u8_resized >0

                ys ,xs =np .where (seg_bool )
                if xs .size ==0 or ys .size ==0 :

                    continue
                x_min ,x_max =int (xs .min ()),int (xs .max ())
                y_min ,y_max =int (ys .min ()),int (ys .max ())
                bbox_w =x_max -x_min +1
                bbox_h =y_max -y_min +1
                bbox =[x_min ,y_min ,bbox_w ,bbox_h ]
                area =int (seg_bool .sum ())
                cx =x_min +bbox_w //2
                cy =y_min +bbox_h //2

                mask ['segmentation']=seg_bool
                mask ['bbox']=bbox
                mask ['area']=area
                mask ['point_coords']=[[cx ,cy ]]
                mask ['crop_box']=[0 ,0 ,W ,H ]
                adjusted_masks .append (mask )
            masks =adjusted_masks

            if not masks :
                return [],[]

            adjusted_embeddings =self .compute_adjusted_embeddings (positive_images ,negative_images )

            mask_embeddings =self .extract_features_from_masks (input_image_np ,masks )

            mask_embeddings =mask_embeddings /np .linalg .norm (mask_embeddings ,axis =1 ,keepdims =True )

            mask_distances =self .compute_distances (mask_embeddings ,adjusted_embeddings )

            bins ,distance_to_bin =self .perform_binning (mask_distances )

            selected_mask_indices =self .detect_concepts (mask_distances ,distance_to_bin )

            selected_masks =[masks [i ]for i in selected_mask_indices if i <len (masks )]

            return selected_masks ,masks

    def visualize_binning_results (self ,bins :List [List [float ]],
    selected_masks :List [int ],
    save_path :Optional [str ]=None )->None :

        bin_averages =[np .mean (bin_distances )for bin_distances in bins ]
        bin_numbers =np .arange (1 ,len (bins )+1 )

        plt .figure (figsize =(10 ,6 ))
        plt .bar (bin_numbers ,bin_averages ,color ='lightgreen')
        plt .title (f'Средние дистанции по бинам (Выбрано масок: {len(selected_masks)})')
        plt .xlabel ('Номер бина')
        plt .ylabel ('Средняя евклидова дистанция')
        plt .grid (True ,alpha =0.3 )

        if save_path :
            plt .savefig (save_path ,dpi =150 ,bbox_inches ='tight')

        plt .show ()

    def visualize_selected_masks (self ,input_image :Image .Image ,
    masks :List [Dict [str ,Any ]],
    selected_indices :List [int ],
    save_path :Optional [str ]=None )->Image .Image :

        img_array =np .array (input_image )

        fig ,ax =plt .subplots (1 ,1 ,figsize =(12 ,8 ))
        ax .imshow (img_array )

        colors =['red','cyan','green','yellow','purple','orange','pink','brown']

        for i ,mask_idx in enumerate (selected_indices ):
            if mask_idx <len (masks ):
                mask =masks [mask_idx ]
                segmentation =mask ['segmentation']
                bbox =mask ['bbox']

                color =colors [i %len (colors )]
                mask_overlay =np .zeros ((*segmentation .shape ,4 ))
                mask_overlay [segmentation ]=[*plt .colors .to_rgba (color ,alpha =0.5 )]

                ax .imshow (mask_overlay )

                rect =patches .Rectangle ((bbox [0 ],bbox [1 ]),bbox [2 ],bbox [3 ],linewidth =2 ,edgecolor =color ,facecolor ='none')
                ax .add_patch (rect )

                ax .text (bbox [0 ],bbox [1 ]-5 ,f'Mask {mask_idx}',
                color =color ,fontsize =12 ,fontweight ='bold')

        ax .set_title (f'Выбранные маски после binning ({len(selected_indices)} из {len(masks)})')
        ax .axis ('off')

        if save_path :
            plt .savefig (save_path ,dpi =150 ,bbox_inches ='tight')

        plt .tight_layout ()
        plt .show ()

        return input_image

    def _compute_bins (self ,hm :np .ndarray ,num_bins :int ,mode :str ="quantile")->np .ndarray :

        vals =hm .flatten ()
        if mode =="quantile":
            qs =np .linspace (0 ,1 ,num_bins +1 )
            edges =np .quantile (vals ,qs )
            edges [0 ],edges [-1 ]=vals .min (),vals .max ()
        else :
            edges =np .linspace (vals .min (),vals .max (),num_bins +1 )
        return edges

    def _soft_assign (self ,hm :np .ndarray ,edges :np .ndarray ,sigma_scale :float =0.5 )->np .ndarray :
        centers =0.5 *(edges [:-1 ]+edges [1 :])
        sigma =(edges [1 :]-edges [:-1 ])*sigma_scale +1e-6
        H ,W =hm .shape
        C =centers .shape [0 ]
        d =hm [...,None ]-centers [None ,None ,:]
        w =np .exp (-0.5 *(d /sigma )**2 )
        w /=(w .sum (axis =-1 ,keepdims =True )+1e-8 )
        return w

    def _hysteresis_mask (self ,hm :np .ndarray ,p_high :float =92.0 ,p_low :float =80.0 )->np .ndarray :

        hi =np .percentile (hm ,p_high )
        lo =np .percentile (hm ,p_low )

        strong =(hm >=hi ).astype (np .uint8 )
        weak =((hm >=lo )&(hm <hi )).astype (np .uint8 )

        mask =strong .copy ()
        num_labels ,labels =cv2 .connectedComponents (weak ,connectivity =8 )

        for lab in range (1 ,num_labels ):
            comp =(labels ==lab ).astype (np .uint8 )

            dilated_strong =cv2 .dilate (strong ,np .ones ((3 ,3 ),np .uint8 ))
            if cv2 .countNonZero (dilated_strong &comp )>0 :
                mask |=comp

        return mask .astype (np .uint8 )

    def _prefilter (self ,hm :np .ndarray ,guide_rgb :np .ndarray =None )->np .ndarray :

        hm_u8 =np .clip (hm *255 ,0 ,255 ).astype (np .uint8 )

        try :
            import cv2 .ximgproc as xi
            if guide_rgb is not None :
                g =cv2 .cvtColor (guide_rgb ,cv2 .COLOR_RGB2BGR )
                filtered =xi .jointBilateralFilter (g ,hm_u8 ,d =7 ,sigmaColor =9 ,sigmaSpace =7 )
                return filtered .astype (np .float32 )/255.0
        except Exception :
            pass

        filtered =cv2 .bilateralFilter (hm_u8 ,d =7 ,sigmaColor =9 ,sigmaSpace =7 )
        return filtered .astype (np .float32 )/255.0

    def _postprocess_cc (self ,mask :np .ndarray ,min_area_ratio :float =1e-4 ,keep_largest :bool =False )->np .ndarray :
        H ,W =mask .shape
        min_area =max (1 ,int (H *W *min_area_ratio ))

        cnts ,_ =cv2 .findContours (mask ,cv2 .RETR_EXTERNAL ,cv2 .CHAIN_APPROX_SIMPLE )
        out =np .zeros_like (mask )

        if keep_largest and cnts :

            c =max (cnts ,key =cv2 .contourArea )
            if cv2 .contourArea (c )>=min_area :
                cv2 .drawContours (out ,[c ],-1 ,255 ,-1 )
            return out

        for c in cnts :
            if cv2 .contourArea (c )>=min_area :
                cv2 .drawContours (out ,[c ],-1 ,255 ,-1 )

        return out