import torch
import numpy as np
from PIL import Image
from typing import List ,Tuple ,Optional ,Union ,Dict ,Any
import cv2
from .heatmap_generator import HeatmapGenerator ,crop_heatmap_region ,merge_masks_with_heatmap ,save_crop_debug_info ,visualize_crop_region
from .scoring import ScoreCalculator ,score_multiclass
from .embeddings import EmbeddingExtractor

class FastSAMHeatmapProcessor :

    def __init__ (self ,heatmap_generator :HeatmapGenerator ,fastsam_model =None ,
    embedding_extractor :EmbeddingExtractor =None ,
    score_calculator :ScoreCalculator =None ,debug_mode :bool =False ):

        self .heatmap_generator =heatmap_generator
        self .fastsam_model =fastsam_model
        self .embedding_extractor =embedding_extractor
        self .score_calculator =score_calculator
        self .debug_mode =debug_mode
        self ._model_loaded =False

        self .max_masks_per_crop =15
        self .min_mask_area =200
        self .confidence_threshold =0.5
        self .iou_threshold =0.8

        self ._model_cache ={}
        self ._embedding_cache ={}
        self ._heatmap_cache ={}
        self .cache_size_limit =10

        self .fast_mode =True
        self .skip_small_crops =True
        self .min_crop_size =64
        self .max_processing_time =0.08

    def _load_fastsam_model (self ):

        if self ._model_loaded :
            return

        model_key ="fastsam_default"
        if model_key in self ._model_cache :
            self .fastsam_model =self ._model_cache [model_key ]
            self ._model_loaded =True
            print ("📦 FastSAM модель загружена из кэша")
            return

        try :
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

    def process_image_with_fastsam (self ,input_image :Image .Image ,positive_images :List [Image .Image ],negative_images :List [Image .Image ]=None ,heatmap_threshold :float =0.5 ,crop_padding :int =20 ,min_overlap_ratio :float =0.7 ,skip_scoring_for_hotspot_masks :bool =False )->Tuple [List [torch .Tensor ],torch .Tensor ,Tuple [int ,int ,int ,int ]]:

        if negative_images is None :
            negative_images =[]

        heatmap =self .heatmap_generator .generate_heatmap (
        input_image ,positive_images ,negative_images
        )

        cropped_image ,bbox =crop_heatmap_region (input_image ,heatmap ,threshold =heatmap_threshold ,padding =crop_padding ,min_crop_size =128 ,max_crop_ratio =0.7 ,adaptive_padding =True)

        if self .debug_mode :
            try :
                save_crop_debug_info (input_image ,heatmap ,bbox ,cropped_image )
            except Exception as debug_e :
                print (f"Ошибка сохранения отладочной информации: {debug_e}")

        fastsam_masks =self ._generate_fastsam_masks (cropped_image )

        filtered_masks =merge_masks_with_heatmap (
        fastsam_masks ,heatmap ,bbox ,input_image .size ,min_overlap_ratio
        )

        return filtered_masks ,heatmap ,bbox

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

    def process_image_fast (self ,image :Image .Image ,pos_by_class :Dict [str ,np .ndarray ],heatmap :torch .Tensor ,neg_imgs :np .ndarray =None ,heatmap_threshold :float =0.3 ,crop_padding :int =50 ,min_overlap_ratio :float =0.7 )->List [torch .Tensor ]:

        import time
        start_time =time .time ()

        try :

            if min (image .size )<self .min_crop_size and self .skip_small_crops :
                print (f"⚡ Пропуск маленького изображения {image.size}")
                return []

            cache_key =self ._get_cache_key (image .size ,heatmap_threshold ,crop_padding )
            if cache_key in self ._heatmap_cache :
                cached_heatmap ,bbox =self ._heatmap_cache [cache_key ]
                print ("📦 Heatmap загружен из кэша")
            else :
                cached_heatmap =heatmap

                cropped_image ,bbox =crop_heatmap_region (
                image ,heatmap ,threshold =heatmap_threshold ,padding =crop_padding
                )

                self ._heatmap_cache [cache_key ]=(heatmap ,bbox )
                self ._cleanup_cache ()

            if time .time ()-start_time >self .max_processing_time *0.5 :
                print ("⏰ Таймаут перед FastSAM")
                return []

            fastsam_masks =self ._generate_fastsam_masks_fast (cropped_image )

            if not fastsam_masks :
                print ("⚠️ FastSAM не сгенерировал маски")
                return []

            if time .time ()-start_time >self .max_processing_time *0.8 :
                print ("⏰ Таймаут перед мерджем")
                return fastsam_masks [:5 ]

            merged_masks =merge_masks_with_heatmap (fastsam_masks [:self .max_masks_per_crop //2 ],heatmap ,bbox ,image .size ,min_overlap_ratio)

            self ._cleanup_gpu_memory ()

            elapsed =time .time ()-start_time
            print (f"⚡ Быстрая обработка завершена за {elapsed*1000:.1f}ms")

            return merged_masks

        except Exception as e :
            print (f"❌ Ошибка в быстрой обработке: {e}")
            return []

    def process_image (self ,image :Image .Image ,pos_by_class :Dict [str ,np .ndarray ],heatmap :Optional [torch .Tensor ]=None ,neg_imgs :np .ndarray =None ,heatmap_threshold :float =0.5 ,crop_padding :int =20 ,min_overlap_ratio :float =0.7 ,skip_scoring_for_hotspot_masks :bool =False )->List [torch .Tensor ]:

        try :
            if heatmap is None :
                print ("⚠️ Heatmap не предоставлена, генерируем заглушку...")
                heatmap =torch .rand (image .size [1 ]//8 ,image .size [0 ]//8 )

            print ("✂️ Кропинг горячей области...")
            cropped_image ,crop_box =crop_heatmap_region (
            image ,heatmap ,threshold =heatmap_threshold ,padding =crop_padding
            )

            print (f"🔍 Применение FastSAM к кропнутой области {crop_box}...")
            fastsam_masks =self ._generate_fastsam_masks (cropped_image )

            if not fastsam_masks :
                print ("⚠️ FastSAM не сгенерировал маски, используем fallback")
                fastsam_masks =self ._generate_fallback_masks (cropped_image )

            print (f"🔗 Мердж {len(fastsam_masks)} FastSAM масок с горячей зоной...")
            merged_masks =merge_masks_with_heatmap (
            fastsam_masks ,heatmap .cpu (),crop_box ,image .size ,min_overlap_ratio =min_overlap_ratio
            )

            if merged_masks and pos_by_class is not None :
                if skip_scoring_for_hotspot_masks :
                    print (f"🎯 Пропуск скоринга для {len(merged_masks)} масок из горячих зон")

                    scoring_decisions =[]
                    for i in range (len (merged_masks )):
                        scoring_decisions .append ({'accepted':True ,'class':'hotspot_mask','pos':1.0 ,'neg':0.0 ,'diff':1.0 ,'mask_index':i})
                    scored_masks =merged_masks
                    print (f"✅ Все {len(scored_masks)} масок из горячих зон приняты без скоринга")
                    return scored_masks
                else :
                    print (f"📊 Применение скоринга к {len(merged_masks)} финальным маскам...")
                    scoring_decisions ,scored_masks =self .score_fastsam_masks(image =image ,masks =merged_masks ,pos_by_class =pos_by_class ,neg_imgs =neg_imgs ,skip_scoring =False)

                    if scored_masks :
                        print (f"✅ Финальный результат: {len(scored_masks)} масок прошли скоринг")
                        return scored_masks
                    else :
                        print ("⚠️ Ни одна маска не прошла скоринг")
                        return []
            else :
                print (f"✅ Финальный результат: {len(merged_masks)} масок после мерджа (без скоринга)")
                return merged_masks

        except Exception as e :
            print (f"❌ Ошибка в process_image: {e}")
            return []

    def _generate_fastsam_masks_fast (self ,cropped_image :Image .Image )->List [torch .Tensor ]:

        self ._load_fastsam_model ()

        if self .fastsam_model is None :
            return self ._generate_fallback_masks (cropped_image )

        try :

            max_size =512 if self .fast_mode else 1024
            if max (cropped_image .size )>max_size :
                ratio =max_size /max (cropped_image .size )
                new_size =(int (cropped_image .size [0 ]*ratio ),int (cropped_image .size [1 ]*ratio ))
                cropped_image =cropped_image .resize (new_size ,Image .Resampling .LANCZOS )

            image_np =np .array (cropped_image )

            results =self .fastsam_model (image_np ,device ='cuda'if torch .cuda .is_available ()else 'cpu',retina_masks =False ,imgsz =target_long ,conf =self .confidence_threshold ,iou =self .iou_threshold ,verbose =False)

            masks =[]
            if len (results )>0 and hasattr (results [0 ],'masks')and results [0 ].masks is not None :
                mask_data =results [0 ].masks .data

                num_masks =min (len (mask_data ),self .max_masks_per_crop //2 )

                for i in range (num_masks ):
                    mask =mask_data [i ].cpu ()

                    if torch .sum (mask >0.5 )>=self .min_mask_area :
                        masks .append (mask )

                    if len (masks )>=10 :
                        break

            return masks

        except Exception as e :
            print (f"Ошибка при генерации FastSAM масок: {e}")
            return self ._generate_fallback_masks (cropped_image )

    def _generate_fastsam_masks_with_hotspots (self ,image :Image .Image ,hotspots :List [Tuple [int ,int ]])->List [torch .Tensor ]:

        self ._load_fastsam_model ()

        if self .fastsam_model is None :
            return self ._generate_fallback_masks (image )

        if not hotspots :
            print ("⚠️ Нет горячих точек для FastSAM")
            return self ._generate_fallback_masks (image )

        try :
            image_np =np .array (image )
            orig_h ,orig_w =image_np .shape [:2 ]
            target_long =1024
            if max (orig_w ,orig_h )>target_long :
                scale =target_long /max (orig_w ,orig_h )
                new_w =int (orig_w *scale )
                new_h =int (orig_h *scale )
            else :
                new_w ,new_h =orig_w ,orig_h
            hotspot_x_orig ,hotspot_y_orig =hotspots [0 ]
            hotspot_x =int (round (hotspot_x_orig *new_w /orig_w ))
            hotspot_y =int (round (hotspot_y_orig *new_h /orig_h ))
            hotspot_x =max (0 ,min (new_w -1 ,hotspot_x ))
            hotspot_y =max (0 ,min (new_h -1 ,hotspot_y ))
            print (f"🎯 Используем горячую точку как prompt: исходная=({hotspot_x_orig}, {hotspot_y_orig}) → масштабированная=({hotspot_x}, {hotspot_y})")

            results =self .fastsam_model (image_np ,device ='cuda'if torch .cuda .is_available ()else 'cpu',retina_masks =True ,imgsz =target_long ,conf =self .confidence_threshold ,iou =self .iou_threshold ,verbose =False)

            if len (results )==0 or not hasattr (results [0 ],'masks')or results [0 ].masks is None :
                return self ._generate_fallback_masks (image )

            from ultralytics .models .fastsam import FastSAMPrompt
            prompt_process =FastSAMPrompt (image_np ,results ,device ='cuda'if torch .cuda .is_available ()else 'cpu')

            point_prompt =[[hotspot_x ,hotspot_y ]]
            point_label =[1 ]

            prompted_masks =prompt_process .point_prompt (points =point_prompt ,pointlabel =point_label )

            if prompted_masks is None or (hasattr (prompted_masks ,'__len__')and len (prompted_masks )==0 ):
                print ("⚠️ Point prompt не дал результатов, используем fallback")
                return self ._generate_fallback_masks (image )

            result_masks =[]
            for mask in prompted_masks :

                if isinstance (mask ,torch .Tensor ):
                    mask_np =mask .detach ().cpu ().numpy ()
                elif isinstance (mask ,np .ndarray ):
                    mask_np =mask
                else :
                    continue

                if mask_np .shape !=(orig_h ,orig_w ):
                    mask_resized =cv2 .resize (mask_np .astype (np .uint8 ),(orig_w ,orig_h ),interpolation =cv2 .INTER_NEAREST )
                    mask_tensor =torch .from_numpy (mask_resized .astype (np .float32 ))
                else :
                    mask_tensor =torch .from_numpy (mask_np .astype (np .float32 ))

                if torch .sum (mask_tensor >0.5 )>=self .min_mask_area :
                    result_masks .append (mask_tensor )

                if len (result_masks )>=self .max_masks_per_crop :
                    break

            print (f"✅ Сгенерировано {len(result_masks)} FastSAM масок с горячими точками")
            return result_masks

        except Exception as e :
            print (f"Ошибка при генерации FastSAM масок с горячими точками: {e}")
            return self ._generate_fallback_masks (image )

    def _generate_fastsam_masks (self ,cropped_image :Image .Image )->List [torch .Tensor ]:

        self ._load_fastsam_model ()

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

    def process_image_with_hotspot_fastsam (self ,input_image :Image .Image ,positive_images :List [Image .Image ],negative_images :List [Image .Image ]=None ,heatmap_threshold :float =0.5 ,crop_padding :int =20 ,min_overlap_ratio :float =0.7 ,use_hotspot_fastsam :bool =True ,max_hotspots_for_fastsam :int =8 ,hotspot_threshold :float =0.7 ,fastsam_heatmap_overlap_threshold :float =0.7 ,prefer_fastsam_on_overlap :bool =True ,fallback_to_heatmap :bool =True ,skip_scoring_for_hotspot_masks :bool =False )->Tuple [List [torch .Tensor ],torch .Tensor ,Tuple [int ,int ,int ,int ]]:

        heatmap =self .heatmap_generator .generate_heatmap (
        input_image ,positive_images ,negative_images
        )

        crop_coords =self ._find_crop_region (heatmap ,heatmap_threshold ,crop_padding )

        if crop_coords is None :
            return [],heatmap ,(0 ,0 ,input_image .width ,input_image .height )

        x1 ,y1 ,x2 ,y2 =crop_coords
        cropped_image =input_image .crop ((x1 ,y1 ,x2 ,y2 ))
        cropped_heatmap =crop_heatmap_region (heatmap ,crop_coords )

        if not use_hotspot_fastsam :

            return self .process_image_with_fastsam (
            input_image ,positive_images ,negative_images ,
            heatmap_threshold ,crop_padding ,min_overlap_ratio ,
            skip_scoring_for_hotspot_masks =False
            )

        hotspots =self ._extract_hotspots_from_heatmap (
        cropped_heatmap ,hotspot_threshold ,max_hotspots_for_fastsam
        )

        if not hotspots :

            if fallback_to_heatmap :
                return self ._generate_heatmap_masks (cropped_heatmap ,heatmap_threshold ),heatmap ,crop_coords
            else :
                return [],heatmap ,crop_coords

        fastsam_masks =self ._generate_fastsam_masks_with_hotspots (cropped_image ,hotspots )

        if not fastsam_masks :

            if fallback_to_heatmap :
                return self ._generate_heatmap_masks (cropped_heatmap ,heatmap_threshold ),heatmap ,crop_coords
            else :
                return [],heatmap ,crop_coords

        overlapping_masks =[]
        non_overlapping_masks =[]
        overlay_details_masks =[]

        for i ,fastsam_mask in enumerate (fastsam_masks ):
            overlap_ratio =self ._calculate_heatmap_mask_overlap (
            fastsam_mask ,cropped_heatmap ,heatmap_threshold
            )

            if overlap_ratio >=0.5 :
                overlay_details_masks .append ((i ,fastsam_mask ,overlap_ratio ))
                print (f"🎯 FastSAM маска {i} перекрывается с heatmap на {overlap_ratio:.1%} - добавляем в overlay_details")

            if overlap_ratio >=fastsam_heatmap_overlap_threshold :
                overlapping_masks .append (fastsam_mask )
            else :
                non_overlapping_masks .append (fastsam_mask )

        if overlay_details_masks :
            self ._save_overlay_details_masks (overlay_details_masks ,input_image ,crop_coords )

        if overlapping_masks :

            if prefer_fastsam_on_overlap :
                return fastsam_masks ,heatmap ,crop_coords
            else :

                return overlapping_masks ,heatmap ,crop_coords
        else :

            if non_overlapping_masks :
                return fastsam_masks ,heatmap ,crop_coords
            elif fallback_to_heatmap :
                return self ._generate_heatmap_masks (cropped_heatmap ,heatmap_threshold ),heatmap ,crop_coords
            else :
                return [],heatmap ,crop_coords

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
    min_overlap_ratio :float =0.7 ,
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

    def _save_overlay_details_masks (self ,overlay_masks :List [Tuple [int ,torch .Tensor ,float ]],
    input_image :Image .Image ,crop_coords :Tuple [int ,int ,int ,int ]):

        try :
            import os
            import json
            from datetime import datetime

            overlay_dir ="overlay_details"
            os .makedirs (overlay_dir ,exist_ok =True )

            timestamp =datetime .now ().strftime ("%Y%m%d_%H%M%S")
            overlay_file =os .path .join (overlay_dir ,f"fastsam_overlay_{timestamp}.json")

            overlay_data ={"timestamp":timestamp ,"image_size":{"width":input_image .width ,"height":input_image .height },"crop_coords":{"x1":crop_coords [0 ],"y1":crop_coords [1 ],"x2":crop_coords [2 ],"y2":crop_coords [3 ]},"fastsam_masks":[]}

            for mask_idx ,mask_tensor ,overlap_ratio in overlay_masks :
                mask_np =mask_tensor .cpu ().numpy ()
                mask_coords =np .where (mask_np >0.5 )
                mask_info ={"mask_index":mask_idx ,"overlap_ratio":float (overlap_ratio ),"mask_area":int (np .sum (mask_np >0.5 )),"bbox":self ._mask_to_bbox (mask_np >0.5 ),"coordinates_count":len (mask_coords [0 ])}
                overlay_data ["fastsam_masks"].append (mask_info )
            with open (overlay_file ,'w',encoding ='utf-8')as f :
                json .dump (overlay_data ,f ,indent =2 ,ensure_ascii =False )

            print (f"💾 Сохранено {len(overlay_masks)} FastSAM масок в overlay_details файл: {overlay_file}")

        except Exception as e :
            print (f"❌ Ошибка при сохранении overlay_details: {e}")

    def _mask_to_bbox (self ,mask :np .ndarray )->List [int ]:

        if not np .any (mask ):
            return [0 ,0 ,0 ,0 ]

        rows =np .any (mask ,axis =1 )
        cols =np .any (mask ,axis =0 )

        y_min ,y_max =np .where (rows )[0 ][[0 ,-1 ]]
        x_min ,x_max =np .where (cols )[0 ][[0 ,-1 ]]

        return [int (x_min ),int (y_min ),int (x_max -x_min +1 ),int (y_max -y_min +1 )]

class FastSAMProcessor :

    def __init__ (self ,fastsam_model =None ,device ='auto'):

        self .fastsam_model =fastsam_model
        self .device =device if device !='auto'else ('cuda'if torch .cuda .is_available ()else 'cpu')
        self ._model_loaded =False

    def _load_fastsam_model (self ):

        if self ._model_loaded :
            return

        try :
            from ultralytics import FastSAM

            if self .fastsam_model is None :
                print ("📦 Загрузка FastSAM модели по умолчанию...")
                self .fastsam_model =FastSAM ('FastSAM-s.pt')

            self ._model_loaded =True
            print ("✅ FastSAM модель загружена")

        except ImportError :
            print ("⚠️ FastSAM не установлен")
            self .fastsam_model =None
        except Exception as e :
            print (f"❌ Ошибка загрузки FastSAM: {e}")
            self .fastsam_model =None

    def process_image (self ,image ,pos_by_class ,neg_imgs =None ,heatmap =None ,**kwargs ):

        all_positive_images =[]
        for class_images in pos_by_class .values ():
            all_positive_images .extend (class_images )

        if neg_imgs is None :
            neg_imgs =[]

        if heatmap is not None :
            try :
                return self ._process_with_existing_heatmap (image =image ,heatmap =heatmap ,all_positive_images =all_positive_images ,neg_imgs =neg_imgs ,**kwargs)
            except Exception as e :
                print (f"❌ Ошибка обработки с готовой heatmap: {e}")

        try :
            masks ,_ ,_ =self .process_image_with_hotspot_fastsam (input_image =image ,positive_images =all_positive_images ,negative_images =neg_imgs ,heatmap_threshold =kwargs .get ('heatmap_threshold',0.5 ),crop_padding =kwargs .get ('crop_padding',20 ),min_overlap_ratio =kwargs .get ('min_overlap_ratio',0.7 ),use_hotspot_fastsam =kwargs .get ('use_hotspot_fastsam',True ),max_hotspots_for_fastsam =kwargs .get ('max_hotspots_for_fastsam',8 ),hotspot_threshold =kwargs .get ('hotspot_threshold',0.7 ),fastsam_heatmap_overlap_threshold =kwargs .get ('fastsam_heatmap_overlap_threshold',0.7 ),prefer_fastsam_on_overlap =kwargs .get ('prefer_fastsam_on_overlap',True ),fallback_to_heatmap =kwargs .get ('fallback_to_heatmap',True ),skip_scoring_for_hotspot_masks =kwargs .get ('skip_scoring_for_hotspot_masks',False ))
            return masks
        except Exception as e :
            print (f"❌ Ошибка обработки с горячими точками: {e}")

            return self ._legacy_process_image (image ,pos_by_class ,neg_imgs ,**kwargs )

    def _process_with_existing_heatmap (self ,image ,heatmap ,all_positive_images ,neg_imgs ,**kwargs ):

        print (f"🔥 Используем готовую heatmap размера: {heatmap.shape}")

        image_size =image .size
        if heatmap .shape [-2 :]!=(image_size [1 ],image_size [0 ]):
            print (f"📐 Масштабируем heatmap с {heatmap.shape[-2:]} до {(image_size[1], image_size[0])}")
            heatmap_scaled =torch .nn .functional .interpolate (heatmap .unsqueeze (0 ).unsqueeze (0 )if heatmap .dim ()==2 else heatmap .unsqueeze (0 ),size =(image_size [1 ],image_size [0 ]),mode ='bilinear',align_corners =False).squeeze ()
        else :
            heatmap_scaled =heatmap
        hotspots =self ._extract_hotspots_from_heatmap (heatmap_scaled ,max_hotspots =kwargs .get ('max_hotspots_for_fastsam',8 ),threshold =kwargs .get ('hotspot_threshold',0.7 ))
        if len (hotspots )==0 :
            print ("⚠️ Горячие точки не найдены, возвращаем маски из heatmap")
            if kwargs .get ('fallback_to_heatmap',True ):
                return self ._generate_masks_from_heatmap (heatmap_scaled ,**kwargs )
            else :
                return []
        print (f"🎯 Найдено горячих точек: {len(hotspots)}")
        fastsam_masks =self ._generate_fastsam_masks_with_hotspots (image ,hotspots)
        if fastsam_masks is None :
            fastsam_masks =[]
            print ("⚠️ FastSAM вернул None, используем пустой список")
        if len (fastsam_masks )==0 :
            print ("⚠️ FastSAM не сгенерировал маски, возвращаем маски из heatmap")
            if kwargs .get ('fallback_to_heatmap',True ):
                return self ._generate_masks_from_heatmap (heatmap_scaled ,**kwargs )
            else :
                return []
        filtered_masks =self ._filter_masks_by_heatmap_overlap (fastsam_masks ,heatmap_scaled ,overlap_threshold =kwargs .get ('fastsam_heatmap_overlap_threshold',0.7 ))
        if len (filtered_masks )>0 :
            print (f"✅ Возвращаем {len(filtered_masks)} FastSAM масок после фильтрации")
            return filtered_masks
        elif kwargs .get ('fallback_to_heatmap',True ):
            print ("⚠️ Нет подходящих FastSAM масок, возвращаем маски из heatmap")
            return self ._generate_masks_from_heatmap (heatmap_scaled ,**kwargs )
        else :
            print ("❌ Нет подходящих масок")
            return []

    def _filter_masks_by_heatmap_overlap (self ,masks :List [torch .Tensor ],heatmap :torch .Tensor ,overlap_threshold :float =0.7 )->List [torch .Tensor ]:

        if not masks or heatmap is None :
            return masks

        try :

            if len (masks )>0 :
                mask_h ,mask_w =masks [0 ].shape [-2 :]
                if heatmap .shape !=(mask_h ,mask_w ):
                    heatmap_resized =torch .nn .functional .interpolate (heatmap .unsqueeze (0 ).unsqueeze (0 ).float (),size =(mask_h ,mask_w ),mode ='bilinear',align_corners =False).squeeze ()
                else :
                    heatmap_resized =heatmap
                hot_zones =(heatmap_resized >0.3 ).float ()
                filtered_masks =[]
                rejected_masks =[]
                for i ,mask in enumerate (masks ):
                    mask_binary =(mask >0.5 ).float ()
                    intersection =torch .sum (mask_binary *hot_zones )
                    mask_area =torch .sum (mask_binary )
                    if mask_area >0 :
                        overlap_ratio =intersection /mask_area
                        if overlap_ratio >=overlap_threshold :
                            filtered_masks .append (mask )
                            print (f"   ✅ Маска {i}: перекрытие {overlap_ratio:.3f} >= {overlap_threshold} - принята")
                        else :
                            rejected_masks .append ((i ,overlap_ratio ))
                            print (f"   ❌ Маска {i}: перекрытие {overlap_ratio:.3f} < {overlap_threshold} - отклонена")
                if rejected_masks :
                    print (f"📝 ЛОГИРОВАНИЕ ОТКЛОНЕННЫХ МАСОК: {len(rejected_masks)} масок не прошли фильтр heatmap:")
                    for mask_idx ,ratio in rejected_masks :
                        print (f"   🚫 Маска #{mask_idx}: перекрытие с heatmap = {ratio:.3f}")
                print (f"🔍 Фильтрация масок: {len(masks)} → {len(filtered_masks)} (порог: {overlap_threshold})")
                return filtered_masks
        except Exception as e :
            print (f"❌ Ошибка при фильтрации масок: {e}")
            return masks
        return masks
    def process_image_fast (self ,image ,pos_by_class ,neg_imgs =None ,heatmap =None ,**kwargs ):
        all_positive_images =[]
        for class_images in pos_by_class .values ():
            all_positive_images .extend (class_images )
        if neg_imgs is None :
            neg_imgs =[]
        if heatmap is not None :
            try :

                fast_kwargs =kwargs .copy ()
                fast_kwargs.update({'max_hotspots_for_fastsam':kwargs .get ('max_hotspots_for_fastsam',5 ),'hotspot_threshold':kwargs .get ('hotspot_threshold',0.7 ),'fastsam_heatmap_overlap_threshold':kwargs .get ('fastsam_heatmap_overlap_threshold',0.7 ),'fallback_to_heatmap':kwargs .get ('fallback_to_heatmap',True )})
                return self ._process_with_existing_heatmap (image =image ,heatmap =heatmap ,all_positive_images =all_positive_images ,neg_imgs =neg_imgs ,**fast_kwargs)
            except Exception as e :
                print (f"❌ Ошибка быстрой обработки с готовой heatmap: {e}")

        try :
            masks ,_ ,_ =self .process_image_with_hotspot_fastsam (input_image =image ,positive_images =all_positive_images ,negative_images =neg_imgs ,heatmap_threshold =kwargs .get ('heatmap_threshold',0.5 ),crop_padding =kwargs .get ('crop_padding',10 ),min_overlap_ratio =kwargs .get ('min_overlap_ratio',0.7 ),use_hotspot_fastsam =kwargs .get ('use_hotspot_fastsam',True ),max_hotspots_for_fastsam =kwargs .get ('max_hotspots_for_fastsam',5 ),hotspot_threshold =kwargs .get ('hotspot_threshold',0.7 ),fastsam_heatmap_overlap_threshold =kwargs .get ('fastsam_heatmap_overlap_threshold',0.7 ),prefer_fastsam_on_overlap =kwargs .get ('prefer_fastsam_on_overlap',True ),fallback_to_heatmap =kwargs .get ('fallback_to_heatmap',True ))
            return masks
        except Exception as e :
            print (f"❌ Ошибка быстрой обработки с горячими точками: {e}")
            return self ._legacy_process_image_fast (image ,pos_by_class ,neg_imgs ,**kwargs )
    def _legacy_process_image (self ,image ,pos_by_class ,neg_imgs =None ,**kwargs ):
        self ._load_fastsam_model ()
        if self .fastsam_model is None :
            return []
        try :
            image_np =np .array (image )
            results =self .fastsam_model (image_np ,device =self .device ,retina_masks =True ,imgsz =1024 ,conf =0.4 ,iou =0.9 ,verbose =False)
            masks =[]
            if len (results )>0 and hasattr (results [0 ],'masks')and results [0 ].masks is not None :
                mask_data =results [0 ].masks .data
                for i in range (len (mask_data )):
                    mask =mask_data [i ].cpu ()
                    if torch .sum (mask >0.5 )>=100 :
                        masks .append (mask )
            return masks
        except Exception as e :
            print (f"❌ Ошибка legacy FastSAM обработки: {e}")
            return []

    def _legacy_process_image_fast (self ,image ,pos_by_class ,neg_imgs =None ,**kwargs ):
        self ._load_fastsam_model ()
        if self .fastsam_model is None :
            return []
        try :
            image_np =np .array (image )
            results =self .fastsam_model (image_np ,device =self .device ,retina_masks =False ,imgsz =512 ,conf =0.5 ,iou =0.8 ,verbose =False)
            masks =[]
            if len (results )>0 and hasattr (results [0 ],'masks')and results [0 ].masks is not None :
                mask_data =results [0 ].masks .data
                num_masks =min (len (mask_data ),15 )
                for i in range (num_masks ):
                    mask =mask_data [i ].cpu ()
                    if torch .sum (mask >0.5 )>=200 :
                        masks .append (mask )
            return masks
        except Exception as e :
            print (f"❌ Ошибка legacy быстрой FastSAM обработки: {e}")
            return []