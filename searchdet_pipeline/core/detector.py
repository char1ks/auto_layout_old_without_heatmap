import os
import sys
import cv2
import time
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict ,List ,Optional ,Union ,Any ,Tuple
sys .path .append ('./searchdet-main')
try :
    from mask_withsearch import initialize_sam as init_searchdet
    SEARCHDET_AVAILABLE =True
except Exception as e :
    print (f"⚠️ SearchDet недоступен: {e}")
    SEARCHDET_AVAILABLE =False
from .mask_generation import MaskGenerator
from .filtering import MaskFilter
from .embeddings import EmbeddingExtractor
from .scoring import ScoreCalculator
from .step7_result_saving import ResultSaver
from .sam_predictor import SAMPredictor
from .utils import get_image_size ,get_feature_map_size ,upsample_feature_map
from .dinov3_encoder import DinoV3Encoder
from .heatmap_generator import HeatmapGenerator
from .binning_processor import BinningProcessor
from .enhanced_heatmap_processor import EnhancedHeatmapProcessor
from .fastsam_integration import FastSAMHeatmapProcessor, FastSAMProcessor
from .models import DetectorConfig ,ProcessingResult ,MaskData ,DetectionResult
from ..utils .validation import ImageValidator ,DirectoryValidator ,ValidationError ,validate_processing_pipeline_inputs
import torch
from .models import MaskBackend ,BackboneType
class SearchDetDetector :
    def __init__ (self ,config :Optional [DetectorConfig ]=None ,**kwargs :Any )->None :
        if not SEARCHDET_AVAILABLE :
            raise RuntimeError ("SearchDet не найден")
        if config is None :
            self .config =DetectorConfig .from_dict (kwargs )
        else :
            if kwargs :
                self .config =config .update (**kwargs )
            else :
                self .config =config
        self .params =self .config .to_dict ()
        self .sam_encoder =self .config .sam_encoder
        self .sam_model =self .config .sam_model
        self .device =self .config .device
        if self .device =="auto":
            self .device ='cuda'if torch .cuda .is_available ()else 'cpu'
        self .half =self .config .half_precision
        self .mask_backend =self .config .mask_backend

        if isinstance (self .mask_backend ,MaskBackend ):
            self .mask_backend =self .mask_backend .value
        if isinstance (self .mask_backend ,str ):
            self .mask_backend =self .mask_backend .replace ('_','-')
        self .backbone =self .config .backbone
        if isinstance (self .backbone ,BackboneType ):
            self .backbone =self .backbone .value
        self .dinov3_ckpt =self .config .dinov3_ckpt
        self .nms_iou =self .config .nms_iou
        if not self .backbone .startswith ('dinov2'):
            feat_short =str (self .config .feat_short_side )
            os .environ ['SEARCHDET_FEAT_SHORT_SIDE']=feat_short
            print (f"🔧 Установлено SEARCHDET_FEAT_SHORT_SIDE={feat_short}")
        else :
            print (f"🔧 DINOv2 бэкенд: используется собственный размер модели")
        print (f"🔧 Выбран SAM энкодер: {self.sam_encoder}")
        self .searchdet_resnet ,self .searchdet_layer ,self .searchdet_transform ,self .searchdet_sam =init_searchdet ()
        if not self .backbone .startswith ('dinov2'):
            import torchvision .transforms as transforms
            feat_short_side_env =os .getenv ('SEARCHDET_FEAT_SHORT_SIDE','384')
            if feat_short_side_env =='None'or feat_short_side_env =='none'or feat_short_side_env is None :
                feat_short_side =384
            else :
                feat_short_side =int (feat_short_side_env )
            self .searchdet_transform =transforms .Compose ([transforms .Resize (feat_short_side ),transforms .ToTensor (),transforms .Normalize (mean =[0.485 ,0.456 ,0.406 ],std =[0.229 ,0.224 ,0.225 ]),])
        else :
            self .searchdet_transform =None

        generator_params ={
        k :v for k ,v in self .params .items ()
        if k not in {"mask_backend","device","fastsam_model","fastsam_device","sam_generator","mask_resize_long_side"}
        }
        self .mask_generator =MaskGenerator (mask_backend =self .mask_backend ,device =self .device ,fastsam_model =self .config .fastsam_model ,fastsam_device =self .config .fastsam_device or self .device ,sam_generator =None ,**generator_params)

        segmentation_backend =getattr (self .config ,'segmentation_backend','fastsam')
        if segmentation_backend =='fastsam':
            self .sam_predictor =SAMPredictor (backend_type ="fastsam",mask_generator =self .mask_generator)
        elif segmentation_backend =='heatmap':
            heatmap_threshold =getattr (self .config ,'heatmap_threshold',0.5 )
            self .sam_predictor =SAMPredictor (backend_type ="heatmap",threshold =heatmap_threshold)
        self .mask_filter =MaskFilter (self .params )
        self .embedding_extractor =EmbeddingExtractor (backbone_name =self .config .dinov3_backbone ,device =self .device ,ckpt_path =self .config .dinov3_ckpt)
        self .score_calculator =ScoreCalculator (self .params )
        self .result_saver =ResultSaver (overlay_alpha =self .config .overlay_alpha ,binning_processor =None ,fast_mode =getattr (self .config ,'fast_mode',False ),save_visualizations =getattr (self .config ,'save_visualizations',True ),save_individual_masks =getattr (self .config ,'save_individual_masks',True ),save_heatmap_debug =getattr (self .config ,'save_heatmap_debug',True ),use_flat_output =getattr (self .config ,'use_flat_output',False ),skip_folder_creation =getattr (self .config ,'skip_folder_creation',False ))
        self .dinov3_encoder =DinoV3Encoder (backbone_name =self .config .dinov3_backbone ,ckpt_path =self .config .dinov3_ckpt ,device =self .device ,half_precision =self .config .dino_half_precision ,vit_pooling =self .config .vit_pooling ,loader =self .config .loader ,repo_dir =self .config .repo_dir)
        optimal_size =self ._get_optimal_image_size (1024 ,1024 )
        self .heatmap_generator =HeatmapGenerator (self .dinov3_encoder ,resize_size =optimal_size ,crop_images =False)
        self .binning_processor =BinningProcessor (self .dinov3_encoder ,concept_threshold =1 )
        self .result_saver .binning_processor =self .binning_processor
        self .enhanced_heatmap_processor =EnhancedHeatmapProcessor (self .dinov3_encoder ,resize_size =optimal_size )
        fastsam_model_instance =getattr (self .mask_generator ,'_fastsam_model',None )
        self .fastsam_processor =FastSAMProcessor (fastsam_model =fastsam_model_instance)
        self ._performance_mode =False
        self ._last_processing_time =0.0
        self ._target_processing_time =0.1

    def _downscale_image (self ,image :Image .Image )->Image .Image :

        original_width ,original_height =image .size
        max_size =self ._get_optimal_image_size (original_width ,original_height )

        if max (original_width ,original_height )<=max_size :
            return image

        if original_width >original_height :
            new_width =max_size
            new_height =int ((original_height *max_size )/original_width )
        else :
            new_height =max_size
            new_width =int ((original_width *max_size )/original_height )

        if self .config .downscale_quality =='bicubic':
            resample =Image .BICUBIC
        elif self .config .downscale_quality =='bilinear':
            resample =Image .BILINEAR
        else :
            resample =Image .NEAREST

        return image .resize ((new_width ,new_height ),resample )

    def _get_optimal_image_size (self ,width :int ,height :int )->int :

        return 512

    def set_references (self ,image_path :Union [str ,Path ],positive_dir :Union [str ,Path ],
    negative_dir :Optional [Union [str ,Path ]]=None )->None :
        try :
            ImageValidator .validate_image_path (str (image_path ))
            DirectoryValidator .validate_input_directory (str (positive_dir ))
            if negative_dir :
                DirectoryValidator .validate_input_directory (str (negative_dir ))
        except ValidationError as e :
            raise ValidationError (f"Ошибка валидации референсных данных: {e}")

        raise NotImplementedError ("Метод set_references еще не реализован")

    def find_present_elements_with_heatmap (self ,image_path :Union [str ,Path ],positive_dir :Union [str ,Path ],
    negative_dir :Optional [Union [str ,Path ]]=None ,
    output_dir :str ="output",
    use_heatmap_masks :bool =True )->Dict [str ,Any ]:

        try :
            validate_processing_pipeline_inputs (image_path =str (image_path ),positive_dir =str (positive_dir ),negative_dir =str (negative_dir )if negative_dir else None ,output_dir =output_dir)
        except ValidationError as e :
            raise ValidationError (f"Ошибка валидации входных данных: {e}")

        print (f"🔍 Анализ с heatmap: {image_path}"+"="*60 )
        print ("🔄 ПОСЛЕДОВАТЕЛЬНОСТЬ ВЫПОЛНЕНИЯ HEATMAP PIPELINE:")
        print ("="*80 )

        timing_info :Dict [str ,float ]={}
        t_total =time .time ()

        t_loading =time .time ()
        img_bgr =cv2 .imread (str (image_path ))
        if img_bgr is None :
            raise FileNotFoundError (f"Не удалось загрузить изображение: {image_path}")
        image_np =cv2 .cvtColor (img_bgr ,cv2 .COLOR_BGR2RGB )
        image_pil =Image .fromarray (image_np .astype (np .uint8 ))

        original_size =image_pil .size
        if self .config .enable_image_downscaling :
            image_pil =self ._downscale_image (image_pil )
            print (f"   📐 Размер исходного изображения: {original_size[0]}x{original_size[1]}")
            print (f"   📐 Размер после даунскейлинга: {image_pil.size[0]}x{image_pil.size[1]}")
        else :
            print (f"   📐 Размер исходного изображения: {original_size[0]}x{original_size[1]}")

        timing_info ['image_loading']=time .time ()-t_loading
        image_name =Path (image_path ).name

        print ("1️⃣ Шаг 1: Загрузка positive/negative примеров")
        t_examples =time .time ()
        pos_by_class =self ._load_positive_by_class (positive_dir )
        if len (pos_by_class )==0 :
            print ("   ❌ Нет положительных примеров — прекращаем.")

            return {"found_elements":[],"masks":[]}

        total_pos =sum (len (v )for v in pos_by_class .values ())
        neg_imgs =self ._load_example_images (negative_dir )if negative_dir else []
        timing_info ['examples_loading']=time .time ()-t_examples
        print (f"   📁 Positive: {total_pos} в {len(pos_by_class)} классах\t📁 Negative: {len(neg_imgs)}")

        print ("2️⃣ Шаг 2: Генерация heatmap с помощью DINOv3")
        t_heatmap =time .time ()

        original_size =image_pil .size
        downscaled_image =self ._downscale_image (image_pil )
        downscaled_size =downscaled_image .size
        print (f"   📐 Размер изображения: {original_size} → {downscaled_size}")

        all_positive_images =[]
        for class_images in pos_by_class .values ():
            all_positive_images .extend (class_images )

        heatmap =self .heatmap_generator .generate_heatmap (input_image =downscaled_image ,positive_images =all_positive_images ,negative_images =neg_imgs)
        timing_info ['heatmap_generation']=time .time ()-t_heatmap
        print (f"   📊 Heatmap сгенерирована: {heatmap.shape}")

        if self .config .use_heatmap_sam_hybrid and self .config .sam_refinement_enabled :
            print ("🔄 Шаг 2.5: Гибридный подход - фильтрация горячих зон через SAM")
            t_hybrid =time .time ()

            hotspots =self ._extract_hotspots_from_heatmap (heatmap ,threshold =self .config .heatmap_sam_threshold ,max_hotspots =self .config .max_hotspots_for_sam)

            if hotspots :

                refined_masks =self ._refine_hotspots_with_sam (image_np ,hotspots)

                if refined_masks :
                    masks =refined_masks
                    timing_info ['hybrid_sam_refinement']=time .time ()-t_hybrid
                    print (f"   ✅ SAM уточнил {len(masks)} масок из {len(hotspots)} горячих зон")
                    use_heatmap_masks =False
                else :
                    print ("   ⚠️ SAM не смог уточнить горячие зоны, используем обычные heatmap маски")
            else :
                print ("   ⚠️ Не найдено горячих зон для SAM уточнения")

        if use_heatmap_masks :

            print ("3️⃣ Шаг 3: Генерация масок из heatmap")
            t_mask_gen =time .time ()

            image_height ,image_width =image_np .shape [:2 ]
            print (f"   📐 Размер исходного изображения: {image_width}x{image_height}")

            min_area =getattr (self .config ,'heatmap_min_area',50 )
            threshold =getattr (self .config ,'heatmap_threshold',0.3 )
            min_solidity =getattr (self .config ,'heatmap_min_solidity',0.3 )
            extent_min =getattr (self .config ,'heatmap_extent_min',0.05 )
            extent_max =getattr (self .config ,'heatmap_extent_max',1.0 )
            max_masks =getattr (self .config ,'heatmap_max_masks',5 )

            masks =self .binning_processor .generate_masks_from_heatmap (heatmap =heatmap ,threshold =threshold ,min_area =min_area ,target_size =(image_height ,image_width ),min_solidity =min_solidity ,extent_range =(extent_min ,extent_max ),max_masks =max_masks ,guide_rgb =image_np)
            timing_info ['mask_generation']=time .time ()-t_mask_gen
            print (f"   📊 Сгенерировано масок из heatmap: {len(masks)}")
        else :

            print ("3️⃣ Шаг 3: Генерация масок через SAM/FastSAM")
            t_masks =time .time ()
            masks =self .mask_generator .generate (image_np )
            timing_info ['mask_generation']=time .time ()-t_masks
            print (f"   📊 Сгенерировано масок SAM: {len(masks)}")

        if not masks :
            print ("   ❌ Нет масок для обработки.")
            _saved =self .result_saver .save_all_results (image_np ,[],output_dir ,image_name ,pipeline_config ={"backend":"heatmap_binning"},heatmap =heatmap ,original_size =original_size)
            return {"found_elements":[],"masks":[],"saved_files":_saved ,"heatmap":heatmap ,"output_directory":output_dir }

        print ("4️⃣ Шаг 4: Фильтрация масок")
        t_filtering =time .time ()
        masks =self .mask_filter .apply_all_filters (masks ,image_np )
        timing_info ['mask_filtering']=time .time ()-t_filtering

        if not masks :
            print ("   ❌ Нет валидных масок после фильтров.")

            results_file_name =Path (image_name ).stem
            _saved =self .result_saver .save_all_results (image_np ,[],output_dir ,image_name ,pipeline_config ={"backend":"heatmap_binning"},heatmap =heatmap ,results_file_name =results_file_name ,class_info =pos_by_class ,original_size =original_size)
            return {"found_elements":[],"masks":[],"saved_files":_saved ,"heatmap":heatmap ,"output_directory":output_dir }

        print ("5️⃣ Шаг 5: Пропуск FastSAM постобработки (отключен)")
        selected_mask_indices =list (range (len (masks )))
        is_index_list =True
        total_candidates =len (masks )

        print (f"   📊 Используется масок: {len(selected_mask_indices)} из {total_candidates}")

        print ("6️⃣ Шаг 6: Формирование результатов")
        t_result =time .time ()

        found =[]
        result_masks =[]
        H ,W =image_np .shape [:2 ]

        if is_index_list :

            for mask_idx in selected_mask_indices :
                if mask_idx >=len (masks ):
                    continue
                mask_dict =masks [mask_idx ].copy ()
                class_name =list (pos_by_class .keys ())[0 ]if pos_by_class else "detected"
                mask_dict ['confidence']=0.8
                mask_dict ['class']=class_name
                if 'area'not in mask_dict and 'segmentation'in mask_dict :
                    mask_dict ['area']=int (np .sum (mask_dict ['segmentation']))
                bx =mask_dict .get ('bbox',[0 ,0 ,0 ,0 ])
                if len (bx )==4 and (bx [2 ]<=W and bx [3 ]<=H ):
                    x1 ,y1 ,w ,h =bx
                    bbox_xyxy =[int (x1 ),int (y1 ),int (x1 +w ),int (y1 +h )]
                else :
                    bbox_xyxy =[int (bx [0 ]),int (bx [1 ]),int (bx [2 ]),int (bx [3 ])]
                found .append ({'mask':mask_dict ,'confidence':float (mask_dict ['confidence']),'bbox':mask_dict ['bbox'],'class':class_name})
                result_masks .append (mask_dict )
        else :

            for i ,mask_tensor in enumerate (selected_mask_indices ):

                if hasattr (mask_tensor ,'cpu'):
                    mask_np =mask_tensor .cpu ().numpy ()
                elif hasattr (mask_tensor ,'numpy'):
                    mask_np =mask_tensor .numpy ()
                else :
                    mask_np =np .array (mask_tensor )

                mask_binary =(mask_np >0.5 ).astype (np .uint8 )

                if mask_binary .sum ()==0 :
                    continue

                class_name =list (pos_by_class .keys ())[0 ]if pos_by_class else "detected"

                ys ,xs =np .where (mask_binary )
                if xs .size and ys .size :
                    x_min ,x_max =int (xs .min ()),int (xs .max ())
                    y_min ,y_max =int (ys .min ()),int (ys .max ())
                    bbox =[x_min ,y_min ,x_max -x_min +1 ,y_max -y_min +1 ]
                else :
                    bbox =[0 ,0 ,0 ,0 ]

                mask_dict ={'segmentation':mask_binary ,'bbox':bbox ,'area':int (mask_binary .sum ()),'confidence':0.8 ,'class':class_name}

                found .append ({'mask':mask_dict ,'confidence':float (mask_dict ['confidence']),'bbox':mask_dict ['bbox'],'class':class_name})
                result_masks .append (mask_dict )

        timing_info ['result_formatting']=time .time ()-t_result

        print ("7️⃣ Шаг 7: Сохранение результатов")
        t_saving =time .time ()

        results_file_name =Path (image_name ).stem

        saved_files =self .result_saver .save_all_results (image_np ,result_masks ,output_dir ,image_name ,pipeline_config ={"backend":"heatmap_binning"},heatmap =heatmap ,results_file_name =results_file_name ,class_info =pos_by_class ,original_size =original_size)
        timing_info ['result_saving']=time .time ()-t_saving

        total_time =time .time ()-t_total
        timing_info ['total_time']=total_time

        print (f"🎯 Найдено элементов: {len(found)}")
        print (f"⏱️ Общее время: {total_time:.2f} сек")
        print (f"💾 Результаты сохранены в: {output_dir}")
        print (f"📁 Сохранено файлов: {len(saved_files)}")
        self ._print_timing_statistics (timing_info )

        return {"found_elements":found ,"masks":result_masks ,"timing_info":timing_info ,"output_directory":output_dir ,"saved_files":saved_files ,"heatmap":heatmap}

    def find_present_elements_with_fastsam_integration (self ,image_path :Union [str ,Path ],positive_dir :Union [str ,Path ],
    negative_dir :Optional [Union [str ,Path ]]=None ,
    output_dir :str ="output")->Dict [str ,Any ]:

        try :
            validate_processing_pipeline_inputs (image_path =str (image_path ),positive_dir =str (positive_dir ),negative_dir =str (negative_dir )if negative_dir else None ,output_dir =output_dir)
        except ValidationError as e :
            raise ValidationError (f"Ошибка валидации входных данных: {e}")

        print (f"🚀 FastSAM интеграция: {image_path}"+"="*60 )
        print ("🔄 ПОСЛЕДОВАТЕЛЬНОСТЬ ВЫПОЛНЕНИЯ FASTSAM PIPELINE:")
        print ("="*80 )

        timing_info :Dict [str ,float ]={}
        t_total =time .time ()

        t_loading =time .time ()
        img_bgr =cv2 .imread (str (image_path ))
        if img_bgr is None :
            raise FileNotFoundError (f"Не удалось загрузить изображение: {image_path}")
        image_np =cv2 .cvtColor (img_bgr ,cv2 .COLOR_BGR2RGB )
        image_pil =Image .fromarray (image_np .astype (np .uint8 ))
        timing_info ['image_loading']=time .time ()-t_loading
        image_name =Path (image_path ).name

        print ("1️⃣ Шаг 1: Загрузка positive/negative примеров")
        t_examples =time .time ()
        pos_by_class =self ._load_positive_by_class (positive_dir )
        if len (pos_by_class )==0 :
            print ("   ❌ Нет положительных примеров — прекращаем.")
            return {"found_elements":[],"masks":[]}

        total_pos =sum (len (v )for v in pos_by_class .values ())
        neg_imgs =self ._load_example_images (negative_dir )if negative_dir else []
        timing_info ['examples_loading']=time .time ()-t_examples
        print (f"   📁 Positive: {total_pos} в {len(pos_by_class)} классах\t📁 Negative: {len(neg_imgs)}")

        print ("2️⃣ Шаг 2: Генерация heatmap с помощью DINOv3")
        t_heatmap =time .time ()

        all_positive_images =[]
        for class_images in pos_by_class .values ():
            all_positive_images .extend (class_images )

        heatmap =self .heatmap_generator .generate_heatmap (input_image =image_pil ,positive_images =all_positive_images ,negative_images =neg_imgs)
        timing_info ['heatmap_generation']=time .time ()-t_heatmap
        print (f"   📊 Heatmap сгенерирована: {heatmap.shape}")

        print ("3️⃣ Шаг 3: FastSAM интеграция с heatmap")
        t_fastsam =time .time ()

        fastsam_masks =self .fastsam_processor .process_image (image =image_pil ,pos_by_class =pos_by_class ,heatmap =heatmap ,neg_imgs =neg_imgs ,skip_scoring_for_hotspot_masks =self .config .skip_scoring_for_hotspot_masks)
        timing_info ['fastsam_integration']=time .time ()-t_fastsam

        if fastsam_masks is None :
            fastsam_masks =[]
            print ("   ⚠️ FastSAM вернул None, используем пустой список")

        print (f"   🎯 FastSAM сгенерировал {len(fastsam_masks)} финальных масок")

        if not fastsam_masks :
            print ("   ❌ Нет FastSAM масок для обработки.")

            _saved =self .result_saver .save_all_results (image_np ,[],output_dir ,image_name ,pipeline_config ={"backend":"fastsam_integration"},heatmap =heatmap)
            return {"found_elements":[],"masks":[],"saved_files":_saved ,"heatmap":heatmap ,"output_directory":output_dir }

        print ("4️⃣ Шаг 4: Формирование результатов")
        t_result =time .time ()

        found =[]
        result_masks =[]

        for mask_item in fastsam_masks :

            if isinstance (mask_item ,torch .Tensor ):
                seg =(mask_item .detach ().cpu ().numpy ()>0.5 ).astype (bool )
                ys ,xs =np .where (seg )
                if xs .size and ys .size :
                    x_min ,x_max =int (xs .min ()),int (xs .max ())
                    y_min ,y_max =int (ys .min ()),int (ys .max ())
                    bbox =[x_min ,y_min ,x_max -x_min +1 ,y_max -y_min +1 ]
                else :
                    bbox =[0 ,0 ,0 ,0 ]
                class_name =list (pos_by_class .keys ())[0 ]if pos_by_class else "detected"
                md ={'segmentation':seg ,'bbox':bbox ,'area':int (seg .sum ()),'confidence':0.9 ,'class':class_name}
                found .append ({'mask':md ,'confidence':float (md ['confidence']),'bbox':md ['bbox'],'class':class_name})
                result_masks .append (md )
            elif isinstance (mask_item ,dict )and 'segmentation'in mask_item :
                class_name =list (pos_by_class .keys ())[0 ]if pos_by_class else "detected"
                md =mask_item .copy ()
                md ['confidence']=float (md .get ('confidence',0.9 ))
                md ['class']=md .get ('class',class_name )
                if 'area'not in md :
                    md ['area']=int (np .sum (md ['segmentation']))
                if 'bbox'not in md :
                    seg =md ['segmentation']
                    ys ,xs =np .where (seg )
                    if xs .size and ys .size :
                        x_min ,x_max =int (xs .min ()),int (xs .max ())
                        y_min ,y_max =int (ys .min ()),int (ys .max ())
                        md ['bbox']=[x_min ,y_min ,x_max -x_min +1 ,y_max -y_min +1 ]
                    else :
                        md ['bbox']=[0 ,0 ,0 ,0 ]
                found .append ({
                'mask':md ,
                'confidence':float (md ['confidence']),
                'bbox':md ['bbox'],
                'class':md ['class']
                })
                result_masks .append (md )
            else :

                continue

        timing_info ['result_formatting']=time .time ()-t_result

        print ("5️⃣ Шаг 5: Сохранение результатов (только FastSAM маски)")
        t_saving =time .time ()
        saved_files =self .result_saver .save_all_results (image_np ,result_masks ,output_dir ,image_name ,pipeline_config ={"backend":"fastsam_integration"},heatmap =heatmap)
        timing_info ['result_saving']=time .time ()-t_saving

        total_time =time .time ()-t_total
        timing_info ['total_time']=total_time

        print (f"🎯 Найдено FastSAM элементов: {len(found)}")
        print (f"⏱️ Общее время: {total_time:.2f} сек")
        print (f"💾 Результаты сохранены в: {output_dir}")
        print (f"📁 Сохранено файлов: {len(saved_files)}")
        self ._print_timing_statistics (timing_info )

        return {"found_elements":found ,"masks":result_masks ,"timing_info":timing_info ,"output_directory":output_dir ,"saved_files":saved_files ,"heatmap":heatmap}

    def find_present_elements_with_fastsam_optimized (self ,image_path :Union [str ,Path ],positive_dir :Union [str ,Path ],negative_dir :Optional [Union [str ,Path ]]=None ,output_dir :str ="output")->Dict [str ,Any ]:

        try :
            validate_processing_pipeline_inputs (image_path =str (image_path ),positive_dir =str (positive_dir ),negative_dir =str (negative_dir )if negative_dir else None ,output_dir =output_dir)
        except ValidationError as e :
            raise ValidationError (f"Ошибка валидации входных данных: {e}")

        print (f"🚀 Оптимизированная FastSAM интеграция: {image_path}"+"="*60 )
        print ("🔄 АДАПТИВНЫЙ РЕЖИМ ПРОИЗВОДИТЕЛЬНОСТИ")
        print ("="*80 )

        timing_info :Dict [str ,float ]={}
        t_total =time .time ()

        t_loading =time .time ()
        img_bgr =cv2 .imread (str (image_path ))
        if img_bgr is None :
            raise FileNotFoundError (f"Не удалось загрузить изображение: {image_path}")
        image_np =cv2 .cvtColor (img_bgr ,cv2 .COLOR_BGR2RGB )
        image_pil =Image .fromarray (image_np .astype (np .uint8 ))
        timing_info ['image_loading']=time .time ()-t_loading
        image_name =Path (image_path ).name

        print ("1️⃣ Шаг 1: Загрузка positive/negative примеров")
        t_examples =time .time ()
        pos_by_class =self ._load_positive_by_class (positive_dir )
        if len (pos_by_class )==0 :
            print ("   ❌ Нет положительных примеров — прекращаем.")
            return {"found_elements":[],"masks":[]}

        total_pos =sum (len (v )for v in pos_by_class .values ())
        neg_imgs =self ._load_example_images (negative_dir )if negative_dir else []
        timing_info ['examples_loading']=time .time ()-t_examples
        print (f"   📁 Positive: {total_pos} в {len(pos_by_class)} классах\t📁 Negative: {len(neg_imgs)}")

        use_fast_mode =self ._performance_mode or self ._last_processing_time >self ._target_processing_time

        if use_fast_mode :
            print ("⚡ Режим: БЫСТРЫЙ (цель: <100ms)")
            processor_method =self .fastsam_processor .process_image_fast
        else :
            print ("🎯 Режим: КАЧЕСТВЕННЫЙ")
            processor_method =self .fastsam_processor .process_image

        print ("2️⃣ Шаг 2: FastSAM обработка изображения")
        t_fastsam =time .time ()

        try :

            all_positive_images =[]
            for class_images in pos_by_class .values ():
                all_positive_images .extend (class_images )

            heatmap =self .heatmap_generator .generate_heatmap (input_image =image_pil ,positive_images =all_positive_images ,negative_images =neg_imgs)
            print (f"   📊 Heatmap сгенерирована: {heatmap.shape}")

            fastsam_masks =processor_method (image =image_pil ,pos_by_class =pos_by_class ,neg_imgs =neg_imgs ,heatmap =heatmap ,heatmap_threshold =0.3 ,crop_padding =50 ,min_overlap_ratio =0.5 ,fastsam_heatmap_overlap_threshold =0.5 ,fallback_to_heatmap =True)

            timing_info ['fastsam_processing']=time .time ()-t_fastsam

            if fastsam_masks is None :
                fastsam_masks =[]
                print ("   ⚠️ FastSAM вернул None, используем пустой список")

            if not fastsam_masks :
                print ("   ⚠️ FastSAM не нашел подходящих масок")
                return {"found_elements":[],"masks":[]}

            print (f"   ✅ FastSAM нашел {len(fastsam_masks)} масок")

        except Exception as e :
            print (f"   ❌ Ошибка FastSAM обработки: {e}")
            return {"found_elements":[],"masks":[]}

        print ("3️⃣ Шаг 3: Форматирование результатов")
        t_format =time .time ()

        formatted_masks =[]
        for i ,mask in enumerate (fastsam_masks ):
            try :

                if hasattr (mask ,'cpu'):
                    mask_np =(mask .cpu ().numpy ()>0.5 ).astype (bool )
                else :
                    mask_np =(mask >0.5 ).astype (bool )

                ys ,xs =np .where (mask_np )
                if len (ys )>0 and len (xs )>0 :
                    x1 ,y1 ,x2 ,y2 =xs .min (),ys .min (),xs .max (),ys .max ()
                    bbox =[int (x1 ),int (y1 ),int (x2 -x1 ),int (y2 -y1 )]
                    area =int (np .sum (mask_np ))

                    formatted_mask ={'segmentation':mask_np ,'bbox':bbox ,'area':area ,'predicted_iou':0.8 ,'stability_score':0.8 ,'crop_box':[0 ,0 ,image_pil .size [0 ],image_pil .size [1 ]]}
                    formatted_masks .append (formatted_mask )

            except Exception as e :
                print (f"   ⚠️ Ошибка форматирования маски {i}: {e}")
                continue

        timing_info ['formatting']=time .time ()-t_format

        print ("4️⃣ Шаг 4: Сохранение результатов")
        t_save =time .time ()

        try :
            saved_files =self .result_saver .save_all_results (image_np ,formatted_masks ,output_dir ,image_name ,pipeline_config ={"backend":"fastsam_optimized"},heatmap =heatmap)
            timing_info ['saving']=time .time ()-t_save
            print (f"   💾 Результаты сохранены в {output_dir}")

        except Exception as e :
            print (f"   ⚠️ Ошибка сохранения: {e}")
            timing_info ['saving']=time .time ()-t_save
            saved_files =[]

        total_time =time .time ()-t_total
        self ._last_processing_time =total_time

        
        if total_time >self ._target_processing_time and not self ._performance_mode :
            self ._performance_mode =True
        elif total_time <self ._target_processing_time *0.7 and self ._performance_mode :
            self ._performance_mode =False

        print ("\n"+"="*80 )
        print ("📊 СТАТИСТИКА ПРОИЗВОДИТЕЛЬНОСТИ:")
        for step ,duration in timing_info .items ():
            print (f"   {step}: {duration*1000:.1f}ms")
        print (f"   ОБЩЕЕ ВРЕМЯ: {total_time*1000:.1f}ms")
        print (f"   НАЙДЕНО ЭЛЕМЕНТОВ: {len(formatted_masks)}")
        print ("="*80 )

        return {"found_elements":formatted_masks ,"masks":fastsam_masks ,"timing_info":timing_info ,"output_directory":output_dir ,"saved_files":saved_files}

    def find_present_elements_with_enhanced_heatmap (self ,image_path :Union [str ,Path ],positive_dir :Union [str ,Path ],
    negative_dir :Optional [Union [str ,Path ]]=None ,
    output_dir :str ="output")->Dict [str ,Any ]:

        try :
            validate_processing_pipeline_inputs (image_path =str (image_path ),positive_dir =str (positive_dir ),negative_dir =str (negative_dir )if negative_dir else None ,output_dir =output_dir)
        except ValidationError as e :
            raise ValidationError (f"Ошибка валидации входных данных: {e}")

        print (f"🔍 Улучшенный анализ с heatmap: {image_path}"+"="*60 )
        print ("🔄 ДЕТАЛЬНАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ ВЫПОЛНЕНИЯ ENHANCED HEATMAP PIPELINE:")
        print ("="*80 )

        timing_info :Dict [str ,float ]={}
        t_total =time .time ()

        print ("1️⃣ Шаг 1: Загрузка изображения")
        t_loading =time .time ()
        img_bgr =cv2 .imread (str (image_path ))
        if img_bgr is None :
            raise FileNotFoundError (f"Не удалось загрузить изображение: {image_path}")
        image_np =cv2 .cvtColor (img_bgr ,cv2 .COLOR_BGR2RGB )
        image_name =Path (image_path ).stem
        timing_info ['image_loading']=time .time ()-t_loading

        print ("2️⃣ Шаг 2: Загрузка positive/negative примеров")
        t_examples =time .time ()
        pos_by_class =self ._load_positive_by_class (positive_dir )
        if len (pos_by_class )==0 :
            print ("   ❌ Нет положительных примеров — прекращаем.")
            return {"found_elements":[],"masks":[]}

        total_pos =sum (len (v )for v in pos_by_class .values ())
        neg_imgs =self ._load_example_images (negative_dir )if negative_dir else []
        timing_info ['examples_loading']=time .time ()-t_examples
        print (f"   📁 Positive: {total_pos} в {len(pos_by_class)} классах\t📁 Negative: {len(neg_imgs)}")
        print ("3️⃣ Шаг 3: Генерация heatmap с DINOv3")
        t_heatmap =time .time ()
        image_pil =Image .fromarray (image_np .astype (np .uint8 ))
        original_size =image_pil .size
        if self .config .enable_image_downscaling :
            image_pil =self ._downscale_image (image_pil )
            print (f"   📐 Размер исходного изображения: {original_size[0]}x{original_size[1]}")
            print (f"   📐 Размер после даунскейлинга: {image_pil.size[0]}x{image_pil.size[1]}")
        else :
            print (f"   📐 Размер исходного изображения: {original_size[0]}x{original_size[1]}")
        all_positive_images =[]
        for class_images in pos_by_class .values ():
            all_positive_images .extend (class_images )
        heatmap =self .heatmap_generator .generate_heatmap (input_image =image_pil ,positive_images =all_positive_images ,negative_images =neg_imgs)
        timing_info ['heatmap_generation']=time .time ()-t_heatmap
        print (f"   🔥 Heatmap размер: {heatmap.shape}, min: {heatmap.min():.3f}, max: {heatmap.max():.3f}")

        print ("4️⃣ Шаг 4: Улучшенная обработка heatmap с пиксельным биннингом")
        t_enhanced =time .time ()

        positive_images =[]
        for class_name ,class_images in pos_by_class .items ():
            for img in class_images :
                if isinstance (img ,np .ndarray ):
                    positive_images .append (Image .fromarray (img .astype (np .uint8 )))
                else :
                    positive_images .append (img )

        negative_images =[]
        for img in neg_imgs :
            if isinstance (img ,np .ndarray ):
                negative_images .append (Image .fromarray (img .astype (np .uint8 )))
            else :
                negative_images .append (img )

        cleaned_heatmap ,zone_masks =self .enhanced_heatmap_processor .process_enhanced_heatmap (image_pil ,positive_images ,negative_images ,save_cleaned_heatmap =True ,output_dir =output_dir)
        timing_info ['enhanced_processing']=time .time ()-t_enhanced

        print (f"   ✨ Получено {len(zone_masks)} зон после улучшенной обработки")

        print ("6️⃣ Шаг 6: Обработка масок и скоринг")
        t_scoring =time .time ()

        found =[]
        result_masks =[]

        for i ,mask_dict in enumerate (zone_masks ):
            image_pil =Image .fromarray (image_np .astype (np .uint8 ))
            try :
                mask_vecs =self .embedding_extractor .extract_mask_embeddings (image_pil ,[mask_dict ])
                if mask_vecs .shape [0 ]>0 :
                    class_scores =self.score_calculator.calculate_scores(mask_vecs ,pos_by_class ,neg_imgs)
                    best_class =None
                    best_score =-1
                    for class_name ,scores in class_scores .items ():
                        if len (scores )>0 and scores [0 ]>best_score :
                            best_score =scores [0 ]
                            best_class =class_name

                    if best_class and best_score >0.5 :
                        found .append ({'class':best_class ,'score':best_score ,'mask_index':i})
                        mask_dict .update ({'class':best_class ,'score':best_score ,'mask_index':i})
                        result_masks .append (mask_dict )

            except Exception as e :
                print (f"   ⚠️ Ошибка обработки маски {i}: {e}")
                continue

        timing_info ['scoring']=time .time ()-t_scoring

        print ("7️⃣ Шаг 7: Сохранение результатов")
        t_saving =time .time ()
        saved_files =self .result_saver .save_all_results (image_np ,result_masks ,output_dir ,image_name ,pipeline_config ={"backend":"enhanced_heatmap"},heatmap =cleaned_heatmap ,positive_images =positive_images ,negative_images =negative_images)
        timing_info ['result_saving']=time .time ()-t_saving

        total_time =time .time ()-t_total
        timing_info ['total_time']=total_time

        print (f"🎯 Найдено элементов: {len(found)}")
        print (f"⏱️ Общее время: {total_time:.2f} сек")
        print (f"💾 Результаты сохранены в: {output_dir}")
        print (f"📁 Сохранено файлов: {len(saved_files)}")
        self ._print_timing_statistics (timing_info )

        return {"found_elements":found ,"masks":result_masks ,"timing_info":timing_info ,"output_directory":output_dir ,"saved_files":saved_files ,"heatmap":cleaned_heatmap ,"zone_masks":zone_masks}

    def find_present_elements (self ,image_path :Union [str ,Path ],positive_dir :Union [str ,Path ],
    negative_dir :Optional [Union [str ,Path ]]=None ,
    output_dir :str ="output")->Dict [str ,Any ]:
        try :
            validate_processing_pipeline_inputs (image_path =str (image_path ),positive_dir =str (positive_dir ),negative_dir =str (negative_dir )if negative_dir else None ,output_dir =output_dir)
        except ValidationError as e :
            raise ValidationError (f"Ошибка валидации входных данных: {e}")

        print (f"🔍 Модульный анализ: {image_path}"+"="*60 )
        print ("🔄 ДЕТАЛЬНАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ ВЫПОЛНЕНИЯ МОДУЛЬНОГО PIPELINE:")
        print ("="*80 )
        print ("8️⃣ searchdet_pipeline/core/detector.py → find_present_elements()")
        timing_info :Dict [str ,float ]={}
        t_total =time .time ()
        t_loading =time .time ()
        img_bgr =cv2 .imread (str (image_path ))
        if img_bgr is None :
            raise FileNotFoundError (f"Не удалось загрузить изображение: {image_path}")
        image_np =cv2 .cvtColor (img_bgr ,cv2 .COLOR_BGR2RGB )
        timing_info ['image_loading']=time .time ()-t_loading
        print ("9️⃣ Шаг 1: _load_example_images() - загрузка positive/negative примеров")
        t_examples =time .time ()
        pos_by_class =self ._load_positive_by_class (positive_dir )
        if len (pos_by_class )==0 :
            print ("   ❌ Нет положительных примеров — прекращаем.")
            return {"found_elements":[],"masks":[]}
        total_pos =sum (len (v )for v in pos_by_class .values ())
        neg_imgs =self ._load_example_images (negative_dir )if negative_dir else []
        timing_info ['examples_loading']=time .time ()-t_examples
        print (f"   📁 Positive: {total_pos} в {len(pos_by_class)} классах	📁 Negative: {len(neg_imgs)}")
        print ("🔟 Шаг 2: MaskGenerator.generate() - генерация масок через SAM/FastSAM")
        t_masks =time .time ()
        masks =self .mask_generator .generate (image_np )
        timing_info ['mask_generation']=time .time ()-t_masks
        print ("1️⃣1️⃣ Шаг 3-7: MaskFilter.apply_all_filters() - все фильтры масок")
        t_filtering =time .time ()
        masks =self .mask_filter .apply_all_filters (masks ,image_np )
        timing_info ['mask_filtering']=time .time ()-t_filtering
        if not masks :
            print ("   ❌ Нет валидных масок после фильтров.")
            return {"found_elements":[],"masks":[]}
        print ("1️⃣2️⃣ Шаг 8: EmbeddingExtractor.extract_mask_embeddings() - эмбеддинги масок")
        t_embeddings =time .time ()
        image_pil =Image .fromarray (image_np .astype (np .uint8 ))

        print (f"   🔍 ДИАГНОСТИКА: Начинаем обработку {len(masks)} масок")
        try :
            mask_vecs =self .embedding_extractor .extract_mask_embeddings (image_pil ,masks )
            print (f"   🔍 ДИАГНОСТИКА: Получено {mask_vecs.shape[0]} валидных векторов из {len(masks)} масок")

            if mask_vecs .shape [0 ]==0 :
                print ("   ❌ Не удалось получить эмбеддинги масок.")
                print ("   📍 ПРИЧИНА: Все маски были отброшены как невалидные (NaN/Inf/нулевая норма)")

                print ("   🔍 ДЕТАЛЬНАЯ ДИАГНОСТИКА МАСОК:")
                for i ,mask in enumerate (masks ):
                    try :
                        print (f"     Маска {i+1}: тип={type(mask)}, размер={getattr(mask, 'shape', 'неизвестно')}")
                        if hasattr (mask ,'segmentation'):
                            seg =mask ['segmentation']
                            if isinstance (seg ,np .ndarray ):
                                print (f"       segmentation: shape={seg.shape}, dtype={seg.dtype}, sum={seg.sum()}")
                            else :
                                print (f"       segmentation: тип={type(seg)} (не numpy array)")
                    except Exception as e :
                        print (f"     Маска {i+1}: ошибка анализа - {e}")

                return {"found_elements":[],"masks":[]}

        except Exception as e :
            import traceback
            print (f"   ❌ КРИТИЧЕСКАЯ ОШИБКА в extract_mask_embeddings: {e}")
            print ("   📍 STACK TRACE:")
            traceback .print_exc ()
            return {"found_elements":[],"masks":[]}
        print (f"   📊 Масок с валидными векторами: {mask_vecs.shape[0]}")
        print ("1️⃣3️⃣ Шаг 9: EmbeddingExtractor.build_queries_multiclass() - эмбеддинги примеров по классам")
        class_pos ,q_neg =self .embedding_extractor .build_queries_multiclass (pos_by_class ,neg_imgs ,pos_as_query_masks =False )
        timing_info ['embedding_extraction']=time .time ()-t_embeddings

        online_negatives =None

        if q_neg is None or q_neg .shape [0 ]==0 :
            print ("   ⚠️ Нет явных негативных примеров, генерируем онлайн-негативы...")

            pos_queries_tensors =[torch .from_numpy (v )for v in class_pos .values ()if v .shape [0 ]>0 ]

            if not pos_queries_tensors :
                print ("   ❌ Нет эмбеддингов для positive-классов, невозможно сгенерировать онлайн-негативы.")
            else :
                all_pos_queries =torch .cat (pos_queries_tensors ,dim =0 )
                if all_pos_queries .shape [0 ]>0 and mask_vecs .shape [0 ]>0 :
                    mask_vecs_torch =torch .from_numpy (mask_vecs )
                    sim_matrix =torch .nn .functional .cosine_similarity (mask_vecs_torch .unsqueeze (1 ),all_pos_queries .unsqueeze (0 ),dim =2 )
                    best_pos_scores ,_ =torch .max (sim_matrix ,dim =1 )
                    num_online_negatives =int (mask_vecs .shape [0 ]*0.4 )
                    if num_online_negatives >0 :
                        k =min (num_online_negatives ,len (best_pos_scores ))
                        if k >0 :
                            _ ,bottom_indices =torch .topk (best_pos_scores ,k =k ,largest =False )
                            online_negatives =mask_vecs [bottom_indices .numpy ()]
                            print (f"   💡 Создано {online_negatives.shape[0]} онлайн-негативов из масок с наихудшими positive-скорами.")
        print ("1️⃣4️⃣ Шаг 10: ScoreCalculator.score_multiclass() - скоринг и принятие решений")
        print ("🔍 ЭТАП 3: Сопоставление с positive/negative по классам...")
        t_scoring =time .time ()
        decisions ,_ =self .score_calculator .score_multiclass (mask_vecs ,class_pos ,q_neg ,online_negatives =online_negatives)
        timing_info ['scoring_and_decisions']=time .time ()-t_scoring
        t_result =time .time ()
        found =[]
        result_masks =[]
        candidates =[]
        H ,W =image_np .shape [:2 ]
        idx_map =list (range (len (masks )))
        print (f"\n🔍 Processing {len(decisions)} decisions...")
        for i ,dec in enumerate (decisions ):
            pos_score =dec .get ('pos',0.0 )
            print (f"  - Decision {i}: accepted={dec.get('accepted')}, class='{dec.get('class')}', pos={pos_score:.3f}")
            if not dec .get ('accepted'):
                print (f"    -> SKIPPED (not accepted)")
                continue
            original_idx =idx_map [i ]
            print (f"    -> ACCEPTED. Original mask index: {original_idx}")
            mask_dict =masks [original_idx ].copy ()
            confidence =float (np .clip (pos_score ,0.0 ,1.0 ))
            mask_dict ['confidence']=confidence
            mask_dict ['class']=dec .get ('class')
            if 'area'not in mask_dict and 'segmentation'in mask_dict :
                mask_dict ['area']=int (np .sum (mask_dict ['segmentation']))
            bx =mask_dict .get ('bbox',[0 ,0 ,0 ,0 ])
            if len (bx )==4 and (bx [2 ]<=W and bx [3 ]<=H ):
                x1 ,y1 ,w ,h =bx
                bbox_xyxy =[int (x1 ),int (y1 ),int (x1 +w ),int (y1 +h )]
            else :
                bbox_xyxy =[int (bx [0 ]),int (bx [1 ]),int (bx [2 ]),int (bx [3 ])]
            cls_label =dec .get ('class')
            try :
                cls_label =str (cls_label )if cls_label is not None else "__unknown__"
            except Exception :
                cls_label ="__unknown__"
            print (f"    -> Appending candidate: class='{cls_label}', confidence={confidence:.3f}")
            candidates .append ({'mask':mask_dict ['segmentation'].astype (bool ),'bbox_xyxy':bbox_xyxy ,'confidence':confidence ,'area':int (mask_dict ['area']),'class':cls_label ,})
        from collections import Counter
        print (f"NMS candidates by class: {Counter([c.get('class') for c in candidates])}")
        print (f"   🔍 DEBUG: Всего кандидатов перед NMS: {len(candidates)}")
        for i ,c in enumerate (candidates ):
            print (f"     Кандидат {i}: class={c.get('class')}, confidence={c.get('confidence'):.3f}, area={c.get('area')}")
        kept =self ._nms (candidates ,class_aware =True )
        print (f"   🔍 DEBUG: Осталось после NMS: {len(kept)}")
        for i ,k in enumerate (kept ):
            print (f"     Оставлен {i}: class={k.get('class')}, confidence={k.get('confidence'):.3f}, area={k.get('area')}")
        for e in kept :
            seg =e ['mask']
            x1 ,y1 ,x2 ,y2 =e ['bbox_xyxy']
            bbox_xywh =[int (x1 ),int (y1 ),int (x2 -x1 ),int (y2 -y1 )]
            mask_dict ={'segmentation':seg ,'bbox':bbox_xywh ,'area':int (seg .sum ()),'confidence':float (e ['confidence']),'class':e .get ('class')}
            found .append ({'mask':mask_dict ,'confidence':float (e ['confidence']),'bbox':bbox_xywh ,'class':e .get ('class')})
            result_masks .append (mask_dict )
        timing_info ['result_formatting']=time .time ()-t_result
        print ("1️⃣5️⃣ Шаг 11: ResultSaver.save_all_results() - сохранение файлов")
        t_saving =time .time ()
        image_name =Path (image_path ).name

        saved_files =self .result_saver .save_all_results (image_np ,result_masks ,output_dir ,image_name ,pipeline_config ={"backend":self .mask_backend })
        timing_info ['result_saving']=time .time ()-t_saving
        total_time =time .time ()-t_total
        timing_info ['total_time']=total_time
        print (f"🎯 Принято масок: {len(found)} (после правил и NMS)")
        print (f"⏱️ Общее время: {total_time:.2f} сек")
        print (f"💾 Результаты сохранены в: {output_dir}")
        print (f"📁 Сохранено файлов: {len(saved_files)}")
        self ._print_timing_statistics (timing_info )
        return {"found_elements":found ,"masks":result_masks ,"timing_info":timing_info ,"output_directory":output_dir ,"saved_files":saved_files}

    def _mask_iou (self ,mask_a ,mask_b ):
        intersection =np .logical_and (mask_a ,mask_b ).sum ()
        union =np .logical_or (mask_a ,mask_b ).sum ()
        return intersection /union if union >0 else 0.0
    def _nms (self ,elements ,class_aware =True ,class_thresholds =None ):
        if not elements :
            print (f"   🔍 NMS DEBUG: Нет элементов для обработки")
            return []
        try :
            import torchvision .ops as ops
            use_torch =True
        except ImportError :
            use_torch =False
        from collections import defaultdict
        def _class_key (v ):
            if v is None :
                return "__unknown__"
            try :
                return str (v )
            except Exception :
                return repr (v )
        groups =defaultdict (list )
        if class_aware :
            for el in elements :
                groups [_class_key (el .get ('class'))].append (el )
        else :
            groups ["__all__"]=list (elements )
        kept_all =[]
        print (f"   🔍 NMS DEBUG: Обрабатываем {len(groups)} групп классов")
        for cls_key ,group in groups .items ():
            if not group :
                print (f"   🔍 NMS DEBUG: Пустая группа для класса '{cls_key}'")
                continue
            iou_thr =(class_thresholds or {}).get (cls_key ,self .nms_iou )
            print (f"   🔍 NMS DEBUG: Класс '{cls_key}': {len(group)} элементов, IoU порог={iou_thr}")

            if use_torch and len (group )>10 :
                print (f"   🔍 NMS DEBUG: Используем torch NMS для класса '{cls_key}'")
                kept_cls =self ._nms_torch (group ,iou_thr )
            else :
                print (f"   🔍 NMS DEBUG: Используем numpy NMS для класса '{cls_key}'")
                kept_cls =self ._nms_numpy (group ,iou_thr )

            print (f"   🔍 NMS DEBUG: Класс '{cls_key}': {len(group)} → {len(kept_cls)} после NMS")
            kept_all .extend (kept_cls )

        return kept_all

    def _nms_torch (self ,elements ,iou_threshold ):

        import torch
        import torchvision .ops as ops

        boxes =[]
        scores =[]
        for el in elements :
            x1 ,y1 ,x2 ,y2 =el ['bbox_xyxy']
            boxes .append ([x1 ,y1 ,x2 ,y2 ])
            scores .append (el .get ('confidence',0.0 ))

        boxes_tensor =torch .tensor (boxes ,dtype =torch .float32 )
        scores_tensor =torch .tensor (scores ,dtype =torch .float32 )

        keep_indices =ops .nms (boxes_tensor ,scores_tensor ,iou_threshold )
        bbox_kept =[elements [i ]for i in keep_indices .tolist ()]

        if len (bbox_kept )<=1 :
            return bbox_kept

        final_kept =[]
        for i ,current in enumerate (bbox_kept ):
            should_keep =True
            current_mask =current ['mask']

            for j in range (i ):
                if j <len (final_kept ):
                    other_mask =final_kept [j ]['mask']
                    if self ._mask_iou (current_mask ,other_mask )>=iou_threshold :
                        should_keep =False
                        break

            if should_keep :
                final_kept .append (current )

        return final_kept

    def _nms_numpy (self ,elements :List [Dict [str ,Any ]],iou_threshold :float )->List [Dict [str ,Any ]]:

        print (f"     🔍 NUMPY NMS: Входных элементов: {len(elements)}, IoU порог: {iou_threshold}")
        if len (elements )<=1 :
            print (f"     🔍 NUMPY NMS: ≤1 элемент, возвращаем как есть")
            return elements

        sorted_elements =sorted (elements ,key =lambda e :float (e .get ('confidence',0.0 )),reverse =True )
        print (f"     🔍 NUMPY NMS: Отсортировано по confidence")

        kept =[]
        masks_kept =[]

        for i ,current in enumerate (sorted_elements ):
            current_mask =current ['mask']
            current_conf =current .get ('confidence',0.0 )
            should_keep =True
            print (f"     🔍 NUMPY NMS: Проверяем элемент {i}: conf={current_conf:.3f}")

            for j ,kept_mask in enumerate (masks_kept ):
                iou =self ._mask_iou (current_mask ,kept_mask )
                print (f"       IoU с элементом {j}: {iou:.3f} (порог: {iou_threshold})")
                if iou >=iou_threshold :
                    should_keep =False
                    print (f"       ❌ Отклонен из-за высокого IoU с элементом {j}")
                    break
            if should_keep :
                print (f"       ✅ Принят элемент {i}")
                kept .append (current )
                masks_kept .append (current_mask )
            else :
                print (f"       ❌ Отклонен элемент {i}")

        print (f"     🔍 NUMPY NMS: Итого принято: {len(kept)} из {len(sorted_elements)}")
        return kept

    def _get_bbox_from_mask (self ,mask :np .ndarray )->List [int ]:

        ys ,xs =np .where (mask )
        if len (ys )==0 :
            return [0 ,0 ,0 ,0 ]

        x_min ,x_max =int (xs .min ()),int (xs .max ())
        y_min ,y_max =int (ys .min ()),int (ys .max ())
        width =x_max -x_min +1
        height =y_max -y_min +1

        return [x_min ,y_min ,width ,height ]

    def _load_example_images (self ,dir_path :Optional [Union [str ,Path ]])->List [Image .Image ]:
        images =[]
        if not dir_path :
            return images
        try :
            DirectoryValidator .validate_input_directory (str (dir_path ))
        except ValidationError as e :
            print (f"   ⚠️ Ошибка валидации директории {dir_path}: {e}")
            return images

        p_dir =Path (dir_path )
        if p_dir .name .startswith ('.'):
            return images

        valid_extensions ={'.jpg','.jpeg','.png','.bmp'}

        for item in p_dir .iterdir ():
            if item .name .startswith ('.'):
                continue

            if item .is_dir ():
                images .extend (self ._load_example_images (item ))
            elif item .is_file ()and item .suffix .lower ()in valid_extensions :
                try :

                    ImageValidator .validate_image_path (str (item ))
                    img =Image .open (item ).convert ('RGB')

                    ImageValidator .validate_image_content (str (item ))
                    images .append (img )
                except ValidationError as e :
                    print (f"   ⚠️ Ошибка валидации изображения {item}: {e}")
                except Exception as e :
                    print (f"   ⚠️ Не удалось загрузить {item}: {e}")
        return images

    def _load_positive_by_class (self ,dir_path :Optional [Union [str ,Path ]])->Dict [str ,List [Image .Image ]]:
        result ={}
        if not dir_path :
            return result
        p_dir =Path (dir_path )
        if not p_dir .exists ()or not p_dir .is_dir ()or p_dir .name .startswith ('.'):
            return result
        subdirs =[p for p in p_dir .iterdir ()if p .is_dir ()and not p .name .startswith ('.')]
        if subdirs :
            print (f"   📂 Режим мульти-класса: подпапки в '{p_dir.name}' считаются классами.")
            for class_dir in subdirs :
                class_name =class_dir .name
                loaded_images =self ._load_example_images (class_dir )
                if loaded_images :
                    result [class_name ]=loaded_images
                    print (f"     -> Класс '{class_name}': найдено {len(loaded_images)} изображений.")
                else :
                    print (f"     -> Класс '{class_name}': 0 изображений.")
        else :
            print (f"   📂 Режим одного класса: все изображения в '{p_dir.name}' будут принадлежать классу '{p_dir.name}'.")
            class_name =p_dir .name
            loaded_images =self ._load_example_images (p_dir )
            if loaded_images :
                result [class_name ]=loaded_images
                print (f"     -> Класс '{class_name}': найдено {len(loaded_images)} изображений.")
            else :
                print (f"     -> Класс '{class_name}': 0 изображений.")
        return result

    def switch_segmentation_backend (self ,backend_type :str ,**kwargs )->None :
        print (f"🔄 Переключение бэкенда сегментации на: {backend_type}")
        if backend_type =='fastsam':
            self .sam_predictor .switch_backend (backend_type ="fastsam",mask_generator =self .mask_generator)
        elif backend_type =='heatmap':
            threshold =kwargs .get ('threshold',0.5 )
            self .sam_predictor .switch_backend (backend_type ="heatmap",threshold =threshold)
        else :
            raise ValueError (f"Неподдерживаемый тип бэкенда: {backend_type}")
        print (f"✅ Бэкенд сегментации переключен на: {backend_type}")
    def get_current_segmentation_backend (self )->str :
        return self .sam_predictor .get_backend_type ()
    def set_heatmap_for_segmentation (self ,heatmap :np .ndarray )->None :
        if self .get_current_segmentation_backend ()=='heatmap':
            self .sam_predictor .set_heatmap (heatmap )
        else :
            print (f"⚠️ Предупреждение: heatmap можно устанавливать только для heatmap бэкенда. "
            f"Текущий бэкенд: {self.get_current_segmentation_backend()}")

    def _extract_hotspots_from_heatmap (self ,heatmap :np .ndarray ,threshold :float =0.7 ,max_hotspots :int =10 )->List [Tuple [int ,int ]]:
        if heatmap .max ()>1.0 :
            heatmap_norm =heatmap /heatmap .max ()
        else :
            heatmap_norm =heatmap .copy ()
        hot_pixels =np .where (heatmap_norm >=threshold )
        if len (hot_pixels [0 ])==0 :
            return []
        from sklearn .cluster import DBSCAN
        coords =np .column_stack ((hot_pixels [1 ],hot_pixels [0 ]))
        if len (coords )<2 :
            return [(int (coords [0 ][0 ]),int (coords [0 ][1 ]))]if len (coords )==1 else []
        clustering =DBSCAN (eps =20 ,min_samples =5 ).fit (coords )
        labels =clustering .labels_
        hotspots =[]
        unique_labels =set (labels )
        for label in unique_labels :
            if label ==-1 :
                continue
            cluster_coords =coords [labels ==label ]
            center_x =int (np .mean (cluster_coords [:,0 ]))
            center_y =int (np .mean (cluster_coords [:,1 ]))
            hotspots .append ((center_x ,center_y ))
        if len (hotspots )>max_hotspots :
            hotspots_with_intensity =[]
            for x ,y in hotspots :
                if 0 <=y <heatmap_norm .shape [0 ]and 0 <=x <heatmap_norm .shape [1 ]:
                    intensity =heatmap_norm [y ,x ]
                    hotspots_with_intensity .append (((x ,y ),intensity ))
            hotspots_with_intensity .sort (key =lambda x :x [1 ],reverse =True )
            hotspots =[coord for coord ,_ in hotspots_with_intensity [:max_hotspots ]]
        return hotspots
    def _refine_hotspots_with_sam (self ,image_np :np .ndarray ,hotspots :List [Tuple [int ,int ]])->List [Dict [str ,Any ]]:
        if not hotspots :
            return []
        try :
            original_backend =self .get_current_segmentation_backend ()
            self .switch_segmentation_backend ('fastsam')
            self .sam_predictor .set_image (image_np )
            refined_masks =[]
            for x ,y in hotspots :
                point_coords =np .array ([[x ,y ]])
                point_labels =np .array ([1 ])
                masks ,scores ,_ =self .sam_predictor .predict (point_coords =point_coords ,point_labels =point_labels ,multimask_output =False)
                if len (masks )>0 and scores [0 ]>0.5 :
                    mask =masks [0 ]
                    mask_dict ={'segmentation':mask .astype (bool ),'area':int (np .sum (mask )),'bbox':self ._mask_to_bbox (mask ),'predicted_iou':float (scores [0 ]),'point_coords':[x ,y ],'stability_score':float (scores [0 ])}
                    refined_masks .append (mask_dict )
            self .switch_segmentation_backend (original_backend )
            return refined_masks
        except Exception as e :
            print (f"   ⚠️ Ошибка при уточнении горячих зон через SAM: {e}")
            return []
    def _extract_hot_regions_from_heatmap (self ,heatmap :np .ndarray ,threshold :float =0.5 ,min_area :int =1000 )->List [Dict [str ,Any ]]:

        try :

            if heatmap .max ()>1.0 :
                heatmap_norm =heatmap /heatmap .max ()
            else :
                if hasattr (heatmap ,'clone'):
                    heatmap_norm =heatmap .clone ().detach ().cpu ().numpy ()if hasattr (heatmap ,'cpu')else heatmap .clone ()
                else :
                    heatmap_norm =heatmap .copy ()

            if hasattr (heatmap_norm ,'detach'):
                heatmap_norm =heatmap_norm .detach ().cpu ().numpy ()
            elif not isinstance (heatmap_norm ,np .ndarray ):
                heatmap_norm =np .array (heatmap_norm )

            hot_mask =(heatmap_norm >=threshold ).astype (np .uint8 )

            kernel =cv2 .getStructuringElement (cv2 .MORPH_ELLIPSE ,(5 ,5 ))
            hot_mask =cv2 .morphologyEx (hot_mask ,cv2 .MORPH_CLOSE ,kernel )
            hot_mask =cv2 .morphologyEx (hot_mask ,cv2 .MORPH_OPEN ,kernel )

            contours ,_ =cv2 .findContours (hot_mask ,cv2 .RETR_EXTERNAL ,cv2 .CHAIN_APPROX_SIMPLE )

            hot_regions =[]
            for contour in contours :
                area =cv2 .contourArea (contour )
                if area >=min_area :

                    x ,y ,w ,h =cv2 .boundingRect (contour )

                    region_mask =np .zeros_like (hot_mask )
                    cv2 .fillPoly (region_mask ,[contour ],1 )

                    hot_regions .append ({
                    'bbox':[x ,y ,w ,h ],
                    'mask':region_mask .astype (bool ),
                    'area':int (area ),
                    'contour':contour
                    })

            print (f"   🔥 Найдено {len(hot_regions)} горячих областей с площадью >= {min_area}")
            return hot_regions

        except Exception as e :
            print (f"   ⚠️ Ошибка при извлечении горячих областей: {e}")
            return []

    def _refine_masks_with_fastsam (self ,heatmap_masks :List [Dict ],image_np :np .ndarray ,heatmap :np .ndarray =None )->List [Dict ]:

        try :
            if not heatmap_masks :
                return []

            if heatmap is not None :
                return self ._generate_fastsam_in_hot_regions (heatmap_masks ,image_np ,heatmap )

            return self ._generate_fastsam_full_image (heatmap_masks ,image_np )

        except Exception as e :
            print (f"   ⚠️ Ошибка при уточнении масок через FastSAM: {e}")
            return heatmap_masks

    def _generate_fastsam_in_hot_regions (self ,heatmap_masks :List [Dict ],image_np :np .ndarray ,heatmap :np .ndarray )->List [Dict ]:

        if not heatmap_masks :
            print ("   ⚠️ Нет масок для использования как горячие области")
            return []

        print (f"   🔥 Используем {len(heatmap_masks)} отфильтрованных масок как горячие области")

        all_fastsam_masks =[]

        for i ,mask_dict in enumerate (heatmap_masks ):

            if 'bbox'in mask_dict :
                x ,y ,w ,h =mask_dict ['bbox']
            else :

                segmentation =mask_dict .get ('segmentation',None )
                if segmentation is None :
                    continue
                ys ,xs =np .where (segmentation )
                if len (xs )==0 or len (ys )==0 :
                    continue
                x ,y =int (xs .min ()),int (ys .min ())
                w ,h =int (xs .max ()-xs .min ()),int (ys .max ()-ys .min ())

            padding =getattr (self .config ,'fastsam_crop_padding',20 )
            x_pad =max (0 ,x -padding )
            y_pad =max (0 ,y -padding )
            w_pad =min (image_np .shape [1 ]-x_pad ,w +2 *padding )
            h_pad =min (image_np .shape [0 ]-y_pad ,h +2 *padding )

            crop_image =image_np [y_pad :y_pad +h_pad ,x_pad :x_pad +w_pad ]

            if crop_image .size ==0 :
                continue

            print (f"   🎯 Генерация FastSAM для маски {i+1}/{len(heatmap_masks)} [{x_pad}:{x_pad+w_pad}, {y_pad}:{y_pad+h_pad}]")

            crop_masks =self .mask_generator .generate (crop_image )

            for mask in crop_masks :
                if 'segmentation'in mask :

                    full_mask =np .zeros ((image_np .shape [0 ],image_np .shape [1 ]),dtype =bool )
                    crop_seg =mask ['segmentation']

                    crop_h ,crop_w =crop_seg .shape
                    if y_pad +crop_h <=image_np .shape [0 ]and x_pad +crop_w <=image_np .shape [1 ]:
                        full_mask [y_pad :y_pad +crop_h ,x_pad :x_pad +crop_w ]=crop_seg
                        mask ['segmentation']=full_mask

                        if 'bbox'in mask :
                            bbox =mask ['bbox']
                            mask ['bbox']=[bbox [0 ]+x_pad ,bbox [1 ]+y_pad ,bbox [2 ],bbox [3 ]]

                        mask ['hot_region_origin']=True
                        mask ['hot_region_id']=i

                        all_fastsam_masks .append (mask )

        print (f"   ✅ Сгенерировано {len(all_fastsam_masks)} масок FastSAM в {len(heatmap_masks)} областях масок")

        return self ._match_heatmap_with_fastsam_masks (heatmap_masks ,all_fastsam_masks )

    def _generate_fastsam_full_image (self ,heatmap_masks :List [Dict ],image_np :np .ndarray )->List [Dict ]:

        fastsam_masks =self .mask_generator .generate (image_np )

        if fastsam_masks is None :
            fastsam_masks =[]
            print ("   ⚠️ FastSAM вернул None, используем пустой список")

        if not fastsam_masks :
            print ("   ⚠️ FastSAM не сгенерировал маски")
            return heatmap_masks

        print (f"   ✅ Сгенерировано {len(fastsam_masks)} масок FastSAM для всего изображения")

        return self ._match_heatmap_with_fastsam_masks (heatmap_masks ,fastsam_masks )

    def _match_heatmap_with_fastsam_masks (self ,heatmap_masks :List [Dict ],fastsam_masks :List [Dict ])->List [Dict ]:

        refined_masks =[]

        for hm_mask in heatmap_masks :
            hm_segmentation =hm_mask .get ('segmentation')
            if hm_segmentation is None :
                continue

            best_match =None
            best_iou =0.0

            for fs_mask in fastsam_masks :
                fs_segmentation =fs_mask .get ('segmentation')
                if fs_segmentation is None :
                    continue

                intersection =np .logical_and (hm_segmentation ,fs_segmentation )
                union =np .logical_or (hm_segmentation ,fs_segmentation )

                if np .sum (union )>0 :
                    iou =np .sum (intersection )/np .sum (union )

                    iou_threshold =getattr (self .config ,'fastsam_refinement_iou_threshold',0.3 )
                    if iou >best_iou and iou >iou_threshold :
                        best_iou =iou
                        best_match =fs_mask

            if best_match is not None :
                refined_mask =best_match .copy ()

                refined_mask ['heatmap_origin']=True
                refined_mask ['iou_with_heatmap']=best_iou
                refined_masks .append (refined_mask )
            else :

                hm_mask ['heatmap_origin']=True
                hm_mask ['iou_with_heatmap']=0.0
                refined_masks .append (hm_mask )

        print (f"   ✅ Уточнено {len(refined_masks)} масок через сопоставление с FastSAM")
        return refined_masks

    def _mask_to_bbox (self ,mask :np .ndarray )->List [int ]:

        rows ,cols =np .where (mask )
        if len (rows )==0 :
            return [0 ,0 ,0 ,0 ]

        y1 ,y2 =rows .min (),rows .max ()
        x1 ,x2 =cols .min (),cols .max ()

        return [int (x1 ),int (y1 ),int (x2 -x1 +1 ),int (y2 -y1 +1 )]

    def _print_timing_statistics (self ,timing_info :Dict [str ,float ])->None :

        print ("\n"+"="*60 )
        print ("⏱️ ДЕТАЛЬНАЯ СТАТИСТИКА ВРЕМЕНИ ВЫПОЛНЕНИЯ:")
        print ("="*60 )
        total_time =timing_info ['total_time']
        stages =[
        ('image_loading','📁 Загрузка изображения'),
        ('examples_loading','🖼️ Загрузка примеров'),
        ('heatmap_generation','🔥 Генерация heatmap'),
        ('hybrid_sam_refinement','🔄 Гибридная обработка SAM'),
        ('mask_generation','🎯 Генерация масок (SAM/FastSAM)'),
        ('mask_filtering','🔍 Фильтрация масок'),
        ('fastsam_processing','⚡ Обработка FastSAM'),
        ('fastsam_integration','⚡ Интеграция FastSAM'),
        ('embedding_extraction','🧠 Извлечение эмбеддингов'),
        ('scoring_and_decisions','📊 Скоринг и решения'),
        ('result_formatting','📋 Формирование результата'),
        ('result_saving','💾 Сохранение файлов')
        ]

        for stage_key ,stage_name in stages :
            if stage_key in timing_info :
                stage_time =timing_info [stage_key ]
                percentage =(stage_time /total_time *100 )if total_time >0 else 0
                print (f"{stage_name:<40}: {stage_time:>6.3f}с ({percentage:>5.1f}%)")
        print ("-"*60 )
        print (f"{'🚀 ОБЩЕЕ ВРЕМЯ':<40}: {total_time:>6.3f}с (100.0%)")
        print ("="*60 )
        stage_times =[(name ,timing_info .get (key ,0 ))for key ,name in stages if key in timing_info ]
        stage_times .sort (key =lambda x :x [1 ],reverse =True )
        if len (stage_times )>1 :
            print ("\n🐌 САМЫЕ МЕДЛЕННЫЕ ЭТАПЫ:")
            for i ,(name ,stage_time )in enumerate (stage_times [:3 ]):
                percentage =(stage_time /total_time *100 )if total_time >0 else 0
                print (f"   {i+1}. {name}: {stage_time:.3f}с ({percentage:.1f}%)")
            print ()  # Добавляем пустую строку после списка медленных этапов
        if 'mask_generation'in timing_info and timing_info ['mask_generation']>total_time *0.5 :
            print ("\n💡 РЕКОМЕНДАЦИИ ПО ОПТИМИЗАЦИИ:")
            print ("   • Генерация масок занимает >50% времени")
            print ("   • Попробуйте FastSAM вместо SAM-HQ для ускорения")
            print ("   • Или уменьшите параметры SAM (points_per_side, imgsz)")
        if 'embedding_extraction'in timing_info and timing_info ['embedding_extraction']>total_time *0.3 :
            print ("\n💡 РЕКОМЕНДАЦИИ ПО ОПТИМИЗАЦИИ:")
            print ("   • Извлечение эмбеддингов занимает >30% времени")
            print ("   • Проверьте размер feature map (SEARCHDET_FEAT_SHORT_SIDE)")
            print ("   • Убедитесь что используется быстрый метод извлечения")
        print ()