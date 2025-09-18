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
from .fastsam_integration import FastSAMHeatmapProcessor
from .models import DetectorConfig ,ProcessingResult ,MaskData ,DetectionResult
from ..utils .validation import ImageValidator ,DirectoryValidator ,ValidationError ,validate_processing_pipeline_inputs
from ..eval_classes.detector_base import DetectorBase 
from ..eval_classes.Context import Context
import torch
from .models import MaskBackend ,BackboneType

class SearchDetDetector(DetectorBase):
    def __init__ (self ,config :Optional [DetectorConfig ]=None ,**kwargs :Any )->None :
        super().__init__(name="SearchDetDetector")
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
        self .fastsam_processor =FastSAMHeatmapProcessor (heatmap_generator =self .heatmap_generator ,fastsam_model =fastsam_model_instance ,embedding_extractor =self .embedding_extractor ,score_calculator =self .score_calculator)
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

    def read_reference_images(
        self,
        positive_dir: Union[str, Path],
        negative_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[Dict[str, List[Image.Image]], List[Image.Image]]:
        pos_by_class = self._load_positive_by_class(positive_dir)
        neg_imgs = []
        if negative_dir:
            neg_imgs = self._load_example_images(negative_dir)
            
        return pos_by_class, neg_imgs

    def set_references(
        self,
        pos_by_class: Dict[str, List[Image.Image]],
        neg_imgs: List[Image.Image],
    ) -> None:
        self.pos_by_class = pos_by_class
        self.neg_imgs = neg_imgs
    def find_present_elements(self, image_np: np.ndarray, context: Context, *args, **kwargs) -> Dict[str, Any]:
        if not hasattr(self, 'pos_by_class') or not self.pos_by_class:
            raise ValueError("References not set. Call set_references() first.")
        
        timing_info: Dict[str, float] = {}
        t_total = time.time()
        image_pil = Image.fromarray(image_np.astype(np.uint8))
        pos_by_class = self.pos_by_class
        neg_imgs = getattr(self, 'neg_imgs', [])
        
        if len(pos_by_class) == 0:
            return {"found_elements": [], "masks": []}
        t_heatmap = time.time()
        all_positive_images = []
        for class_images in pos_by_class.values():
            all_positive_images.extend(class_images)
        heatmap = self.heatmap_generator.generate_heatmap(
            input_image=image_pil, 
            positive_images=all_positive_images, 
            negative_images=neg_imgs
        )
        timing_info['heatmap_generation'] = time.time() - t_heatmap
        t_embeddings = time.time()
        class_pos, q_neg = self.embedding_extractor.build_queries_multiclass(
            pos_by_class, neg_imgs, pos_as_query_masks=False
        )
        timing_info['class_embeddings'] = time.time() - t_embeddings
        t_masks = time.time()
        masks = self.mask_generator.generate(image_np)
        timing_info['mask_generation'] = time.time() - t_masks
        t_filtering = time.time()
        masks = self.mask_filter.apply_all_filters(masks, image_np)
        timing_info['mask_filtering'] = time.time() - t_filtering
        
        if not masks:
            return {"found_elements": [], "masks": [], "timing_info": timing_info, "heatmap": heatmap}
        t_embeddings = time.time()
        try:
            mask_vecs = self.embedding_extractor.extract_mask_embeddings(image_pil, masks)
            if mask_vecs.shape[0] == 0:
                for i, mask in enumerate(masks):
                    try:
                        print(f"     Маска {i+1}: тип={type(mask)}, размер={getattr(mask, 'shape', 'неизвестно')}")
                        if hasattr(mask, 'segmentation'):
                            seg = mask['segmentation']
                    except Exception as e:
                        print(f"     Маска {i+1}: ошибка анализа - {e}")
                return {"found_elements": [], "masks": [], "timing_info": timing_info, "heatmap": heatmap}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"found_elements": [], "masks": [], "timing_info": timing_info, "heatmap": heatmap}
        timing_info['embedding_extraction'] = time.time() - t_embeddings
        online_negatives = None
        if q_neg is None or q_neg.shape[0] == 0:
            pos_queries_tensors = [torch.from_numpy(v) for v in class_pos.values() if v.shape[0] > 0]
            if not pos_queries_tensors:
                print("   ❌ Нет эмбеддингов для positive-классов, невозможно сгенерировать онлайн-негативы.")
            else:
                all_pos_queries = torch.cat(pos_queries_tensors, dim=0)
                if all_pos_queries.shape[0] > 0 and mask_vecs.shape[0] > 0:
                    mask_vecs_torch = torch.from_numpy(mask_vecs)
                    sim_matrix = torch.nn.functional.cosine_similarity(
                        mask_vecs_torch.unsqueeze(1), all_pos_queries.unsqueeze(0), dim=2
                    )
                    best_pos_scores, _ = torch.max(sim_matrix, dim=1)
                    num_online_negatives = int(mask_vecs.shape[0] * 0.4)
                    if num_online_negatives > 0:
                        k = min(num_online_negatives, len(best_pos_scores))
                        if k > 0:
                            _, bottom_indices = torch.topk(best_pos_scores, k=k, largest=False)
                            online_negatives = mask_vecs[bottom_indices.numpy()]
        t_scoring = time.time()
        decisions, _ = self.score_calculator.score_multiclass(
            mask_vecs, class_pos, q_neg, online_negatives=online_negatives
        )
        timing_info['scoring_and_decisions'] = time.time() - t_scoring
        t_result =time .time ()
        found =[]
        result_masks =[]
        candidates =[]
        H ,W =image_np .shape [:2 ]
        idx_map =list (range (len (masks )))
        
        for i ,dec in enumerate (decisions ):
            pos_score =dec.get ('pos',0.0 )
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
            candidates .append ({
                'mask':mask_dict ['segmentation'].astype (bool ),
                'bbox_xyxy':bbox_xyxy ,
                'confidence':confidence ,
                'area':int (mask_dict ['area']),
                'class':cls_label ,
            })
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
            mask_dict ={
                'segmentation':seg ,
                'bbox':bbox_xywh ,
                'area':int (seg .sum ()),
                'confidence':float (e ['confidence']),
                'class':e .get ('class')
            }
            found .append ({
                'mask':mask_dict ,
                'confidence':float (e ['confidence']),
                'bbox':bbox_xywh ,
                'class':e .get ('class')
            })
            result_masks .append (mask_dict )
        timing_info ['result_formatting']=time .time ()-t_result
        total_time = time.time() - t_total
        timing_info['total_time'] = total_time
        self._print_timing_statistics(timing_info)
        return {
            "found_elements": found,
            "masks": result_masks,
            "timing_info": timing_info,
            "heatmap": heatmap
        }

    def _convert_to_coco_annotations(self, detection_result: Dict[str, Any], context) -> List:
        from ..eval_classes.COCOAnnotations import COCOAnnotation
        annotations = []
        found_elements = detection_result.get('found_elements', [])
    
        image_np = context.extra['original_image']
        height, width = image_np.shape[:2]
        
        # Prefer file_name provided via Dataset_Point -> DetectorBase.detect(context.extra['file_name'])
        file_name = None
        try:
            if hasattr(context, 'extra') and isinstance(context.extra, dict):
                file_name = context.extra.get('file_name') or None
        except Exception:
            file_name = None
        if not file_name:
            if hasattr(context, 'image_path') and context.image_path:
                file_name = os.path.basename(str(context.image_path))
            else:
                file_name = 'unknown.jpg'
        
        for element in found_elements:
            mask_data = element.get('mask', {})
            
            segmentation = mask_data.get('segmentation')
            if segmentation is None:
                continue
                
            if isinstance(segmentation, np.ndarray):
                mask = segmentation.astype(bool)
            else:
                mask = np.array(segmentation, dtype=bool)
            
            bbox = element.get('bbox', mask_data.get('bbox', [0, 0, 0, 0]))
            if len(bbox) != 4:
                if mask.any():
                    y_indices, x_indices = np.where(mask)
                    x_min, x_max = x_indices.min(), x_indices.max()
                    y_min, y_max = y_indices.min(), y_indices.max()
                    bbox = [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]
                else:
                    bbox = [0, 0, 0, 0]
            
            area = element.get('area', mask_data.get('area', int(mask.sum())))
            label = element.get('class', mask_data.get('class', 'unknown'))
            annotation = COCOAnnotation(
                img=image_np,
                mask=mask,
                label=label,
                image_size=(width, height),
                width=width,
                height=height,
                area=float(area),
                file_name=file_name,
                bbox=[float(x) for x in bbox]
            )
            
            annotations.append(annotation)
        
        return annotations


    def set_references_from_dirs(self, image_path: Union[str, Path], positive_dir: Union[str, Path],
    negative_dir: Optional[Union[str, Path]] = None) -> None:
        try :
            ImageValidator .validate_image_path (str (image_path ))
            DirectoryValidator .validate_input_directory (str (positive_dir ))
            if negative_dir :
                DirectoryValidator .validate_input_directory (str (negative_dir ))
        except ValidationError as e :
            raise ValidationError (f"Ошибка валидации референсных данных: {e}")

        raise NotImplementedError ("Метод set_references еще не реализован")
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
            for class_dir in subdirs :
                class_name =class_dir .name
                loaded_images =self ._load_example_images (class_dir )
                if loaded_images :
                    result [class_name ]=loaded_images
                else :
                    pass
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

    def get_class_from_positive_query(self, positive_dir: Union[str, Path]) -> Optional[str]:
        if not positive_dir:
            return None
            
        p_dir = Path(positive_dir)
        if not p_dir.exists() or not p_dir.is_dir() or p_dir.name.startswith('.'):
            return None
        subdirs = [p for p in p_dir.iterdir() if p.is_dir() and not p.name.startswith('.')]
        
        if subdirs:
            for class_dir in subdirs:
                class_name = class_dir.name
                loaded_images = self._load_example_images(class_dir)
                if loaded_images:
                    print(f"   🎯 Определен класс из positive query: '{class_name}'")
                    return class_name
            return None
        else:
            # Одно-класс режим: название папки = название класса
            class_name = p_dir.name
            loaded_images = self._load_example_images(p_dir)
            if loaded_images:
                print(f"   🎯 Определен класс из positive query: '{class_name}'")
                return class_name
            return None

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
            print ()
