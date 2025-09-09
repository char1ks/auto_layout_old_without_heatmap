
from dataclasses import dataclass ,field
from typing import List ,Dict ,Any ,Optional ,Tuple ,Union
from pathlib import Path
import numpy as np
from enum import Enum
from ..utils .validation import ModelValidator ,ValidationError

class MaskBackend (Enum ):

    FASTSAM ="fastsam"
    SAM ="sam"
    SAM2 ="sam2"
    SAM_HQ ="sam_hq"

class BackboneType (Enum ):

    RESNET101 ="resnet101"
    DINOV2_S ="dinov2_s"
    DINOV2_B ="dinov2_b"
    DINOV2_L ="dinov2_l"
    DINOV2_G ="dinov2_g"
    DINOV3_VITB16 ="dinov3_vitb16"
    DINOV3_VITL16 ="dinov3_vitl16"
    DINOV3_VITH14 ="dinov3_vith14"
    DINOV3_CONVNEXT_TINY ="dinov3_convnext_tiny"
    DINOV3_CONVNEXT_SMALL ="dinov3_convnext_small"
    DINOV3_CONVNEXT_BASE ="dinov3_convnext_base"
    DINOV3_CONVNEXT_LARGE ="dinov3_convnext_large"

@dataclass
class BoundingBox :

    x :int
    y :int
    width :int
    height :int

    @property
    def x2 (self )->int :

        return self .x +self .width

    @property
    def y2 (self )->int :

        return self .y +self .height

    @property
    def area (self )->int :

        return self .width *self .height

    @property
    def center (self )->Tuple [float ,float ]:

        return (self .x +self .width /2 ,self .y +self .height /2 )

@dataclass
class MaskData :

    mask :np .ndarray
    bbox :BoundingBox
    area :int
    confidence :float =0.0
    stability_score :Optional [float ]=None
    predicted_iou :Optional [float ]=None

    def __post_init__ (self ):

        if self .mask .dtype !=bool and self .mask .dtype !=np .uint8 :
            raise ValueError (f"Маска должна быть bool или uint8, получен {self.mask.dtype}")
        if self .area <=0 :
            raise ValueError (f"Площадь маски должна быть положительной, получена {self.area}")
        if not 0.0 <=self .confidence <=1.0 :
            raise ValueError (f"Уверенность должна быть в диапазоне [0, 1], получена {self.confidence}")

@dataclass
class EmbeddingData :

    embedding :np .ndarray
    mask_id :int
    source_type :str
    source_path :Optional [str ]=None

    def __post_init__ (self ):

        if self .embedding .ndim !=1 :
            raise ValueError (f"Эмбеддинг должен быть 1D массивом, получен {self.embedding.ndim}D")
        if self .source_type not in ['positive','negative','target']:
            raise ValueError (f"Неизвестный тип источника: {self.source_type}")

@dataclass
class DetectionResult :

    mask :np .ndarray
    bbox :BoundingBox
    confidence :float
    score :float
    positive_scores :List [float ]=field (default_factory =list )
    negative_scores :List [float ]=field (default_factory =list )
    embedding :Optional [np .ndarray ]=None
    metadata :Dict [str ,Any ]=field (default_factory =dict )

    def __post_init__ (self ):

        if not 0.0 <=self .confidence <=1.0 :
            raise ValueError (f"Уверенность должна быть в диапазоне [0, 1], получена {self.confidence}")

@dataclass
class ProcessingResult :

    success :bool
    detections :List [DetectionResult ]=field (default_factory =list )
    processing_time :float =0.0
    error :Optional [str ]=None
    timing_info :Dict [str ,float ]=field (default_factory =dict )
    saved_files :Dict [str ,str ]=field (default_factory =dict )
    output_directory :Optional [str ]=None

    @property
    def num_detections (self )->int :

        return len (self .detections )

    @property
    def average_confidence (self )->float :

        if not self .detections :
            return 0.0
        return sum (d .confidence for d in self .detections )/len (self .detections )

@dataclass
class DetectorConfig :

    mask_backend :MaskBackend =MaskBackend .FASTSAM
    backbone :BackboneType =BackboneType .DINOV2_B
    device :str ="auto"
    half_precision :bool =False

    sam_model :Optional [str ]=None
    sam_encoder :str ="vit_l"
    fastsam_model :Optional [str ]=None
    fastsam_device :Optional [str ]=None
    dinov3_ckpt :Optional [str ]=None
    loader :str ="timm"
    repo_dir :Optional [str ]=None
    overlay_alpha :float =0.5

    segmentation_backend :str ="sam"
    heatmap_threshold :float =0.55

    heatmap_min_area :int =200
    heatmap_min_solidity :float =0.5
    heatmap_extent_min :float =0.05
    heatmap_extent_max :float =1.0
    heatmap_max_masks :int =5

    use_heatmap_sam_hybrid :bool =False
    heatmap_sam_threshold :float =0.7
    sam_refinement_enabled :bool =True
    max_hotspots_for_sam :int =10
    use_hotspots :bool =True
    point_prompts :bool =True

    use_hotspot_fastsam :bool =True
    max_hotspots_for_fastsam :int =8
    hotspot_threshold :float =0.7
    fastsam_heatmap_overlap_threshold :float =0.8
    prefer_fastsam_on_overlap :bool =True
    fallback_to_heatmap :bool =True
    skip_scoring_for_hotspot_masks :bool =True

    fastsam_refinement_iou_threshold :float =0.3
    min_overlap_ratio :float =0.8

    sam_model_path :Optional [str ]=None
    sam_encoder_path :Optional [str ]=None
    sam2_weights_path :Optional [str ]=None
    fastsam_model_path :Optional [str ]=None
    dinov3_checkpoint_path :Optional [str ]=None

    min_confidence :float =0.5
    max_masks :int =1000
    min_area_fraction :float =0.001
    max_area_fraction :float =0.9
    containment_iou :float =0.8

    score_margin :float =0.1
    score_ratio :float =1.5
    score_confidence :float =0.7
    min_positive_score :float =0.3
    decision_threshold :float =0.5
    adaptive_ratio :float =0.8
    adaptive_diff_floor :float =0.05
    topk :int =5
    positive_aggregation :str ="mean"

    dinov3_backbone :Optional [str ]=None
    vit_pooling :str ="cls"
    layer :Optional [int ]=None
    feature_short_side :Optional [int ]=None
    dino_half_precision :bool =False

    consensus_k :Optional [int ]=None
    consensus_threshold :Optional [float ]=None
    nms_iou :float =0.5

    fastsam_image_size :int =1024
    fastsam_confidence :float =0.4
    fastsam_iou :float =0.9
    fastsam_retina :bool =True
    fastsam_points_per_side :int =32

    sam_long_side :int =1024

    enable_image_downscaling :bool =True
    max_image_size :int =512
    downscale_quality :str ="bilinear"

    ban_border_masks :bool =False
    border_width :int =10

    smart_rectangle_filter :bool =True
    rectangle_bbox_iou_threshold :float =0.95
    rectangle_straight_line_ratio :float =0.8
    rectangle_area_ratio_threshold :float =0.95
    rectangle_angle_tolerance :float =10.0
    rectangle_side_ratio_threshold :float =0.9
    perfect_rectangle_iou_threshold :float =0.99
    rectangle_similarity_iou_threshold :float =0.94
    square_similarity_iou_threshold :float =0.94
    rectangle_use_silhouette :bool =True
    hole_area_ratio_threshold :float =0.03

    defect_mode :bool =False
    positive_as_query_masks :bool =True
    min_mask_area :int =100
    max_embedding_size :Optional [int ]=None

    @property
    def feat_short_side (self )->Optional [int ]:
        return self .feature_short_side

    @feat_short_side .setter
    def feat_short_side (self ,value :Optional [int ])->None :
        self .feature_short_side =value

    @classmethod
    def from_dict (cls ,config_dict :Dict [str ,Any ])->'DetectorConfig':

        data =dict (config_dict or {})

        if 'feature_short_side'not in data and 'feat_short_side'in data :
            data ['feature_short_side']=data ['feat_short_side']

        valid_fields ={field_name for field_name in cls .__dataclass_fields__ }
        filtered ={k :v for k ,v in data .items ()if k in valid_fields }
        return cls (**filtered )

    def to_dict (self )->Dict [str ,Any ]:

        return {name :getattr (self ,name )for name in self .__dataclass_fields__ }

    def update (self ,**kwargs )->'DetectorConfig':

        base =self .to_dict ()
        base .update (kwargs or {})
        return self .from_dict (base )

@dataclass
class BatchProcessingConfig :

    input_directory :Path
    output_directory :Path
    positive_examples_directory :Optional [Path ]=None
    negative_examples_directory :Optional [Path ]=None
    file_extensions :List [str ]=field (default_factory =lambda :['.jpg','.jpeg','.png','.bmp','.tiff'])
    recursive :bool =False
    max_workers :int =1
    save_intermediate :bool =True
    overwrite_existing :bool =False

    def __post_init__ (self ):

        if not self .input_directory .exists ():
            raise ValueError (f"Входная директория не существует: {self.input_directory}")
        if self .max_workers <1 :
            raise ValueError (f"max_workers должен быть >= 1, получен {self.max_workers}")

@dataclass
class LegacyDetectorConfig :

    device :str ="auto"
    mask_backend :str ="sam-hq"

    sam_model :Optional [str ]=None
    sam_encoder :str ="vit_l"
    fastsam_model :Optional [str ]=None
    fastsam_device :Optional [str ]=None
    dinov3_backbone :str ="vitb16"
    dinov3_ckpt :Optional [str ]=None

    sam_long_side :Optional [int ]=None
    fastsam_imgsz :int =1024
    fastsam_conf :float =0.4
    fastsam_iou :float =0.9
    fastsam_retina :bool =True

    perfect_rectangle_iou_threshold :float =0.99
    border_width :int =2
    ban_border_masks :bool =True

    min_pos_score :float =0.62
    decision_threshold :float =0.06
    class_separation :float =0.04
    neg_cap :float =0.90
    topk :int =5
    consensus_k :int =0
    consensus_thr :float =0.45
    adaptive_ratio :float =0.85
    adaptive_diff_floor :float =0.04
    adaptive_trigger_pos_range :float =0.20
    adaptive_trigger_neg_range :float =0.20
    score_margin :float =0.00
    score_ratio :float =1.01
    score_confidence :float =0.60
    allow_unknown :bool =True
    verbose :bool =True

    overlay_alpha :float =0.5

    @classmethod
    def from_dict (cls ,config_dict :Dict [str ,Any ])->'LegacyDetectorConfig':
        valid_fields ={field .name for field in cls .__dataclass_fields__ .values ()}
        filtered_dict ={k :v for k ,v in (config_dict or {}).items ()if k in valid_fields }
        return cls (**filtered_dict )

    def to_dict (self )->Dict [str ,Any ]:
        return {field .name :getattr (self ,field .name )for field in self .__dataclass_fields__ .values ()}

    def update (self ,**kwargs )->'LegacyDetectorConfig':
        config_dict =self .to_dict ()
        config_dict .update (kwargs )
        return self .from_dict (config_dict )

@dataclass
class TimingInfo :

    total_time :float =0.0
    mask_generation :float =0.0
    embedding_extraction :float =0.0
    scoring_and_decisions :float =0.0
    result_saving :float =0.0
    image_loading :float =0.0
    preprocessing :float =0.0
    postprocessing :float =0.0

    def to_dict (self )->Dict [str ,float ]:

        return {
        'total_time':self .total_time ,
        'mask_generation':self .mask_generation ,
        'embedding_extraction':self .embedding_extraction ,
        'scoring_and_decisions':self .scoring_and_decisions ,
        'result_saving':self .result_saving ,
        'image_loading':self .image_loading ,
        'preprocessing':self .preprocessing ,
        'postprocessing':self .postprocessing
        }
