
import torch
import numpy as np
from typing import Optional ,Tuple ,List ,Dict ,Any
from abc import ABC ,abstractmethod

class BaseSegmentationBackend (ABC ):

    @abstractmethod
    def set_image (self ,image :np .ndarray )->None :

        pass

    @abstractmethod
    def predict (self ,**kwargs )->Tuple [np .ndarray ,np .ndarray ,np .ndarray ]:

        pass

    @abstractmethod
    def reset_image (self )->None :

        pass

class SAMBackend (BaseSegmentationBackend ):

    def __init__ (self ,sam_model ):
        self .sam_model =sam_model
        self .device =torch .device ('cuda'if torch .cuda .is_available ()else 'cpu')

    def set_image (self ,image :np .ndarray )->None :
        if self .sam_model is not None :
            self .sam_model .set_image (image )

    def predict (self ,point_coords =None ,point_labels =None ,box =None ,
    mask_input =None ,multimask_output =True ,return_logits =False ,**kwargs ):
        if self .sam_model is None :
            return np .array ([]),np .array ([]),np .array ([])

        return self .sam_model .predict (
        point_coords =point_coords ,
        point_labels =point_labels ,
        box =box ,
        mask_input =mask_input ,
        multimask_output =multimask_output ,
        return_logits =return_logits
        )

    def reset_image (self )->None :
        if hasattr (self .sam_model ,'reset_image'):
            self .sam_model .reset_image ()

class FastSAMBackend (BaseSegmentationBackend ):

    def __init__ (self ,mask_generator ):
        self .mask_generator =mask_generator
        self .current_image =None

    def set_image (self ,image :np .ndarray )->None :
        self .current_image =image

    def predict (self ,point_coords =None ,point_labels =None ,box =None ,**kwargs ):
        if self .current_image is None :
            return np .array ([]),np .array ([]),np .array ([])

        masks_data =self .mask_generator .generate (self .current_image )

        if not masks_data :
            return np .array ([]),np .array ([]),np .array ([])

        masks =np .array ([mask_data ['segmentation']for mask_data in masks_data ])
        scores =np .array ([mask_data .get ('predicted_iou',0.5 )for mask_data in masks_data ])
        logits =np .zeros_like (masks ,dtype =np .float32 )

        return masks ,scores ,logits

    def reset_image (self )->None :
        self .current_image =None

class HeatmapBackend (BaseSegmentationBackend ):

    def __init__ (self ,threshold :float =0.5 ):
        self .threshold =threshold
        self .current_image =None
        self .current_heatmap =None

    def set_image (self ,image :np .ndarray )->None :
        self .current_image =image

    def set_heatmap (self ,heatmap :np .ndarray )->None :

        self .current_heatmap =heatmap

    def predict (self ,heatmap =None ,**kwargs ):
        if self .current_image is None :
            return np .array ([]),np .array ([]),np .array ([])

        target_heatmap =heatmap if heatmap is not None else self .current_heatmap

        if target_heatmap is None :
            return np .array ([]),np .array ([]),np .array ([])

        binary_mask =target_heatmap >self .threshold

        try :
            from scipy import ndimage
            labeled_mask ,num_features =ndimage .label (binary_mask )
        except ImportError :

            labeled_mask =binary_mask .astype (np .int32 )
            num_features =1

        masks =[]
        scores =[]

        for i in range (1 ,num_features +1 ):
            mask =(labeled_mask ==i )
            if mask .sum ()>100 :
                masks .append (mask )
                scores .append (float (target_heatmap [mask ].mean ()))

        if not masks :
            return np .array ([]),np .array ([]),np .array ([])

        masks_array =np .array (masks )
        scores_array =np .array (scores )
        logits_array =np .zeros_like (masks_array ,dtype =np .float32 )

        return masks_array ,scores_array ,logits_array

    def reset_image (self )->None :
        self .current_image =None
        self .current_heatmap =None

class SAMPredictor :

    def __init__ (self ,sam_model =None ,backend_type :str ="sam",**kwargs ):

        self .backend_type =backend_type
        self .device =torch .device ('cuda'if torch .cuda .is_available ()else 'cpu')

        if backend_type =="sam":
            self .backend =SAMBackend (sam_model )
        elif backend_type =="fastsam":
            mask_generator =kwargs .get ('mask_generator')
            if mask_generator is None :
                raise ValueError ("mask_generator требуется для FastSAM бэкенда")
            self .backend =FastSAMBackend (mask_generator )
        elif backend_type =="heatmap":
            threshold =kwargs .get ('threshold',0.5 )
            self .backend =HeatmapBackend (threshold )
        else :
            raise ValueError (f"Неподдерживаемый тип бэкенда: {backend_type}")

    def set_image (self ,image :np .ndarray )->None :

        self .backend .set_image (image )

    def predict (
    self ,
    point_coords :Optional [np .ndarray ]=None ,
    point_labels :Optional [np .ndarray ]=None ,
    box :Optional [np .ndarray ]=None ,
    mask_input :Optional [np .ndarray ]=None ,
    multimask_output :bool =True ,
    return_logits :bool =False ,
    **kwargs
    )->Tuple [np .ndarray ,np .ndarray ,np .ndarray ]:

        return self .backend .predict (
        point_coords =point_coords ,
        point_labels =point_labels ,
        box =box ,
        mask_input =mask_input ,
        multimask_output =multimask_output ,
        return_logits =return_logits ,
        **kwargs
        )

    def reset_image (self )->None :

        self .backend .reset_image ()

    def set_heatmap (self ,heatmap :np .ndarray )->None :

        if hasattr (self .backend ,'set_heatmap'):
            self .backend .set_heatmap (heatmap )
        else :
            raise ValueError (f"Метод set_heatmap не поддерживается для бэкенда {self.backend_type}")

    def get_backend_type (self )->str :

        return self .backend_type

    def switch_backend (self ,backend_type :str ,**kwargs )->None :

        if backend_type =="sam":
            sam_model =kwargs .get ('sam_model')
            if sam_model is None :
                raise ValueError ("sam_model требуется для SAM бэкенда")
            self .backend =SAMBackend (sam_model )
        elif backend_type =="fastsam":
            mask_generator =kwargs .get ('mask_generator')
            if mask_generator is None :
                raise ValueError ("mask_generator требуется для FastSAM бэкенда")
            self .backend =FastSAMBackend (mask_generator )
        elif backend_type =="heatmap":
            threshold =kwargs .get ('threshold',0.5 )
            self .backend =HeatmapBackend (threshold )
        else :
            raise ValueError (f"Неподдерживаемый тип бэкенда: {backend_type}")

        self .backend_type =backend_type