import torch
import torch .nn .functional as F
import numpy as np
from typing import Tuple ,Union

def get_image_size (image :Union [np .ndarray ,torch .Tensor ])->Tuple [int ,int ]:
    if isinstance (image ,np .ndarray ):
        if len (image .shape )==3 :
            return image .shape [:2 ]
        elif len (image .shape )==2 :
            return image .shape
        else :
            raise ValueError (f"Unsupported image shape: {image.shape}")
    elif isinstance (image ,torch .Tensor ):
        if len (image .shape )==4 :
            return image .shape [2 :4 ]
        elif len (image .shape )==3 :
            return image .shape [1 :3 ]
        elif len (image .shape )==2 :
            return image .shape
        else :
            raise ValueError (f"Unsupported tensor shape: {image.shape}")
    else :
        raise TypeError (f"Unsupported image type: {type(image)}")

def get_feature_map_size (feature_map :torch .Tensor )->Tuple [int ,int ]:
    if len (feature_map .shape )==4 :
        return feature_map .shape [2 :4 ]
    elif len (feature_map .shape )==3 :
        return feature_map .shape [1 :3 ]
    elif len (feature_map .shape )==2 :
        return feature_map .shape
    else :
        raise ValueError (f"Unsupported feature map shape: {feature_map.shape}")

def upsample_feature_map (
feature_map :torch .Tensor ,
target_size :Tuple [int ,int ],
mode :str ='bilinear',
align_corners :bool =False
)->torch .Tensor :
    if len (feature_map .shape )==2 :
        feature_map =feature_map .unsqueeze (0 ).unsqueeze (0 )
        squeeze_output =True
    elif len (feature_map .shape )==3 :
        feature_map =feature_map .unsqueeze (0 )
        squeeze_output =False
    elif len (feature_map .shape )==4 :
        squeeze_output =False
    else :
        raise ValueError (f"Unsupported feature map shape: {feature_map.shape}")

    upsampled =F .interpolate (
    feature_map ,
    size =target_size ,
    mode =mode ,
    align_corners =align_corners if mode =='bilinear'else None
    )

    if squeeze_output :
        if len (feature_map .shape )==2 :
            upsampled =upsampled .squeeze (0 ).squeeze (0 )
        elif len (feature_map .shape )==3 :
            upsampled =upsampled .squeeze (0 )

    return upsampled

def normalize_tensor (tensor :torch .Tensor ,dim :int =-1 )->torch .Tensor :
    return F .normalize (tensor ,p =2 ,dim =dim )

def compute_cosine_similarity (
tensor1 :torch .Tensor ,
tensor2 :torch .Tensor
)->torch .Tensor :
    return F .cosine_similarity (tensor1 ,tensor2 ,dim =-1 )

def batch_process (
data :list ,
batch_size :int ,
process_fn :callable
)->list :
    results =[]
    for i in range (0 ,len (data ),batch_size ):
        batch =data [i :i +batch_size ]
        batch_result =process_fn (batch )
        results .extend (batch_result if isinstance (batch_result ,list )else [batch_result ])
    return results
