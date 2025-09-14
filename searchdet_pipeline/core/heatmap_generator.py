import numpy as np
import torch
import torch .nn as nn
import torch .nn .functional as F
from torchvision import transforms
from PIL import Image
from typing import List ,Optional ,Union ,Tuple
import math
from sklearn .metrics .pairwise import cosine_similarity
from .dinov3_encoder import DinoV3Encoder

IMAGENET_DEFAULT_MEAN =(0.485 ,0.456 ,0.406 )
IMAGENET_DEFAULT_STD =(0.229 ,0.224 ,0.225 )

class CenterPadding (torch .nn .Module ):

    def __init__ (self ,multiple :int =14 ):
        super ().__init__ ()
        self .multiple =multiple

    def _get_pad (self ,size :int )->Tuple [int ,int ]:
        new_size =math .ceil (size /self .multiple )*self .multiple
        pad_size =new_size -size
        pad_size_left =pad_size //2
        pad_size_right =pad_size -pad_size_left
        return pad_size_left ,pad_size_right

    @torch .no_grad ()
    def forward (self ,x :torch .Tensor )->torch .Tensor :

        if x .dim ()==3 or x .dim ()==4 :
            h ,w =x .shape [-2 ],x .shape [-1 ]
        else :
            raise ValueError (f"CenterPadding expects 3D (C,H,W) or 4D (B,C,H,W) tensor, got {x.dim()}D")
        pad_h =self ._get_pad (h )
        pad_w =self ._get_pad (w )
        return F .pad (x ,(pad_w [0 ],pad_w [1 ],pad_h [0 ],pad_h [1 ]))

class _MaybeToTensor (transforms .ToTensor ):

    def __call__ (self ,pic ):
        if isinstance (pic ,torch .Tensor ):
            return pic
        return super ().__call__ (pic )

def _make_normalize_transform (
mean :tuple =IMAGENET_DEFAULT_MEAN ,
std :tuple =IMAGENET_DEFAULT_STD ,
)->transforms .Normalize :

    return transforms .Normalize (mean =mean ,std =std )

def get_dinov3_transform (
crop_img :bool ,
*,
padding_multiple :int =14 ,
resize_img :bool =True ,
resize_size :int =256 ,
resize_max_size :int =800 ,
interpolation =transforms .InterpolationMode .BICUBIC ,
crop_size :int =224 ,
mean :tuple =IMAGENET_DEFAULT_MEAN ,
std :tuple =IMAGENET_DEFAULT_STD ,
)->transforms .Compose :

    transform_list =[_MaybeToTensor ()]

    if resize_img :
        if crop_img :

            transform_list .append (
            transforms .Resize (resize_size ,interpolation =interpolation ,max_size =resize_max_size )
            )
        else :

            transform_list .append (
            transforms .Resize ((resize_size ,resize_size ),interpolation =interpolation )
            )

    if crop_img :
        transform_list .append (transforms .CenterCrop (crop_size ))
    else :
        transform_list .append (CenterPadding (padding_multiple ))

    transform_list .append (_make_normalize_transform (mean ,std ))

    return transforms .Compose (transform_list )

class DinoV3FeatureExtractor (nn .Module ):

    def __init__ (self ,dinov3_encoder :DinoV3Encoder ,resize_images :bool =True ,crop_images :bool =False ,resize_size :int =512 ):
        super ().__init__ ()
        self .dinov3_encoder =dinov3_encoder
        self .transform =get_dinov3_transform (
        crop_img =crop_images ,
        resize_img =resize_images ,
        resize_size =resize_size
        )

    @property
    def device (self ):
        return self .dinov3_encoder .device

    def forward (self ,images :List [Image .Image ])->Tuple [torch .Tensor ,torch .Tensor ]:

        if not images :
            raise ValueError ("Список изображений не может быть пустым")

        tensors =[]
        for img in images :
            if isinstance (img ,Image .Image ):
                tensor =self .transform (img )
            else :
                tensor =self .transform (Image .fromarray (img ))
            tensors .append (tensor )

        batch =torch .stack (tensors ).to (self .device )

        with torch .no_grad ():
            cls_tokens ,patch_tokens =self .dinov3_encoder .extract_cls_and_patches (batch )

        return cls_tokens ,patch_tokens

    def forward_from_tensor (self ,image :torch .Tensor )->Tuple [torch .Tensor ,torch .Tensor ]:

        if image .dim ()==3 :
            image =image .unsqueeze (0 )

        image =image .to (self .device )

        with torch .no_grad ():
            cls_tokens ,patch_tokens =self .dinov3_encoder .extract_cls_and_patches (image )

        return cls_tokens ,patch_tokens

def rescale_features (
features :torch .Tensor ,
img :Optional [Image .Image ]=None ,
height :Optional [int ]=None ,
width :Optional [int ]=None ,
do_resize :bool =False ,
resize_size :Union [int ,Tuple [int ,int ]]=256
)->torch .Tensor :

    if img is not None :
        target_height ,target_width =img .size [1 ],img .size [0 ]
    elif height is not None and width is not None :
        target_height ,target_width =height ,width
    else :
        raise ValueError ("Необходимо указать либо изображение, либо размеры")

    if do_resize :
        if isinstance (resize_size ,int ):

            aspect_ratio =target_width /target_height
            if aspect_ratio >1 :
                target_width =resize_size
                target_height =int (resize_size /aspect_ratio )
            else :
                target_height =resize_size
                target_width =int (resize_size *aspect_ratio )
        else :
            target_height ,target_width =resize_size

    if features .dim ()==3 :
        features =features .unsqueeze (0 )

    batch_size ,seq_len ,dim =features .shape

    grid_side =int (math .sqrt (seq_len ))
    valid =grid_side *grid_side
    if seq_len !=valid :
        features =features [:,:valid ,:]

    features =features .view (features .shape [0 ],grid_side ,grid_side ,dim )
    features =features .permute (0 ,3 ,1 ,2 )

    features =F .interpolate (
    features ,
    size =(target_height ,target_width ),
    mode ='bilinear',
    align_corners =False
    )

    return features

def calculate_attention_weights_softmax (query_embedding :torch .Tensor ,example_embeddings :torch .Tensor ,temperature :float =2.0 )->torch .Tensor :

    similarities =F .cosine_similarity (
    query_embedding .unsqueeze (0 ),
    example_embeddings ,
    dim =1
    )

    similarities =similarities /temperature

    weights =F .softmax (similarities ,dim =0 )

    return weights

def adjust_embedding (query_embedding :torch .Tensor ,
positive_embeddings :torch .Tensor ,
negative_embeddings :torch .Tensor ,
positive_weight :float =3.0 ,
negative_weight :float =2.0 ,
query_weight :float =0.5 )->torch .Tensor :

    positive_weights =calculate_attention_weights_softmax (query_embedding ,positive_embeddings ,temperature =2.5 )

    positive_adjustment =positive_weight *torch .sum (positive_weights .unsqueeze (1 )*positive_embeddings ,dim =0 )

    if negative_embeddings .numel ()>0 :
        negative_weights =calculate_attention_weights_softmax (query_embedding ,negative_embeddings ,temperature =2.5 )

        negative_adjustment =negative_weight *torch .sum (negative_weights .unsqueeze (1 )*negative_embeddings ,dim =0 )

        combined_adjustment =query_weight *query_embedding +positive_adjustment -negative_adjustment
    else :
        combined_adjustment =query_weight *query_embedding +positive_adjustment

    combined_adjustment =F .normalize (combined_adjustment ,p =2 ,dim =0 )

    return combined_adjustment

class HeatmapGenerator :

    def __init__ (
    self ,
    dinov3_encoder :DinoV3Encoder ,
    attention_pool_examples :bool =False ,
    use_cosine_similarity_for_heatmap :bool =True ,
    enable_expansion :bool =False ,
    morph_smoothing_kernel :int =5 ,
    resize_size :int =512 ,
    crop_images :bool =False
    ):

        self .dinov3_fe =DinoV3FeatureExtractor (
        dinov3_encoder ,
        resize_images =True ,
        crop_images =crop_images ,
        resize_size =resize_size
        )
        self .attention_pool_examples =attention_pool_examples
        self .use_cosine_similarity_for_heatmap =use_cosine_similarity_for_heatmap
        self .enable_expansion =enable_expansion
        self .morph_smoothing_kernel =morph_smoothing_kernel

    @torch .no_grad ()
    def generate_heatmap (
    self ,
    input_image :Image .Image ,
    positive_images :List [Image .Image ],
    negative_images :List [Image .Image ]=None
    )->torch .Tensor :

        if negative_images is None :
            negative_images =[]

        input_cls ,input_patches =self .dinov3_fe ([input_image ])

        tokens =input_patches [0 ]
        side =int (math .sqrt (tokens .shape [0 ]))
        valid =side *side
        if tokens .shape [0 ]!=valid :
            tokens =tokens [:valid ]

        if self .attention_pool_examples :
            pooled_embed =self ._get_pooled_embed (tokens ,positive_images +negative_images )
            positive_embed =pooled_embed [:len (positive_images )]
            negative_embed =pooled_embed [len (positive_images ):]if negative_images else torch .empty (0 ,device =self .dinov3_fe .device )
        else :
            positive_cls ,_ =self .dinov3_fe (positive_images )
            positive_embed =positive_cls

            if negative_images :
                negative_cls ,_ =self .dinov3_fe (negative_images )
                negative_embed =negative_cls
            else :
                negative_embed =torch .empty (0 ,device =self .dinov3_fe .device )

        adjusted_embed =adjust_embedding (
        input_cls [0 ],
        positive_embed ,
        negative_embed ,
        positive_weight =3.0 ,
        negative_weight =2.0 ,
        query_weight =0.5
        )

        if self .use_cosine_similarity_for_heatmap :
            similarities =F .cosine_similarity (
            adjusted_embed .unsqueeze (0 ),
            tokens ,
            dim =1
            )
        else :

            distances =torch .norm (tokens -adjusted_embed .unsqueeze (0 ),dim =1 )
            similarities =1.0 /(1.0 +distances )

        similarities =self ._enhance_heatmap_contrast (similarities )

        if similarities .numel ()!=side *side :
            similarities =similarities [:side *side ]
        heatmap =similarities .view (side ,side )

        heatmap =self ._adaptive_normalize_heatmap (heatmap )
        heatmap =self ._apply_morphological_filtering (heatmap )

        if self .enable_expansion :
            heatmap =self .expand_hot_zones (heatmap ,expansion_factor =2.5 ,dilation_iterations =4 )

        return heatmap

    def _enhance_heatmap_contrast (self ,similarities :torch .Tensor ,
    contrast_factor :float =2.0 ,
    sharpening_factor :float =1.5 )->torch .Tensor :

        # ОПТИМИЗАЦИЯ: Упрощенная нормализация без сложных операций
        similarities_norm =(similarities -similarities .min ())/(similarities .max ()-similarities .min ()+1e-8 )
        return similarities_norm

    def _adaptive_normalize_heatmap (self ,heatmap :torch .Tensor ,
    percentile_low :float =2.0 ,
    percentile_high :float =98.0 )->torch .Tensor :

        flat_heatmap =heatmap .flatten ()
        low_val =torch .quantile (flat_heatmap ,percentile_low /100.0 )
        high_val =torch .quantile (flat_heatmap ,percentile_high /100.0 )

        heatmap_clipped =torch .clamp (heatmap ,low_val ,high_val )

        if high_val >low_val :
            heatmap_normalized =(heatmap_clipped -low_val )/(high_val -low_val )
        else :
            heatmap_normalized =heatmap_clipped

        return heatmap_normalized

    def _apply_morphological_filtering (self ,heatmap :torch .Tensor ,
    threshold :float =0.6 ,
    kernel_size :int =3 )->torch .Tensor :

        # ОПТИМИЗАЦИЯ: Упрощенное сглаживание без создания гауссового ядра
        kernel_size =3  # Фиксированный размер для ускорения
        heatmap_expanded =heatmap .unsqueeze (0 ).unsqueeze (0 )
        
        # Простое усреднение с фиксированным ядром 3x3
        kernel =torch .ones (1 ,1 ,kernel_size ,kernel_size ,device =heatmap .device )/(kernel_size **2 )
        smoothed =F .conv2d (heatmap_expanded ,kernel ,padding =kernel_size //2 )
        smoothed =smoothed .squeeze (0 ).squeeze (0 )

        if smoothed .max ()>smoothed .min ():
            smoothed =(smoothed -smoothed .min ())/(smoothed .max ()-smoothed .min ())

        return smoothed

    def expand_hot_zones (self ,heatmap :torch .Tensor ,
    expansion_factor :float =2.0 ,
    hot_zone_threshold :float =0.7 ,
    dilation_iterations :int =3 )->torch .Tensor :

        hot_zones =(heatmap >hot_zone_threshold ).float ()

        kernel_size =max (3 ,int (expansion_factor *2 )+1 )
        kernel =torch .ones (1 ,1 ,kernel_size ,kernel_size ,device =heatmap .device )

        hot_zones_expanded =hot_zones .unsqueeze (0 ).unsqueeze (0 )

        for _ in range (dilation_iterations ):
            dilated =F .conv2d (hot_zones_expanded ,kernel ,padding =kernel_size //2 )
            hot_zones_expanded =(dilated >0.1 ).float ()

        expanded_mask =hot_zones_expanded .squeeze (0 ).squeeze (0 )

        if expanded_mask .shape !=heatmap .shape :

            expanded_mask =F .interpolate (
            expanded_mask .unsqueeze (0 ).unsqueeze (0 ),
            size =heatmap .shape ,
            mode ='nearest'
            ).squeeze (0 ).squeeze (0 )

        expanded_heatmap =heatmap .clone ()

        expansion_value =hot_zone_threshold *0.8
        expanded_heatmap =torch .where (
        (expanded_mask >0 )&(heatmap <=hot_zone_threshold ),
        torch .full_like (heatmap ,expansion_value ),
        expanded_heatmap
        )

        return expanded_heatmap

    def get_expanded_crop_bbox (self ,heatmap :torch .Tensor ,
    expansion_threshold :float =0.5 ,
    padding_ratio :float =0.1 )->Tuple [int ,int ,int ,int ]:

        crop_mask =(heatmap >expansion_threshold ).float ()

        nonzero_coords =torch .nonzero (crop_mask ,as_tuple =False )

        if len (nonzero_coords )==0 :

            h ,w =heatmap .shape
            center_h ,center_w =h //2 ,w //2
            size =min (h ,w )//4
            return (center_w -size ,center_h -size ,center_w +size ,center_h +size )

        y_coords =nonzero_coords [:,0 ]
        x_coords =nonzero_coords [:,1 ]

        y_min ,y_max =y_coords .min ().item (),y_coords .max ().item ()
        x_min ,x_max =x_coords .min ().item (),x_coords .max ().item ()

        h ,w =heatmap .shape
        bbox_h =y_max -y_min +1
        bbox_w =x_max -x_min +1

        pad_h =int (bbox_h *padding_ratio )
        pad_w =int (bbox_w *padding_ratio )

        y_min =max (0 ,y_min -pad_h )
        y_max =min (h -1 ,y_max +pad_h )
        x_min =max (0 ,x_min -pad_w )
        x_max =min (w -1 ,x_max +pad_w )

        return (x_min ,y_min ,x_max ,y_max )

    def crop_image_by_heatmap (self ,image :Image .Image ,heatmap :torch .Tensor ,
    expansion_threshold :float =0.5 ,
    padding_ratio :float =0.15 )->Tuple [Image .Image ,Tuple [int ,int ,int ,int ]]:

        x_min ,y_min ,x_max ,y_max =self .get_expanded_crop_bbox (
        heatmap ,expansion_threshold ,padding_ratio
        )

        img_w ,img_h =image .size
        heatmap_h ,heatmap_w =heatmap .shape

        scale_x =img_w /heatmap_w
        scale_y =img_h /heatmap_h

        x_min_scaled =int (x_min *scale_x )
        y_min_scaled =int (y_min *scale_y )
        x_max_scaled =int (x_max *scale_x )
        y_max_scaled =int (y_max *scale_y )

        x_min_scaled =max (0 ,x_min_scaled )
        y_min_scaled =max (0 ,y_min_scaled )
        x_max_scaled =min (img_w ,x_max_scaled )
        y_max_scaled =min (img_h ,y_max_scaled )

        cropped_image =image .crop ((x_min_scaled ,y_min_scaled ,x_max_scaled ,y_max_scaled ))

        return cropped_image ,(x_min_scaled ,y_min_scaled ,x_max_scaled ,y_max_scaled )

    def generate_heatmap_with_crop (self ,input_image :Image .Image ,
    positive_images :List [Image .Image ],
    negative_images :List [Image .Image ]=None )->Tuple [torch .Tensor ,Image .Image ,Tuple [int ,int ,int ,int ]]:

        heatmap =self .generate_heatmap (input_image ,positive_images ,negative_images )

        cropped_image ,bbox =self .crop_image_by_heatmap (input_image ,heatmap )

        return heatmap ,cropped_image ,bbox

    def image_from_heatmap (
    self ,
    heatmap :torch .Tensor ,
    image :Image .Image ,
    use_relative_heatmap :bool =False ,
    center :float =0 ,
    clamp_min :float =0 ,
    clamp_max :float =0.3 ,
    scale :float =1 ,
    heatmap_cmap :str ='inferno',
    heatmap_blend_ratio :float =0.5 ,
    **kwargs
    )->Image .Image :

        import matplotlib .pyplot as plt
        import matplotlib .cm as cm

        if isinstance (heatmap ,torch .Tensor ):
            heatmap_np =heatmap .cpu ().numpy ()
        else :
            heatmap_np =heatmap

        if use_relative_heatmap :
            heatmap_np =heatmap_np -center

        heatmap_np =np .clip (heatmap_np ,clamp_min ,clamp_max )

        heatmap_np =heatmap_np *scale

        heatmap_normalized =(heatmap_np -heatmap_np .min ())/(heatmap_np .max ()-heatmap_np .min ()+1e-8 )

        from scipy .ndimage import zoom
        target_height ,target_width =image .size [1 ],image .size [0 ]
        zoom_factors =(target_height /heatmap_normalized .shape [0 ],target_width /heatmap_normalized .shape [1 ])
        heatmap_resized =zoom (heatmap_normalized ,zoom_factors ,order =1 )

        colormap =cm .get_cmap (heatmap_cmap )
        heatmap_colored =colormap (heatmap_resized )
        heatmap_colored =(heatmap_colored [:,:,:3 ]*255 ).astype (np .uint8 )

        image_np =np .array (image )

        blended =(1 -heatmap_blend_ratio )*image_np +heatmap_blend_ratio *heatmap_colored
        blended =np .clip (blended ,0 ,255 ).astype (np .uint8 )

        return Image .fromarray (blended )

    def _get_pooled_embed (self ,query_feats :torch .Tensor ,images :List [Image .Image ])->torch .Tensor :

        pooled_features =self ._generate_pooled_patch_features (images )
        pooled_embed =self ._attention_pool_keys (query_feats ,pooled_features )
        return pooled_embed

    def _generate_pooled_patch_features (self ,images :List [Image .Image ])->torch .Tensor :

        all_patches =[]
        for img in images :
            _ ,patches =self .dinov3_fe ([img ])
            tokens =patches [0 ]
            grid_side =int (math .sqrt (tokens .shape [0 ]))
            num_patches =grid_side *grid_side
            tokens =tokens [:num_patches ]
            all_patches .append (tokens )
        return torch .stack (all_patches )

    def _attention_pool_keys (self ,query :torch .Tensor ,keys :torch .Tensor )->torch .Tensor :

        attention_scores =torch .einsum ('sd,nsd->ns',query ,keys )
        attention_weights =F .softmax (attention_scores ,dim =1 )

        pooled =torch .einsum ('ns,nsd->nd',attention_weights ,keys )

        return pooled

def open_image (img_path :str )->Image .Image :

    return Image .open (img_path ).convert ('RGB')

def crop_heatmap_region (image :Image .Image ,heatmap :torch .Tensor ,
threshold :float =0.5 ,padding :int =20 ,
min_crop_size :int =64 ,max_crop_ratio :float =0.8 ,
adaptive_padding :bool =True )->Tuple [Image .Image ,Tuple [int ,int ,int ,int ]]:

    img_h ,img_w =image .size [1 ],image .size [0 ]
    heatmap_h ,heatmap_w =heatmap .shape

    import cv2
    heatmap_np =heatmap .cpu ().numpy ()if isinstance (heatmap ,torch .Tensor )else heatmap

    hot_mask =(heatmap_np >threshold ).astype (np .uint8 )

    kernel_size =max (3 ,min (heatmap_h ,heatmap_w )//20 )
    kernel =cv2 .getStructuringElement (cv2 .MORPH_ELLIPSE ,(kernel_size ,kernel_size ))
    hot_mask =cv2 .morphologyEx (hot_mask ,cv2 .MORPH_CLOSE ,kernel )
    hot_mask =cv2 .morphologyEx (hot_mask ,cv2 .MORPH_OPEN ,kernel //2 )

    nonzero_coords =np .nonzero (hot_mask )

    if len (nonzero_coords [0 ])==0 :

        center_x ,center_y =img_w //2 ,img_h //2
        crop_size =max (min_crop_size ,min (img_w ,img_h )//3 )
        x1 =max (0 ,center_x -crop_size //2 )
        y1 =max (0 ,center_y -crop_size //2 )
        x2 =min (img_w ,x1 +crop_size )
        y2 =min (img_h ,y1 +crop_size )
        bbox =(x1 ,y1 ,x2 ,y2 )
        cropped =image .crop (bbox )
        print (f"   ⚠️ Нет горячих зон, используем центральную область: {bbox}")
        return cropped ,bbox

    min_y ,max_y =np .min (nonzero_coords [0 ]),np .max (nonzero_coords [0 ])
    min_x ,max_x =np .min (nonzero_coords [1 ]),np .max (nonzero_coords [1 ])

    scale_x =img_w /heatmap_w
    scale_y =img_h /heatmap_h

    x1 =int (min_x *scale_x )
    y1 =int (min_y *scale_y )
    x2 =int ((max_x +1 )*scale_x )
    y2 =int ((max_y +1 )*scale_y )

    bbox_w =x2 -x1
    bbox_h =y2 -y1

    if adaptive_padding :
        adaptive_pad_x =max (padding ,int (bbox_w *0.15 ))
        adaptive_pad_y =max (padding ,int (bbox_h *0.15 ))
    else :
        adaptive_pad_x =adaptive_pad_y =padding

    x1 =max (0 ,x1 -adaptive_pad_x )
    y1 =max (0 ,y1 -adaptive_pad_y )
    x2 =min (img_w ,x2 +adaptive_pad_x )
    y2 =min (img_h ,y2 +adaptive_pad_y )

    final_w =x2 -x1
    final_h =y2 -y1

    if final_w <min_crop_size or final_h <min_crop_size :

        center_x =(x1 +x2 )//2
        center_y =(y1 +y2 )//2
        half_size =max (min_crop_size //2 ,max (final_w ,final_h )//2 )

        x1 =max (0 ,center_x -half_size )
        y1 =max (0 ,center_y -half_size )
        x2 =min (img_w ,center_x +half_size )
        y2 =min (img_h ,center_y +half_size )

    max_w =int (img_w *max_crop_ratio )
    max_h =int (img_h *max_crop_ratio )

    if (x2 -x1 )>max_w or (y2 -y1 )>max_h :

        center_x =(x1 +x2 )//2
        center_y =(y1 +y2 )//2

        half_w =min (max_w //2 ,(x2 -x1 )//2 )
        half_h =min (max_h //2 ,(y2 -y1 )//2 )

        x1 =max (0 ,center_x -half_w )
        y1 =max (0 ,center_y -half_h )
        x2 =min (img_w ,center_x +half_w )
        y2 =min (img_h ,center_y +half_h )

    bbox =(x1 ,y1 ,x2 ,y2 )
    cropped =image .crop (bbox )

    coverage =_calculate_hotzone_coverage (heatmap_np ,bbox ,(img_w ,img_h ),threshold )
    print (f"   🎯 ROI bbox: {bbox}, размер: {x2-x1}x{y2-y1}, покрытие горячих зон: {coverage:.1%}")

    return cropped ,bbox

def _calculate_hotzone_coverage (heatmap :np .ndarray ,bbox :Tuple [int ,int ,int ,int ],
image_size :Tuple [int ,int ],threshold :float )->float :

    img_w ,img_h =image_size
    heatmap_h ,heatmap_w =heatmap .shape
    x1 ,y1 ,x2 ,y2 =bbox

    scale_x =heatmap_w /img_w
    scale_y =heatmap_h /img_h

    hm_x1 =int (x1 *scale_x )
    hm_y1 =int (y1 *scale_y )
    hm_x2 =int (x2 *scale_x )
    hm_y2 =int (y2 *scale_y )

    hm_x1 =max (0 ,min (hm_x1 ,heatmap_w -1 ))
    hm_y1 =max (0 ,min (hm_y1 ,heatmap_h -1 ))
    hm_x2 =max (hm_x1 +1 ,min (hm_x2 ,heatmap_w ))
    hm_y2 =max (hm_y1 +1 ,min (hm_y2 ,heatmap_h ))

    hot_mask =heatmap >threshold
    total_hot_pixels =np .sum (hot_mask )

    if total_hot_pixels ==0 :
        return 0.0

    bbox_hot_pixels =np .sum (hot_mask [hm_y1 :hm_y2 ,hm_x1 :hm_x2 ])

    return bbox_hot_pixels /total_hot_pixels

def visualize_crop_region (image :Image .Image ,heatmap :torch .Tensor ,bbox :Tuple [int ,int ,int ,int ],
output_path :Optional [str ]=None ,show_heatmap :bool =True )->Image .Image :

    import matplotlib .pyplot as plt
    import matplotlib .patches as patches
    from matplotlib .colors import LinearSegmentedColormap

    fig ,axes =plt .subplots (1 ,2 if show_heatmap else 1 ,figsize =(15 if show_heatmap else 8 ,6 ))
    if not show_heatmap :
        axes =[axes ]

    axes [0 ].imshow (image )
    axes [0 ].set_title ('Исходное изображение с ROI bbox',fontsize =12 )

    x1 ,y1 ,x2 ,y2 =bbox
    rect =patches .Rectangle ((x1 ,y1 ),x2 -x1 ,y2 -y1 ,
    linewidth =3 ,edgecolor ='red',facecolor ='none',alpha =0.8 )
    axes [0 ].add_patch (rect )

    axes [0 ].text (x1 ,y1 -10 ,f'ROI: {x2-x1}×{y2-y1}',
    bbox =dict (boxstyle ='round',facecolor ='red',alpha =0.7 ),
    fontsize =10 ,color ='white',weight ='bold')

    axes [0 ].set_xlim (0 ,image .size [0 ])
    axes [0 ].set_ylim (image .size [1 ],0 )
    axes [0 ].axis ('off')

    if show_heatmap :

        heatmap_np =heatmap .cpu ().numpy ()if isinstance (heatmap ,torch .Tensor )else heatmap

        colors =['black','green','cyan','yellow','red']
        n_bins =256
        cmap =LinearSegmentedColormap .from_list ('heatmap',colors ,N =n_bins )

        im =axes [1 ].imshow (heatmap_np ,cmap =cmap ,alpha =0.8 ,
        extent =[0 ,image .size [0 ],image .size [1 ],0 ])
        axes [1 ].set_title ('Heatmap с ROI bbox',fontsize =12 )

        rect2 =patches .Rectangle ((x1 ,y1 ),x2 -x1 ,y2 -y1 ,
        linewidth =3 ,edgecolor ='white',facecolor ='none',alpha =1.0 )
        axes [1 ].add_patch (rect2 )

        plt .colorbar (im ,ax =axes [1 ],fraction =0.046 ,pad =0.04 )

        axes [1 ].set_xlim (0 ,image .size [0 ])
        axes [1 ].set_ylim (image .size [1 ],0 )
        axes [1 ].axis ('off')

    plt .tight_layout ()

    if output_path :
        plt .savefig (output_path ,dpi =150 ,bbox_inches ='tight')
        print (f"   💾 Визуализация сохранена: {output_path}")

    fig .canvas .draw ()
    buf =np .frombuffer (fig .canvas .tostring_rgb (),dtype =np .uint8 )
    buf =buf .reshape (fig .canvas .get_width_height ()[::-1 ]+(3 ,))
    result_image =Image .fromarray (buf )

    plt .close (fig )
    return result_image

def save_crop_debug_info (image :Image .Image ,heatmap :torch .Tensor ,bbox :Tuple [int ,int ,int ,int ],
cropped_image :Image .Image ,debug_dir :str ="debug_crops")->str :

    import os
    from datetime import datetime

    os .makedirs (debug_dir ,exist_ok =True )

    timestamp =datetime .now ().strftime ("%Y%m%d_%H%M%S")
    viz_path =os .path .join (debug_dir ,f"crop_debug_{timestamp}.png")
    crop_path =os .path .join (debug_dir ,f"cropped_{timestamp}.png")

    visualize_crop_region (image ,heatmap ,bbox ,viz_path ,show_heatmap =True )

    cropped_image .save (crop_path )

    info_path =os .path .join (debug_dir ,f"crop_info_{timestamp}.txt")
    x1 ,y1 ,x2 ,y2 =bbox
    with open (info_path ,'w',encoding ='utf-8')as f :
        f .write (f"Отладочная информация кропинга\n")
        f .write (f"Время: {datetime.now()}\n")
        f .write (f"Исходное изображение: {image.size}\n")
        f .write (f"Heatmap размер: {heatmap.shape}\n")
        f .write (f"ROI bbox: {bbox}\n")
        f .write (f"ROI размер: {x2-x1}×{y2-y1}\n")
        f .write (f"Кропнутое изображение: {cropped_image.size}\n")
        f .write (f"Покрытие области: {(x2-x1)*(y2-y1)/(image.size[0]*image.size[1]):.1%}\n")

    print (f"   📁 Отладочная информация сохранена в {debug_dir}/")
    return viz_path

def _save_fastsam_overlay_details (overlay_masks ,image_size ,bbox ):

    import json
    import os
    from datetime import datetime

    overlay_dir ="output/overlay_details"
    os .makedirs (overlay_dir ,exist_ok =True )

    overlay_data ={
    "timestamp":datetime .now ().isoformat (),
    "image_size":image_size ,
    "crop_bbox":bbox ,
    "total_masks":len (overlay_masks ),
    "masks":[]
    }

    for idx ,mask ,overlap_ratio in overlay_masks :
        mask_info ={
        "mask_id":idx ,
        "overlap_ratio":round (overlap_ratio ,4 ),
        "mask_shape":list (mask .shape )if hasattr (mask ,'shape')else None
        }
        overlay_data ["masks"].append (mask_info )

    filename =f"fastsam_overlay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath =os .path .join (overlay_dir ,filename )

    with open (filepath ,'w',encoding ='utf-8')as f :
        json .dump (overlay_data ,f ,indent =2 ,ensure_ascii =False )

    print (f"   💾 Сохранена информация о {len(overlay_masks)} масках в {filepath}")

def merge_masks_with_heatmap (fastsam_masks :List [torch .Tensor ],
heatmap :torch .Tensor ,
bbox :Tuple [int ,int ,int ,int ],
image_size :Tuple [int ,int ],
min_overlap_ratio :float =0.5 )->List [torch .Tensor ]:

    img_w ,img_h =image_size
    x1 ,y1 ,x2 ,y2 =bbox

    hot_mask =(heatmap >0.5 ).float ()
    hot_mask_resized =F .interpolate (hot_mask .unsqueeze (0 ).unsqueeze (0 ),size =(img_h ,img_w ),mode ='nearest').squeeze (0 ).squeeze (0 )

    filtered_masks =[]
    overlay_masks =[]

    print (f"   🔍 Анализ перекрытия {len(fastsam_masks)} FastSAM масок с heatmap...")

    for i ,mask in enumerate (fastsam_masks ):

        full_mask =torch .zeros ((img_h ,img_w ),device =mask .device ,dtype =mask .dtype )

        mask_h ,mask_w =mask .shape [-2 :]
        crop_h ,crop_w =y2 -y1 ,x2 -x1

        if mask .dim ()==2 :
            mask_resized =F .interpolate (
            mask .unsqueeze (0 ).unsqueeze (0 ),
            size =(crop_h ,crop_w ),
            mode ='nearest'
            ).squeeze (0 ).squeeze (0 )
        else :
            mask_resized =F .interpolate (
            mask .unsqueeze (0 )if mask .dim ()==3 else mask ,
            size =(crop_h ,crop_w ),
            mode ='nearest'
            ).squeeze (0 )
            if mask_resized .dim ()==3 :
                mask_resized =mask_resized [0 ]

        full_mask [y1 :y2 ,x1 :x2 ]=mask_resized

        merged_mask =full_mask *hot_mask_resized

        mask_area =torch .sum (full_mask >0.5 ).float ()
        overlap_area =torch .sum (merged_mask >0.5 ).float ()

        if mask_area >0 :
            overlap_ratio =overlap_area /mask_area

            overlay_masks .append ((i ,mask ,float (overlap_ratio )))

            if overlap_ratio >=min_overlap_ratio :

                filtered_masks .append (full_mask )
                print (f"   ✅ Маска {i}: перекрытие {overlap_ratio:.3f} >= {min_overlap_ratio} - принята")
            else :
                print (f"   ❌ Маска {i}: перекрытие {overlap_ratio:.3f} < {min_overlap_ratio} - отклонена")

    overlay_50_masks =[(idx ,mask ,ratio )for idx ,mask ,ratio in overlay_masks if ratio >=0.5 ]
    if overlay_50_masks :
        try :
            _save_fastsam_overlay_details (overlay_50_masks ,image_size ,bbox )
        except Exception as e :
            print (f"   ⚠️ Ошибка сохранения overlay_details: {e}")

    if len (filtered_masks )>1 :
        print (f"   🔗 Объединение {len(filtered_masks)} масок по IoU...")
        merged_masks =merge_overlapping_masks (filtered_masks ,iou_threshold =0.3 )
        print (f"   📊 Результат: {len(fastsam_masks)} -> {len(filtered_masks)} -> {len(merged_masks)} масок")
        return merged_masks
    else :
        print(f"   📊 Результат фильтрации: {len(filtered_masks)}/{len(fastsam_masks)} масок прошли фильтр (порог {min_overlap_ratio})")
        return filtered_masks


def merge_overlapping_masks(masks: List[torch.Tensor], iou_threshold: float = 0.3) -> List[torch.Tensor]:
    if not masks:
        return []
    
    print(f"   🔗 Объединение {len(masks)} масок по IoU (порог {iou_threshold})...")

    binary_masks = [(mask > 0.5).float() for mask in masks]
    
    merged_masks = []
    used_indices = set()
    
    for i, mask_i in enumerate(binary_masks):
        if i in used_indices:
            continue
            
        current_merged = mask_i.clone()
        merged_group = [i]
        
        for j, mask_j in enumerate(binary_masks):
            if j <= i or j in used_indices:
                continue
                
            intersection = torch.sum(current_merged * mask_j)
            union = torch.sum((current_merged + mask_j) > 0.5)
            
            if union > 0:
                iou = intersection / union
                
                if iou >= iou_threshold:
                    current_merged = torch.clamp(current_merged + mask_j, 0, 1)
                    merged_group.append(j)
                    used_indices.add(j)
                    print(f" маски {i} и {j} (IoU: {iou:.3f})")
        
        merged_masks.append(current_merged)
        used_indices.add(i)
        
        if len(merged_group) > 1:
            print(f"Группа масок {merged_group} объединена в одну")
    
    print(f"Результат объединения: {len(masks)} -> {len(merged_masks)} масок")
    return merged_masks