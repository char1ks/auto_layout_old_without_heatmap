import sys
from pathlib import Path
import torch
import torchvision .transforms as T
import torch .nn .functional as F
from PIL import Image
import numpy as np
from typing import Any ,Dict ,Optional
import re
import math

try :
    torch .backends .cuda .matmul .allow_tf32 =True
    torch .backends .cudnn .allow_tf32 =True
except Exception :
    pass

try:
    from dinov3.hub import backbones as dino_backbones
except ImportError:
    project_root =Path (__file__ ).resolve ().parent .parent .parent
    dinov3_repo_path =project_root
    if str (dinov3_repo_path )not in sys .path :
        sys .path .insert (0 ,str (dinov3_repo_path ))
    inner_dinov3_path =project_root /'dinov3'
    if str (inner_dinov3_path )not in sys .path :
        sys .path .insert (0 ,str (inner_dinov3_path ))
    from dinov3.hub import backbones as dino_backbones

def _to_pil_any (x :object )->Image .Image :

    if isinstance (x ,Image .Image ):
        return x .convert ("RGB")
    if isinstance (x ,str ):
        return Image .open (x ).convert ("RGB")
    if isinstance (x ,np .ndarray ):
        arr =x
        if arr .ndim ==2 :
            arr =np .stack ([arr ]*3 ,axis =-1 )
        if arr .dtype !=np .uint8 :

            if arr .max ()<=1.0 :
                arr =(np .clip (arr ,0 ,1 )*255 ).astype (np .uint8 )
            else :
                arr =np .clip (arr ,0 ,255 ).astype (np .uint8 )
        return Image .fromarray (arr ,mode ="RGB")
    if isinstance (x ,torch .Tensor ):
        t =x .detach ().cpu ()
        if t .ndim ==2 :
            t =t .unsqueeze (-1 ).repeat (1 ,1 ,3 )
        if t .ndim ==3 :
            if t .shape [0 ]in (1 ,3 ):
                t =(t .clamp (0 ,1 )*255 ).byte ().permute (1 ,2 ,0 ).numpy ()
            else :
                t =(t .clamp (0 ,1 )*255 ).byte ().numpy ()
        return Image .fromarray (t ,mode ="RGB")
    raise TypeError (f"Unsupported image type: {type(x)}")

class DinoV3Encoder:
    def __init__ (self ,backbone_name ='vitb16',device ='cpu',ckpt_path =None ,half_precision :bool =False ,vit_pooling :str ='cls',loader :str ='hub',repo_dir :Optional [str ]=None ):
        self .device =torch .device (device if (device =="cpu"or torch .cuda .is_available ())else "cpu")

        self .half =bool (half_precision )
        self .pooling =str (vit_pooling or "cls").lower ()
        self .loader =loader
        self .repo_dir =repo_dir
        self .is_vit =True

        name =(backbone_name or 'vitb16').lower ()

        aliases ={
        'vits16':'dinov3_vits16',
        'vits16plus':'dinov3_vits16plus',
        'vitb16':'dinov3_vitb16',
        'vitl16':'dinov3_vitl16',
        'vitl16plus':'dinov3_vitl16plus',
        'vith16plus':'dinov3_vith16plus',
        'vit7b16':'dinov3_vit7b16',
        'convnext_tiny':'dinov3_convnext_tiny',
        'convnext_small':'dinov3_convnext_small',
        'convnext_base':'dinov3_convnext_base',
        'convnext_large':'dinov3_convnext_large',
        }

        if name in aliases :
            fn_name =aliases [name ]
        elif name .startswith ('dinov3_'):
            fn_name =name
        else :

            if 'vit7b16'in name :
                fn_name ='dinov3_vit7b16'
            elif 'vitb16'in name :
                fn_name ='dinov3_vitb16'
            elif 'vits16plus'in name :
                fn_name ='dinov3_vits16plus'
            elif 'vits16'in name :
                fn_name ='dinov3_vits16'
            elif 'vitl16plus'in name :
                fn_name ='dinov3_vitl16plus'
            elif 'vitl16'in name :
                fn_name ='dinov3_vitl16'
            elif 'vith16plus'in name :
                fn_name ='dinov3_vith16plus'
            elif 'convnext_large'in name :
                fn_name ='dinov3_convnext_large'
            elif 'convnext_base'in name :
                fn_name ='dinov3_convnext_base'
            elif 'convnext_small'in name :
                fn_name ='dinov3_convnext_small'
            elif 'convnext_tiny'in name :
                fn_name ='dinov3_convnext_tiny'
            else :
                fn_name ='dinov3_vitb16'
        self .is_vit =not fn_name .startswith ('dinov3_convnext')

        if not hasattr (dino_backbones ,fn_name ):
            print (f"⚠️ DINOv3: Неизвестный backbone '{backbone_name}', используем vitb16")
            fn_name ='dinov3_vitb16'
        backbone_fn =getattr (dino_backbones ,fn_name )
        print (f"🔧 DINOv3: используем архитектуру {fn_name}")

        if not ckpt_path :
            cached_path =self ._find_cached_dinov3_ckpt (fn_name )
            if cached_path is not None :
                ckpt_path =str (cached_path )
                print (f"🔧 DINOv3: найден локальный checkpoint в cache: {ckpt_path}")
            else :

                any_arch ,any_path =self ._find_any_cached_dinov3_ckpt ()
                if any_arch and any_path :
                    print (f"🔧 DINOv3: локальные веса обнаружены для {any_arch}, переключаемся с {fn_name} → {any_arch} для оффлайн-режима")
                    fn_name =any_arch
                    backbone_fn =getattr (dino_backbones ,fn_name )
                    self .is_vit =not fn_name .startswith ('dinov3_convnext')
                    ckpt_path =str (any_path )

        if ckpt_path :
            print (f"🔧 DINOv3: загружаем checkpoint из {ckpt_path}")
            self .model =backbone_fn (pretrained =False ).to (self .device )
            ckpt =torch .load (ckpt_path ,map_location =self .device )

            if isinstance (ckpt ,dict ):
                if 'model'in ckpt and isinstance (ckpt ['model'],dict ):
                    state_dict =ckpt ['model']
                    print (f"   📦 Извлекли state_dict из ключа 'model'")
                elif 'state_dict'in ckpt and isinstance (ckpt ['state_dict'],dict ):
                    state_dict =ckpt ['state_dict']
                    print (f"   📦 Извлекли state_dict из ключа 'state_dict'")
                elif 'backbone'in ckpt and isinstance (ckpt ['backbone'],dict ):
                    state_dict =ckpt ['backbone']
                    print (f"   📦 Извлекли state_dict из ключа 'backbone'")
                else :

                    state_dict ={k :v for k ,v in ckpt .items ()if isinstance (v ,torch .Tensor )}
                    print (f"   📦 Используем checkpoint как state_dict напрямую")
            else :
                state_dict =ckpt
                print (f"   📦 Checkpoint не является словарем, используем как есть")

            model_keys =set (self .model .state_dict ().keys ())
            ckpt_keys =set (state_dict .keys ())
            matching_keys =model_keys &ckpt_keys

            print (f"   🔍 Анализ совместимости: модель имеет {len(model_keys)} ключей, checkpoint - {len(ckpt_keys)}")
            print (f"   🔍 Совпадающих ключей: {len(matching_keys)} из {len(model_keys)}")

            if len (matching_keys )<len (model_keys )*0.1 :
                print (f"   ❌ КРИТИЧЕСКОЕ ПРЕДУПРЕЖДЕНИЕ: Очень мало совпадающих ключей ({len(matching_keys)}/{len(model_keys)})!")
                print (f"   ❌ Возможно, это checkpoint не для backbone DINOv3, а для другой модели (например, DETR head)")
                print (f"   ❌ Рекомендуется использовать правильный backbone checkpoint или убрать --dinov3-ckpt")

                print (f"   🔍 Примеры ключей модели: {list(model_keys)[:5]}")
                print (f"   🔍 Примеры ключей checkpoint: {list(ckpt_keys)[:5]}")

            missing ,unexpected =self .model .load_state_dict (state_dict ,strict =False )
            try :
                n_total =sum (p .numel ()for p in self .model .state_dict ().values ())
                n_loaded =sum (state_dict [k ].numel ()for k in self .model .state_dict ().keys ()if k in state_dict )
                load_percentage =(n_loaded /n_total )*100 if n_total >0 else 0
                print (f"   📦 DINOv3 ckpt: загружено параметров ~{n_loaded}/{n_total} ({load_percentage:.1f}%)")
                print (f"   📦 Отсутствующих ключей: {len(missing)}, лишних ключей: {len(unexpected)}")

                if len (missing )>0 :
                    print (f"   ⚠️ Отсутствуют ключи (первые 5): {list(missing)[:5]}")
                if len (unexpected )>0 :
                    print (f"   ⚠️ Лишние ключи (первые 5): {list(unexpected)[:5]}")

                if load_percentage <50 :
                    print (f"   ❌ ВНИМАНИЕ: Загружено менее 50% параметров! Модель может работать некорректно.")
                    print (f"   ❌ Рекомендуется проверить совместимость checkpoint с выбранной архитектурой {fn_name}")

            except Exception as e :
                print (f"   ⚠️ Ошибка при подсчете статистики загрузки: {e}")
        else :
            print (f"🔧 DINOv3: используем предобученные веса из hub")
            self .model =backbone_fn (pretrained =True ).to (self .device )

        self .model .eval ().float ()

        self .transform =T .Compose ([
        T .Resize (224 ),
        T .CenterCrop (224 ),
        T .ToTensor (),
        T .Normalize (mean =[0.485 ,0.456 ,0.406 ],std =[0.229 ,0.224 ,0.225 ]),
        ])

    def _compute_dtype (self ):

        if self .device .type =="cuda"and hasattr (self ,'half')and self .half :
            return torch .float16
        return torch .float32

    @staticmethod
    def _pick_feature (feats :Any ,pooling :str ="cls")->torch .Tensor :

        if isinstance (feats ,dict ):
            if 'x_norm_clstoken'in feats :
                z =feats ['x_norm_clstoken']
            elif 'x_prenorm'in feats :
                z =feats ['x_prenorm']
                if z .ndim ==3 :
                    z =z [:,0 ,:]
            elif 'x_norm_patchtokens'in feats :
                z =feats ['x_norm_patchtokens']
                if z .ndim ==3 :
                    z =z [:,1 :,:].mean (dim =1 )if z .shape [1 ]>1 else z [:,0 ,:]
            elif 'x'in feats :
                z =feats ['x']
                if z .ndim ==4 :
                    z =z .mean (dim =(2 ,3 ))
                elif z .ndim ==3 :
                    z =z [:,0 ,:]if pooling =="cls"else z [:,1 :,:].mean (dim =1 )
                elif z .ndim ==2 :
                    pass
                else :
                    z =z .flatten (1 )
            else :

                z =None
                for v in feats .values ():
                    if isinstance (v ,torch .Tensor ):
                        vv =v
                        if vv .ndim ==4 :
                            vv =vv .mean (dim =(2 ,3 ))
                        elif vv .ndim ==3 :
                            vv =vv [:,0 ,:]
                        elif vv .ndim ==2 :
                            pass
                        else :
                            vv =vv .flatten (1 )
                        z =vv if z is None else (z +vv )/2
                if z is None :
                    raise ValueError ("Unsupported feats dict content")
        else :
            z =feats
            if z .ndim ==4 :
                z =z .mean (dim =(2 ,3 ))
            elif z .ndim ==3 :
                z =z [:,0 ,:]
            elif z .ndim ==2 :
                pass
            else :
                z =z .flatten (1 )
        return z

    def _find_cached_dinov3_ckpt (self ,fn_name :str )->Optional [Path ]:

        try :
            import torch .hub as th
            hub_dir =Path (th .get_dir ())/"checkpoints"
        except Exception :
            hub_dir =Path .home ()/".cache"/"torch"/"hub"/"checkpoints"
        if not hub_dir .exists ():
            return None

        exact =sorted (hub_dir .glob (f"{fn_name}_pretrain*.pth"))
        if exact :
            return exact [-1 ]

        short =fn_name .replace ("dinov3_","")
        variants =[]
        variants +=sorted (hub_dir .glob (f"dinov3_{short}_pretrain*.pth"))
        variants +=sorted (hub_dir .glob (f"*{fn_name}*pretrain*.pth"))
        variants +=sorted (hub_dir .glob (f"*{short}*pretrain*.pth"))
        return variants [-1 ]if variants else None

    def _find_any_cached_dinov3_ckpt (self )->tuple [Optional [str ],Optional [Path ]]:

        try :
            import torch .hub as th
            hub_dir =Path (th .get_dir ())/"checkpoints"
        except Exception :
            hub_dir =Path .home ()/".cache"/"torch"/"hub"/"checkpoints"
        if not hub_dir .exists ():
            return (None ,None )

        for p in sorted (hub_dir .glob ("dinov3_*_pretrain*.pth")):
            m =re .search (r"(dinov3_[^/\\]+?)_pretrain",p .name )
            if m :
                arch =m .group (1 )
                if hasattr (dino_backbones ,arch ):
                    return (arch ,p )
        return (None ,None )

    @staticmethod
    def _pick_feature (feats :Any ,pooling :str ="cls")->torch .Tensor :

        if isinstance (feats ,dict ):
            if 'x_norm_clstoken'in feats :
                z =feats ['x_norm_clstoken']
            elif 'x_prenorm'in feats :
                z =feats ['x_prenorm']
                if z .ndim ==3 :
                    z =z [:,0 ,:]
            elif 'x_norm_patchtokens'in feats :
                z =feats ['x_norm_patchtokens']
                if z .ndim ==3 :
                    z =z [:,1 :,:].mean (dim =1 )if z .shape [1 ]>1 else z [:,0 ,:]
            elif 'x'in feats :
                z =feats ['x']
                if z .ndim ==4 :
                    z =z .mean (dim =(2 ,3 ))
                elif z .ndim ==3 :
                    z =z [:,0 ,:]if pooling =="cls"else z [:,1 :,:].mean (dim =1 )
                elif z .ndim ==2 :
                    pass
                else :
                    z =z .flatten (1 )
            else :

                z =None
                for v in feats .values ():
                    if isinstance (v ,torch .Tensor ):
                        vv =v
                        if vv .ndim ==4 :
                            vv =vv .mean (dim =(2 ,3 ))
                        elif vv .ndim ==3 :
                            vv =vv [:,0 ,:]
                        elif vv .ndim ==2 :
                            pass
                        else :
                            vv =vv .flatten (1 )
                        z =vv if z is None else (z +vv )/2
                if z is None :
                    raise ValueError ("Unsupported feats dict content")
        else :
            z =feats
            if z .ndim ==4 :
                z =z .mean (dim =(2 ,3 ))
            elif z .ndim ==3 :
                z =z [:,0 ,:]
            elif z .ndim ==2 :
                pass
            else :
                z =z .flatten (1 )
        return z

    @torch .no_grad ()
    def _prep (self ,img_pil ):
        x =_to_pil_any (img_pil )
        return self .transform (x ).unsqueeze (0 ).to (self .device )

    @torch .no_grad ()
    def _prep_tensor (self ,img_pil :Image .Image )->torch .Tensor :
        x =_to_pil_any (img_pil )
        return self .transform (x ).unsqueeze (0 ).to (self .device )

    @torch .no_grad ()
    def _forward_safe (self ,x :torch .Tensor )->torch .Tensor :
        dtype =self ._compute_dtype ()
        x =x .to (self .device ,non_blocking =True )
        try :
            with torch .autocast (device_type =self .device .type ,dtype =dtype ,enabled =(dtype ==torch .float16 )):
                feats =self .model (x )
        except Exception :
            feats =self .model (x )
        return self ._pick_feature (feats ,pooling =self .pooling )

    @torch .no_grad ()
    def _forward_features_safe (self ,x :torch .Tensor )->torch .Tensor :
        dtype =self ._compute_dtype ()
        x =x .to (self .device ,non_blocking =True )
        try :
            with torch .autocast (device_type =self .device .type ,dtype =dtype ,enabled =(dtype ==torch .float16 )):
                feats =self .model .forward_features (x )
        except Exception :
            feats =self .model .forward_features (x )
        return self ._pick_feature (feats ,pooling =self .pooling )

    @torch .no_grad ()
    def extract_features (self ,x :torch .Tensor )->torch .Tensor :

        dtype =self ._compute_dtype ()
        x =x .to (self .device ,non_blocking =True )
        try :
            with torch .autocast (device_type =self .device .type ,dtype =dtype ,enabled =(dtype ==torch .float16 )):
                feats =self .model .forward_features (x )
        except Exception :
            feats =self .model .forward_features (x )

        if isinstance (feats ,dict ):
            if 'x_prenorm'in feats :
                return feats ['x_prenorm']
            elif 'x'in feats :
                return feats ['x']
            else :

                for v in feats .values ():
                    if isinstance (v ,torch .Tensor )and v .ndim ==3 :
                        return v
                raise ValueError ("Не удалось найти подходящие признаки в dict")
        else :
            return feats

    @torch .no_grad ()
    def extract_cls_and_patches (self ,x :torch .Tensor ):

        dtype =self ._compute_dtype ()
        x =x .to (self .device ,non_blocking =True )
        try :
            with torch .autocast (device_type =self .device .type ,dtype =dtype ,enabled =(dtype ==torch .float16 )):
                feats =self .model .forward_features (x )
        except Exception :
            feats =self .model .forward_features (x )

        if isinstance (feats ,dict )and ('x_norm_clstoken'in feats and 'x_norm_patchtokens'in feats ):
            cls =feats ['x_norm_clstoken']
            patches =feats ['x_norm_patchtokens']
            return cls ,patches

        if isinstance (feats ,dict ):
            if 'x_prenorm'in feats :
                tokens =feats ['x_prenorm']
            elif 'x'in feats and isinstance (feats ['x'],torch .Tensor )and feats ['x'].ndim ==3 :
                tokens =feats ['x']
            else :
                tokens =None
                for v in feats .values ():
                    if isinstance (v ,torch .Tensor )and v .ndim ==3 :
                        tokens =v
                        break
                if tokens is None :
                    raise ValueError ("Не удалось извлечь последовательность токенов вида [B,T,D] из выходов DINOv3")
        else :
            tokens =feats

        cls =tokens [:,0 ,:]
        seq_no_cls =tokens [:,1 :,:]

        patch_size =None
        if hasattr (self .model ,'patch_embed')and hasattr (self .model .patch_embed ,'patch_size'):
            p =self .model .patch_embed .patch_size
            if isinstance (p ,(tuple ,list )):
                patch_size =int (p [0 ])
            else :
                patch_size =int (p )
        if not patch_size or patch_size <=0 :
            patch_size =16

        H ,W =int (x .shape [-2 ]),int (x .shape [-1 ])
        grid_h ,grid_w =H //patch_size ,W //patch_size
        num_patches =int (grid_h *grid_w )

        if seq_no_cls .shape [1 ]>=num_patches :

            patches =seq_no_cls [:,:num_patches ,:]
        else :
            patches =seq_no_cls

        t =patches .shape [1 ]
        side =int (math .sqrt (t ))
        patches =patches [:,:side *side ,:]

        return cls ,patches

    @torch .no_grad ()
    def encode (self ,img_pil )->np .ndarray :
        x =self ._prep (img_pil )
        z =self ._forward_safe (x )
        return z .detach ().cpu ().numpy ()[0 ]
