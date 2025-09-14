
import os
import glob
import numpy as np
from PIL import Image
import cv2
import torch

from .dinov3_encoder import DinoV3Encoder

def _to_pil_any (x ):

    if isinstance (x ,Image .Image ):
        return x
    if isinstance (x ,str ):

        return Image .open (x ).convert ("RGB")
    if isinstance (x ,np .ndarray ):
        if x .ndim ==2 :
            x =np .stack ([x ,x ,x ],axis =-1 )
        if x .dtype !=np .uint8 :
            x =np .clip (x ,0 ,255 ).astype (np .uint8 )
        return Image .fromarray (x ,mode ="RGB")
    if torch .is_tensor (x ):
        t =x .detach ().cpu ()
        if t .ndim ==3 and t .shape [0 ]in (1 ,3 ):
            t =t .permute (1 ,2 ,0 ).contiguous ()
        t =t .numpy ()
        return _to_pil_any (t )

    raise TypeError (f"Cannot convert type {type(x)} to PIL.Image")

def _l2n (x :np .ndarray ,axis :int =-1 ,eps :float =1e-12 )->np .ndarray :
    n =np .linalg .norm (x ,axis =axis ,keepdims =True )
    return x /np .maximum (n ,eps )

def _iter_images (root ):
    exts =("*.jpg","*.jpeg","*.png","*.bmp","*.webp","*.tif","*.tiff")
    paths =[]
    for e in exts :
        paths .extend (glob .glob (os .path .join (root ,e )))
    return sorted (paths )

def extract_features_from_masks (encoder ,masked_crops ):
    feats =[]

    p =next (encoder .model .parameters ())
    dev ,dt =p .device ,p .dtype

    for im in masked_crops :
        x =encoder .transform (im ).unsqueeze (0 )
        x =x .to (device =dev ,dtype =dt ,non_blocking =True )
        with torch .no_grad ():
            f =encoder .model .forward_features (x )

            if hasattr (encoder .model ,"forward_head"):
                out =encoder .model .forward_head (f ,pre_logits =True )
            else :
                out =f

            if isinstance (out ,dict ):
                if 'x_norm_clstoken'in out :
                    v =out ['x_norm_clstoken'][0 ]
                elif 'x_prenorm'in out :
                    v =out ['x_prenorm'][0 ,0 ]
                elif 'x_norm_patchtokens'in out :
                    v =out ['x_norm_patchtokens'][0 ].mean (dim =0 )
                elif 'x'in out :
                    x_out =out ['x']
                    if x_out .ndim ==4 :
                        v =x_out .mean (dim =(2 ,3 ))[0 ]
                    elif x_out .ndim ==3 :
                        v =x_out [0 ,0 ]if encoder .pooling !="mean"else x_out [0 ,1 :].mean (dim =0 )
                    else :
                        v =x_out [0 ]
                else :

                    v =next (iter (out .values ()))[0 ]
                    if v .ndim >1 :
                        v =v .flatten ()
            else :

                if out .ndim ==2 :
                    v =out [0 ]
                elif out .ndim ==3 :
                    if encoder .pooling =="mean"and out .shape [1 ]>1 :
                        v =out [0 ,1 :].mean (dim =0 )
                    else :
                        v =out [0 ,0 ]
                elif out .ndim ==4 :
                    v =out .mean (dim =(2 ,3 ))[0 ]
                else :
                    v =out .flatten (1 )[0 ]

            v =torch .nn .functional .normalize (v .float (),dim =0 )
            feats .append (v .cpu ().numpy ().astype (np .float32 ))

    if not feats :
        return np .zeros ((0 ,1 ),dtype =np .float32 )
    return np .stack (feats ,axis =0 )

class EmbeddingExtractor :

    def __init__ (self ,backbone_name :str ='vitb16',device :str ='cpu',ckpt_path :str =None ):

        self.encoder = DinoV3Encoder(
            backbone_name=backbone_name,
            device=device,
            ckpt_path=ckpt_path,
        )

    def _safe_stack (self ,arrs ,axis =0 ):
        if not arrs:
            return np .zeros ((0 ,768 ),dtype =np .float32 )
        return np .stack (arrs ,axis =axis )

    def _filter_bad (self ,mat :np .ndarray ,cls_name :str =None ,kind :str =""):

        if mat .size ==0 :
            return mat
        bad =~np .isfinite (mat ).all (axis =1 )
        norms =np .linalg .norm (mat ,axis =1 )
        bad |=(norms <1e-6 )
        if bad .any ():
            print (f"   ⚠️ Dropped {bad.sum()} bad {kind} embeddings"
            +(f" in class '{cls_name}'"if cls_name else ""))
        return mat [~bad ]

    def _encode_pil_list (self ,pil_list ):

        out =[]
        for i ,im in enumerate (pil_list ):
            try :
                v =self .encoder .encode (im )
                if not np .isfinite (v ).all ():
                    v =np .zeros_like (v ,dtype =np .float32 )
                out .append (v .astype (np .float32 ))
            except Exception as e :
                print (f"   ⚠️ Pos/Neg encode failed ({i}): {e}")
        if not out :
            return np .zeros ((0 ,768 ),dtype =np .float32 )
        M =self ._safe_stack (out ,axis =0 )
        M =self ._filter_bad (M ,kind ="example")
        return M

    def _encode_mask_roi_with_dinov3 (self ,image_np :np .ndarray ,mask_bool :np .ndarray ,pad :int =8 )->np .ndarray :

        H ,W ,_ =image_np .shape
        mb =mask_bool .astype (bool )

        if mb .sum ()==0 :
            return np .zeros ((getattr (self .encoder ,"feat_dim",768 ),),dtype =np .float32 )

        ys ,xs =np .where (mb )
        y0 ,y1 =max (0 ,ys .min ()-pad ),min (H ,ys .max ()+1 +pad )
        x0 ,x1 =max (0 ,xs .min ()-pad ),min (W ,xs .max ()+1 +pad )

        crop =image_np [y0 :y1 ,x0 :x1 ].copy ()
        m_crop =mb [y0 :y1 ,x0 :x1 ]

        mean_color =crop [m_crop ].mean (axis =0 )if m_crop .any ()else np .array ([128 ,128 ,128 ],dtype =np .float32 )
        bg =np .tile (mean_color .reshape (1 ,1 ,3 ),(crop .shape [0 ],crop .shape [1 ],1 ))
        crop =np .where (m_crop [...,None ],crop ,bg ).astype (np .uint8 )

        try :
            pil =Image .fromarray (crop ,mode ="RGB")
            v =self .encoder .encode (pil )

            if not np .isfinite (v ).all ():
                v =np .nan_to_num (v ,nan =0.0 ,posinf =0.0 ,neginf =0.0 ).astype (np .float32 )

            n =np .linalg .norm (v )
            if not np .isfinite (n )or n <1e-6 :
                v =np .zeros_like (v ,dtype =np .float32 )

        except Exception :
            v =np .zeros ((getattr (self .encoder ,"feat_dim",768 ),),dtype =np .float32 )
        return v .astype (np .float32 )

    def _encode_masks_batch_optimized (self ,image_np :np .ndarray ,masks :list ,batch_size :int =32 )->np .ndarray :

        if not masks :
            return np .zeros ((0 ,getattr (self .encoder ,"feat_dim",768 )),dtype =np .float32 )

        all_embeddings =[]

        for i in range (0 ,len (masks ),batch_size ):
            batch_masks =masks [i :i +batch_size ]
            batch_crops =[]

            for mask_data in batch_masks :
                if isinstance (mask_data ,dict )and 'segmentation'in mask_data :
                    mask_bool =mask_data ['segmentation'].astype (bool )
                    if mask_bool .sum ()==0 :
                        continue

                    ys ,xs =np .where (mask_bool )
                    y0 ,y1 =max (0 ,ys .min ()-8 ),min (image_np .shape [0 ],ys .max ()+9 )
                    x0 ,x1 =max (0 ,xs .min ()-8 ),min (image_np .shape [1 ],xs .max ()+9 )

                    crop =image_np [y0 :y1 ,x0 :x1 ].copy ()
                    m_crop =mask_bool [y0 :y1 ,x0 :x1 ]

                    if m_crop .any ():
                        mean_color =crop [m_crop ].mean (axis =0 )
                    else :
                        mean_color =np .array ([128 ,128 ,128 ],dtype =np .float32 )

                    crop =np .where (m_crop [...,None ],crop ,mean_color ).astype (np .uint8 )
                    batch_crops .append (Image .fromarray (crop ,mode ="RGB"))

            if batch_crops :
                batch_embeddings =extract_features_from_masks (self .encoder ,batch_crops )
                all_embeddings .append (batch_embeddings )

        if not all_embeddings :
            return np .zeros ((0 ,getattr (self .encoder ,"feat_dim",768 )),dtype =np .float32 )

        return np .vstack (all_embeddings )

    def extract_mask_embeddings (self ,image_pil ,masks ):

        if not masks :
            D =getattr (self .encoder ,"feat_dim",768 )
            return np .zeros ((0 ,D ),dtype =np .float32 )

        image_np =np .array (image_pil )

        sam_masks =[]
        other_masks =[]

        for m in masks :
            if isinstance (m ,dict )and 'segmentation'in m :
                sam_masks .append (m )
            else :
                other_masks .append (m )

        vecs =[]

        if sam_masks :
            try :
                batch_embeddings =self ._encode_masks_batch_optimized (image_np ,sam_masks ,batch_size =32 )
                if batch_embeddings .size >0 :
                    vecs .append (batch_embeddings )
            except Exception as e :
                print (f"   ⚠️ Batch processing failed, falling back to individual processing: {e}")

                for m in sam_masks :
                    try :
                        seg =m ['segmentation']
                        if isinstance (seg ,np .ndarray ):
                            v =self ._encode_mask_roi_with_dinov3 (image_np ,seg .astype (bool ),pad =8 )
                            vecs .append (v .astype (np .float32 ))
                    except Exception :
                        continue

        for m in other_masks :
            try :

                if hasattr (m ,"crop"):
                    crop_any =m .crop (image_pil )
                    crop_pil =_to_pil_any (crop_any )
                    v =self .encoder .encode (crop_pil )
                    v =np .nan_to_num (v ,nan =0.0 ,posinf =0.0 ,neginf =0.0 ).astype (np .float32 )
                    vecs .append (v )

                elif isinstance (m ,(str ,np .ndarray ,torch .Tensor ,Image .Image )):
                    crop_pil =_to_pil_any (m )
                    v =self .encoder .encode (crop_pil )
                    v =np .nan_to_num (v ,nan =0.0 ,posinf =0.0 ,neginf =0.0 ).astype (np .float32 )
                    vecs .append (v )

                elif isinstance (m ,(tuple ,list ))and len (m )==2 :
                    crop_pil =_to_pil_any (m [0 ])
                    v =self .encoder .encode (crop_pil )
                    v =np .nan_to_num (v ,nan =0.0 ,posinf =0.0 ,neginf =0.0 ).astype (np .float32 )
                    vecs .append (v )

                elif isinstance (m ,dict ):
                    crop_any =m .get ("image",m .get ("crop",None ))
                    if crop_any is not None :
                        crop_pil =_to_pil_any (crop_any )
                        v =self .encoder .encode (crop_pil )
                        v =np .nan_to_num (v ,nan =0.0 ,posinf =0.0 ,neginf =0.0 ).astype (np .float32 )
                        vecs .append (v )
            except Exception :
                continue

        if not vecs :
            D =getattr (self .encoder ,"feat_dim",768 )
            return np .zeros ((0 ,D ),dtype =np .float32 )

        if len (vecs )==1 and vecs [0 ].ndim ==2 :
            V =vecs [0 ]
        else :

            processed_vecs =[]
            for v in vecs :
                if isinstance (v ,np .ndarray ):
                    if v .ndim ==1 :
                        processed_vecs .append (v [None ,:])
                    else :
                        processed_vecs .append (v )
            if processed_vecs :
                V =np .vstack (processed_vecs )
            else :
                D =getattr (self .encoder ,"feat_dim",768 )
                return np .zeros ((0 ,D ),dtype =np .float32 )

        finite =np .isfinite (V ).all (axis =1 )
        norms =np .linalg .norm (V ,axis =1 )
        good =finite &(norms >=1e-6 )

        if not good .all ():
            dropped_count =(~good ).sum ()
            if dropped_count >0 :
                print (f"   ⚠️ Dropped {dropped_count} bad mask embeddings")

        return V [good ].astype (np .float32 )

    def build_queries_multiclass(self, pos_dict, neg_list):
        q_pos = {}
        for cls, pil_list in pos_dict.items():
            M = self._encode_pil_list(pil_list)
            M = self._filter_bad(M, cls_name=cls, kind="q_pos")
            if M.shape[0] > 0:
                q_pos[cls] = M
            else :
                print(f"   ⚠️ Класс '{cls}' пуст после фильтрации — пропущен")

        q_neg = self._encode_pil_list(neg_list) if neg_list else np.zeros((0, 768), dtype=np.float32)
        q_neg = self._filter_bad(q_neg, kind="q_neg")

        if not q_pos:
            print("   ❌ Нет валидных классов с эмбеддингами после фильтрации.")
        return q_pos, q_neg
