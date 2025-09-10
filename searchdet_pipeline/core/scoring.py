
from __future__ import annotations
import numpy as np
from typing import Dict ,List ,Tuple ,Any
from ..utils .config import ScoringConfig

def _l2n (x :np .ndarray ,axis :int =-1 ,eps :float =1e-12 )->np .ndarray :
    n =np .linalg .norm (x ,axis =axis ,keepdims =True )
    return x /np .maximum (n ,eps )

def _aggregate (sim_mat :np .ndarray ,mode :str ="max",topk :int =3 )->np .ndarray :
    if sim_mat .size ==0 :
        return np .full ((sim_mat .shape [0 ]if sim_mat .ndim else 0 ,),-1.0 ,dtype =np .float32 )

    if mode =="max":
        return sim_mat .max (axis =1 )
    elif mode =="mean":
        return sim_mat .mean (axis =1 )
    elif mode =="mean_topk":
        k =min (topk ,sim_mat .shape [1 ])
        if k <=0 :
            return np .full ((sim_mat .shape [0 ],),-1.0 ,dtype =np .float32 )
        part =np .partition (sim_mat ,-k ,axis =1 )[:,-k :]
        return part .mean (axis =1 )
    else :
        return sim_mat .max (axis =1 )

def _to_matrix (X :np .ndarray ,D :int |None =None )->np .ndarray :
    X =np .asarray (X ,dtype =np .float32 )
    if X .size ==0 :
        return X .reshape (0 ,(D if D is not None else 0 ))
    if X .ndim ==1 :
        X =X .reshape (1 ,-1 )
    elif X .ndim >2 :
        X =X .reshape (X .shape [0 ],-1 )
    if D is not None and X .shape [1 ]!=D :
        raise ValueError (f"Dim mismatch: expected D={D}, got {X.shape}")
    return X

def _cosine_matrix (A :np .ndarray ,B :np .ndarray )->np .ndarray :

    A =np .asarray (A ,dtype =np .float32 )
    B =np .asarray (B ,dtype =np .float32 )

    if A .ndim ==1 :A =A .reshape (1 ,-1 )
    elif A .ndim >2 :A =A .reshape (A .shape [0 ],-1 )
    if B .ndim ==1 :B =B .reshape (1 ,-1 )
    elif B .ndim >2 :B =B .reshape (B .shape [0 ],-1 )

    if A .size ==0 or B .size ==0 :
        return np .zeros ((A .shape [0 ],B .shape [0 ]),dtype =np .float32 )
    if A .shape [1 ]!=B .shape [1 ]:
        raise ValueError (f"Dim mismatch in _cosine_matrix: A{A.shape} vs B{B.shape}")

    A =A /(np .linalg .norm (A ,axis =1 ,keepdims =True )+1e-12 )
    B =B /(np .linalg .norm (B ,axis =1 ,keepdims =True )+1e-12 )

    sims =A @B .T
    sims =np .clip (sims ,-1.0 ,1.0 ).astype (np .float32 )
    return sims

def _l2n_torch (x :torch .Tensor )->np .ndarray :
    return x .detach ().cpu ().numpy ()

def _cos (A :np .ndarray ,B :np .ndarray ):
    if A .size ==0 or B .size ==0 :
        return np .full ((A .shape [0 ],B .shape [0 ]),-1.0 ,dtype =np .float32 )

    A =A .astype (np .float32 ,copy =False )
    B =B .astype (np .float32 ,copy =False )

    a_norm =np .linalg .norm (A ,axis =1 ,keepdims =True )
    b_norm =np .linalg .norm (B ,axis =1 ,keepdims =True ).T

    a_zero =(a_norm <1e-6 )
    b_zero =(b_norm <1e-6 )

    a_norm_safe =np .where (a_zero ,1.0 ,a_norm )
    b_norm_safe =np .where (b_zero ,1.0 ,b_norm )

    S =(A @B .T )/(a_norm_safe *b_norm_safe )
    S =np .clip (S ,-1.0 ,1.0 )
    S =np .nan_to_num (S ,nan =-1.0 ,posinf =-1.0 ,neginf =-1.0 )

    if a_zero .any ():
        S [a_zero [:,0 ],:]=-1.0
    if b_zero .any ():
        S [:,b_zero [0 ,:]]=-1.0
    return S .astype (np .float32 ,copy =False )

def _agg_scores (scores :np .ndarray ,aggregation :str ,topk :int =3 )->np .ndarray :
    if aggregation =='mean':
        return np .mean (scores ,axis =1 )
    elif aggregation =='max':
        return np .max (scores ,axis =1 )
    elif aggregation =='topk':
        k =min (topk ,scores .shape [1 ])

        return np .mean (np .partition (scores ,-k ,axis =1 )[:,-k :],axis =1 )
    else :
        raise ValueError (f'Unknown aggregation: {aggregation}')

def _safe_get_int (d :Dict [str ,Any ],key :str ,default :int )->int :
    v =(d or {}).get (key ,default )
    if v is None :
        return default
    try :
        return int (v )
    except Exception :
        return default

def _safe_get_float (d :Dict [str ,Any ],key :str ,default :float )->float :
    v =(d or {}).get (key ,default )
    if v is None :
        return default
    try :
        return float (v )
    except Exception :
        return default

def _safe_get_bool (d :Dict [str ,Any ],key :str ,default :bool )->bool :
    v =(d or {}).get (key ,default )
    if v is None :
        return default
    if isinstance (v ,bool ):
        return v

    if isinstance (v ,str ):
        lv =v .strip ().lower ()
        if lv in ("1","true","yes","y","on"):return True
        if lv in ("0","false","no","n","off"):return False

        return default
    try :
        return bool (v )
    except Exception :
        return default

class ScoreCalculator :

    def __init__ (self ,params :Dict [str ,Any ]|None =None ,config :ScoringConfig |None =None ):

        if config is not None :
            self .config =config
        else :
            if params is None :
                params ={}
            self .config =ScoringConfig (
            min_pos_score =_safe_get_float (params ,'min_positive_score',_safe_get_float (params ,'min_pos_score',0.62 )),
            decision_threshold =_safe_get_float (params ,'decision_threshold',0.06 ),
            class_separation =_safe_get_float (params ,'class_separation',0.04 ),
            neg_cap =_safe_get_float (params ,'neg_cap',0.90 ),
            topk =_safe_get_int (params ,'topk',5 ),
            consensus_k =_safe_get_int (params ,'consensus_k',0 ),
            consensus_thr =_safe_get_float (params ,'consensus_thr',0.45 ),
            adaptive_ratio =_safe_get_float (params ,'adaptive_ratio',0.85 ),
            adaptive_diff_floor =_safe_get_float (params ,'adaptive_diff_floor',0.04 ),
            adaptive_trigger_pos_range =_safe_get_float (params ,'adaptive_trigger_pos_range',0.20 ),
            adaptive_trigger_neg_range =_safe_get_float (params ,'adaptive_trigger_neg_range',0.20 ),
            margin =_safe_get_float (params ,'score_margin',0.00 ),
            ratio =_safe_get_float (params ,'score_ratio',1.01 ),
            confidence =_safe_get_float (params ,'score_confidence',0.50 ),
            allow_unknown =_safe_get_bool (params ,'allow_unknown',True ),
            verbose =_safe_get_bool (params ,'verbose',True )
            )

        self .min_pos_score =self .config .min_pos_score
        self .decision_threshold =self .config .decision_threshold
        self .class_separation =self .config .class_separation
        self .neg_cap =self .config .neg_cap
        self .topk =self .config .topk
        self .consensus_k =self .config .consensus_k
        self .consensus_thr_01 =self .config .consensus_thr
        self .adaptive_ratio =self .config .adaptive_ratio
        self .adaptive_diff_floor =self .config .adaptive_diff_floor
        self .adaptive_trigger_pos_range =self .config .adaptive_trigger_pos_range
        self .adaptive_trigger_neg_range =self .config .adaptive_trigger_neg_range
        self .margin =self .config .margin
        self .ratio =self .config .ratio
        self .confidence_min =self .config .confidence
        self .allow_unknown =self .config .allow_unknown
        self .verbose =self .config .verbose

        self .consensus_thr_cos =float (2 *self .consensus_thr_01 -1.0 )

        self .pos_agg =str ((((params or {}).get ('positive_aggregation'))or ((params or {}).get ('pos_agg'))or 'max')).lower ()
        if self .verbose :
            print (f"   ⚙️ POS_AGG_MODE = {self.pos_agg}")

    def _aggregate_positive (self ,sims_pos :np .ndarray )->np .ndarray :

        if sims_pos .size ==0 or sims_pos .shape [1 ]==0 :
            return np .zeros ((sims_pos .shape [0 ]if sims_pos .ndim >0 else 0 ,),dtype =np .float32 )

        mode =self .pos_agg
        if mode in ("max","top1"):
            pos =sims_pos .max (axis =1 )
        elif mode in ("mean_topk","topk"):
            k =min (self .topk ,sims_pos .shape [1 ])if sims_pos .shape [1 ]>0 else 1
            if k >1 :
                part =np .partition (sims_pos ,-k ,axis =1 )[:,-k :]
                pos =part .mean (axis =1 )
            else :
                pos =sims_pos .max (axis =1 )
        elif mode =="mean":
            pos =sims_pos .mean (axis =1 )
        else :

            maxv =sims_pos .max (axis =1 )
            k =min (self .topk ,sims_pos .shape [1 ])if sims_pos .shape [1 ]>0 else 1
            if k >1 :
                part =np .partition (sims_pos ,-k ,axis =1 )[:,-k :]
                topk_mean =part .mean (axis =1 )
            else :
                topk_mean =maxv
            pos =0.7 *maxv +0.3 *topk_mean

        return np .clip (pos .astype (np .float32 ),-1.0 ,1.0 )

    def _aggregate_negative (self ,sims_neg :np .ndarray )->np .ndarray :

        if sims_neg .size ==0 or sims_neg .shape [1 ]==0 :
            return np .zeros ((sims_neg .shape [0 ]if sims_neg .ndim >0 else 0 ,),dtype =np .float32 )
        neg_raw =np .max (sims_neg ,axis =1 )
        neg_raw =np .clip (neg_raw ,-1.0 ,1.0 ).astype (np .float32 )
        return neg_raw

    def _consensus_count (self ,sims_pos_cls :np .ndarray )->np .ndarray :

        if sims_pos_cls .size ==0 :
            return np .zeros ((0 ,),dtype =np .int32 )
        return (sims_pos_cls >=self .consensus_thr_cos ).sum (axis =1 ).astype (np .int32 )

    def _aggregate_pos (self ,sims_vec :np .ndarray ):

        if sims_vec .size ==0 :
            return -1.0
        s =np .sort (sims_vec )[::-1 ]
        if self .pos_agg =="max":
            return float (s [0 ])
        if self .pos_agg =="mean":
            return float (s .mean ())
        if self .pos_agg =="mean_topk"and self .topk and self .topk >0 :
            k =min (self .topk ,s .shape [0 ])
            return float (s [:k ].mean ())
        return float (s [0 ])

    def score_multiclass (self ,mask_vecs :np .ndarray ,q_pos :dict ,q_neg :np .ndarray ,online_negatives =None ):

        M =mask_vecs .shape [0 ]
        decisions =[]
        if M ==0 or not q_pos :
            return decisions

        if not isinstance (q_neg ,np .ndarray ):
            print (f"   ⚠️ ПРЕДУПРЕЖДЕНИЕ: q_neg не является numpy массивом: {type(q_neg)}")
            try :
                q_neg =np .array (q_neg ,dtype =np .float32 )
                if q_neg .ndim ==1 :
                    q_neg =q_neg [None ,:]
                print (f"   ✅ Исправлено: q_neg теперь имеет форму {q_neg.shape}")
            except Exception as e :
                print (f"   ❌ Ошибка при конвертации q_neg: {e}")
                q_neg =np .zeros ((0 ,mask_vecs .shape [1 ]),dtype =np .float32 )

        neg_scores =None
        if q_neg .size >0 :
            neg_scores =_cos (mask_vecs ,q_neg ).max (axis =1 )
        else :

            neg_scores =np .zeros ((M ,),dtype =np .float32 )

        for i in range (M ):
            mv =mask_vecs [i :i +1 ]
            best_cls =None
            best_pos =-1.0

            for cls ,Q in q_pos .items ():
                sims =_cos (mv ,Q )[0 ]
                pos =self ._aggregate_pos (sims )
                if pos >best_pos :
                    best_pos =pos
                    best_cls =cls

            neg =float (neg_scores [i ])
            diff =best_pos -neg
            accepted =(best_pos >=self .min_pos_score )and (diff >=self .decision_threshold )

            decisions .append ({
            "mask_index":i ,
            "class":best_cls ,
            "pos":float (best_pos ),
            "neg":float (neg ),
            "diff":float (diff ),
            "accepted":bool (accepted ),
            "confidence":float (best_pos ),
            })

        debug ={
        "num_masks":M ,
        "num_classes":len (q_pos ),
        "has_negatives":(hasattr (q_neg ,'size')and q_neg .size >0 )if q_neg is not None else False
        }
        return decisions ,debug

    def score_and_decide (self ,mask_vecs :np .ndarray ,q_pos :np .ndarray ,q_neg :np .ndarray
    )->Tuple [List [int ],np .ndarray ,np .ndarray ]:
        mask_vecs =_to_matrix (mask_vecs )
        q_pos =_to_matrix (q_pos ,mask_vecs .shape [1 ])
        q_neg =_to_matrix (q_neg ,mask_vecs .shape [1 ])

        sims_pos =_cosine_matrix (mask_vecs ,q_pos )
        pos_scores =self ._aggregate_positive (sims_pos )

        if q_neg .shape [0 ]>0 :
            sims_neg =_cosine_matrix (mask_vecs ,q_neg )
            neg_raw =self ._aggregate_negative (sims_neg )
        else :
            neg_raw =np .zeros (mask_vecs .shape [0 ],dtype =np .float32 )

        neg_for_decision =np .clip (np .maximum (neg_raw ,0.0 ),0.0 ,
        (self .neg_cap if self .neg_cap is not None else 1.0 )).astype (np .float32 )

        accepted =[]
        eps =1e-8
        for i ,(p ,n )in enumerate (zip (pos_scores ,neg_for_decision )):
            diff =p -n -self .margin
            ratio_ok =(p /(n +eps ))>=float (self .ratio )if n >0 else (p >=self .min_pos_score )
            if (p >=self .min_pos_score )and ((diff >=self .decision_threshold )or ratio_ok )and (p >n ):
                accepted .append (i )

        return accepted ,pos_scores ,neg_for_decision

def score_multiclass (
mask_vecs :np .ndarray ,
q_pos :dict [str ,np .ndarray ],
q_neg :np .ndarray |None ,
*,
pos_agg :str ="max",
topk :int =3 ,
min_pos_score :float =0.70 ,
decision_threshold :float =0.10 ,
clamp_neg_to_zero :bool =True ,
verbose :bool =True ,
):

    if mask_vecs .ndim ==1 :
        mask_vecs =mask_vecs [None ,:]
    mask_vecs =mask_vecs .astype (np .float32 ,copy =False )
    mask_vecs =_l2n (mask_vecs ,axis =1 )
    M ,D =mask_vecs .shape

    try :
        print (f"   🔍 DEBUG: q_neg тип = {type(q_neg)}, значение = {q_neg}")
        if q_neg is not None and not isinstance (q_neg ,np .ndarray ):
            print (f"   🔍 DEBUG: Конвертируем q_neg из {type(q_neg)} в numpy массив")
            q_neg =np .array (q_neg ,dtype =np .float32 )
            if q_neg .ndim ==1 :
                q_neg =q_neg [None ,:]
            print (f"   🔍 DEBUG: q_neg после конвертации: форма = {q_neg.shape}")
    except Exception as e :
        print (f"   ❌ DEBUG: Ошибка при конвертации q_neg: {e}")
        import traceback
        traceback .print_exc ()
        q_neg =None

    if q_neg is None or (hasattr (q_neg ,'size')and q_neg .size ==0 ):

        neg_pool =[]
        for cls ,Q in q_pos .items ():
            if Q is not None :

                if not isinstance (Q ,np .ndarray ):
                    try :
                        Q =np .array (Q ,dtype =np .float32 )
                    except Exception :
                        continue
                if hasattr (Q ,'size')and Q .size >0 :
                    neg_pool .append (Q )
        if neg_pool :
            q_neg_eff =_l2n (np .concatenate (neg_pool ,axis =0 ),axis =1 )
        else :
            q_neg_eff =np .zeros ((0 ,D ),dtype =np .float32 )
        used_online_negs =True
    else :
        q_neg_eff =q_neg .astype (np .float32 ,copy =False )
        if q_neg_eff .ndim ==1 :
            q_neg_eff =q_neg_eff [None ,:]
        q_neg_eff =_l2n (q_neg_eff ,axis =1 )
        used_online_negs =False

    per_class_pos ={}
    per_class_best ={}
    all_pos_vals =[]

    for cls ,Q in q_pos .items ():
        try :
            print (f"   🔍 DEBUG: Обрабатываем класс '{cls}', Q тип = {type(Q)}")

            if isinstance (Q ,list )and len (Q )>0 :

                first_item =Q [0 ]
                if hasattr (first_item ,'mode')and hasattr (first_item ,'size'):
                    print (f"   ❌ DEBUG: Класс '{cls}' содержит PIL изображения вместо эмбеддингов")
                    print (f"   ❌ DEBUG: Q = {Q}, тип = {type(Q)}")
                    if verbose :
                        print (f"   ⚠️ Пропускаем класс '{cls}': получены PIL изображения вместо эмбеддингов")
                    continue

            Q =np .asarray (Q ,dtype =np .float32 )
            if Q .ndim ==1 :
                Q =Q [None ,:]
            print (f"   🔍 DEBUG: Q после asarray: форма = {Q.shape}, тип = {type(Q)}")
            if Q .size ==0 :
                if verbose :
                    print (f"   📊 Класс '{cls}': 0 примеров")
                per_class_pos [cls ]=np .full ((M ,),-1.0 ,dtype =np .float32 )
                continue
        except Exception as e :
            print (f"   ❌ DEBUG: Ошибка при обработке класса '{cls}': {e}")
            print (f"   ❌ DEBUG: Q = {Q}, тип = {type(Q)}")
            import traceback
            traceback .print_exc ()
            continue

        Q =_l2n (Q ,axis =1 )

        sim =mask_vecs @Q .T
        pos_cls =_aggregate (sim ,mode =pos_agg ,topk =topk )
        per_class_pos [cls ]=pos_cls
        all_pos_vals .extend (pos_cls .tolist ())

        if verbose :

            flat =sim .reshape (-1 )
            cls_list =[f"{v:+.3f}"if v <0 else f"{v:.3f}"for v in pos_cls ]
            print (f"   📊 Класс '{cls}': {Q.shape[0]} примеров → скоры={cls_list[:8] + (['...'] if len(cls_list)>8 else [])}")

    if not per_class_pos :
        if verbose :
            print ("   ❌ Нет валидных классов с эмбеддингами - возвращаем пустой результат")
        return [],{'pos_avg':0.0 ,'neg_avg':0.0 }

    best_cls =[]
    best_pos =[]
    for m in range (M ):
        cls_scores ={cls :per_class_pos [cls ][m ]for cls in per_class_pos .keys ()}
        cls =max (cls_scores ,key =cls_scores .get )
        best_cls .append (cls )
        best_pos .append (cls_scores [cls ])
        if verbose :

            vals =", ".join ([f"{c}={cls_scores[c]:.3f}"for c in [cls ]])
            print (f"     Маска {m}: [{vals}] → выбран {cls} ({cls_scores[cls]:.3f})")

    best_pos =np .array (best_pos ,dtype =np .float32 )

    if not used_online_negs :
        if q_neg_eff .size >0 :
            neg_sim =mask_vecs @q_neg_eff .T
            neg_raw =neg_sim .max (axis =1 )
        else :
            neg_raw =np .zeros ((M ,),dtype =np .float32 )
    else :

        if len (q_pos )<=1 :

            neg_raw =np .zeros ((M ,),dtype =np .float32 )
        else :

            neg_raw_by_class :dict [str ,np .ndarray ]={}

            q_by_class :dict [str ,np .ndarray ]={}
            for cls_key ,Qc in q_pos .items ():
                Qc_arr =np .asarray (Qc ,dtype =np .float32 )
                if Qc_arr .ndim ==1 :
                    Qc_arr =Qc_arr [None ,:]

                q_by_class [cls_key ]=_l2n (Qc_arr ,axis =1 )

            for cls_key in q_by_class .keys ():
                others =[q_by_class [k ]for k in q_by_class .keys ()if k !=cls_key and q_by_class [k ].size >0 ]
                if not others :
                    neg_raw_by_class [cls_key ]=np .zeros ((M ,),dtype =np .float32 )
                    continue
                neg_mat_c =np .concatenate (others ,axis =0 )

                neg_sim_c =mask_vecs @neg_mat_c .T
                neg_raw_by_class [cls_key ]=neg_sim_c .max (axis =1 )

            neg_raw =np .zeros ((M ,),dtype =np .float32 )
            for m in range (M ):
                cls_m =best_cls [m ]
                neg_raw [m ]=neg_raw_by_class .get (cls_m ,np .zeros ((),dtype =np .float32 )).reshape (-1 )[m ]if cls_m in neg_raw_by_class else 0.0

    neg =np .maximum (neg_raw ,0.0 )if clamp_neg_to_zero else neg_raw
    diff =best_pos -neg

    pos_avg =float (np .mean (best_pos ))if len (best_pos )else 0.0
    neg_avg =float (np .mean (neg_raw ))if len (neg_raw )else 0.0
    if verbose :
        print (f"📊 Скоры мультикласса: pos_avg={pos_avg:.3f}, neg_avg={neg_avg:.3f}")

    decisions =[]
    for m in range (M ):
        accepted =(best_pos [m ]>=min_pos_score )and (diff [m ]>=decision_threshold )
        decisions .append ({
        'class':best_cls [m ],
        'pos':float (best_pos [m ]),
        'neg_raw':float (neg_raw [m ]),
        'neg':float (neg [m ]),
        'diff':float (diff [m ]),
        'accepted':bool (accepted ),
        })
        if verbose :
            sep =best_pos [m ]
            cons ="1/1"
            print (f"   Маска {m}: класс={best_cls[m]}, pos={best_pos[m]:.3f}, "
            f"neg_raw={neg_raw[m]:+.3f}, neg={neg[m]:.3f}, "
            f"diff={diff[m]:+.3f}, sep=+{sep:.3f}, cons={cons} → принято={accepted}")

    debug ={
    'pos_avg':pos_avg ,
    'neg_avg':neg_avg ,
    'used_online_negs':used_online_negs ,
    'best_pos':best_pos ,
    'neg_raw':neg_raw ,
    'diff':diff ,
    }
    return decisions ,debug