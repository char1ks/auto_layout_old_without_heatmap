from __future__ import annotations
import abc, numpy as np
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import jaccard_score, f1_score, classification_report
from evaluate.DatasetModel import DatasetModel
from evaluate.COCOAnnotations import COCOAnnotation

def _xywh2xyxy(b): x,y,w,h=b; return [x,y,x+w,y+h]
def _iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0.0,ix2-ix1),max(0.0,iy2-iy1); inter=iw*ih
    if inter<=0: return 0.0
    ua=max(0.0,ax2-ax1)*max(0.0,ay2-ay1); ub=max(0.0,bx2-bx1)*max(0.0,by2-by1)
    return float(inter/max(ua+ub-inter,1e-12))

@dataclass
class Stats:
    categories: List[str]=field(default_factory=list)
    gt_counts: List[int]=field(default_factory=list)
    pred_counts: List[int]=field(default_factory=list)
    per_class_ap: List[float]=field(default_factory=list)
    per_class_ap_avg: List[float]=field(default_factory=list)
    map: Optional[float]=None
    map_50: Optional[float]=None
    map_75: Optional[float]=None
    ap_iou_macro: List[float]=field(default_factory=list)
    ap_iou_micro: List[float]=field(default_factory=list)
    iou_thresholds: List[float]=field(default_factory=list)
    pr_macro: Dict[str,Dict[str,List[float]]]=field(default_factory=dict)
    pr_micro: Dict[str,Dict[str,List[float]]]=field(default_factory=dict)
    pr_curves_per_threshold: Dict[float,Dict[str,Dict[str,List[float]]]]=field(default_factory=dict)
    doc: str=""
    formula: Optional[str]=None

class MetricOutputModel:
    def __init__(self, metric_name: str, score: float, stats: Dict[str,Any]):
        self.metric_name=metric_name; self.score=score; self.stats=stats; self.metadata=stats

class Metric(abc.ABC):
    name: str
    @abc.abstractmethod
    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs)->MetricOutputModel: ...

class MeanAveragePrecision(Metric):
    name="map"
    def __init__(self, iou_thresholds: Optional[List[float]]=None):
        self.iou_thresholds=sorted(set([0.5,0.75] if not iou_thresholds else [float(x) for x in iou_thresholds]))

    def _split(self, gt, pr):
        g, p, labels = {}, {}, set()
        for a in (gt.data_points or []):
            fn=getattr(a,"file_name",None); lab=str(getattr(a,"label",None)); bb=getattr(a,"bbox",None)
            if fn is None or bb is None: continue
            g.setdefault(lab,{}).setdefault(str(fn),[]).append(_xywh2xyxy([float(bb[0]),float(bb[1]),float(bb[2]),float(bb[3])]))
            labels.add(lab)
        for a in (pr or []):
            fn=getattr(a,"file_name",None); lab=str(getattr(a,"label",None)); bb=getattr(a,"bbox",None)
            sc=float(getattr(a,"score",getattr(a,"confidence",0.0)) or 0.0)
            if fn is None or bb is None: continue
            p.setdefault(lab,[]).append((str(fn),_xywh2xyxy([float(bb[0]),float(bb[1]),float(bb[2]),float(bb[3])]),sc))
            labels.add(lab)
        return g,p,sorted(labels)

    def _match_tp_fp_scores(self, preds, gt_by_file, thr)->Tuple[np.ndarray,np.ndarray,np.ndarray]:
        preds=sorted(preds,key=lambda x:x[2],reverse=True)
        used={fn:set() for fn in gt_by_file}
        tp=[]; fp=[]; sc=[]
        for fn,bb,s in preds:
            sc.append(float(s)); gts=gt_by_file.get(fn,[]) or []; bi=-1; bv=0.0
            for i,gb in enumerate(gts):
                if i in used.get(fn,set()): continue
                v=_iou(bb,gb)
                if v>=thr and v>bv: bv,bi=v,i
            if bi>=0: tp.append(1.0); fp.append(0.0); used.setdefault(fn,set()).add(bi)
            else: fp.append(1.0); tp.append(0.0)
        return np.asarray(tp,float), np.asarray(fp,float), np.asarray(sc,float)

    def _step_pr(self, tp: np.ndarray, fp: np.ndarray, num_gt: int)->Tuple[np.ndarray,np.ndarray]:
        if num_gt<=0 or tp.size==0: return np.array([0.0]), np.array([0.0])
        ctp=np.cumsum(tp); cfp=np.cumsum(fp); k=np.arange(1, tp.size+1, dtype=float)
        prec=ctp/np.maximum(k,1e-12); rec=ctp/float(num_gt)
        ur, idx, cnts = np.unique(rec, return_index=True, return_counts=True)
        up=np.zeros_like(ur); s=0
        for i,c in enumerate(cnts):
            up[i]=float(np.max(prec[s:s+c])); s+=c
        return ur, up

    def _ap_from_step(self, r: np.ndarray, p: np.ndarray)->float:
        if r.size<2 or p.size<2: return 0.0
        env=np.maximum.accumulate(p[::-1])[::-1]
        return float(np.trapz(env, r))

    def _macro_union_step(self, per_curves: List[Tuple[np.ndarray,np.ndarray]])->Tuple[List[float],List[float]]:
        if not per_curves: return [0.0],[0.0]
        # явные границы 0 и 1
        base = [np.asarray([0.0,1.0],float)]
        grid=np.unique(np.clip(np.concatenate(base+[r for r,_ in per_curves if r.size]),0,1))
        P=[]
        for r,p in per_curves:
            if r.size==0 or p.size==0: continue
            # интерполяция со значениями по краям как константы
            P.append(np.interp(grid, r, p, left=float(p[0]), right=float(p[-1])))
        if not P: return [0.0],[0.0]
        macro=np.mean(np.vstack(P),axis=0)
        return list(grid), [float(x) for x in macro]

    def compute(self, gt, prediction, **kwargs):
        g,p,labels=self._split(gt,prediction)
        gt_counts=[sum(len(v) for v in (g.get(l,{}) or {}).values()) for l in labels]
        pred_counts=[len(p.get(l,[]) or []) for l in labels]
        pr_macro: Dict[str,Dict[str,List[float]]]={}
        pr_micro: Dict[str,Dict[str,List[float]]]={}
        pr_thr: Dict[float,Dict[str,Dict[str,List[float]]]]={}
        ap_iou_macro=[]; ap_iou_micro=[]; per_class_ap_05=[]

        for thr in self.iou_thresholds:
            key=f"{thr:.2f}"; pr_thr[thr]={}
            per_cls_curves=[]; per_cls_ap=[]
            all_tp=[]; all_fp=[]; all_scores=[]; total_gt=0

            for lab, support in zip(labels, gt_counts):
                gt_by_file=g.get(lab,{}) or {}
                tp,fp,scores=self._match_tp_fp_scores(p.get(lab,[]) or [], gt_by_file, float(thr))
                r,pr=self._step_pr(tp,fp,support)
                pr_thr[thr][lab]={"precision":[float(x) for x in pr],"recall":[float(x) for x in r]}
                per_cls_curves.append((r,pr))
                per_cls_ap.append(self._ap_from_step(r,pr))
                if tp.size:
                    all_tp.append(tp); all_fp.append(fp); all_scores.append(scores); total_gt+=int(support)

            rM,pM=self._macro_union_step(per_cls_curves)
            pr_macro[key]={"precision":pM,"recall":rM}
            ap_iou_macro.append(float(np.mean(per_cls_ap) if per_cls_ap else 0.0))

            if total_gt>0 and all_scores:
                scores=np.concatenate(all_scores); tp=np.concatenate(all_tp); fp=np.concatenate(all_fp)
                order=np.argsort(-scores); tp,fp=tp[order],fp[order]
                rmi,pmi=self._step_pr(tp,fp,total_gt)
                pr_micro[key]={"precision":[float(x) for x in pmi],"recall":[float(x) for x in rmi]}
                ap_iou_micro.append(self._ap_from_step(rmi,pmi))
            else:
                pr_micro[key]={"precision":[0.0],"recall":[0.0]}; ap_iou_micro.append(0.0)

            if abs(thr-0.5)<1e-6:
                per_class_ap_05=[float(x) for x in per_cls_ap]

        overall=float(np.mean(ap_iou_macro) if ap_iou_macro else 0.0)
        def pick(t):
            for i,x in enumerate(self.iou_thresholds):
                if abs(x-t)<1e-6: return ap_iou_macro[i] if i<len(ap_iou_macro) else 0.0
            return 0.0
        map50,map75=pick(0.5),pick(0.75)

        st=Stats(categories=labels,gt_counts=gt_counts,pred_counts=pred_counts,per_class_ap=per_class_ap_05,per_class_ap_avg=list(per_class_ap_05),map=overall,map_50=map50,map_75=map75,ap_iou_macro=[float(x) for x in ap_iou_macro],ap_iou_micro=[float(x) for x in ap_iou_micro],iou_thresholds=[float(x) for x in self.iou_thresholds],pr_macro=pr_macro,pr_micro=pr_micro,pr_curves_per_threshold=pr_thr,doc="",formula="")
        stats=asdict(st); stats["mAP@0.5"]=map50; stats["mAP@0.75"]=map75
        return MetricOutputModel(self.name,overall,stats)

class MeanIntersectionOverUnion(Metric):
    name="mIoU"
    def _to_hw(self, m, H, W):
        a=np.asarray(m)
        if a.ndim==3: a=np.squeeze(a)
        if a.ndim==2 and a.shape==(H,W): return (a>0).astype(np.uint8)
        if a.ndim==2 and a.shape==(W,H): return (a.T>0).astype(np.uint8)
        if a.size==H*W:
            try: return (a.reshape(H,W)>0).astype(np.uint8)
            except: return None
        return None
    def _data(self, gt, pr):
        H=int(getattr(gt,"image_height",0) or 0); W=int(getattr(gt,"image_width",0) or 0)
        files=set()
        for a in (gt.data_points or []):
            fn=getattr(a,"file_name",None)
            if fn is not None: files.add(str(fn))
        for a in (pr or []):
            fn=getattr(a,"file_name",None)
            if fn is not None: files.add(str(fn))
        files=sorted(files)
        if H<=0 or W<=0:
            for a in (gt.data_points or [])+(pr or []):
                m=getattr(a,"mask",None)
                if m is None: continue
                arr=np.asarray(m)
                if arr.ndim==3: arr=np.squeeze(arr)
                if arr.ndim==2: H,W=int(arr.shape[0]),int(arr.shape[1]); break
        if H<=0 or W<=0: return np.zeros((1,),np.uint8), np.zeros((1,),np.uint8)
        y_true_list=[]; y_pred_list=[]
        for fn in files:
            g=np.zeros((H,W),np.uint8); p=np.zeros((H,W),np.uint8)
            for a in (gt.data_points or []):
                if getattr(a,"file_name",None)!=fn: continue
                mh=self._to_hw(getattr(a,"mask",None),H,W) if getattr(a,"mask",None) is not None else None
                if mh is not None: g|=mh
            for a in (pr or []):
                if getattr(a,"file_name",None)!=fn: continue
                mh=self._to_hw(getattr(a,"mask",None),H,W) if getattr(a,"mask",None) is not None else None
                if mh is not None: p|=mh
            y_true_list.append(g.reshape(-1)); y_pred_list.append(p.reshape(-1))
        y_true=np.concatenate(y_true_list,0) if y_true_list else np.zeros((1,),np.uint8)
        y_pred=np.concatenate(y_pred_list,0) if y_pred_list else np.zeros((1,),np.uint8)
        return y_true,y_pred
    def compute(self, gt, prediction, **kwargs):
        y_true,y_pred=self._data(gt,prediction); s=float(jaccard_score(y_true,y_pred,average="binary"))
        return MetricOutputModel(self.name,s,asdict(Stats()))

class DiceCoefficient(Metric):
    name="dice"
    def compute(self, gt, prediction, **kwargs):
        miou=MeanIntersectionOverUnion(); y_true,y_pred=miou._data(gt,prediction); s=float(f1_score(y_true,y_pred,average="binary"))
        return MetricOutputModel(self.name,s,asdict(Stats()))

class ClassificationReportMetric(Metric):
    name="classification_report"
    def compute(self, gt, prediction, **kwargs):
        gb,pb,files={}, {}, set()
        for a in (gt.data_points or []):
            fn=getattr(a,"file_name",None); lab=getattr(a,"label",None)
            if fn is None or lab is None: continue
            gb.setdefault(str(fn),[]).append(str(lab)); files.add(str(fn))
        for a in (prediction or []):
            fn=getattr(a,"file_name",None); lab=getattr(a,"label",None)
            if fn is None or lab is None: continue
            pb.setdefault(str(fn),[]).append(str(lab)); files.add(str(fn))
        files=sorted(files)
        def maj(x): 
            return max(((v,x.count(v)) for v in set(x)), key=lambda t:t[1])[0] if x else "none"
        y_t,y_p=[],[]
        for fn in files:
            g=maj(gb.get(fn,[])); p=maj(pb.get(fn,[]))
            if g!="none" and p!="none": y_t.append(g); y_p.append(p)
        if not y_t: return MetricOutputModel(self.name,0.0,asdict(Stats()))
        rep_dict=classification_report(y_t,y_p,output_dict=True,zero_division=0)
        rep_text=classification_report(y_t,y_p,output_dict=False,zero_division=0)
        acc=float(sum(1 for a,b in zip(y_t,y_p) if a==b))/float(len(y_t))
        st=asdict(Stats()); st.update({"num_files":len(files),"dict":rep_dict,"text":rep_text,"labels":sorted(list(set(y_t)|set(y_p)))})
        return MetricOutputModel(self.name,acc,st)
