from __future__ import annotations
import abc, warnings
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score, jaccard_score, f1_score
from evaluate.DatasetModel import DatasetModel
from evaluate.COCOAnnotations import COCOAnnotation

@dataclass
class Meta:
    doc: str = ""
    formula: Optional[str] = None
    def to_dict(self)->Dict[str,Any]:
        d=asdict(self); return {k:v for k,v in d.items() if v not in (None,"",[])}

@dataclass
class Stats:
    data: Dict[str, Any]=field(default_factory=dict)
    meta: Meta=field(default_factory=Meta)
    def to_dict(self)->Dict[str,Any]:
        out=dict(self.data); out.update(self.meta.to_dict()); return out

@dataclass
class MetricOutputModel:
    metric_name:str
    score:float
    stats:Stats

def _f(x,default=0.0)->float:
    try:
        v=float(x); return v if np.isfinite(v) else default
    except Exception:
        return default

def _box(bb)->Optional[List[float]]:
    if not isinstance(bb,(list,tuple)) or len(bb)!=4: return None
    return [_f(bb[0]),_f(bb[1]),_f(bb[2]),_f(bb[3])]

def _xywh2xyxy(b:List[float])->List[float]:
    x,y,w,h=b; return [x,y,x+w,y+h]

def _iou(a:List[float],b:List[float])->float:
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0.0,ix2-ix1),max(0.0,iy2-iy1); inter=iw*ih
    if inter<=0: return 0.0
    ua=max(0.0,ax2-ax1)*max(0.0,ay2-ay1); ub=max(0.0,bx2-bx1)*max(0.0,by2-by1)
    return float(inter/max(ua+ub-inter,1e-12))

class Metric(abc.ABC):
    name:str="metric"
    @abc.abstractmethod
    def compute(self, gt:DatasetModel, prediction:List[COCOAnnotation], **kwargs)->MetricOutputModel: ...

class MeanAveragePrecision(Metric):
    name="mAP"
    def __init__(self,iou_thresholds:Optional[List[float]]=None)->None:
        self.iou_thresholds=iou_thresholds or np.arange(0.5,0.95+1e-9,0.05).tolist()

    def _group(self,gt:DatasetModel,pr:List[COCOAnnotation]):
        g_by:Dict[str,Dict[str,List[List[float]]]]={}
        p_by:Dict[str,List[Tuple[str,List[float],float]]]={}
        labels:Set[str]=set()
        files:Set[str]=set()
        for a in (gt.data_points or []):
            fn=getattr(a,'file_name',None); lab=str(getattr(a,'label',None))
            bb=_box(getattr(a,'bbox',None))
            if fn is None or bb is None: continue
            files.add(str(fn)); labels.add(lab)
            g_by.setdefault(lab,{}).setdefault(str(fn),[]).append(_xywh2xyxy(bb))
        for a in (pr or []):
            fn=getattr(a,'file_name',None); lab=str(getattr(a,'label',None))
            bb=_box(getattr(a,'bbox',None))
            if fn is None or bb is None: continue
            files.add(str(fn)); labels.add(lab)
            score=_f(getattr(a,'score',getattr(a,'confidence',1.0)))
            p_by.setdefault(lab,[]).append((str(fn),_xywh2xyxy(bb),score))
        for l in p_by: p_by[l].sort(key=lambda t:t[2],reverse=True)
        return g_by,p_by,sorted(files),sorted(labels)

    def _match(self,preds:List[Tuple[str,List[float],float]], gt_per_file:Dict[str,List[List[float]]], thr:float):
        used={fn:set() for fn in gt_per_file}; yt:List[int]=[]; ys:List[float]=[]
        for fn,bb,sc in preds:
            gts=gt_per_file.get(fn,[])
            if not gts: yt.append(0); ys.append(float(sc)); continue
            best_iou,best_idx=0.0,-1
            for i,gb in enumerate(gts):
                if i in used[fn]: continue
                v=_iou(bb,gb)
                if v>best_iou: best_iou,best_idx=v,i
            hit=(best_idx>=0 and best_iou>=thr)
            yt.append(1 if hit else 0); ys.append(float(sc))
            if hit: used[fn].add(best_idx)
        return yt,ys

    def _ap_pr(self,yt:List[int],ys:List[float]):
        if not yt: return 0.0,[1.0],[0.0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            ap=_f(average_precision_score(yt,ys)); p,r,_=precision_recall_curve(yt,ys)
        r=np.asarray(r,float); p=np.asarray(p,float)
        o=np.argsort(r); r,p=r[o],p[o]
        r,u=np.unique(r,return_index=True); p=p[u]
        p=np.maximum.accumulate(p[::-1])[::-1]
        return float(ap),[float(x) for x in p],[float(x) for x in r]

    def compute(self,gt:DatasetModel,prediction:List[COCOAnnotation],**kwargs)->MetricOutputModel:
        g,p,files,labels=self._group(gt,prediction)
        if not files: 
            return MetricOutputModel(self.name,0.0,Stats(data={"error":"empty GT or predictions"},meta=Meta(doc=(self.__class__.__doc__ or '').strip())))
        ap_lab_thr={l:[0.0]*len(self.iou_thresholds) for l in labels}
        pr_thr:Dict[float,Dict[str,Dict[str,List[float]]]]={}
        ystore:Dict[float,Dict[str,Tuple[List[int],List[float]]]]={}
        for ti,thr in enumerate(self.iou_thresholds):
            pr_thr[thr]={}; ystore[thr]={}
            for l in labels:
                yt,ys=self._match(p.get(l,[]), g.get(l,{}) or {}, float(thr))
                ystore[thr][l]=(yt,ys)
                ap,pp,rr=self._ap_pr(yt,ys)
                ap_lab_thr[l][ti]=ap; pr_thr[thr][l]={"precision":pp,"recall":rr}
        rec_grid=np.linspace(0,1,101)
        pr_micro:Dict[str,Dict[str,List[float]]]={}; pr_macro:Dict[str,Dict[str,List[float]]]={}
        ap_micro:List[float]=[]; ap_macro:List[float]=[]
        for thr in self.iou_thresholds:
            key=f"{_f(thr):.2f}"
            yt_all=sum((ystore[thr][l][0] for l in labels),[]); ys_all=sum((ystore[thr][l][1] for l in labels),[])
            ap_mi,p_mi,r_mi=self._ap_pr(yt_all,ys_all) if yt_all else (0.0,[1.0],[0.0])
            ap_micro.append(ap_mi); pr_micro[key]={"precision":[float(x) for x in p_mi],"recall":[float(x) for x in r_mi]}
            stack=[]
            for l in labels:
                rr=np.asarray(pr_thr[thr][l]["recall"],float); pp=np.asarray(pr_thr[thr][l]["precision"],float)
                if rr.size>1 and pp.size>1:
                    o=np.argsort(rr); pi=np.interp(rec_grid,rr[o],pp[o],left=pp[o][0],right=pp[o][-1])
                    pi=np.maximum.accumulate(pi[::-1])[::-1]; stack.append(pi)
            pm=np.mean(np.stack(stack,0),0) if stack else np.zeros_like(rec_grid)
            pr_macro[key]={"precision":[float(x) for x in pm],"recall":[float(x) for x in rec_grid]}
            ti=int(np.argmin([abs(x-thr) for x in self.iou_thresholds]))
            ap_macro.append(float(np.mean([ap_lab_thr[l][ti] for l in labels])) if labels else 0.0)
        overall=float(np.mean(ap_macro)) if ap_macro else 0.0
        pick=lambda t: float(ap_macro[int(np.argmin([abs(x-t) for x in self.iou_thresholds]))]) if ap_macro else 0.0
        map50,map75=pick(0.5),pick(0.75)
        i050=int(np.argmin([abs(x-0.5) for x in self.iou_thresholds]))
        per_class_05=[float(ap_lab_thr[l][i050]) for l in labels] if labels else []
        gt_counts=[sum(len(v) for v in (g.get(l,{}) or {}).values()) for l in labels]
        pred_counts=[len(p.get(l,[]) or []) for l in labels]
        data={
            'categories':labels,
            'gt_counts':gt_counts,
            'pred_counts':pred_counts,
            'per_class_ap':per_class_05,
            'per_class_ap_avg':[float(np.mean(ap_lab_thr[l])) if ap_lab_thr[l] else 0.0 for l in labels],
            'map':overall,
            'map_50':map50,'map_75':map75,'mAP@0.5':map50,'mAP@0.75':map75,
            'ap_iou_macro':[float(x) for x in ap_macro],
            'ap_iou_micro':[float(x) for x in ap_micro],
            'iou_thresholds':[float(x) for x in self.iou_thresholds],
            'pr_macro':pr_macro,'pr_micro':pr_micro,'pr_curves_per_threshold':pr_thr
        }
        return MetricOutputModel(self.name,overall,Stats(data=data,meta=Meta(doc=(self.__class__.__doc__ or '').strip())))

class MeanIntersectionOverUnion(Metric):
    name="mIoU"
    def _stack(self,gt:DatasetModel,pr:List[COCOAnnotation]):
        gb:Dict[str,List[Any]]={} 
        pb:Dict[str,List[Any]]={}
        for a in (gt.data_points or []): 
            gb.setdefault(str(getattr(a,'file_name',None) or 'f'),[]).append(a)
        for a in (pr or []): 
            pb.setdefault(str(getattr(a,'file_name',None) or 'f'),[]).append(a)
        files=sorted(set(gb)|set(pb))
        H=int(getattr(gt,'image_height',0) or 0); W=int(getattr(gt,'image_width',0) or 0)
        if H<=0 or W<=0:
            for a in (gt.data_points or [])+(pr or []):
                m=getattr(a,'mask',None)
                if isinstance(m,np.ndarray) and m.ndim>=2: H,W=int(m.shape[0]),int(m.shape[1]); break
        if H<=0 or W<=0: return np.zeros((1,),np.uint8),np.zeros((1,),np.uint8)
        gs,ps=[],[]
        for fn in files:
            g=np.zeros((H,W),np.uint8); p=np.zeros((H,W),np.uint8)
            for a in gb.get(fn,[]):
                m=getattr(a,'mask',None)
                if isinstance(m,np.ndarray) and m.shape[:2]==(H,W): g|=(m>0).astype(np.uint8)
            for a in pb.get(fn,[]):
                m=getattr(a,'mask',None)
                if isinstance(m,np.ndarray) and m.shape[:2]==(H,W): p|=(m>0).astype(np.uint8)
            gs.append(g.reshape(-1)); ps.append(p.reshape(-1))
        yt=np.concatenate(gs,0) if gs else np.zeros((1,),np.uint8)
        yp=np.concatenate(ps,0) if ps else np.zeros((1,),np.uint8)
        return yt,yp
    def compute(self,gt:DatasetModel,prediction:List[COCOAnnotation],**kwargs)->MetricOutputModel:
        yt,yp=self._stack(gt,prediction); s=float(jaccard_score(yt,yp,average="binary"))
        return MetricOutputModel(self.name,s,Stats(meta=Meta(doc=(self.__class__.__doc__ or '').strip(),formula='IoU = |X∩Y|/|X∪Y|')))

class DiceCoefficient(Metric):
    name="dice"
    def compute(self,gt:DatasetModel,prediction:List[COCOAnnotation],**kwargs)->MetricOutputModel:
        yt,yp=MeanIntersectionOverUnion()._stack(gt,prediction); s=float(f1_score(yt,yp,average="binary"))
        return MetricOutputModel(self.name,s,Stats(meta=Meta(doc=(self.__class__.__doc__ or '').strip(),formula='Dice = 2PR/(P+R)')))

class ClassificationReportMetric(Metric):
    name="classification_report"
    def compute(self,gt:DatasetModel,prediction:List[COCOAnnotation],**kwargs)->MetricOutputModel:
        gb:Dict[str,List[Union[int,str]]]={}
        pb:Dict[str,List[Union[int,str]]]={}
        for a in (gt.data_points or []):
            fn,lab=getattr(a,'file_name',None),getattr(a,'label',None)
            if fn is not None and lab is not None: 
                gb.setdefault(str(fn),[]).append(lab)
        for a in (prediction or []):
            fn,lab=getattr(a,'file_name',None),getattr(a,'label',None)
            if fn is not None and lab is not None: 
                pb.setdefault(str(fn),[]).append(lab)
        files=sorted(set(gb)|set(pb))
        if not files: 
            return MetricOutputModel(self.name,0.0,Stats(data={"error":"no files to compare","dict":{},"text":""},meta=Meta(doc=(self.__class__.__doc__ or '').strip())))
        def maj(v): 
            if not v: return "none"
            u,c=np.unique([str(x) for x in v],return_counts=True); return str(u[int(np.argmax(c))])
        yt,yp=[],[]
        for fn in files:
            g,p=maj(gb.get(fn,[])),maj(pb.get(fn,[]))
            if g!="none" and p!="none": yt.append(g); yp.append(p)
        if not yt:
            return MetricOutputModel(self.name,0.0,Stats(data={"error":"no valid labels after filtering","dict":{},"text":""},meta=Meta(doc=(self.__class__.__doc__ or '').strip())))
        rep_dict=classification_report(yt,yp,output_dict=True,zero_division=0)
        rep_text=classification_report(yt,yp,output_dict=False,zero_division=0)
        acc=float(np.mean([a==b for a,b in zip(yt,yp)]))
        data={"num_files":len(files),"dict":rep_dict,"text":rep_text,"labels":sorted(set(yt)|set(yp))}
        return MetricOutputModel(self.name,acc,Stats(data=data,meta=Meta(doc=(self.__class__.__doc__ or '').strip())))
