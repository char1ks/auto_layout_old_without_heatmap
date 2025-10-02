from __future__ import annotations
import abc, warnings
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, jaccard_score, f1_score, classification_report
from evaluate.DatasetModel import DatasetModel
from evaluate.COCOAnnotations import COCOAnnotation

def _sf(x: Any, d: float=0.0)->float:
    try:
        v=float(x);return v if np.isfinite(v) else d
    except: return d

def _v4(a: Any)->Optional[List[float]]:
    try:
        r=[float(x) for x in a]; 
        return r if len(r)==4 and np.all(np.isfinite(r)) else None
    except: return None

def _xywh2xyxy(b: List[float])->List[float]:
    x,y,w,h=b;return [x,y,x+w,y+h]

def _iou(a: List[float], b: List[float])->float:
    ax1,ay1,ax2,ay2=a;bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1);ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0.0,ix2-ix1),max(0.0,iy2-iy1); inter=iw*ih
    if inter<=0: return 0.0
    ua=max(0.0,ax2-ax1)*max(0.0,ay2-ay1); ub=max(0.0,bx2-bx1)*max(0.0,by2-by1)
    return float(inter/max(ua+ub-inter,1e-12))

@dataclass
class Stats:
    categories: List[str]=field(default_factory=list)
    gt_counts: List[int]=field(default_factory=list)
    pred_counts: List[int]=field(default_factory=list)
    per_class_ap: Dict[str,float]=field(default_factory=dict)
    map: Optional[float]=None
    map_50: Optional[float]=None
    map_75: Optional[float]=None
    ap_iou_macro: List[float]=field(default_factory=list)
    ap_iou_micro: List[float]=field(default_factory=list)
    iou_thresholds: List[float]=field(default_factory=list)
    pr_macro: Dict[str,Dict[str,List[float]]]=field(default_factory=dict)
    pr_micro: Dict[str,Dict[str,List[float]]]=field(default_factory=dict)
    pr_curves_per_threshold: Dict[float,Dict[str,List[float]]]=field(default_factory=dict)
    doc: str=""
    formula: Optional[str]=None

class MetricOutputModel:
    def __init__(self, metric_name: str, score: float, stats: Dict[str,Any]): self.metric_name=metric_name; self.score=score; self.stats=stats; self.metadata=stats

class Metric(abc.ABC):
    name: str
    @abc.abstractmethod
    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs)->MetricOutputModel: ...

class MeanAveragePrecision(Metric):
    name="map"
    def __init__(self, iou_thresholds: Optional[List[float]]=None): self.iou_thresholds=sorted(set([0.5,0.75] if not iou_thresholds else [float(x) for x in iou_thresholds]))
    def _group(self, gt: DatasetModel, pr: List[COCOAnnotation]):
        gt_by: Dict[str,List[COCOAnnotation]]={}; pr_by: Dict[str,List[COCOAnnotation]]={}; files=[]
        for a in gt.data_points or []:
            fn=getattr(a,"file_name",None); 
            if fn is not None: gt_by.setdefault(str(fn),[]).append(a); files.append(str(fn))
        for a in pr or []:
            fn=getattr(a,"file_name",None); 
            if fn is not None: pr_by.setdefault(str(fn),[]).append(a); files.append(str(fn))
        files=sorted(set(files))
        labels=set()
        for arr in (gt.data_points or []): lab=getattr(arr,"label",None); 
        for arr in (pr or []): lab=getattr(arr,"label",None)
        for a in (gt.data_points or [])+(pr or []): 
            lab=getattr(a,"label",None)
            if lab is not None: labels.add(str(lab))
        return gt_by,pr_by,files,sorted(labels)

    def _prep(self, gt_by, pr_by, files, labels):
        gt_boxes={lab:{} for lab in labels}; preds={lab:[] for lab in labels}
        for fn in files:
            for a in gt_by.get(fn,[]) or []:
                lab=str(getattr(a,"label","")); bb=_v4(getattr(a,"bbox",None))
                if lab and bb is not None: gt_boxes.setdefault(lab,{}).setdefault(fn,[]).append(_xywh2xyxy(bb))
            for a in pr_by.get(fn,[]) or []:
                lab=str(getattr(a,"label","")); bb=_v4(getattr(a,"bbox",None)); sc=float(getattr(a,"score",getattr(a,"confidence",0.0)) or 0.0)
                if lab and bb is not None: preds.setdefault(lab,[]).append((str(fn),_xywh2xyxy(bb),sc))
        return gt_boxes,preds

    def _targets(self, preds_list, gt_per_file, thr):
        ps=sorted(preds_list,key=lambda x:x[2],reverse=True); y_t=[]; y_s=[]; used={fn:set() for fn in gt_per_file}
        for fn,bb,sc in ps:
            y_s.append(_sf(sc)); gts=gt_per_file.get(fn,[]) or []; bi=-1; bv=0.0
            for i,gb in enumerate(gts):
                if i in used.get(fn,set()): continue
                v=_iou(bb,gb)
                if v>=thr and v>bv: bv=v; bi=i
            if bi>=0: y_t.append(1); used.setdefault(fn,set()).add(bi)
            else: y_t.append(0)
        return y_t,y_s
        
    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs)->MetricOutputModel:
        gt_by,pr_by,files,labels=self._group(gt,prediction); gt_boxes,preds=self._prep(gt_by,pr_by,files,labels)
        per_class_05={}; ap_macro=[]; ap_micro=[]; pr_macro={}; pr_micro={}; pr_thr={}
        for thr in self.iou_thresholds:
            key=f"{thr:.2f}"; pr_macro[key]={}; pr_micro[key]={}; pr_thr[thr]={}
            all_t=[]; all_s=[]; ap_per={}; rec=np.linspace(0,1,101); macro_p=[]
            for lab in labels:
                yt,ys=self._targets(preds.get(lab,[]),gt_boxes.get(lab,{}) or {},float(thr))
                if yt:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        ap=_sf(average_precision_score(yt,ys)); p,r,_=precision_recall_curve(yt,ys)
                    ap_per[lab]=ap; pr_thr[thr][lab]={"precision":[_sf(x) for x in p],"recall":[_sf(x) for x in r]}
                    macro_p.append(np.interp(rec,r,p,left=p[0],right=p[-1]))
                else:
                    ap_per[lab]=0.0; pr_thr[thr][lab]={"precision":[1.0],"recall":[0.0]}; macro_p.append(np.ones_like(rec))
                all_t+=yt; all_s+=ys
            if abs(thr-0.5)<1e-6: per_class_05={lab:float(ap) for lab,ap in ap_per.items()}
            P=np.vstack(macro_p) if macro_p else np.ones((1,len(rec))); pr_macro[key]={"precision":[float(x) for x in P.mean(0)],"recall":[float(x) for x in rec]}
            if all_t:
                p_m,r_m,_=precision_recall_curve(np.asarray(all_t,dtype=int),np.asarray(all_s,dtype=float))
                pr_micro[key]={"precision":[_sf(x) for x in p_m],"recall":[_sf(x) for x in r_m]}
                ap_micro.append(_sf(average_precision_score(all_t,all_s)))
            else:
                pr_micro[key]={"precision":[1.0],"recall":[0.0]}; ap_micro.append(0.0)
            ap_macro.append(float(np.mean(list(ap_per.values()))) if ap_per else 0.0)
        overall=float(np.mean(ap_macro)) if ap_macro else 0.0
        def tget(t): 
            for i,v in enumerate(self.iou_thresholds):
                if abs(v-t)<1e-6: return ap_macro[i] if i<len(ap_macro) else 0.0
            return 0.0
        map50, map75=tget(0.5), tget(0.75)
        gt_counts=[sum(len(v) for v in (gt_boxes.get(l,{}) or {}).values()) for l in labels]
        pred_counts=[len(preds.get(l,[]) or []) for l in labels]
        st=Stats(categories=labels,gt_counts=gt_counts,pred_counts=pred_counts,per_class_ap=per_class_05,map=overall,map_50=map50,map_75=map75,ap_iou_macro=[float(x) for x in ap_macro],ap_iou_micro=[float(x) for x in ap_micro],iou_thresholds=[float(x) for x in self.iou_thresholds],pr_macro=pr_macro,pr_micro=pr_micro,pr_curves_per_threshold=pr_thr,doc=(self.__class__.__doc__ or '').strip(),formula="AP via sklearn average_precision_score on matched TP/FP; mAP = mean over classes/IoU thresholds")
        return MetricOutputModel(self.name, overall, asdict(st))

class MeanIntersectionOverUnion(Metric):
    name="miou"
    def __init__(self, iou_threshold: float=0.5): self.iou_threshold=float(iou_threshold)
    def _data(self, gt: DatasetModel, pr: List[COCOAnnotation])->Tuple[np.ndarray,np.ndarray]:
        gt_by={}; pr_by={}; files=[]
        for a in gt.data_points or []:
            fn=getattr(a,"file_name",None); 
            if fn is not None: gt_by.setdefault(str(fn),[]).append(a); files.append(str(fn))
        for a in pr or []:
            fn=getattr(a,"file_name",None); 
            if fn is not None: pr_by.setdefault(str(fn),[]).append(a); files.append(str(fn))
        files=sorted(set(files)); 
        H=int(getattr(gt,"image_height",1) or 1); W=int(getattr(gt,"image_width",1) or 1)
        if not files: return np.zeros((1,),np.uint8),np.zeros((1,),np.uint8)
        gt_all=[]; pr_all=[]
        for fn in files:
            g=np.zeros((H,W),np.uint8); p=np.zeros((H,W),np.uint8)
            for a in gt_by.get(fn,[]) or []:
                m=getattr(a,"mask",None)
                if isinstance(m,np.ndarray): g|=(m.astype(np.uint8)>0).astype(np.uint8)
            for a in pr_by.get(fn,[]) or []:
                m=getattr(a,"mask",None)
                if isinstance(m,np.ndarray): p|=(m.astype(np.uint8)>0).astype(np.uint8)
            gt_all.append(g); pr_all.append(p)
        y_true=np.concatenate([x.reshape(-1) for x in gt_all],0); y_pred=np.concatenate([x.reshape(-1) for x in pr_all],0)
        return y_true,y_pred
    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs)->MetricOutputModel:
        y_true,y_pred=self._data(gt,prediction); s=float(jaccard_score(y_true,y_pred,average="binary"))
        st=Stats(doc=(self.__class__.__doc__ or '').strip(),formula="IoU = |A∩B| / |A∪B|")
        return MetricOutputModel(self.name,s,asdict(st))

class DiceCoefficient(Metric):
    name="dice"
    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs)->MetricOutputModel:
        y_true,y_pred=MeanIntersectionOverUnion()._data(gt,prediction); s=float(f1_score(y_true,y_pred,average="binary"))
        st=Stats(doc=(self.__class__.__doc__ or '').strip(),formula="Dice = 2TP/(2TP+FP+FN)")
        return MetricOutputModel(self.name,s,asdict(st))

class ClassificationReportMetric(Metric):
    name="classification_report"
    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs)->MetricOutputModel:
        gb: Dict[str,List[str]]={}; pb: Dict[str,List[str]]={}; files=[]
        for a in (gt.data_points or []):
            fn=getattr(a,"file_name",None); lab=getattr(a,"label",None)
            if fn is None or lab is None: continue
            gb.setdefault(str(fn),[]).append(str(lab)); files.append(str(fn))
        for a in (prediction or []):
            fn=getattr(a,"file_name",None); lab=getattr(a,"label",None)
            if fn is None or lab is None: continue
            pb.setdefault(str(fn),[]).append(str(lab)); files.append(str(fn))
        files=sorted(set(files))
        def maj(arr: List[str])->str:
            if not arr: return "none"
            d: Dict[str,int]={}
            for s in arr: d[s]=d.get(s,0)+1
            return max(d.items(),key=lambda x:x[1])[0]
        y_t=[]; y_p=[]
        for fn in files:
            gt_l,pr_l=gb.get(fn,[]),pb.get(fn,[])
            if not gt_l and not pr_l: continue
            g=maj(gt_l); p=maj(pr_l)
            if g=="none" or p=="none": continue
            y_t.append(g); y_p.append(p)
        if not y_t:
            return MetricOutputModel(self.name,0.0,asdict(Stats(doc=(self.__class__.__doc__ or '').strip())))
        rep_dict=classification_report(y_t,y_p,output_dict=True,zero_division=0)
        rep_text=classification_report(y_t,y_p,output_dict=False,zero_division=0)
        acc=float(sum(1 for a,b in zip(y_t,y_p) if a==b))/float(len(y_t))
        st=asdict(Stats(doc=(self.__class__.__doc__ or '').strip()))
        st.update({"num_files":len(files),"dict":rep_dict,"text":rep_text,"labels":sorted(list(set(y_t)|set(y_p)))})
        return MetricOutputModel(self.name,acc,st)
