from __future__ import annotations
import abc
import numpy as np
from collections import Counter
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional,Tuple
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score, jaccard_score, f1_score
from evaluation.dataset_model import DatasetModel
from evaluation.coco_annotation import CocoAnnotation
from shapely.geometry import box as _box_poly

@dataclass
class Meta:
    doc: str = ""
    formula: Optional[str] = None
    def to_dict(self)->Dict[str,Any]:
        d=asdict(self)
        return {k:v for k,v in d.items() if v not in (None,"",[])}

@dataclass
class Stats:
    data: Dict[str, Any]=field(default_factory=dict)
    meta: Meta=field(default_factory=Meta)
    def to_dict(self)->Dict[str,Any]:
        out=dict(self.data)
        out.update(self.meta.to_dict())
        return out

@dataclass
class MetricOutputModel:
    metric_name:str
    score:float
    stats:Stats

class Metric(abc.ABC):
    name:str="metric"
    @abc.abstractmethod
    def compute(self, gt:DatasetModel, prediction:List[CocoAnnotation], **kwargs)->MetricOutputModel: ...

def _iou(a: List[float], b: List[float]) -> float:
    pa, pb = _box_poly(a[0], a[1], a[2], a[3]), _box_poly(b[0], b[1], b[2], b[3])
    inter = pa.intersection(pb).area
    return float(inter / (pa.area + pb.area - inter)) if inter > 0 else 0.0

class MeanAveragePrecision(Metric):
    name = "mAP"

    def __init__(self, iou_thresholds: Optional[List[float]] = None) -> None:
        self.iou_thresholds = iou_thresholds or np.arange(0.5, 0.95 + 1e-9, 0.05).tolist()

    # group data
    def _group(self, ground_truth: DatasetModel, predictions: List[CocoAnnotation]):
        ground_truth_by: dict[str, dict[str, List[List[float]]]] = {}
        predictions_by: dict[str, List[Tuple[str, List[float], float]]] = {}
        label_names = set()

        for annotation in (ground_truth.data_points or []):
            bbox = ([float(x) if np.isfinite(float(x)) else 0.0 for x in annotation.bbox] if isinstance(annotation.bbox,(list,tuple)) and len(annotation.bbox)==4 else None)
            if annotation.file_name is None or bbox is None: 
                continue
            file_key = str(annotation.file_name)
            label_key = str(annotation.label)
            label_names.add(label_key)

            ground_truth_by.setdefault(label_key, {}).setdefault(file_key, []).append([
                float(bbox[0]), float(bbox[1]), float(bbox[0]) + float(bbox[2]), float(bbox[1]) + float(bbox[3])
            ])

        for annotation in (predictions or []):
            bbox = ([float(x) if np.isfinite(float(x)) else 0.0 for x in annotation.bbox] if isinstance(annotation.bbox,(list,tuple)) and len(annotation.bbox)==4 else None)
            if annotation.file_name is None or bbox is None: 
                continue
            file_key = str(annotation.file_name)
            label_key = str(annotation.label)
            label_names.add(label_key)
            score = annotation.score
            if score is None:
                score = annotation.confidence if annotation.confidence is not None else 1.0
            score_value = float(score)

            predictions_by.setdefault(label_key, []).append((
                file_key,
                [float(bbox[0]), float(bbox[1]), float(bbox[0]) + float(bbox[2]), float(bbox[1]) + float(bbox[3])],
                float(score_value)
            ))

        for label_key in predictions_by:
            predictions_by[label_key].sort(key=lambda t: t[2], reverse=True)

        return ground_truth_by, predictions_by, sorted(label_names)
        
    # match data
    def _match(self,predictions_for_label: List[Tuple[str, List[float], float]],ground_truth_by_file: dict[str, List[List[float]]],iou_threshold: float,):
        used_indices: Dict[str, set[int]] = {fname: set() for fname in ground_truth_by_file}
        y_true: List[int] = []
        y_score: List[float] = []

        for file_name, pred_box, score in predictions_for_label:
            gt_boxes = ground_truth_by_file.get(file_name, [])
            if not gt_boxes:
                y_true.append(0)
                y_score.append(float(score))
                continue

            candidates = [(i, _iou(pred_box, gt)) for i, gt in enumerate(gt_boxes) if i not in used_indices[file_name]]
            best_index, best_iou = max(candidates, key=lambda t: t[1], default=(-1, 0.0))

            hit = (best_index >= 0) and (best_iou >= iou_threshold)
            y_true.append(1 if hit else 0)
            y_score.append(float(score))
            if hit:
                used_indices[file_name].add(best_index)
        return y_true, y_score

    def _ap_pr(self, y_true: List[int], y_score: List[float]):
        # empty input
        if not y_true:
            return 0.0, [1.0], [0.0]
        # no positives
        if int(np.sum(np.asarray(y_true))) == 0:
            return 0.0, [1.0], [0.0]

        # AP score
        ap_value = float(average_precision_score(y_true, y_score))
        # PR curve
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        
        # to float
        recall = np.asarray(recall, float)
        precision = np.asarray(precision, float)
        # sort recall
        order = np.argsort(recall)
        recall, precision = recall[order], precision[order]
        # drop duplicates
        recall, uniq_idx = np.unique(recall, return_index=True)
        precision = precision[uniq_idx]
        # smooth precision
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        
        return float(ap_value), [float(x) for x in precision], [float(x) for x in recall]

    def compute(self, gt: DatasetModel, prediction: List[CocoAnnotation], **kwargs) -> MetricOutputModel:
        gt_by, pr_by, label_names = self._group(gt, prediction)
        thresholds = [float(t) for t in self.iou_thresholds]
        num_labels, num_thresholds = len(label_names), len(thresholds)

        # PR/AP for all IoU

        # storage
        ap_per_label_per_thr = {lab: [0.0] * num_thresholds for lab in label_names}
        pr_curves_per_thr: Dict[float, Dict[str, Dict[str, List[float]]]] = {thr: {} for thr in thresholds}
        y_store: Dict[float, Dict[str, Tuple[List[int], List[float]]]] = {thr: {} for thr in thresholds}

        # per-label loop
        for threshold_index, thr in enumerate(thresholds):
            for lab in label_names:
                preds_lab = pr_by.get(lab, [])
                gts_lab = gt_by.get(lab, {})
                y_true, y_score = self._match(preds_lab, gts_lab, thr)
                y_store[thr][lab] = (y_true, y_score)
                ap_val, prec, rec = self._ap_pr(y_true, y_score)
                pr_curves_per_thr[thr][lab] = {"precision": prec, "recall": rec}
                ap_per_label_per_thr[lab][threshold_index] = ap_val

        # micro/macro
        recall_grid = np.linspace(0, 1, 101)
        pr_micro = {}
        pr_macro = {}
        ap_micro, ap_macro = [], []

        for ti, thr in enumerate(thresholds):
            key = f"{thr:.2f}"

            # micro merge
            y_true_all = [y for lab in label_names for y in y_store[thr][lab][0]]
            y_score_all = [s for lab in label_names for s in y_store[thr][lab][1]]
            ap_mi, p_mi, r_mi = self._ap_pr(y_true_all, y_score_all) if y_true_all else (0.0, [1.0], [0.0])
            pr_micro[key] = {"precision": [float(x) for x in p_mi], "recall": [float(x) for x in r_mi]}
            ap_micro.append(ap_mi)

            # macro avg curve
            stack = []
            for lab in label_names:
                rr = np.asarray(pr_curves_per_thr[thr].get(lab, {}).get("recall", []), float)
                pp = np.asarray(pr_curves_per_thr[thr].get(lab, {}).get("precision", []), float)
                if rr.size > 1 and pp.size > 1:
                    o = np.argsort(rr)
                    pi = np.interp(recall_grid, rr[o], pp[o], left=pp[o][0], right=pp[o][-1])
                    pi = np.maximum.accumulate(pi[::-1])[::-1]
                    stack.append(pi)
            pm = np.mean(np.stack(stack, 0), 0) if stack else np.zeros_like(recall_grid)
            pr_macro[key] = {"precision": [float(x) for x in pm], "recall": [float(x) for x in recall_grid]}

            # macro AP avg
            ap_macro.append(float(np.mean([ap_per_label_per_thr[lab][ti] for lab in label_names])) if num_labels else 0.0)

        # aggregates
        avg_mode = str(kwargs.get("average", "macro")).lower()
        map_macro = float(np.mean(ap_macro)) if ap_macro else 0.0
        map_micro = float(np.mean(ap_micro)) if ap_micro else 0.0

        def pick_macro(t: float) -> float:
            return float(ap_macro[int(np.argmin([abs(x - t) for x in thresholds]))]) if ap_macro else 0.0

        def pick_micro(t: float) -> float:
            return float(ap_micro[int(np.argmin([abs(x - t) for x in thresholds]))]) if ap_micro else 0.0

        map_macro_50, map_macro_75 = pick_macro(0.5), pick_macro(0.75)
        map_micro_50, map_micro_75 = pick_micro(0.5), pick_micro(0.75)
        idx_05 = int(np.argmin([abs(x - 0.5) for x in thresholds])) if num_thresholds else 0
        per_class_ap_05 = [float(ap_per_label_per_thr[lab][idx_05]) for lab in label_names] if num_labels and num_thresholds else []
        gt_counts = [sum(len(v) for v in (gt_by.get(lab, {}) or {}).values()) for lab in label_names]
        pred_counts = [len(pr_by.get(lab, []) or []) for lab in label_names]
        data = {
            "categories": label_names,
            "gt_counts": gt_counts,
            "pred_counts": pred_counts,
            "per_class_ap": per_class_ap_05,
            "per_class_ap_avg": [float(np.mean(ap_per_label_per_thr[lab])) if num_thresholds else 0.0 for lab in label_names],
            "map_macro": map_macro,
            "map_micro": map_micro,
            "mAP_macro@0.5": map_macro_50,
            "mAP_macro@0.75": map_macro_75,
            "mAP_micro@0.5": map_micro_50,
            "mAP_micro@0.75": map_micro_75,
            "ap_iou_macro": [float(x) for x in ap_macro],
            "ap_iou_micro": [float(x) for x in ap_micro],
            "iou_thresholds": [float(x) for x in thresholds],
            "pr_macro": pr_macro,
            "pr_micro": pr_micro,
            "pr_curves_per_threshold": {f"{thr:.2f}": pr_curves_per_thr[thr] for thr in thresholds},
        }

        # compatibility
        if avg_mode == "micro":
            data.update({
                "map": map_micro,
                "map_50": map_micro_50,
                "map_75": map_micro_75,
                "mAP@0.5": map_micro_50,
                "mAP@0.75": map_micro_75,
            })
            final_score = map_micro
        else:
            data.update({
                "map": map_macro,
                "map_50": map_macro_50,
                "map_75": map_macro_75,
                "mAP@0.5": map_macro_50,
                "mAP@0.75": map_macro_75,
            })
            final_score = map_macro

        return MetricOutputModel(self.name, final_score, Stats(data=data, meta=Meta(doc=(self.__class__.__doc__ or '').strip())))
class MeanIntersectionOverUnion(Metric):
    name = "mIoU"

    # flat mask
    def _flat_mask(self, points: List[CocoAnnotation], file_key: str, h: int, w: int, label_key: Optional[str] = None) -> np.ndarray:
        acc = np.zeros((h, w), np.uint8)
        for ann in points:
            same_file = str(ann.file_name or "f") == file_key
            same_label = (label_key is None) or (str(ann.label) == label_key)
            if not (same_file and same_label):
                continue
            m = ann.mask
            if isinstance(m, np.ndarray) and m.shape[:2] == (h, w):
                acc |= (m > 0).astype(np.uint8)
        return acc.reshape(-1)

    # gather space
    def _space(self, gt: DatasetModel, pr: List[CocoAnnotation]) -> Tuple[List[CocoAnnotation], List[CocoAnnotation], List[str], List[str], int, int]:
        gt_pts = gt.data_points or []
        pr_pts = pr or []
        files = sorted({str(a.file_name or "f") for a in (gt_pts + pr_pts)})
        labels = sorted({str(a.label) for a in (gt_pts + pr_pts)})
        h, w = next(((int(a.mask.shape[0]), int(a.mask.shape[1]))
                     for a in (gt_pts + pr_pts)
                     if isinstance(a.mask, np.ndarray) and a.mask.ndim >= 2),
                    (0, 0))
        return gt_pts, pr_pts, files, labels, h, w

    def compute(self, gt: DatasetModel, prediction: List[CocoAnnotation], **kwargs) -> MetricOutputModel:
        gt_pts, pr_pts, files, labels, h, w = self._space(gt, prediction)
        # MICRO
        y_true_micro = np.concatenate([self._flat_mask(gt_pts, f, h, w, None) for f in files], 0)
        y_pred_micro = np.concatenate([self._flat_mask(pr_pts, f, h, w, None) for f in files], 0)
        micro_iou = float(jaccard_score(y_true_micro, y_pred_micro, average="binary", zero_division=0))

        # MACRO
        per_class = []
        for lab in labels:
            y_true_lab = np.concatenate([self._flat_mask(gt_pts, f, h, w, lab) for f in files], 0)
            y_pred_lab = np.concatenate([self._flat_mask(pr_pts, f, h, w, lab) for f in files], 0)
            iou_lab = float(jaccard_score(y_true_lab, y_pred_lab, average="binary", zero_division=0))
            per_class.append({"label": lab, "iou": iou_lab})
        macro_iou = float(np.mean([c["iou"] for c in per_class])) if per_class else micro_iou

        data = {
            "micro_iou": micro_iou,
            "macro_iou": macro_iou,
            "per_class": per_class,
            "labels": labels
        }
        return MetricOutputModel(
            self.name,
            micro_iou,
            Stats(data=data, meta=Meta(doc=(self.__class__.__doc__ or "").strip(), formula="IoU = |X∩Y| / |X∪Y|"))
        )
class DiceCoefficient(Metric):
    name="dice"
    def compute(self,gt:DatasetModel,prediction:List[CocoAnnotation],**kwargs)->MetricOutputModel:

        miou = MeanIntersectionOverUnion()
        gt_pts, pr_pts, files, labels, h, w = miou._space(gt, prediction)


        y_true = np.concatenate([miou._flat_mask(gt_pts, f, h, w, None) for f in files], 0) if files else np.array([], dtype=np.uint8)
        y_pred = np.concatenate([miou._flat_mask(pr_pts, f, h, w, None) for f in files], 0) if files else np.array([], dtype=np.uint8)
        
        s = float(f1_score(y_true, y_pred, average="binary", zero_division=0)) if y_true.size and y_pred.size else 0.0
        return MetricOutputModel(
            self.name,
            s,
            Stats(meta=Meta(doc=(self.__class__.__doc__ or '').strip(), formula='Dice = 2PR/(P+R)'))
        )

class ClassificationReportMetric(Metric):
    name="classification_report"
    def compute(self, gt: DatasetModel, prediction: List[CocoAnnotation], **kwargs) -> MetricOutputModel:
    # inputs
        gt_pts = gt.data_points or []
        pr_pts = prediction or []

        # files list
        gt_files = [str(a.file_name) for a in gt_pts if a.file_name is not None]
        pr_files = [str(a.file_name) for a in pr_pts if a.file_name is not None]
        files = sorted(set(gt_files) | set(pr_files))
        # file labels
        def labels_for(points: List[CocoAnnotation], fname: str) -> List[str]:
            return [
                str(a.label)
                for a in points
                if a.file_name is not None and str(a.file_name) == fname and a.label is not None
            ]

        # build pairs
        y_true: List[str] = []
        y_pred: List[str] = []
        for fname in files:
            gt_labels = labels_for(gt_pts, fname)
            pr_labels = labels_for(pr_pts, fname)

            g = Counter(gt_labels).most_common(1)[0][0] if gt_labels else "none"
            p = Counter(pr_labels).most_common(1)[0][0] if pr_labels else "none"
            if g != "none" and p != "none":
                y_true.append(g)
                y_pred.append(p)
        
        report_text = classification_report(y_true, y_pred, output_dict=False, zero_division=0)
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        accuracy = float(np.mean([a == b for a, b in zip(y_true, y_pred)]))

        all_labels = sorted(set(y_true) | set(y_pred))
        data = {
            "num_files": len(files),
            "labels": all_labels,
            "text": report_text,
            "dict": report_dict,
            "accuracy": accuracy,
        }
        return MetricOutputModel(
            self.name,
            accuracy,
            Stats(data=data, meta=Meta(doc=(self.__class__.__doc__ or "").strip())),
        )
