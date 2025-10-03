from __future__ import annotations
import abc
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional,Tuple
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score, jaccard_score, f1_score
from evaluate.DatasetModel import DatasetModel
from evaluate.COCOAnnotations import COCOAnnotation
from shapely.geometry import box as _box_poly

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

class Metric(abc.ABC):
    name:str="metric"
    @abc.abstractmethod
    def compute(self, gt:DatasetModel, prediction:List[COCOAnnotation], **kwargs)->MetricOutputModel: ...

def _iou(a: List[float], b: List[float]) -> float:
    pa, pb = _box_poly(a[0], a[1], a[2], a[3]), _box_poly(b[0], b[1], b[2], b[3])
    inter = pa.intersection(pb).area
    return float(inter / (pa.area + pb.area - inter)) if inter > 0 else 0.0

class MeanAveragePrecision(Metric):
    name = "mAP"

    def __init__(self, iou_thresholds: Optional[List[float]] = None) -> None:
        self.iou_thresholds = iou_thresholds or np.arange(0.5, 0.95 + 1e-9, 0.05).tolist()
    def _group(self, ground_truth: DatasetModel, predictions: List[COCOAnnotation]):
        ground_truth_by: dict[str, dict[str, List[List[float]]]] = {}
        predictions_by: dict[str, List[Tuple[str, List[float], float]]] = {}
        label_names = set()

        for annotation in (ground_truth.data_points or []):
            bbox = ([float(x) if np.isfinite(float(x)) else 0.0 for x in annotation.bbox] if isinstance(annotation.bbox,(list,tuple)) and len(annotation.bbox)==4 else None)
            if annotation.file_name is None or bbox is None: 
                continue
            file_key = str(annotation.file_name); label_key = str(annotation.label)
            label_names.add(label_key)

            ground_truth_by.setdefault(label_key, {}).setdefault(file_key, []).append([
                float(bbox[0]), float(bbox[1]), float(bbox[0]) + float(bbox[2]), float(bbox[1]) + float(bbox[3])
            ])

        for annotation in (predictions or []):
            bbox = ([float(x) if np.isfinite(float(x)) else 0.0 for x in annotation.bbox] if isinstance(annotation.bbox,(list,tuple)) and len(annotation.bbox)==4 else None)
            if annotation.file_name is None or bbox is None: 
                continue
            file_key = str(annotation.file_name); label_key = str(annotation.label)
            label_names.add(label_key)
            score_value = float(
                annotation.score if hasattr(annotation, 'score') and annotation.score is not None
                else (annotation.confidence if hasattr(annotation, 'confidence') and annotation.confidence is not None else 1.0)
            )

            predictions_by.setdefault(label_key, []).append((
                file_key,
                [float(bbox[0]), float(bbox[1]), float(bbox[0]) + float(bbox[2]), float(bbox[1]) + float(bbox[3])],
                float(score_value)
            ))

        for label_key in predictions_by:
            predictions_by[label_key].sort(key=lambda t: t[2], reverse=True)

        return ground_truth_by, predictions_by, sorted(label_names)
    def _match(self,predictions_for_label: List[Tuple[str, List[float], float]],ground_truth_by_file: dict[str, List[List[float]]],iou_threshold: float,):
        used_indices = {fname: set() for fname in ground_truth_by_file}
        y_true: List[int] = []
        y_score: List[float] = []

        for file_name, pred_box, score in predictions_for_label:
            gt_boxes = ground_truth_by_file.get(file_name, [])
            if not gt_boxes:
                y_true.append(0); y_score.append(float(score)); continue

            candidates = [(i, _iou(pred_box, gt)) for i, gt in enumerate(gt_boxes) if i not in used_indices[file_name]]
            best_index, best_iou = max(candidates, key=lambda t: t[1], default=(-1, 0.0))

            hit = (best_index >= 0) and (best_iou >= iou_threshold)
            y_true.append(1 if hit else 0)
            y_score.append(float(score))
            if hit:
                used_indices[file_name].add(best_index)
        return y_true, y_score

    def _ap_pr(self, y_true: List[int], y_score: List[float]):
        # вернём нулевой AP и базовую PR-кривую,если вход пустой
        if not y_true:
            return 0.0, [1.0], [0.0]

        # AP по меткам и скору
        ap_value = float(average_precision_score(y_true, y_score))
        # Строим PR-кривую
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        
        # Приводим к float-массивам
        recall = np.asarray(recall, float)
        precision = np.asarray(precision, float)
        # Сортируем по recall
        order = np.argsort(recall)
        recall, precision = recall[order], precision[order]
        # Удаляем дубликаты по recall
        recall, uniq_idx = np.unique(recall, return_index=True)
        precision = precision[uniq_idx]
        # Монотонное сглаживание precision (не возрастает с ростом recall)
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        
        return float(ap_value), [float(x) for x in precision], [float(x) for x in recall]

    def compute(self, ground_truth: DatasetModel, prediction: List[COCOAnnotation], **kwargs) -> MetricOutputModel:
        gt_by, pr_by, label_names = self._group(ground_truth, prediction)
        thresholds = [float(t) for t in self.iou_thresholds]
        L, T = len(label_names), len(thresholds)

        # AP/PR хранилища
        ap_per_label_per_thr = {lab: [0.0] * T for lab in label_names}
        pr_curves_per_thr = {thr: {} for thr in thresholds}
        y_store = {thr: {} for thr in thresholds}

        # посчитаем per-label для каждого порога
        for threshold_index, thr in enumerate(thresholds):
            for lab in label_names:
                preds_lab = pr_by.get(lab, [])
                gts_lab = gt_by.get(lab, {})
                y_true, y_score = self._match(preds_lab, gts_lab, thr)
                y_store[thr][lab] = (y_true, y_score)
                ap_val, prec, rec = self._ap_pr(y_true, y_score)
                ap_per_label_per_thr[lab][threshold_index] = ap_val
                pr_curves_per_thr[thr][lab] = {"precision": prec, "recall": rec}

        # micro / macro
        recall_grid = np.linspace(0, 1, 101)
        pr_micro = {}
        pr_macro = {}
        ap_micro, ap_macro = [], []

        for ti, thr in enumerate(thresholds):
            key = f"{thr:.2f}"

            # micro: объединяем все метки
            y_true_all = [y for lab in label_names for y in y_store[thr][lab][0]]
            y_score_all = [s for lab in label_names for s in y_store[thr][lab][1]]
            ap_mi, p_mi, r_mi = self._ap_pr(y_true_all, y_score_all) if y_true_all else (0.0, [1.0], [0.0])
            ap_micro.append(ap_mi)
            pr_micro[key] = {"precision": [float(x) for x in p_mi], "recall": [float(x) for x in r_mi]}

            # macro: усредняем интерполированные PR по меткам
            stack = []
            for lab in label_names:
                rr = np.asarray(pr_curves_per_thr[thr][lab]["recall"], float)
                pp = np.asarray(pr_curves_per_thr[thr][lab]["precision"], float)
                if rr.size > 1 and pp.size > 1:
                    o = np.argsort(rr)
                    pi = np.interp(recall_grid, rr[o], pp[o], left=pp[o][0], right=pp[o][-1])
                    pi = np.maximum.accumulate(pi[::-1])[::-1]
                    stack.append(pi)
            pm = np.mean(np.stack(stack, 0), 0) if stack else np.zeros_like(recall_grid)
            pr_macro[key] = {"precision": [float(x) for x in pm], "recall": [float(x) for x in recall_grid]}

            # macro-AP как среднее AP по меткам для данного порога
            ap_macro.append(float(np.mean([ap_per_label_per_thr[lab][ti] for lab in label_names])) if L else 0.0)

        # агрегаты
        map_overall = float(np.mean(ap_macro)) if ap_macro else 0.0
        pick = lambda t: float(ap_macro[int(np.argmin([abs(x - t) for x in thresholds]))]) if ap_macro else 0.0
        map_50, map_75 = pick(0.5), pick(0.75)
        idx_05 = int(np.argmin([abs(x - 0.5) for x in thresholds])) if T else 0
        per_class_ap_05 = [float(ap_per_label_per_thr[lab][idx_05]) for lab in label_names] if L and T else []

        # counts
        gt_counts = [sum(len(v) for v in (gt_by.get(lab, {}) or {}).values()) for lab in label_names]
        pred_counts = [len(pr_by.get(lab, []) or []) for lab in label_names]

        data = {
            "categories": label_names,
            "gt_counts": gt_counts,
            "pred_counts": pred_counts,
            "per_class_ap": per_class_ap_05,
            "per_class_ap_avg": [float(np.mean(ap_per_label_per_thr[lab])) if T else 0.0 for lab in label_names],
            "map": map_overall,
            "map_50": map_50,
            "map_75": map_75,
            "mAP@0.5": map_50,
            "mAP@0.75": map_75,
            "ap_iou_macro": [float(x) for x in ap_macro],
            "ap_iou_micro": [float(x) for x in ap_micro],
            "iou_thresholds": [float(x) for x in thresholds],
            "pr_macro": pr_macro,
            "pr_micro": pr_micro,
            "pr_curves_per_threshold": {f"{thr:.2f}": pr_curves_per_thr[thr] for thr in thresholds},
        }

        return MetricOutputModel(self.name, map_overall, Stats(payload=data, meta=Meta(doc=(self.__class__.__doc__ or '').strip())))

class MeanIntersectionOverUnion(Metric):
    name="mIoU"

    # Собирает бинарные векторы y_true и y_pred для IoU/Dice мержит маски по файлам и конкатенирует
    def _stack(self, gt: DatasetModel, pr: List[COCOAnnotation]):
        gt_pts = gt.data_points or []
        pr_pts = pr or []
        files = sorted({str(a.file_name or 'f') for a in (gt_pts + pr_pts)})

        height = int(gt.image_height or 0)
        width  = int(gt.image_width  or 0)
        if height <= 0 or width <= 0:
            for a in (gt_pts + pr_pts):
                mask = a.mask
                if isinstance(mask, np.ndarray) and mask.ndim >= 2:
                    height, width = int(mask.shape[0]), int(mask.shape[1])
                    break
        if height <= 0 or width <= 0:
            return np.zeros((1,), np.uint8), np.zeros((1,), np.uint8)

        # Мержит все валидные маски одного файла в один плоский вектор (0/1)
        def merged_mask(points: List[COCOAnnotation], fname: str) -> np.ndarray:
            acc = np.zeros((height, width), np.uint8)
            for a in points:
                if str(a.file_name or 'f') != fname:
                    continue
                mask = a.mask
                if isinstance(mask, np.ndarray) and mask.shape[:2] == (height, width):
                    acc |= (mask > 0).astype(np.uint8)
            return acc.reshape(-1)

        y_true_list, y_pred_list = [], []
        for fname in files:
            y_true_list.append(merged_mask(gt_pts, fname))
            y_pred_list.append(merged_mask(pr_pts, fname))

        y_true = np.concatenate(y_true_list, 0) if y_true_list else np.zeros((1,), np.uint8)
        y_pred = np.concatenate(y_pred_list, 0) if y_pred_list else np.zeros((1,), np.uint8)

        return y_true, y_pred
    
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
    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation], **kwargs) -> MetricOutputModel:
    # входные точки
        gt_pts = gt.data_points or []
        pr_pts = prediction or []

        # список файлов (уникальные имена из GT и предсказаний)
        gt_files = [str(a.file_name) for a in gt_pts if a.file_name is not None]
        pr_files = [str(a.file_name) for a in pr_pts if a.file_name is not None]
        files = sorted(set(gt_files) | set(pr_files))
        # вытаскиваем все метки по файлу 
        def labels_for(points: List[COCOAnnotation], fname: str) -> List[str]:
            return [
                str(a.label)
                for a in points
                if a.file_name is not None and str(a.file_name) == fname and a.label is not None
            ]

        # метка по списку значений
        def majority(v: List[str]) -> str:
            if not v:
                return "none"
            u, c = np.unique([str(x) for x in v], return_counts=True)
            return str(u[int(np.argmax(c))])

        # собираем пары истинной/предсказанной меток на уровне файла
        y_true: List[str] = []
        y_pred: List[str] = []
        for fname in files:
            g = majority(labels_for(gt_pts, fname))
            p = majority(labels_for(pr_pts, fname))
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
