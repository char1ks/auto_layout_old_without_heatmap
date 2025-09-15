from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, List, Dict, Tuple, Iterable, Optional

from .DatasetModel import DatasetModel
from .COCOAnnotations import COCOAnnotation


@dataclass
class MetricOutputModel:
    metric_name: str
    score: float
    uid: Optional[str] = None
    stats: dict[str, Any] = field(default_factory=dict)


class Metric(abc.ABC):
    name: str = "metric"

    def compute(self, gt: DatasetModel, prediction: List[COCOAnnotation]) -> MetricOutputModel:
        gt_by_file: Dict[str, List[COCOAnnotation]] = {}
        for ann in gt.data_points:
            fname = getattr(ann, "file_name", "") or ""
            gt_by_file.setdefault(fname, []).append(ann)

        pred_by_file: Dict[str, List[COCOAnnotation]] = {}
        for ann in prediction:
            fname = getattr(ann, "file_name", "") or ""
            pred_by_file.setdefault(fname, []).append(ann)

        ious: List[float] = []
        pairs: List[dict] = []
        matched_images = 0

        for fname, gt_list in gt_by_file.items():
            preds = pred_by_file.get(fname)
            if not preds:
                continue
            matched_images += 1
            n = min(len(gt_list), len(preds))
            for i in range(n):
                iou = self._iou(gt_list[i].bbox, preds[i].bbox)
                ious.append(iou)
                pairs.append({
                    "file_name": fname,
                    "gt_uid": getattr(gt_list[i], "uid", None),
                    "pred_uid": getattr(preds[i], "uid", None),
                    "iou": iou,
                })

        mean_iou = (sum(ious) / len(ious)) if ious else 0.0

        # mAP по file_name/label
        map_score = self._map(gt.data_points, prediction)

        stats = {
            "mean_iou": mean_iou,
            "mAP": map_score,
            "num_images_matched": matched_images,
            "num_pairs": len(pairs),
            "pairs": pairs,
            "map_iou_thresholds": [t / 100 for t in range(50, 100, 5)],
        }

        return MetricOutputModel(metric_name="combined", score=mean_iou, stats=stats)

    def _iou(self, box_a: Iterable[float], box_b: Iterable[float]) -> float:
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        ax2, ay2 = ax + max(0.0, aw), ay + max(0.0, ah)
        bx2, by2 = bx + max(0.0, bw), by + max(0.0, bh)

        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)

        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, (ax2 - ax)) * max(0.0, (ay2 - ay))
        area_b = max(0.0, (bx2 - bx)) * max(0.0, (by2 - by))
        union = area_a + area_b - inter
        return 0.0 if union <= 0.0 else inter / union

    def _map(self,gt_anns: List[COCOAnnotation],pred_anns: List[COCOAnnotation],iou_thresholds: List[float] = [t / 100 for t in range(50, 100, 5)], ) -> float:
        gt_by_key: Dict[Tuple[Any, Any], List[COCOAnnotation]] = {}
        for g in gt_anns:
            fname = getattr(g, "file_name", None)
            label = getattr(g, "label", None)
            gt_by_key.setdefault((fname, label), []).append(g)
        categories = sorted({getattr(g, "label", None) for g in gt_anns})
        if not categories:
            return 0.0
        aps_per_t: List[float] = []
        for thr in iou_thresholds:
            aps_per_cat: List[float] = []
            for cat in categories:
                gt_used: Dict[Tuple[Any, int], bool] = {}
                gt_per_img: Dict[Any, List[COCOAnnotation]] = {}
                for (fname, c), lst in gt_by_key.items():
                    if c == cat:
                        gt_per_img[fname] = lst
                preds_cat = [p for p in pred_anns if getattr(p, "label", None) == cat]
                preds_cat.sort(key=lambda x: getattr(x, "score", 0.0), reverse=True)
                tps: List[int] = []
                fps: List[int] = []
                for p in preds_cat:
                    best_iou = 0.0
                    best_idx = -1
                    p_fname = getattr(p, "file_name", None)
                    gts = gt_per_img.get(p_fname, [])
                    for gi, g in enumerate(gts):
                        key = (p_fname, gi)
                        if gt_used.get(key, False):
                            continue
                        iou = self._iou(p.bbox, g.bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = gi
                    if best_iou >= thr and best_idx >= 0:
                        gt_used[(p_fname, best_idx)] = True
                        tps.append(1)
                        fps.append(0)
                    else:
                        tps.append(0)
                        fps.append(1)
                tp_cum, fp_cum = 0, 0
                precisions: List[float] = []
                recalls: List[float] = []
                total_gt = sum(len(v) for v in gt_per_img.values())
                if total_gt == 0:
                    aps_per_cat.append(0.0)
                    continue
                for tp, fp in zip(tps, fps):
                    tp_cum += tp
                    fp_cum += fp
                    prec = tp_cum / max(1, (tp_cum + fp_cum))
                    rec = tp_cum / total_gt
                    precisions.append(prec)
                    recalls.append(rec)
                # 101-точечная аппроксимация AP
                ap = 0.0
                for r_i in range(101):
                    r_target = r_i / 100.0
                    prec_at_r = 0.0
                    for pr, rc in zip(precisions, recalls):
                        if rc >= r_target and pr > prec_at_r:
                            prec_at_r = pr
                    ap += prec_at_r
                ap /= 101.0
                aps_per_cat.append(ap)
            aps_per_t.append(sum(aps_per_cat) / len(aps_per_cat) if aps_per_cat else 0.0)
        return sum(aps_per_t) / len(aps_per_t) if aps_per_t else 0.0
