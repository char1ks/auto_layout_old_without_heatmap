import json
from pathlib import Path
from typing import List
import numpy as np
import xml.etree.ElementTree as ET
import sys
from pathlib import Path as _P
EVAL_CLASSES_PATH = _P(__file__).resolve().parents[2] / "evaluate"
sys.path.insert(0, str(EVAL_CLASSES_PATH))
from COCOAnnotations import COCOAnnotation
from DatasetModel import DatasetModel
from Example_datasets.ArchiveVOCDataset import ArchiveVOCDataset
STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
RESULTS_DIR = STATIC_DIR / "results_fruit1"
FRUIT_XML = STATIC_DIR / "fruit1.xml"
ANN_JSON = RESULTS_DIR / "annotations.json"

def _voc_gt_dataset() -> DatasetModel:
    anns, _ = ArchiveVOCDataset._parse_single_voc_xml(FRUIT_XML, STATIC_DIR)
    return DatasetModel(data_points=anns, meta={"name": "fruit1_gt"})

def _preds_from_annotations() -> List[COCOAnnotation]:
    with ANN_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("metadata", {})
    file_name = meta.get("image_name", "fruit1.png")
    W = int(meta.get("image_width", 0) or 0)
    H = int(meta.get("image_height", 0) or 0)
    img_arr = np.zeros((H, W, 3), dtype=np.uint8)
    preds: List[COCOAnnotation] = []
    for d in data.get("detections", []):
        bbox = d.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x, y, w, h = [float(v) for v in bbox]
        mask = np.zeros((H, W), dtype=np.uint8)
        x1 = int(max(0, np.floor(x)))
        y1 = int(max(0, np.floor(y)))
        x2 = int(min(W, np.ceil(x + w)))
        y2 = int(min(H, np.ceil(y + h)))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
        label = d.get("class", "object")
        conf = float(d.get("confidence", 1.0) or 1.0)
        area = float(d.get("area", w * h))
        preds.append(COCOAnnotation(
            img=img_arr,
            mask=mask,
            label=label,
            image_size=(W, H),
            width=W,
            height=H,
            area=area,
            file_name=file_name,
            bbox=[x, y, w, h],
            confidence=conf,
            score=conf,
        ))
    return preds
def build_gt_dataset() -> DatasetModel:
    return _voc_gt_dataset()

def build_predictions() -> List[COCOAnnotation]:
    return _preds_from_annotations()

def build_perfect_predictions_from_gt() -> List[COCOAnnotation]:
    gt = _voc_gt_dataset()
    preds: List[COCOAnnotation] = []
    for a in gt.data_points:
        mask = (a.mask > 0).astype(np.uint8)
        preds.append(COCOAnnotation(
            img=a.img,
            mask=mask,
            label=a.label,
            image_size=a.image_size,
            width=a.width,
            height=a.height,
            area=float(a.area),
            file_name=a.file_name,
            bbox=list(a.bbox),
            confidence=1.0,
            score=1.0,
        ))
    return preds


def build_empty_predictions() -> List[COCOAnnotation]:
    return []


def build_empty_gt() -> DatasetModel:
    return DatasetModel(data_points=[], meta={"name": "empty_gt"})