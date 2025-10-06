import json
from pathlib import Path
from typing import List
import numpy as np
import pytest
from evaluate.coco_annotation import CocoAnnotation
from evaluate.dataset_model import DatasetModel
from evaluate.dataset_meta import DatasetMeta
from evaluate.example_datasets.voc_dataset import VocDataset

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
RESULTS_DIR = STATIC_DIR / "results_fruit1"
FRUIT_XML = STATIC_DIR / "fruit1.xml"
ANN_JSON = RESULTS_DIR / "annotations.json"


def _voc_gt_dataset() -> DatasetModel:
    anns, _ = VocDataset._parse_single_voc_xml(FRUIT_XML, STATIC_DIR)
    return DatasetModel(data_points=anns, meta=DatasetMeta(name="fruit1_gt"))


def _preds_from_annotations() -> List[CocoAnnotation]:
    with ANN_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("metadata", {})
    file_name = meta.get("image_name", "fruit1.png")
    W = int(meta.get("image_width", 0) or 0)
    H = int(meta.get("image_height", 0) or 0)
    img_arr = np.zeros((H, W, 3), dtype=np.uint8)
    preds: List[CocoAnnotation] = []
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
        preds.append(CocoAnnotation(
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


@pytest.fixture
def build_gt_dataset() -> DatasetModel:
    return _voc_gt_dataset()


@pytest.fixture
def build_predictions() -> List[CocoAnnotation]:
    return _preds_from_annotations()


@pytest.fixture
def build_perfect_predictions_from_gt() -> List[CocoAnnotation]:
    gt = _voc_gt_dataset()
    preds: List[CocoAnnotation] = []
    for a in gt.data_points:
        mask = (a.mask > 0).astype(np.uint8)
        preds.append(CocoAnnotation(
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


@pytest.fixture
def build_empty_predictions() -> List[CocoAnnotation]:
    return []


@pytest.fixture
def build_empty_gt() -> DatasetModel:
    return DatasetModel(data_points=[], meta=DatasetMeta(name="empty_gt"))