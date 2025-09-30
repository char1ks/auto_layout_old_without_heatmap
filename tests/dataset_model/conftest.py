import pytest
import numpy as np
from typing import List
from pathlib import Path
from PIL import Image
import sys
from COCOAnnotations import COCOAnnotation
from DatasetModel import DatasetModel
from DatasetMeta import DatasetMeta
EVAL_CLASSES_PATH = Path(__file__).parent.parent.parent / "evaluate"
sys.path.insert(0, str(EVAL_CLASSES_PATH))

@pytest.fixture
def coco_annotations_list() -> List[COCOAnnotation]:
    static_dir = Path(__file__).parent.parent / "static"
    width_default, height_default = 400, 300
    static_data = [
        ("fruit0.png", "pineapple", [38, 82, 233, 145]),
        ("fruit0.png", "snake fruit", [244, 174, 36, 33]),
        ("fruit0.png", "dragon fruit", [254, 228, 97, 72]),
        ("fruit1.png", "pineapple", [38, 87, 237, 154]),
        ("fruit1.png", "snake fruit", [240, 185, 39, 35]),
        ("fruit1.png", "dragon fruit", [256, 244, 90, 56]),
        ("fruit2.png", "pineapple", [92, 115, 156, 83]),
        ("fruit2.png", "snake fruit", [217, 185, 26, 24]),
        ("fruit2.png", "dragon fruit", [212, 217, 59, 52]),
        ("fruit3.png", "pineapple", [82, 101, 162, 96]),
        ("fruit3.png", "snake fruit", [216, 169, 26, 23]),
        ("fruit3.png", "dragon fruit", [215, 202, 62, 59]),
        ("fruit4.png", "pineapple", [70, 80, 184, 109]),
        ("fruit4.png", "snake fruit", [228, 153, 27, 27]),
        ("fruit4.png", "dragon fruit", [230, 191, 72, 60]),
        ("fruit5.png", "pineapple", [52, 69, 203, 130]),
        ("fruit5.png", "snake fruit", [228, 147, 34, 29]),
        ("fruit5.png", "dragon fruit", [241, 189, 80, 67]),
    ]
    anns: List[COCOAnnotation] = []
    for file_name, label, bbox in static_data:
        img_path = static_dir / file_name
        if img_path.exists():
            img = np.array(Image.open(img_path).convert("RGB"))
            height, width = img.shape[:2]
        else:
            width, height = width_default, height_default
            img = np.zeros((height, width, 3), dtype=np.uint8)
        x, y, w, h = bbox
        mask = np.zeros((height, width), dtype=np.uint8)
        x0 = int(max(0, np.floor(x)))
        y0 = int(max(0, np.floor(y)))
        x1 = int(min(width, np.ceil(x + w)))
        y1 = int(min(height, np.ceil(y + h)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1
        area = float(np.sum(mask > 0))
        anns.append(
            COCOAnnotation(
                img=img,
                mask=mask,
                label=label,
                image_size=(width, height),
                width=width,
                height=height,
                area=area,
                file_name=file_name,
                bbox=[float(x), float(y), float(w), float(h)],
                score=None,
                confidence=None,
            )
        )
    return anns

@pytest.fixture
def meta_valid_dict():
    return {
        "name": "StaticVOC",
        "categories": ["pineapple", "snake fruit", "dragon fruit"],
        "color_channels": ["RGB"],
        "total_images": 6,
        "total_annotations": 18,
        "dataset_type": "voc_static",
        "url": None,
    }

@pytest.fixture
def meta_mixed_types_dict():
    return {
        "name": "MixedDS",
        "categories": ["A", "B"],
        "color_channels": ["R", "G", "B"],
        "total_images": "not_a_number",
        "total_annotations": "oops",
    }

@pytest.fixture
def dataset_model_from_meta_dict(coco_annotations_list, meta_valid_dict):
    return DatasetModel(data_points=coco_annotations_list, meta=meta_valid_dict)

@pytest.fixture
def dataset_model_from_meta_obj(coco_annotations_list, meta_valid_dict):
    meta = DatasetMeta.from_dict(meta_valid_dict)
    return DatasetModel(data_points=coco_annotations_list, meta=meta)