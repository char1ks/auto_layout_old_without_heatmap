from pathlib import Path
import numpy as np
import pytest
from evaluate.Example_datasets.voc_dataset import voc_dataset

STATIC_DIR = Path(__file__).resolve().parents[2] / "static"

def test_from_path_meta_and_annotations(tmp_path: Path):
    ds = voc_dataset.from_path(path=STATIC_DIR, ann_dir=STATIC_DIR, img_dir=STATIC_DIR)
    model = ds.data
    assert model.meta.dataset_type == "voc_detection"
    assert model.meta.total_images is not None and model.meta.total_images >= 1
    assert model.meta.total_annotations is not None and model.meta.total_annotations >= 3
    cats = set(model.meta.categories)
    assert {"pineapple", "snake fruit", "dragon fruit"}.issubset(cats)
    anns = [a for a in model.data_points if a.file_name == "fruit1.png"]
    assert len(anns) == 3
    pine = next(a for a in anns if str(a.label) == "pineapple")
    assert pine.bbox == [pytest.approx(37.0), pytest.approx(86.0), pytest.approx(238.0), pytest.approx(155.0)]
    x, y, w, h = pine.bbox
    x1, y1 = int(np.floor(x)), int(np.floor(y))
    x2, y2 = int(np.ceil(x + w)), int(np.ceil(y + h))
    area_mask = int((pine.mask[y1:y2, x1:x2] == 1).sum())
    assert area_mask == int(pine.area)


def test__parse_single_voc_xml_swapped():
    xml_path = STATIC_DIR / "fruit1.xml"
    anns, cats = voc_dataset._parse_single_voc_xml(xml_path, img_dir=STATIC_DIR)
    assert len(anns) == 3
    ann = next(a for a in anns if str(a.label) == "pineapple")
    assert ann.file_name == "fruit1.png"
    assert ann.width == 400 and ann.height == 300
    assert {"pineapple", "snake fruit", "dragon fruit"}.issubset(set(cats))
    assert ann.bbox == [pytest.approx(37.0), pytest.approx(86.0), pytest.approx(238.0), pytest.approx(155.0)]
    x1, y1 = int(np.floor(ann.bbox[0])), int(np.floor(ann.bbox[1]))
    x2, y2 = int(np.ceil(ann.bbox[0] + ann.bbox[2])), int(np.ceil(ann.bbox[1] + ann.bbox[3]))
    area_mask = int((ann.mask[y1:y2, x1:x2] == 1).sum())
    assert area_mask == int(ann.area)


def test__parse_single_voc_xml_swapped_dims():
    xml_path = STATIC_DIR / "fruit1.xml"
    anns, cats = voc_dataset._parse_single_voc_xml(xml_path, img_dir=STATIC_DIR)
    assert len(anns) == 3
    assert {"pineapple", "snake fruit", "dragon fruit"}.issubset(set(cats))