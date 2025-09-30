import json
from pathlib import Path
import numpy as np
from evaluate.Example_datasets.COCODataset import ChickenDataset


def build_coco_like(minimal: bool = False) -> dict:
    images = [
        {"id": 1, "file_name": "img1.jpg", "width": 100, "height": 80},
        {"id": 2, "file_name": "img2.jpg", "width": 50, "height": 40},
    ]
    categories = [
        {"id": 10, "name": "cat"},
        {"id": 20, "name": "dog"},
    ]
    if minimal:
        annotations = [
            {"id": 1, "image_id": 1, "category_id": 10, "bbox": [10, 5, 20, 10]},
        ]
    else:
        annotations = [
            {"id": 1, "image_id": 1, "category_id": 10, "bbox": [10, 5, 20, 10]},
            {"id": 2, "image_id": 2, "category_id": 20, "segmentation": [[0, 0, 30, 0, 30, 20, 0, 20]], "area": 600},
        ]
    return {"images": images, "annotations": annotations, "categories": categories}


def test_build_annotations_bbox_and_poly_mask():
    obj = build_coco_like(minimal=False)
    anns = ChickenDataset._build_annotations(obj, base_dir=None)
    assert len(anns) == 2
    a0 = next(a for a in anns if a.file_name == "img1.jpg")
    assert a0.bbox == [10.0, 5.0, 20.0, 10.0]
    x, y, w, h = a0.bbox
    x1, y1 = int(np.floor(x)), int(np.floor(y))
    x2, y2 = int(np.ceil(x + w)), int(np.ceil(y + h))
    assert int((a0.mask[y1:y2, x1:x2] == 1).sum()) == int(a0.area)
    a1 = next(a for a in anns if a.file_name == "img2.jpg")
    assert a1.width == 50 and a1.height == 40
    assert a1.label == "dog"
    assert int(a1.area) == 600
    assert int(a1.mask.sum()) > 0


def test_from_json_meta_counts():
    obj = build_coco_like(minimal=True)
    ds = ChickenDataset.from_json(obj)
    model = ds.data
    assert model.meta.dataset_type == "coco_dataset"
    assert model.meta.total_images == 2
    assert model.meta.total_annotations == 1
    cat_names = model.meta.categories
    assert set(cat_names) == {"{'id': 10, 'name': 'cat'}", "{'id': 20, 'name': 'dog'}"}


def test_from_path_reads_and_sets_meta(tmp_path: Path):
    obj = build_coco_like(minimal=False)
    json_path = tmp_path / "demo.json"
    json_path.write_text(json.dumps(obj), encoding="utf-8")
    ds = ChickenDataset.from_path(json_path)
    model = ds.data
    assert model.meta.dataset_type == "chicken_detection"
    assert model.meta.source_path == str(json_path)
    assert model.meta.base_directory == str(json_path.parent)
    assert model.meta.total_images == 2
    assert model.meta.total_annotations == 2
    cat_names = model.meta.categories
    assert set(cat_names) == {"{'id': 10, 'name': 'cat'}", "{'id': 20, 'name': 'dog'}"}
    anns = model.data_points
    assert len(anns) == 2
    assert set(a.file_name for a in anns) == {"img1.jpg", "img2.jpg"}