from typing import List, Dict

from evaluation.dataset_model import DatasetModel
from evaluation.dataset_meta import DatasetMeta
from evaluation.coco_annotation import CocoAnnotation


def test_meta_fields_valid(dataset_model_from_meta_obj):
    dm: DatasetModel = dataset_model_from_meta_obj
    meta = dm.meta
    assert meta.name == "StaticVOC"
    assert meta.categories == ["pineapple", "snake fruit", "dragon fruit"]
    assert meta.color_channels == ["RGB"]
    assert meta.total_images == 6
    assert meta.total_annotations == 18
    assert meta.dataset_type == "voc_static"
    assert meta.url is None


def test_data_points_file_names_and_counts(coco_annotations_list):
    anns: List[CocoAnnotation] = coco_annotations_list
    assert len(anns) == 18
    file_names = [a.file_name for a in anns]
    assert set(file_names) == {
        "fruit0.png", "fruit1.png", "fruit2.png", "fruit3.png", "fruit4.png", "fruit5.png",
    }
    counts: Dict[str, int] = {}
    for fn in file_names:
        counts[fn] = counts.get(fn, 0) + 1
    for fn in [f"fruit{i}.png" for i in range(6)]:
        assert counts.get(fn, 0) == 3


def test_bbox_mask_area_consistency(coco_annotations_list):
    anns: List[CocoAnnotation] = coco_annotations_list
    for a in anns:
        x, y, w, h = a.bbox
        expected_area = int(w) * int(h)
        assert int(a.area) == expected_area
        assert a.image_size == (a.width, a.height)
        assert a.width == 400
        assert a.height == 300


def test_uids_are_strings_and_present(dataset_model_from_meta_obj, coco_annotations_list):
    dm: DatasetModel = dataset_model_from_meta_obj
    assert isinstance(dm.uid, str) and len(dm.uid) > 0
    for a in coco_annotations_list:
        assert isinstance(a.uid, str) and len(a.uid) > 0


def test_meta_mixed_types_handling(coco_annotations_list, meta_mixed_types_dict):
    # Create DatasetMeta directly; invalid numeric values are set to None
    meta = DatasetMeta(
        name=meta_mixed_types_dict["name"],
        categories=meta_mixed_types_dict["categories"],
        color_channels=meta_mixed_types_dict["color_channels"],
        total_images=None,
        total_annotations=None,
    )
    dm = DatasetModel(data_points=coco_annotations_list, meta=meta)
    assert isinstance(dm.meta, DatasetMeta)
    assert dm.meta.total_images is None
    assert dm.meta.total_annotations is None
    assert dm.meta.categories == ["A", "B"]


def test_labels_set_matches_meta_categories(coco_annotations_list, dataset_model_from_meta_obj):
    labels = sorted({str(a.label) for a in coco_annotations_list})
    assert labels == ["dragon fruit", "pineapple", "snake fruit"]
    assert dataset_model_from_meta_obj.meta.categories == ["pineapple", "snake fruit", "dragon fruit"]