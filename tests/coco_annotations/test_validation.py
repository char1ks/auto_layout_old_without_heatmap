import numpy as np
from evaluation.coco_annotation import CocoAnnotation


def test_image_and_mask_shapes_must_match(sample_image, sample_mask):
    h, w = sample_image.shape[:2]
    ann = CocoAnnotation(
        img=sample_image,
        mask=sample_mask,
        label="obj",
        image_size=(w, h),
        width=w,
        height=h,
        area=float(np.sum(sample_mask > 0)),
        file_name="img1.jpg",
    )
    assert ann.image_size == (ann.width, ann.height)
    assert sample_mask.shape == (ann.height, ann.width)


def test_bbox_format_and_values(coco_annotation_factory):
    ann = coco_annotation_factory(bbox=[5, 10, 20, 15])
    assert isinstance(ann.bbox, list) and len(ann.bbox) in (0, 4)
    if ann.bbox:
        x, y, w, h = ann.bbox
        assert w >= 0 and h >= 0
        assert 0 <= x <= ann.width
        assert 0 <= y <= ann.height


def test_label_types(coco_annotation_factory):
    a_int = coco_annotation_factory(label=3)
    a_str = coco_annotation_factory(label="class")
    assert isinstance(a_int.label, (int, str))
    assert isinstance(a_str.label, (int, str))


def test_area_is_non_negative(sample_coco_annotation):
    assert sample_coco_annotation.area >= 0