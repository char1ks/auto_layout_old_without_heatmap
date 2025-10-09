import numpy as np
from evaluation.coco_annotation import CocoAnnotation


def test_basic_creation(sample_image, sample_mask):
    h, w = sample_image.shape[:2]
    ann = CocoAnnotation(
        img=sample_image,
        mask=sample_mask,
        label=1,
        image_size=(w, h),
        width=w,
        height=h,
        area=float(np.sum(sample_mask > 0)),
        file_name="img1.jpg",
    )
    assert isinstance(ann.uid, str) and len(ann.uid) > 0
    assert ann.width == w and ann.height == h
    assert ann.image_size == (w, h)
    assert isinstance(ann.label, (int, str))
    assert ann.bbox == []


def test_optional_fields(sample_coco_annotation):
    ann = sample_coco_annotation
    assert ann.score is not None or ann.confidence is not None
    assert isinstance(ann.file_name, str)


def test_uid_uniqueness(coco_annotation_factory):
    a1 = coco_annotation_factory(file_name="a1.jpg")
    a2 = coco_annotation_factory(file_name="a2.jpg")
    assert a1.uid != a2.uid