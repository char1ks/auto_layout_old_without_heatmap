import json
import numpy as np
from evaluate.COCOAnnotations import COCOAnnotation


def to_serializable_dict(ann: COCOAnnotation) -> dict:
    return {
        "uid": ann.uid,
        "file_name": ann.file_name,
        "label": ann.label,
        "image_size": list(ann.image_size),
        "width": ann.width,
        "height": ann.height,
        "area": float(ann.area),
        "bbox": list(ann.bbox),
        "score": float(ann.score) if ann.score is not None else None,
        "confidence": float(ann.confidence) if ann.confidence is not None else None,
        "img_shape": list(ann.img.shape),
        "mask_shape": list(ann.mask.shape),
    }


def test_basic_json_serialization(sample_coco_annotation):
    ann = sample_coco_annotation
    d = to_serializable_dict(ann)
    s = json.dumps(d)
    loaded = json.loads(s)
    assert loaded["uid"] == ann.uid
    assert loaded["file_name"] == ann.file_name
    assert loaded["image_size"] == [ann.width, ann.height]
    assert loaded["img_shape"] == list(ann.img.shape)
    assert loaded["mask_shape"] == list(ann.mask.shape)


def test_serialization_without_optional_fields(sample_image, sample_mask):
    h, w = sample_image.shape[:2]
    ann = COCOAnnotation(
        img=sample_image,
        mask=sample_mask,
        label=1,
        image_size=(w, h),
        width=w,
        height=h,
        area=float(np.sum(sample_mask > 0)),
        file_name="img2.jpg",
    )
    d = to_serializable_dict(ann)
    s = json.dumps(d)
    loaded = json.loads(s)
    assert loaded["score"] is None
    assert loaded["confidence"] is None