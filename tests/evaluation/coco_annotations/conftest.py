import numpy as np
import pytest
from evaluation.coco_annotation import CocoAnnotation
@pytest.fixture
def sample_image() -> np.ndarray:
    return (np.random.rand(50, 30, 3) * 255).astype(np.uint8)


@pytest.fixture
def sample_mask() -> np.ndarray:
    mask = np.zeros((50, 30), dtype=np.uint8)
    mask[10:20, 5:15] = 1
    return mask


@pytest.fixture
def sample_coco_annotation(sample_image: np.ndarray, sample_mask: np.ndarray) -> CocoAnnotation:
    height, width = sample_image.shape[:2]
    ann = CocoAnnotation(
        img=sample_image,
        mask=sample_mask,
        label="object",
        image_size=(width, height),
        width=width,
        height=height,
        area=float(np.sum(sample_mask > 0)),
        file_name="img1.jpg",
        bbox=[5.0, 10.0, 10.0, 10.0],
        score=0.9,
        confidence=None,
    )
    return ann


@pytest.fixture
def coco_annotation_factory():
    def _factory(
        *,
        width: int = 30,
        height: int = 50,
        label: int | str = "object",
        bbox: list[float] | None = None,
        score: float | None = None,
        confidence: float | None = None,
        file_name: str = "factory_img.jpg",
    ) -> CocoAnnotation:
        img = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[0 : height // 2, 0 : width // 2] = 1
        return CocoAnnotation(
            img=img,
            mask=mask,
            label=label,
            image_size=(width, height),
            width=width,
            height=height,
            area=float(np.sum(mask > 0)),
            file_name=file_name,
            bbox=bbox or [],
            score=score,
            confidence=confidence,
        )

    return _factory