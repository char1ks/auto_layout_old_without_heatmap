import pytest
from pathlib import Path
from PIL import Image

from flashbone.core.classification.base import ClassData
from flashbone.core.classification.vision_text_classifier_knn import VisionTextClassifierKNN
from flashbone.core.encoding.dinov3.multimodal import DinoV3VisionTextEncoderGaz


@pytest.fixture(scope="session")
def test_data_dir():
   return Path(__file__).parent.parent / "data"


@pytest.fixture(scope="session")
def image_left(test_data_dir):
    return Image.open(test_data_dir / "image_left.jpg").convert("RGB")


@pytest.fixture(scope="session")
def image_right(test_data_dir):
    return Image.open(test_data_dir / "image_right.jpg").convert("RGB")


@pytest.fixture(scope="session")
def image_example(test_data_dir):
    return Image.open(test_data_dir / "example.jpg").convert("RGB")


@pytest.fixture(scope="session")
def vt_encoder():
    return DinoV3VisionTextEncoderGaz()


@pytest.fixture(scope="session")
def vt_classifier(vt_encoder):
    return VisionTextClassifierKNN(encoder=vt_encoder, d=2048)


@pytest.fixture(scope="function")
def fresh_vt_classifier(vt_encoder):
    return VisionTextClassifierKNN(encoder=vt_encoder, d=2048)


@pytest.fixture(scope="session")
def indexed_text_only(vt_classifier):
    vt_classifier.index([
        ClassData(
            class_id=0,
            texts=["donkey"],
            negative_texts=["green grass", "blue sky"],
        )
    ])
    return vt_classifier