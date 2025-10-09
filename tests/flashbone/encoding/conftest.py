import pytest
from PIL import Image
from pathlib import Path

@pytest.fixture
def test_data_dir():
    return Path(__file__).parent.parent / "data"


@pytest.fixture
def example_image(test_data_dir):
    return Image.open(test_data_dir / "example.jpg").convert("RGB")


@pytest.fixture
def image_left(test_data_dir):
    return Image.open(test_data_dir / "image_left.jpg").convert("RGB")


@pytest.fixture
def mask_left(test_data_dir):
    return Image.open(test_data_dir / "image_left_fg.png").split()[-1] 



@pytest.fixture
def encoded_features(dino_encoder, image_left):
    return dino_encoder.encode([image_left])