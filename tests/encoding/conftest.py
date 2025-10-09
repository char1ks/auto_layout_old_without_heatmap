import pytest
from PIL import Image
import os
from flashbone.core.encoding.dinov3.image import DinoV3EncoderGaz

@pytest.fixture
def test_data_dir():
    return "tests/data/"


@pytest.fixture
def example_image(test_data_dir):
    image_path = os.path.join(test_data_dir, "example.jpg")
    return Image.open(image_path).convert("RGB")


@pytest.fixture
def image_left(test_data_dir):
    image_path = os.path.join(test_data_dir, "image_left.jpg")
    return Image.open(image_path).convert("RGB")


@pytest.fixture
def mask_left(test_data_dir):
    mask_path = os.path.join(test_data_dir, "image_left_fg.png")
    mask = Image.open(mask_path)
    return mask.split()[-1] 

@pytest.fixture
def dino_encoder():
    return DinoV3EncoderGaz()


@pytest.fixture
def encoded_features(dino_encoder, image_left):
    return dino_encoder.encode([image_left])