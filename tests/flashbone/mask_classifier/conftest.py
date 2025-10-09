import pytest
from pathlib import Path
from PIL import Image
from flashbone.core.classification.base import ClassData
from flashbone.core.classification.mask_classifier_knn import MaskClassifierKNN


@pytest.fixture(scope="session")
def test_data_dir():
    return Path(__file__).parent.parent / "data"


@pytest.fixture(scope="session")
def example_image(test_data_dir):
    return Image.open(test_data_dir / "example.jpg").convert("RGB")


@pytest.fixture(scope="session")
def image_left(test_data_dir):
    return Image.open(test_data_dir / "image_left.jpg").convert("RGB")


@pytest.fixture(scope="session")
def image_right(test_data_dir):
    return Image.open(test_data_dir / "image_right.jpg").convert("RGB")


@pytest.fixture(scope="session")
def mask_left(test_data_dir):
    return Image.open(test_data_dir / "image_left_fg.png").split()[-1]

@pytest.fixture(scope="session")
def classifier(dino_encoder):
    return MaskClassifierKNN(encoder=dino_encoder, d=1024)


@pytest.fixture(scope="session")
def train_image_neg_1(image_right):
    return image_right.crop((0, 0, 150, 150))


@pytest.fixture(scope="session")
def train_image_neg_2(image_right):
    return image_right.crop((image_right.width - 150, image_right.height - 150, image_right.width, image_right.height))


@pytest.fixture(scope="session")
def indexed_classifier(classifier, image_right, train_image_neg_1, train_image_neg_2):
    classifier.index([
        ClassData(
            class_id=0,
            images=[image_right],
            negative_images=[train_image_neg_1, train_image_neg_2],
        ),
    ])
    return classifier