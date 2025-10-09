import pytest
from pathlib import Path
from PIL import Image
from flashbone.core.heatmap_generation import HeatmapGenerator, crop_by_mask


@pytest.fixture(scope="session")
def test_data_dir():
    return Path(__file__).parent.parent / "data"


@pytest.fixture(scope="session")
def sample_image(test_data_dir):
    return Image.open(test_data_dir / "example.jpg").convert("RGB")


@pytest.fixture(scope="session")
def image_left(test_data_dir):
    return Image.open(test_data_dir / "image_left.jpg").convert("RGB")


@pytest.fixture(scope="session")
def image_right(test_data_dir):
    return Image.open(test_data_dir / "image_right.jpg").convert("RGB")


@pytest.fixture(scope="session")
def mask_left(test_data_dir):
    mask = Image.open(test_data_dir / "image_left_fg.png")
    return mask.split()[-1]  

@pytest.fixture(scope="session")
def heatmap_generator(dino_encoder):
    return HeatmapGenerator(
        dino_fe=dino_encoder,
        use_cosine_similarity_for_heatmap=False,
        threshold_dotp=10,
    )


@pytest.fixture(scope="session")
def trained_heatmap_generator(heatmap_generator, image_left, mask_left):
    train_image_pos = crop_by_mask(image_left, mask_left)
    train_image_neg_1 = image_left.crop((0, 0, 150, 150))  # небо
    train_image_neg_2 = image_left.crop((
        image_left.width-150, 
        image_left.height-150, 
        image_left.width, 
        image_left.height
    )) 
    heatmap_generator.init_pooled_features_train(
        positive_images=[train_image_pos], 
        negative_images=[train_image_neg_1, train_image_neg_2]
    )
    
    return heatmap_generator