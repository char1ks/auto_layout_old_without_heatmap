import os
import pytest
from PIL import Image

from flashbone.core.encoding.dinov3.image import DinoV3EncoderGaz
from flashbone.core.segmentation import SegmenterConfig, SamSegmenter
from flashbone.core.heatmap_generation import HeatmapGenerator, crop_by_mask
from flashbone.core.classification.mask_classifier_knn import MaskClassifierKNN
from flashbone.core.image_resizing import ImageResizer
from flashbone.core.detection.searchdet_detector import SearchDetDetector


@pytest.fixture(scope="session")
def test_data_dir():
    return os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture(scope="session")
def example_image(test_data_dir):
    return Image.open(os.path.join(test_data_dir, "example.jpg")).convert("RGB")


@pytest.fixture(scope="session")
def image_left(test_data_dir):
    return Image.open(os.path.join(test_data_dir, "image_left.jpg")).convert("RGB")


@pytest.fixture(scope="session")
def image_right(test_data_dir):
    return Image.open(os.path.join(test_data_dir, "image_right.jpg")).convert("RGB")


@pytest.fixture(scope="session")
def mask_left(test_data_dir):
    return Image.open(os.path.join(test_data_dir, "image_left_fg.png")).split()[-1] # alpha channel


@pytest.fixture(scope="session")
def fastsam_weights_path():
    path = os.environ.get("FASTSAM_WEIGHTS", "FastSAM-x.pt")
    if not os.path.exists(path):
        pytest.skip(f"FastSAM weights not found: {path}")
    return path


@pytest.fixture(scope="session")
def sam_model(fastsam_weights_path):
    pytest.importorskip("ultralytics")
    from ultralytics import FastSAM
    return FastSAM(fastsam_weights_path)


@pytest.fixture(scope="session")
def dino_encoder(example_image):
    encoder = DinoV3EncoderGaz()
    
    _ = encoder.encode([example_image])
    return encoder


@pytest.fixture(scope="session")
def heatmap_generator(dino_encoder):
    return HeatmapGenerator(
        dino_fe=dino_encoder,
        use_cosine_similarity_for_heatmap=True,
        threshold_cosine=0.3,
    )


@pytest.fixture(scope="session")
def sam_segmenter(sam_model):
    return SamSegmenter(
        sam_model=sam_model,
        config=SegmenterConfig(
            min_mask_area=200,
            confidence_threshold=0.5,
            iou_threshold=0.8,
            mask_threshold=0.5,
        ),
    )


@pytest.fixture(scope="session")
def classifier(dino_encoder):
    return MaskClassifierKNN(encoder=dino_encoder, d=1024)


@pytest.fixture(scope="session")
def image_resizer():
    return ImageResizer(max_side=1024)


@pytest.fixture(scope="session")
def train_image_pos(image_left, mask_left):
    return crop_by_mask(image_left, mask_left)


@pytest.fixture(scope="session")
def train_image_neg_1(image_left):
    return image_left.crop((0, 0, 150, 150))


@pytest.fixture(scope="session")
def train_image_neg_2(image_left):
    return image_left.crop((image_left.width - 150, image_left.height - 150, image_left.width, image_left.height))


@pytest.fixture(scope="session")
def detector(sam_segmenter, classifier, heatmap_generator, image_resizer, train_image_pos, train_image_neg_1, train_image_neg_2):
    det = SearchDetDetector(
        segmenter=sam_segmenter,
        classifier=classifier,
        heatmap_generator=heatmap_generator,
        image_resizer=image_resizer,
    )
    det.set_references(pos_by_class={0: [train_image_pos]}, neg_imgs=[train_image_neg_1, train_image_neg_2])
    return det