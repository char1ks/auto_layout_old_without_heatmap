import pytest
from pathlib import Path
from PIL import Image
from flashbone.core.encoding.dinov3.image import DinoV3EncoderGaz


@pytest.fixture(scope="session")
def dino_encoder():
    encoder = DinoV3EncoderGaz()
    img = Image.open(Path(__file__).parent / "data" / "example.jpg").convert("RGB")
    _ = encoder.encode([img]) # прогрели чуточку
    return encoder