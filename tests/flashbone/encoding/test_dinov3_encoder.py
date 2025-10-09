import time
import torch
from flashbone.core.encoding.base import DinoFeaturesPT

def test_single_image_encode(dino_encoder, image_left):
    start = time.perf_counter()
    feats = dino_encoder.encode([image_left])
    end = time.perf_counter()
    assert isinstance(feats, DinoFeaturesPT)
    assert feats.patches.shape[0] == 1  # batch size = 1
    assert feats.cls.shape[0] == 1      # batch size = 1
    execution_time_ms = int((end - start) * 1000)
    assert execution_time_ms < 10000
    
    print(f"Shapes: patches={feats.patches.shape}, cls={feats.cls.shape}")
    print(f"Execution time: {execution_time_ms} ms.")
def test_mask_encoding(dino_encoder, mask_left, encoded_features):
    mask_features = dino_encoder.encode_mask([mask_left], encoded_features)
    
    assert isinstance(mask_features, torch.Tensor)
    assert len(mask_features.shape) == 2 
    assert mask_features.shape[0] == 1   
def test_cosine_similarity(dino_encoder, example_image, image_left, mask_left):
    feats_ex = dino_encoder.encode([example_image])
    feats = dino_encoder.encode([image_left])
    
    mask_features = dino_encoder.encode_mask([mask_left], feats)
    
    sim1 = torch.cosine_similarity(feats_ex.cls, feats.cls, dim=-1)  # should be low
    sim2 = torch.cosine_similarity(feats.cls, mask_features, dim=-1)  # should be high
    
    assert isinstance(sim1, torch.Tensor)
    assert isinstance(sim2, torch.Tensor)
    assert sim1.shape == torch.Size([1])
    assert sim2.shape == torch.Size([1])

    assert -1 <= sim1.item() <= 1
    assert -1 <= sim2.item() <= 1
def test_target_size_computation(dino_encoder, image_left, example_image):
    target_size = dino_encoder._compute_target_size([image_left, example_image])
    
    assert isinstance(target_size, tuple)
    assert len(target_size) == 2
    h_target, w_target = target_size
    assert h_target % dino_encoder.patch_size == 0
    assert w_target % dino_encoder.patch_size == 0
def test_resize_transform(dino_encoder, image_left):
    target_size = dino_encoder._compute_target_size([image_left])
    transformed = dino_encoder.resize_transform(image_left, target_size)
    
    assert isinstance(transformed, torch.Tensor)
    assert len(transformed.shape) == 3  # (C, H, W)
    assert transformed.shape[1] == target_size[0]  # height
    assert transformed.shape[2] == target_size[1]  # width