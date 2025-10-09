import torch
import numpy as np
import cv2
from PIL import Image
from flashbone.core.heatmap_generation import (
    HeatmapGenerator,
    crop_by_mask,
)
def test_model_creation_and_warmup(dino_encoder, sample_image):
    feats_ex = dino_encoder.encode([sample_image])
    assert feats_ex is not None

    assert hasattr(feats_ex, "cls") and isinstance(feats_ex.cls, torch.Tensor)
    assert hasattr(feats_ex, "patches") and isinstance(feats_ex.patches, torch.Tensor)
    assert feats_ex.patches.ndim == 4
    patch0 = feats_ex.patches[0]
    assert isinstance(patch0, torch.Tensor) and patch0.ndim == 3
    
def test_heatmap_generator_creation(dino_encoder):
    heatmap_generator = HeatmapGenerator(
        dino_fe=dino_encoder, 
        use_cosine_similarity_for_heatmap=False,
        threshold_dotp=10,
    )
    
    assert heatmap_generator.dino_fe is dino_encoder
    assert not heatmap_generator.use_cosine_similarity_for_heatmap
    assert heatmap_generator.threshold_dotp == 10
def test_training_data_preparation(image_left, mask_left):
    train_image_pos = crop_by_mask(image_left, mask_left)

    train_image_neg_1 = image_left.crop((0, 0, 150, 150)) 
    train_image_neg_2 = image_left.crop((
        image_left.width-150, 
        image_left.height-150, 
        image_left.width, 
        image_left.height
    )) 
    assert isinstance(train_image_pos, Image.Image)
    assert isinstance(train_image_neg_1, Image.Image)
    assert isinstance(train_image_neg_2, Image.Image)
    assert train_image_neg_1.size == (150, 150)
    assert train_image_neg_2.size == (150, 150)
def test_pooled_features_training(heatmap_generator, image_left, mask_left):
    train_image_pos = crop_by_mask(image_left, mask_left)
    train_image_neg_1 = image_left.crop((0, 0, 150, 150))
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
    # Проверяем корректную инициализацию внутренних пулов признаков
    assert heatmap_generator.pooled_patch_features_pos is not None
    assert heatmap_generator.pooled_patch_features_neg is not None
def test_heatmap_generation(trained_heatmap_generator, image_right):
    heatmap, heatmap_resized = trained_heatmap_generator.generate_heatmap(image_right)
    
    assert heatmap is not None
    assert heatmap_resized is not None
    assert isinstance(heatmap, torch.Tensor)
    assert isinstance(heatmap_resized, torch.Tensor)
    print(heatmap_resized.shape, heatmap_resized.min(), heatmap_resized.max())
    
def test_threshold_application(trained_heatmap_generator, image_right):
    heatmap, heatmap_resized = trained_heatmap_generator.generate_heatmap(image_right)
    heatmap_resized = trained_heatmap_generator.apply_threshold(heatmap_resized)
    
    assert isinstance(heatmap_resized, torch.Tensor)
    
    heatmap_np = heatmap_resized.cpu().numpy()
    assert isinstance(heatmap_np, np.ndarray)

def test_image_saving_preparation(trained_heatmap_generator, image_left, mask_left, image_right):
    train_image_pos = crop_by_mask(image_left, mask_left)
    heatmap, heatmap_resized = trained_heatmap_generator.generate_heatmap(image_right)
    heatmap_resized = trained_heatmap_generator.apply_threshold(heatmap_resized)
    heatmap_np = heatmap_resized.cpu().numpy()

    crop_array = np.asarray(train_image_pos)
    crop_bgr = cv2.cvtColor(crop_array, cv2.COLOR_RGB2BGR)
    heatmap_uint8 = (heatmap_np*255).astype(np.uint8)
    
    assert crop_bgr.dtype == np.uint8
    assert heatmap_uint8.dtype == np.uint8
    assert len(crop_bgr.shape) == 3  # H, W, C
    assert len(heatmap_uint8.shape) == 2  # H, W
    #Проверяем тока типы данных и ничего не сохраняем
    assert crop_bgr.min() >= 0 and crop_bgr.max() <= 255
    assert heatmap_uint8.min() >= 0 and heatmap_uint8.max() <= 255