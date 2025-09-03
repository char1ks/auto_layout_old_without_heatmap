import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from typing import List, Optional, Union, Tuple
import math
from sklearn.metrics.pairwise import cosine_similarity
from .dinov3_encoder import DinoV3Encoder


# Константы для нормализации изображений
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class CenterPadding(torch.nn.Module):
    """Центральное дополнение изображения до размерa, кратного заданному числу."""
    
    def __init__(self, multiple: int = 14):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size: int) -> Tuple[int, int]:
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Поддерживаем как 3D (C,H,W), так и 4D (B,C,H,W) тензоры
        if x.dim() == 3 or x.dim() == 4:
            h, w = x.shape[-2], x.shape[-1]
        else:
            raise ValueError(f"CenterPadding expects 3D (C,H,W) or 4D (B,C,H,W) tensor, got {x.dim()}D")
        pad_h = self._get_pad(h)
        pad_w = self._get_pad(w)
        return F.pad(x, (pad_w[0], pad_w[1], pad_h[0], pad_h[1]))


class _MaybeToTensor(transforms.ToTensor):
    """Преобразование в тензор, если входные данные не являются тензором."""
    
    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


def _make_normalize_transform(
    mean: tuple = IMAGENET_DEFAULT_MEAN,
    std: tuple = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    """Создает трансформацию нормализации."""
    return transforms.Normalize(mean=mean, std=std)


def get_dinov3_transform(
    crop_img: bool,
    *,
    padding_multiple: int = 14,  # размер патча DINOv3
    resize_img: bool = True,
    resize_size: int = 256,
    resize_max_size: int = 800,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: tuple = IMAGENET_DEFAULT_MEAN,
    std: tuple = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    """Создает композицию трансформаций для DINOv3."""
    
    transform_list = [_MaybeToTensor()]
    
    if resize_img:
        transform_list.append(
            transforms.Resize(resize_size, interpolation=interpolation, max_size=resize_max_size)
        )
    
    if crop_img:
        transform_list.append(transforms.CenterCrop(crop_size))
    else:
        transform_list.append(CenterPadding(padding_multiple))
    
    transform_list.append(_make_normalize_transform(mean, std))
    
    return transforms.Compose(transform_list)


class DinoV3FeatureExtractor(nn.Module):
    """Экстрактор признаков на основе DINOv3."""
    
    def __init__(self, dinov3_encoder: DinoV3Encoder, resize_images: bool = True, crop_images: bool = False):
        super().__init__()
        self.dinov3_encoder = dinov3_encoder
        self.transform = get_dinov3_transform(
            crop_img=crop_images,
            resize_img=resize_images
        )
    
    @property
    def device(self):
        return self.dinov3_encoder.device
    
    def forward(self, images: List[Image.Image]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Извлекает признаки из списка изображений.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (cls_tokens, patch_features)
        """
        if not images:
            raise ValueError("Список изображений не может быть пустым")
        
        # Преобразуем изображения в тензоры
        tensors = []
        for img in images:
            if isinstance(img, Image.Image):
                tensor = self.transform(img)
            else:
                tensor = self.transform(Image.fromarray(img))
            tensors.append(tensor)
        
        # Объединяем в батч
        batch = torch.stack(tensors).to(self.device)
        
        # Извлекаем признаки через DINOv3 (устойчиво к register-токенам)
        with torch.no_grad():
            cls_tokens, patch_tokens = self.dinov3_encoder.extract_cls_and_patches(batch)
        
        return cls_tokens, patch_tokens
    
    def forward_from_tensor(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Извлекает признаки из тензора изображения."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        with torch.no_grad():
            cls_tokens, patch_tokens = self.dinov3_encoder.extract_cls_and_patches(image)
        
        return cls_tokens, patch_tokens


def rescale_features(
    features: torch.Tensor,
    img: Optional[Image.Image] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    do_resize: bool = False,
    resize_size: Union[int, Tuple[int, int]] = 256
) -> torch.Tensor:
    """Изменяет размер карты признаков до размеров изображения."""
    
    if img is not None:
        target_height, target_width = img.size[1], img.size[0]
    elif height is not None and width is not None:
        target_height, target_width = height, width
    else:
        raise ValueError("Необходимо указать либо изображение, либо размеры")
    
    if do_resize:
        if isinstance(resize_size, int):
            # Пропорциональное изменение размера
            aspect_ratio = target_width / target_height
            if aspect_ratio > 1:
                target_width = resize_size
                target_height = int(resize_size / aspect_ratio)
            else:
                target_height = resize_size
                target_width = int(resize_size * aspect_ratio)
        else:
            target_height, target_width = resize_size
    
    # Интерполируем признаки до целевого размера
    if features.dim() == 3:
        features = features.unsqueeze(0)
    
    # Предполагаем, что features имеет форму [batch, seq_len, dim]
    # Нужно преобразовать в [batch, dim, height, width]
    batch_size, seq_len, dim = features.shape
    
    # Вычисляем размеры квадратной сетки признаков, обрезая лишние токены
    grid_side = int(math.sqrt(seq_len))
    valid = grid_side * grid_side
    if seq_len != valid:
        features = features[:, :valid, :]
    
    # Преобразуем в пространственный формат
    features = features.view(features.shape[0], grid_side, grid_side, dim)
    features = features.permute(0, 3, 1, 2)  # [batch, dim, height, width]
    
    # Интерполируем до целевого размера
    features = F.interpolate(
        features,
        size=(target_height, target_width),
        mode='bilinear',
        align_corners=False
    )
    
    return features


def calculate_attention_weights_softmax(query_embedding: torch.Tensor, example_embeddings: torch.Tensor) -> torch.Tensor:
    """Вычисляет веса внимания с использованием softmax."""
    # Вычисляем косинусное сходство
    similarities = F.cosine_similarity(
        query_embedding.unsqueeze(0), 
        example_embeddings, 
        dim=1
    )
    
    # Применяем softmax для получения весов
    weights = F.softmax(similarities, dim=0)
    
    return weights


def adjust_embedding(query_embedding: torch.Tensor, 
                    positive_embeddings: torch.Tensor, 
                    negative_embeddings: torch.Tensor) -> torch.Tensor:
    """Корректирует эмбеддинг запроса на основе положительных и отрицательных примеров."""
    
    # Вычисляем веса для положительных примеров
    positive_weights = calculate_attention_weights_softmax(query_embedding, positive_embeddings)
    
    # Вычисляем взвешенную сумму положительных эмбеддингов
    positive_adjustment = torch.sum(positive_weights.unsqueeze(1) * positive_embeddings, dim=0)
    
    # Если есть отрицательные примеры
    if negative_embeddings.numel() > 0:
        negative_weights = calculate_attention_weights_softmax(query_embedding, negative_embeddings)
        negative_adjustment = torch.sum(negative_weights.unsqueeze(1) * negative_embeddings, dim=0)
        
        # Вычитаем отрицательную корректировку
        combined_adjustment = positive_adjustment - negative_adjustment
    else:
        combined_adjustment = positive_adjustment
    
    return combined_adjustment


class HeatmapGenerator:
    """Генератор тепловых карт на основе DINOv3."""
    
    def __init__(
        self,
        dinov3_encoder: DinoV3Encoder,
        attention_pool_examples: bool = False,
        use_cosine_similarity_for_heatmap: bool = True
    ):
        # Включаем центр-кроп до фиксированного размера, чтобы все тензоры имели одинаковую форму
        self.dinov3_fe = DinoV3FeatureExtractor(dinov3_encoder, resize_images=True, crop_images=True)
        self.attention_pool_examples = attention_pool_examples
        self.use_cosine_similarity_for_heatmap = use_cosine_similarity_for_heatmap
    
    @torch.no_grad()
    def generate_heatmap(
        self,
        input_image: Image.Image,
        positive_images: List[Image.Image],
        negative_images: List[Image.Image] = None
    ) -> torch.Tensor:
        """Генерирует тепловую карту для входного изображения.
        
        Args:
            input_image: Входное изображение
            positive_images: Список положительных примеров
            negative_images: Список отрицательных примеров (опционально)
            
        Returns:
            torch.Tensor: Тепловая карта
        """
        if negative_images is None:
            negative_images = []
        
        # Извлекаем признаки из входного изображения
        input_cls, input_patches = self.dinov3_fe([input_image])

        # Оставляем только пространственные патчи (уже без register-токенов), дополнительно обрезаем до квадрата
        tokens = input_patches[0]
        side = int(math.sqrt(tokens.shape[0]))
        valid = side * side
        if tokens.shape[0] != valid:
            tokens = tokens[:valid]
        
        # Извлекаем признаки из примеров
        if self.attention_pool_examples:
            pooled_embed = self._get_pooled_embed(tokens, positive_images + negative_images)
            positive_embed = pooled_embed[:len(positive_images)]
            negative_embed = pooled_embed[len(positive_images):] if negative_images else torch.empty(0, device=self.dinov3_fe.device)
        else:
            positive_cls, _ = self.dinov3_fe(positive_images)
            positive_embed = positive_cls
            
            if negative_images:
                negative_cls, _ = self.dinov3_fe(negative_images)
                negative_embed = negative_cls
            else:
                negative_embed = torch.empty(0, device=self.dinov3_fe.device)
        
        # Корректируем эмбеддинг
        adjusted_embed = adjust_embedding(input_cls[0], positive_embed, negative_embed)
        
        # Вычисляем сходство между скорректированным эмбеддингом и патчами
        if self.use_cosine_similarity_for_heatmap:
            similarities = F.cosine_similarity(
                adjusted_embed.unsqueeze(0),
                tokens,
                dim=1
            )
        else:
            # Используем евклидово расстояние
            distances = torch.norm(tokens - adjusted_embed.unsqueeze(0), dim=1)
            similarities = 1.0 / (1.0 + distances)  # Преобразуем расстояние в сходство
        
        # Преобразуем в пространственную тепловую карту
        # На всякий случай обрежем до side*side перед reshape
        if similarities.numel() != side * side:
            similarities = similarities[: side * side]
        heatmap = similarities.view(side, side)
        
        return heatmap
    
    def image_from_heatmap(
        self,
        heatmap: torch.Tensor,
        image: Image.Image,
        use_relative_heatmap: bool = False,
        center: float = 0,
        clamp_min: float = 0,
        clamp_max: float = 0.3,
        scale: float = 1,
        heatmap_cmap: str = 'inferno',
        heatmap_blend_ratio: float = 0.5,
        **kwargs
    ) -> Image.Image:
        """Создает изображение с наложенной тепловой картой."""
        
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        # Преобразуем тензор в numpy
        if isinstance(heatmap, torch.Tensor):
            heatmap_np = heatmap.cpu().numpy()
        else:
            heatmap_np = heatmap
        
        # Применяем относительную тепловую карту если нужно
        if use_relative_heatmap:
            heatmap_np = heatmap_np - center
        
        # Обрезаем значения
        heatmap_np = np.clip(heatmap_np, clamp_min, clamp_max)
        
        # Масштабируем
        heatmap_np = heatmap_np * scale
        
        # Нормализуем для цветовой карты
        heatmap_normalized = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
        
        # Изменяем размер тепловой карты до размера изображения
        from scipy.ndimage import zoom
        target_height, target_width = image.size[1], image.size[0]
        zoom_factors = (target_height / heatmap_normalized.shape[0], target_width / heatmap_normalized.shape[1])
        heatmap_resized = zoom(heatmap_normalized, zoom_factors, order=1)
        
        # Применяем цветовую карту
        colormap = cm.get_cmap(heatmap_cmap)
        heatmap_colored = colormap(heatmap_resized)
        heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
        
        # Преобразуем исходное изображение в numpy
        image_np = np.array(image)
        
        # Смешиваем изображения
        blended = (1 - heatmap_blend_ratio) * image_np + heatmap_blend_ratio * heatmap_colored
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        return Image.fromarray(blended)
    
    def _get_pooled_embed(self, query_feats: torch.Tensor, images: List[Image.Image]) -> torch.Tensor:
        """Получает объединенные эмбеддинги с использованием attention pooling."""
        pooled_features = self._generate_pooled_patch_features(images)
        pooled_embed = self._attention_pool_keys(query_feats, pooled_features)
        return pooled_embed
    
    def _generate_pooled_patch_features(self, images: List[Image.Image]) -> torch.Tensor:
        """Генерирует объединенные признаки патчей."""
        all_patches = []
        for img in images:
            _, patches = self.dinov3_fe([img])
            tokens = patches[0]
            grid_side = int(math.sqrt(tokens.shape[0]))
            num_patches = grid_side * grid_side
            tokens = tokens[:num_patches]
            all_patches.append(tokens)
        return torch.stack(all_patches)
    
    def _attention_pool_keys(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """Выполняет attention pooling для ключей."""
        # query: [seq_len, dim]
        # keys: [num_images, seq_len, dim]
        
        # Вычисляем attention веса
        attention_scores = torch.einsum('sd,nsd->ns', query, keys)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Применяем веса к ключам
        pooled = torch.einsum('ns,nsd->nd', attention_weights, keys)
        
        return pooled


def open_image(img_path: str) -> Image.Image:
    """Открывает изображение из файла."""
    return Image.open(img_path).convert('RGB')