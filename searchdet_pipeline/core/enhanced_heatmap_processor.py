import numpy as np
import torch
import cv2
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional
from sklearn.cluster import DBSCAN
from scipy import ndimage
from skimage.measure import label, regionprops
from .dinov3_encoder import DinoV3Encoder
from .heatmap_generator import HeatmapGenerator, DinoV3FeatureExtractor
from .binning_processor import BinningProcessor


class EnhancedHeatmapProcessor:
    """Улучшенный процессор для обработки heatmap с пиксельным биннингом и разбиением горячих зон."""
    
    def __init__(self, dinov3_encoder: DinoV3Encoder, 
                 hot_zone_threshold: float = 0.7,
                 pixel_similarity_threshold: float = 0.8,
                 min_zone_area: int = 100,
                 connectivity: int = 8):
        """
        Args:
            dinov3_encoder: Энкодер DINOv3
            hot_zone_threshold: Порог для определения "горячих" зон в heatmap
            pixel_similarity_threshold: Порог схожести пикселей для биннинга
            min_zone_area: Минимальная площадь зоны для сохранения
            connectivity: Связность для поиска компонент (4 или 8)
        """
        self.dinov3_encoder = dinov3_encoder
        self.heatmap_generator = HeatmapGenerator(dinov3_encoder)
        self.feature_extractor = DinoV3FeatureExtractor(dinov3_encoder, resize_images=True, crop_images=False)
        self.binning_processor = BinningProcessor(dinov3_encoder)
        
        self.hot_zone_threshold = hot_zone_threshold
        self.pixel_similarity_threshold = pixel_similarity_threshold
        self.min_zone_area = min_zone_area
        self.connectivity = connectivity
    
    def extract_hot_zones(self, heatmap: torch.Tensor) -> np.ndarray:
        """Извлекает "горячие" зоны из heatmap.
        
        Args:
            heatmap: Тепловая карта
            
        Returns:
            np.ndarray: Бинарная маска горячих зон
        """
        # Преобразуем в numpy если нужно
        if isinstance(heatmap, torch.Tensor):
            heatmap_np = heatmap.cpu().numpy()
        else:
            heatmap_np = heatmap
        
        # Нормализуем heatmap
        heatmap_normalized = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
        
        # Создаем бинарную маску горячих зон
        hot_zones = heatmap_normalized > self.hot_zone_threshold
        
        return hot_zones.astype(np.uint8)
    
    def get_pixel_features(self, image: np.ndarray, patch_size: int = 14) -> np.ndarray:
        """Извлекает признаки для каждого пикселя/патча изображения.
        
        Args:
            image: Входное изображение
            patch_size: Размер патча для извлечения признаков
            
        Returns:
            np.ndarray: Массив признаков для каждого патча [H/patch_size, W/patch_size, feature_dim]
        """
        # Преобразуем в PIL если нужно
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Извлекаем признаки патчей с помощью DINOv3
        with torch.no_grad():
            _, patch_features = self.feature_extractor([pil_image])
        
        # patch_features имеет размер [1, num_patches, feature_dim]
        patch_features = patch_features.squeeze(0).cpu().numpy()  # [num_patches, feature_dim]
        
        # Определяем размеры сетки патчей
        h, w = pil_image.size[1], pil_image.size[0]  # PIL возвращает (width, height)
        patch_h = h // patch_size
        patch_w = w // patch_size
        
        # Изменяем форму на [patch_h, patch_w, feature_dim]
        patch_features_reshaped = patch_features.reshape(patch_h, patch_w, -1)
        
        return patch_features_reshaped
    
    def perform_pixel_binning(self, hot_zones: np.ndarray, 
                             pixel_features: np.ndarray,
                             reference_features: np.ndarray) -> np.ndarray:
        """Выполняет пиксельный биннинг для очистки горячих зон от непохожих пикселей.
        
        Args:
            hot_zones: Бинарная маска горячих зон
            pixel_features: Признаки пикселей/патчей
            reference_features: Эталонные признаки для сравнения
            
        Returns:
            np.ndarray: Очищенная маска горячих зон
        """
        # Приводим размеры hot_zones к размеру pixel_features
        if hot_zones.shape != pixel_features.shape[:2]:
            hot_zones_resized = cv2.resize(hot_zones.astype(np.uint8), 
                                         (pixel_features.shape[1], pixel_features.shape[0]), 
                                         interpolation=cv2.INTER_NEAREST)
        else:
            hot_zones_resized = hot_zones
        
        # Создаем копию для результата
        cleaned_zones = hot_zones_resized.copy()
        
        # Получаем координаты горячих пикселей
        hot_coords = np.where(hot_zones_resized > 0)
        
        if len(hot_coords[0]) == 0:
            return cleaned_zones
        
        # Вычисляем средний эталонный признак
        if len(reference_features.shape) == 3:  # Если несколько эталонных признаков
            mean_reference = np.mean(reference_features, axis=0)
        else:
            mean_reference = reference_features
        
        # Проверяем каждый горячий пиксель
        for i, j in zip(hot_coords[0], hot_coords[1]):
            pixel_feature = pixel_features[i, j]
            
            # Вычисляем косинусное сходство
            similarity = np.dot(pixel_feature, mean_reference) / (
                np.linalg.norm(pixel_feature) * np.linalg.norm(mean_reference) + 1e-8
            )
            
            # Если пиксель недостаточно похож, удаляем его
            if similarity < self.pixel_similarity_threshold:
                cleaned_zones[i, j] = 0
        
        return cleaned_zones
    
    def split_zones_by_connectivity(self, zones_mask: np.ndarray) -> List[np.ndarray]:
        """Разбивает горячие зоны на отдельные маски по разрывам.
        
        Args:
            zones_mask: Бинарная маска зон
            
        Returns:
            List[np.ndarray]: Список отдельных масок для каждой связной компоненты
        """
        # Находим связные компоненты
        labeled_zones = label(zones_mask, connectivity=self.connectivity)
        
        # Получаем свойства регионов
        regions = regionprops(labeled_zones)
        
        individual_masks = []
        
        for region in regions:
            # Проверяем минимальную площадь
            if region.area < self.min_zone_area:
                continue
            
            # Создаем маску для этого региона
            mask = (labeled_zones == region.label).astype(np.uint8)
            individual_masks.append(mask)
        
        return individual_masks
    
    def convert_masks_to_sam_format(self, masks: List[np.ndarray], 
                                   original_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Преобразует маски в формат SAM.
        
        Args:
            masks: Список бинарных масок
            original_size: Размер оригинального изображения (height, width)
            
        Returns:
            List[Dict[str, Any]]: Маски в формате SAM
        """
        sam_masks = []
        
        for mask in masks:
            # Приводим маску к размеру оригинального изображения
            if mask.shape != original_size:
                mask_resized = cv2.resize(mask, (original_size[1], original_size[0]), 
                                        interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = mask
            
            # Преобразуем в boolean
            segmentation = mask_resized.astype(bool)
            
            # Вычисляем площадь
            area = int(segmentation.sum())
            
            if area == 0:
                continue
            
            # Вычисляем bounding box
            ys, xs = np.where(segmentation)
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            bbox_w = x_max - x_min + 1
            bbox_h = y_max - y_min + 1
            bbox = [x_min, y_min, bbox_w, bbox_h]
            
            # Центр bbox
            cx = x_min + bbox_w // 2
            cy = y_min + bbox_h // 2
            
            # Создаем словарь в формате SAM
            sam_mask = {
                'segmentation': segmentation,
                'area': area,
                'bbox': bbox,
                'predicted_iou': 1.0,
                'point_coords': [[cx, cy]],
                'stability_score': 1.0,
                'crop_box': [0, 0, original_size[1], original_size[0]]
            }
            
            sam_masks.append(sam_mask)
        
        return sam_masks
    
    def process_enhanced_heatmap(self, input_image: Image.Image,
                               positive_images: List[Image.Image],
                               negative_images: List[Image.Image] = None,
                               save_cleaned_heatmap: bool = True,
                               output_dir: str = None) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Полный процесс обработки heatmap с улучшенной логикой.
        
        Args:
            input_image: Входное изображение
            positive_images: Положительные примеры
            negative_images: Отрицательные примеры
            save_cleaned_heatmap: Сохранять ли очищенную heatmap
            output_dir: Директория для сохранения
            
        Returns:
            Tuple[torch.Tensor, List[Dict[str, Any]]]: (очищенная_heatmap, маски_в_формате_SAM)
        """
        if negative_images is None:
            negative_images = []
        
        print("[EnhancedHeatmapProcessor] Генерация исходной heatmap...")
        # Генерируем исходную heatmap
        original_heatmap = self.heatmap_generator.generate_heatmap(
            input_image, positive_images, negative_images
        )
        
        print("[EnhancedHeatmapProcessor] Извлечение горячих зон...")
        # Извлекаем горячие зоны
        hot_zones = self.extract_hot_zones(original_heatmap)
        
        print("[EnhancedHeatmapProcessor] Извлечение признаков пикселей...")
        # Извлекаем признаки пикселей
        pixel_features = self.get_pixel_features(np.array(input_image))
        
        print("[EnhancedHeatmapProcessor] Вычисление эталонных признаков...")
        # Вычисляем эталонные признаки из положительных примеров
        reference_features = []
        for pos_img in positive_images:
            features = self.get_pixel_features(np.array(pos_img))
            reference_features.append(features)
        
        if reference_features:
            reference_features = np.stack(reference_features)
        else:
            # Если нет положительных примеров, используем признаки самого изображения
            reference_features = pixel_features
        
        print("[EnhancedHeatmapProcessor] Выполнение пиксельного биннинга...")
        # Выполняем пиксельный биннинг
        cleaned_zones = self.perform_pixel_binning(hot_zones, pixel_features, reference_features)
        
        print("[EnhancedHeatmapProcessor] Разбиение зон по связности...")
        # Разбиваем зоны на отдельные маски
        individual_masks = self.split_zones_by_connectivity(cleaned_zones)
        
        print(f"[EnhancedHeatmapProcessor] Найдено {len(individual_masks)} отдельных зон")
        
        # Преобразуем в формат SAM
        original_size = (input_image.size[1], input_image.size[0])  # (height, width)
        sam_masks = self.convert_masks_to_sam_format(individual_masks, original_size)
        
        print(f"[EnhancedHeatmapProcessor] Создано {len(sam_masks)} масок в формате SAM")
        
        # Создаем очищенную heatmap
        cleaned_heatmap = torch.from_numpy(cleaned_zones.astype(np.float32))
        
        # Сохраняем очищенную heatmap если требуется
        if save_cleaned_heatmap and output_dir:
            self.save_cleaned_heatmap(cleaned_heatmap, output_dir)
        
        return cleaned_heatmap, sam_masks
    
    def save_cleaned_heatmap(self, cleaned_heatmap: torch.Tensor, output_dir: str) -> None:
        """Сохраняет очищенную heatmap.
        
        Args:
            cleaned_heatmap: Очищенная тепловая карта
            output_dir: Директория для сохранения
        """
        import os
        
        # Преобразуем в numpy
        heatmap_np = cleaned_heatmap.cpu().numpy() if isinstance(cleaned_heatmap, torch.Tensor) else cleaned_heatmap
        
        # Нормализуем для визуализации
        heatmap_normalized = (heatmap_np * 255).astype(np.uint8)
        
        # Применяем цветовую карту
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        
        # Сохраняем
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "cleaned_heatmap.png")
        cv2.imwrite(save_path, heatmap_colored)
        
        print(f"[EnhancedHeatmapProcessor] Очищенная heatmap сохранена: {save_path}")