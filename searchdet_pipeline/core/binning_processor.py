import numpy as np
import torch
import cv2
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from .dinov3_encoder import DinoV3Encoder
from .heatmap_generator import DinoV3FeatureExtractor, adjust_embedding


class BinningProcessor:
    """Процессор для бинаризации хитмап и генерации масок."""
    
    def __init__(self, dinov3_encoder: DinoV3Encoder, num_bins: int = 5, concept_threshold: int = 4):
        """
        Args:
            dinov3_encoder: Энкодер DINOv3
            num_bins: Количество бинов для разделения дистанций
            concept_threshold: Минимальное количество дистанций в первых двух бинах для детекции концепта
        """
        self.dinov3_encoder = dinov3_encoder
        self.feature_extractor = DinoV3FeatureExtractor(dinov3_encoder, resize_images=True, crop_images=False)
        self.num_bins = num_bins
        self.concept_threshold = concept_threshold
    
    def get_dinov3_vector(self, image: Image.Image) -> np.ndarray:
        """Извлекает вектор признаков из изображения с помощью DINOv3.
        
        Args:
            image: Входное изображение
            
        Returns:
            np.ndarray: Вектор признаков
        """
        with torch.no_grad():
            cls_token, _ = self.feature_extractor([image])
        return cls_token.squeeze().cpu().numpy()
    
    def extract_features_from_masks(self, image: np.ndarray, masks: List[Dict[str, Any]]) -> np.ndarray:
        """Извлекает признаки из масок с помощью DINOv3.
        
        Args:
            image: Исходное изображение в формате numpy
            masks: Список масок с ключом 'segmentation'
            
        Returns:
            np.ndarray: Массив признаков для каждой маски
        """
        features = []
        for mask in masks:
            segmentation = mask['segmentation']
            # Создаем маскированное изображение
            mask_image = np.full_like(image, 255)  # Белый фон
            # Убеждаемся, что segmentation - boolean массив
            seg_bool = segmentation.astype(bool) if segmentation.dtype != bool else segmentation
            mask_image[seg_bool] = image[seg_bool]  # Применяем маску
            
            # Преобразуем в PIL Image
            pil_image = Image.fromarray(mask_image)
            
            # Извлекаем признаки
            feature_vector = self.get_dinov3_vector(pil_image)
            features.append(feature_vector)
        
        return np.array(features)
    
    def extract_pixel_features_from_masks(self, image: np.ndarray, masks: List[Dict[str, Any]]) -> List[Tuple[int, List[Tuple[Tuple[int, int], np.ndarray]]]]:
        """Извлекает признаки для каждого пикселя внутри масок с помощью DINOv3 patch tokens.
        
        Args:
            image: Исходное изображение в формате numpy
            masks: Список масок с ключом 'segmentation'
            
        Returns:
            List[Tuple[int, List[Tuple[Tuple[int, int], np.ndarray]]]]: 
            Список кортежей (индекс_маски, список_пикселей_с_признаками)
            где каждый пиксель представлен как ((x, y), feature_vector)
        """
        # Преобразуем изображение в PIL
        pil_image = Image.fromarray(image)
        
        # Извлекаем patch tokens для всего изображения
        with torch.no_grad():
            cls_tokens, patch_tokens = self.feature_extractor([pil_image])
        
        # patch_tokens имеет размер [batch_size, num_patches, feature_dim]
        patch_features = patch_tokens[0].cpu().numpy()  # Берем первое изображение из батча
        
        # Определяем размеры патчей
        H, W = image.shape[:2]
        # DINOv3 использует патчи 14x14 пикселей
        patch_size = 14
        num_patches_h = (H + patch_size - 1) // patch_size
        num_patches_w = (W + patch_size - 1) // patch_size
        
        mask_pixel_features = []
        
        for mask_idx, mask in enumerate(masks):
            segmentation = mask['segmentation']
            seg_bool = segmentation.astype(bool) if segmentation.dtype != bool else segmentation
            
            pixel_features = []
            
            # Проходим по всем пикселям в маске
            y_coords, x_coords = np.where(seg_bool)
            
            for y, x in zip(y_coords, x_coords):
                # Определяем, к какому патчу относится пиксель
                patch_y = min(y // patch_size, num_patches_h - 1)
                patch_x = min(x // patch_size, num_patches_w - 1)
                patch_idx = patch_y * num_patches_w + patch_x
                
                # Проверяем, что индекс патча в допустимых пределах
                if patch_idx < len(patch_features):
                    feature_vector = patch_features[patch_idx]
                    pixel_features.append(((int(x), int(y)), feature_vector))
            
            mask_pixel_features.append((mask_idx, pixel_features))
        
        return mask_pixel_features
    
    def compute_adjusted_embeddings(self, positive_images: List[Image.Image], 
                                  negative_images: List[Image.Image] = None) -> np.ndarray:
        """Вычисляет скорректированные эмбеддинги на основе положительных и отрицательных примеров.
        
        Args:
            positive_images: Список положительных примеров
            negative_images: Список отрицательных примеров (опционально)
            
        Returns:
            np.ndarray: Массив скорректированных эмбеддингов
        """
        if negative_images is None:
            negative_images = []
        
        # Извлекаем эмбеддинги для положительных примеров
        pos_embeddings = []
        for img in positive_images:
            embedding = self.get_dinov3_vector(img)
            pos_embeddings.append(embedding)
        pos_embeddings = np.array(pos_embeddings)
        
        # Извлекаем эмбеддинги для отрицательных примеров
        neg_embeddings = []
        if negative_images:
            for img in negative_images:
                embedding = self.get_dinov3_vector(img)
                neg_embeddings.append(embedding)
            neg_embeddings = np.array(neg_embeddings)
        
        # Корректируем каждый положительный эмбеддинг
        adjusted_embeddings = []
        for pos_emb in pos_embeddings:
            # Преобразуем в torch тензоры для функции adjust_embedding
            pos_emb_tensor = torch.from_numpy(pos_emb).to(self.dinov3_encoder.device)
            pos_embeddings_tensor = torch.from_numpy(pos_embeddings).to(self.dinov3_encoder.device)
            
            if len(neg_embeddings) > 0:
                neg_embeddings_tensor = torch.from_numpy(neg_embeddings).to(self.dinov3_encoder.device)
            else:
                neg_embeddings_tensor = torch.empty(0, device=self.dinov3_encoder.device)
            
            adjusted_emb = adjust_embedding(pos_emb_tensor, pos_embeddings_tensor, neg_embeddings_tensor)
            adjusted_embeddings.append(adjusted_emb.cpu().numpy())
        
        adjusted_embeddings = np.array(adjusted_embeddings).astype('float32')
        
        # Нормализуем скорректированные эмбеддинги
        adjusted_embeddings = adjusted_embeddings / np.linalg.norm(adjusted_embeddings, axis=1, keepdims=True)
        
        return adjusted_embeddings
    
    def compute_distances(self, mask_embeddings: np.ndarray, adjusted_embeddings: np.ndarray) -> List[Tuple[int, List[float]]]:
        """Вычисляет дистанции (1 - cosine similarity) между эмбеддингами масок и скорректированными эмбеддингами.
        
        Args:
            mask_embeddings: Эмбеддинги масок
            adjusted_embeddings: Скорректированные эмбеддинги
            
        Returns:
            List[Tuple[int, List[float]]]: Список кортежей (индекс_маски, список_дистанций)
        """
        mask_distances = []
        
        for i, mask_emb in enumerate(mask_embeddings):
            distances = []
            for adj_emb in adjusted_embeddings:
                # Используем косинусную меру близости, превращая её в "дистанцию"
                sim = cosine_similarity(mask_emb.reshape(1, -1), adj_emb.reshape(1, -1))[0][0]
                distance = 1.0 - float(sim)
                distances.append(distance)
            mask_distances.append((i, distances))
        
        return mask_distances

    def compute_pixel_distances(self, mask_pixel_features: List[Tuple[int, List[Tuple[Tuple[int, int], np.ndarray]]]], 
                               adjusted_embeddings: np.ndarray) -> List[Tuple[int, List[Tuple[Tuple[int, int], List[float]]]]]:
        """Вычисляет дистанции (1 - cosine similarity) между признаками пикселей и скорректированными эмбеддингами.
        
        Args:
            mask_pixel_features: Список признаков пикселей для каждой маски
            adjusted_embeddings: Скорректированные эмбеддинги
            
        Returns:
            List[Tuple[int, List[Tuple[Tuple[int, int], List[float]]]]]: 
            Список кортежей (индекс_маски, список_пикселей_с_дистанциями)
            где каждый пиксель представлен как ((x, y), список_дистанций)
        """
        mask_pixel_distances = []
        
        for mask_idx, pixel_features in mask_pixel_features:
            pixel_distances = []
            
            for (x, y), pixel_feature in pixel_features:
                distances = []
                for adj_emb in adjusted_embeddings:
                    sim = cosine_similarity(pixel_feature.reshape(1, -1), adj_emb.reshape(1, -1))[0][0]
                    distance = 1.0 - float(sim)
                    distances.append(distance)
                pixel_distances.append(((x, y), distances))
            
            mask_pixel_distances.append((mask_idx, pixel_distances))
        
        return mask_pixel_distances
    
    def perform_binning(self, mask_distances: List[Tuple[int, List[float]]]) -> Tuple[List[List[float]], Dict[float, int]]:
        """Выполняет биннинг дистанций.
        
        Args:
            mask_distances: Список дистанций для каждой маски
            
        Returns:
            Tuple[List[List[float]], Dict[float, int]]: (список_бинов, словарь_дистанция_к_бину)
        """
        # Собираем все дистанции
        all_distances = []
        for _, distances in mask_distances:
            all_distances.extend(distances)
        
        if len(all_distances) == 0:
            return [], {}
        
        # Сортируем дистанции
        sorted_distances = sorted(all_distances)
        
        # Создаем не более num_bins бинов с помощью равномерного разбиения
        num_bins = min(self.num_bins, len(sorted_distances))
        np_bins = np.array_split(np.array(sorted_distances, dtype=float), num_bins)
        bins = [arr.tolist() for arr in np_bins if arr.size > 0]
        
        # Создаем словарь для маппинга дистанции к номеру бина
        distance_to_bin = {}
        for bin_idx, bin_distances in enumerate(bins):
            for distance in bin_distances:
                distance_to_bin[float(distance)] = bin_idx
        
        return bins, distance_to_bin
    
    def detect_concepts(self, mask_distances: List[Tuple[int, List[float]]], 
                       distance_to_bin: Dict[float, int]) -> List[int]:
        """Детектирует концепты на основе биннинга с адаптивным порогом.
        
        Args:
            mask_distances: Список дистанций для каждой маски
            distance_to_bin: Словарь маппинга дистанции к номеру бина
            
        Returns:
            List[int]: Список индексов масок, где обнаружены концепты
        """
        selected_masks = []
        
        for mask_idx, distances in mask_distances:
            if not distances:
                continue
            # Определяем, в какие бины попадают дистанции этой маски
            bins_for_mask = [distance_to_bin[distance] for distance in distances if distance in distance_to_bin]
            
            # Считаем количество дистанций в первых двух бинах (бины 0 и 1)
            counts_in_first_bins = sum(1 for bin_idx in bins_for_mask if bin_idx <= 1)
            
            # Адаптивный порог: не больше кол-ва дистанций (на случай 1-2 позитивов)
            adaptive_threshold = min(self.concept_threshold, len(distances))
            
            # Если достаточно дистанций в первых бинах, считаем концепт обнаруженным
            if counts_in_first_bins >= adaptive_threshold:
                selected_masks.append(mask_idx)
        
        return selected_masks
    
    def perform_pixel_binning(self, mask_pixel_distances: List[Tuple[int, List[Tuple[Tuple[int, int], List[float]]]]]) -> Tuple[List[List[float]], Dict[float, int]]:
        """Выполняет биннинг дистанций на уровне пикселей.
        
        Args:
            mask_pixel_distances: Список дистанций для каждого пикселя в каждой маске
            
        Returns:
            Tuple[List[List[float]], Dict[float, int]]: (список_бинов, словарь_дистанция_к_бину)
        """
        # Собираем все дистанции со всех пикселей
        all_distances = []
        for mask_idx, pixel_distances in mask_pixel_distances:
            for (x, y), distances in pixel_distances:
                all_distances.extend(distances)
        
        if len(all_distances) == 0:
            return [], {}
        
        # Сортируем дистанции
        sorted_distances = sorted(all_distances)
        
        # Создаем не более num_bins бинов с помощью равномерного разбиения
        num_bins = min(self.num_bins, len(sorted_distances))
        np_bins = np.array_split(np.array(sorted_distances, dtype=float), num_bins)
        bins = [arr.tolist() for arr in np_bins if arr.size > 0]
        
        # Создаем словарь для маппинга дистанции к номеру бина
        distance_to_bin = {}
        for bin_idx, bin_distances in enumerate(bins):
            for distance in bin_distances:
                distance_to_bin[float(distance)] = bin_idx
        
        return bins, distance_to_bin
    
    def detect_pixel_concepts(self, mask_pixel_distances: List[Tuple[int, List[Tuple[Tuple[int, int], List[float]]]]], 
                             distance_to_bin: Dict[float, int]) -> List[Tuple[int, List[Tuple[int, int]]]]:
        """Детектирует концепты на уровне пикселей с адаптивным порогом.
        
        Args:
            mask_pixel_distances: Список дистанций для каждого пикселя в каждой маске
            distance_to_bin: Словарь маппинга дистанции к номеру бина
            
        Returns:
            List[Tuple[int, List[Tuple[int, int]]]]: Список кортежей (индекс_маски, список_координат_пикселей_концепта)
        """
        selected_pixel_masks = []
        
        for mask_idx, pixel_distances in mask_pixel_distances:
            concept_pixels = []
            
            for (x, y), distances in pixel_distances:
                if not distances:
                    continue
                # Определяем, в какие бины попадают дистанции этого пикселя
                bins_for_pixel = []
                for distance in distances:
                    if distance in distance_to_bin:
                        bins_for_pixel.append(distance_to_bin[distance])
                
                # Считаем количество дистанций в первых двух бинах (бины 0 и 1)
                counts_in_first_bins = sum(1 for bin_idx in bins_for_pixel if bin_idx <= 1)
                
                # Адаптивный порог: не больше кол-ва дистанций (на случай 1-2 позитивов)
                adaptive_threshold = min(self.concept_threshold, len(distances))
                
                # Если достаточно дистанций в первых бинах, считаем пиксель принадлежащим концепту
                if counts_in_first_bins >= adaptive_threshold:
                    concept_pixels.append((x, y))
            
            # Добавляем маску только если в ней есть пиксели концепта
            if concept_pixels:
                selected_pixel_masks.append((mask_idx, concept_pixels))
        
        return selected_pixel_masks

    def _fill_holes(self, binary):
        h, w = binary.shape[:2]
        flood = binary.copy()
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, mask, (0, 0), 255)
        flood_inv = cv2.bitwise_not(flood)
        return cv2.bitwise_or(binary, flood_inv)

    def _extract_valid_contours(self, binary, min_area=50, min_pts=5,
                                min_solidity=0.75, extent_range=(0.08, 0.995)):
        # contours: список экстерн-контуров без дыр
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in (contours or []):
            if c is None or len(c) < min_pts:
                continue
            area = float(cv2.contourArea(c))
            if area < min_area:
                continue
            hull = cv2.convexHull(c)
            hull_area = float(cv2.contourArea(hull)) + 1e-6
            solidity = area / hull_area  # насколько контур «плотный», без щелей
            x, y, w, h = cv2.boundingRect(c)
            extent = area / float(w * h + 1e-6)  # доля заполнения bbox (убирает длинные полоски)
            if solidity < min_solidity:
                continue
            if not (extent_range[0] <= extent <= extent_range[1]):
                continue
            out.append((c, {"area": area, "bbox": [x, y, w, h], "solidity": solidity, "extent": extent}))
        return out
    
    def create_masks_from_pixels(self, selected_pixels: List[Tuple[int, List[Tuple[int, int]]]], 
                                original_masks: List[Dict[str, Any]], 
                                image_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Создает новые маски на основе выбранных пикселей.
        
        Args:
            selected_pixels: Список кортежей (mask_idx, selected_pixel_coords)
            original_masks: Исходные маски
            image_shape: Размер изображения (height, width)
            
        Returns:
            Список новых масок в формате SAM
        """
        new_masks = []
        
        for mask_idx, pixel_coords in selected_pixels:
            if mask_idx >= len(original_masks):
                continue
                
            # Создаем новую маску того же размера, заполненную нулями
            new_mask = np.zeros(image_shape, dtype=bool)
            
            # Устанавливаем True для выбранных пикселей
            for x, y in pixel_coords:
                if 0 <= y < new_mask.shape[0] and 0 <= x < new_mask.shape[1]:
                    new_mask[y, x] = True
            
            # Добавляем маску только если она содержит пиксели
            if np.any(new_mask):
                # Вычисляем bounding box
                ys, xs = np.where(new_mask)
                x_min, x_max = int(xs.min()), int(xs.max())
                y_min, y_max = int(ys.min()), int(ys.max())
                bbox_w = x_max - x_min + 1
                bbox_h = y_max - y_min + 1
                bbox = [x_min, y_min, bbox_w, bbox_h]
                area = int(new_mask.sum())
                cx = x_min + bbox_w // 2
                cy = y_min + bbox_h // 2
                
                # Создаем словарь в формате SAM
                mask_dict = {
                    'segmentation': new_mask,
                    'area': area,
                    'bbox': bbox,
                    'predicted_iou': 1.0,
                    'point_coords': [[cx, cy]],
                    'stability_score': 1.0,
                    'crop_box': [0, 0, image_shape[1], image_shape[0]]
                }
                
                new_masks.append(mask_dict)
        
        return new_masks

    def process_with_pixel_binning(self, input_image: Image.Image,
                                  positive_images: List[Image.Image],
                                  negative_images: List[Image.Image] = None,
                                  masks: List[Dict[str, Any]] = None,
                                  heatmap: torch.Tensor = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Полный процесс биннинга на уровне пикселей для детекции концептов.
        
        Args:
            input_image: Входное изображение
            positive_images: Положительные примеры
            negative_images: Отрицательные примеры
            masks: Существующие маски (если есть)
            heatmap: Тепловая карта для генерации масок (если masks не предоставлены)
            
        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: (новые_маски_из_пикселей, все_исходные_маски)
        """
        if negative_images is None:
            negative_images = []
        
        # Если маски не предоставлены, генерируем их из хитмапы
        if masks is None:
            if heatmap is None:
                raise ValueError("Необходимо предоставить либо маски, либо хитмапу")
            # Получаем размер исходного изображения для правильного масштабирования
            input_image_np = np.array(input_image)
            image_height, image_width = input_image_np.shape[:2]
            masks = self.generate_masks_from_heatmap(
                heatmap=heatmap,
                target_size=(image_height, image_width)
            )
        
        if not masks:
            return [], []
        
        # Преобразуем входное изображение в numpy
        input_image_np = np.array(input_image)
        H, W = input_image_np.shape[:2]
        
        # Вычисляем скорректированный эмбеддинг для входного изображения
        adjusted_embeddings = self.compute_adjusted_embeddings([input_image] + positive_images, negative_images)
        
        # Извлекаем признаки пикселей из масок
        mask_pixel_features = self.extract_pixel_features_from_masks(input_image_np, masks)
        
        # Вычисляем дистанции для пикселей
        mask_pixel_distances = self.compute_pixel_distances(mask_pixel_features, adjusted_embeddings)
        
        # Выполняем биннинг на уровне пикселей
        pixel_bins, pixel_distance_to_bin = self.perform_pixel_binning(mask_pixel_distances)
        
        # Детектируем концепты на уровне пикселей
        selected_pixel_masks = self.detect_pixel_concepts(mask_pixel_distances, pixel_distance_to_bin)
        
        # Создаем новые маски из выбранных пикселей
        new_masks = self.create_masks_from_pixels(selected_pixel_masks, masks, (H, W))
        
        return new_masks, masks
    
    def generate_masks_from_heatmap(self, heatmap: torch.Tensor,
                                   threshold: float = 0.5,
                                   min_area: int = 100,
                                   target_size: Tuple[int, int] = None,
                                   min_solidity: float = 0.5,
                                   extent_range: Tuple[float, float] = (0.1, 1.0)) -> List[Dict[str, Any]]:
        """Генерирует маски из тепловой карты с улучшенной фильтрацией контуров."""
        if isinstance(heatmap, torch.Tensor):
            heatmap_np = heatmap.cpu().numpy()
        else:
            heatmap_np = heatmap

        if target_size is not None:
            H, W = target_size
            Hm_up = cv2.resize(heatmap_np, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            H, W = heatmap_np.shape
            Hm_up = heatmap_np

        # Нормализуем хитмапу
        Hm_up = (Hm_up - Hm_up.min()) / (Hm_up.max() - Hm_up.min() + 1e-8)

        # 1) бинаризация
        B = (Hm_up >= threshold).astype(np.uint8) * 255

        # 2) морфология + заполнение дыр
        B = cv2.morphologyEx(B, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        B = cv2.morphologyEx(B, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        B = self._fill_holes(B)

        # 3) контуры + геометрические фильтры
        conts = self._extract_valid_contours(
            B,
            min_area=min_area,
            min_pts=5,
            min_solidity=min_solidity,
            extent_range=extent_range
        )

        masks = []
        # 4) берём крупнейший валидный контур
        conts.sort(key=lambda x: x[1]["area"], reverse=True)
        
        if not conts:
            return []
            
        c, st = conts[0]

        # 5) аккуратная сегментация из контура
        seg = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(seg, [c], -1, 255, thickness=-1, lineType=cv2.LINE_AA)
        seg = seg.astype(bool)

        x, y, w, h = st["bbox"]

        masks.append({
            "segmentation": seg,
            "area": int(st["area"]),
            "bbox": [x, y, w, h],
            "predicted_iou": 1.0,
            "point_coords": [[x + w // 2, y + h // 2]],
            "stability_score": 1.0,
            "crop_box": [0, 0, W, H],
        })
        
        return masks

    def process_with_binning(self, input_image: Image.Image,
                           positive_images: List[Image.Image],
                           negative_images: List[Image.Image] = None,
                           masks: List[Dict[str, Any]] = None,
                           heatmap: torch.Tensor = None,
                           use_pixel_level: bool = True) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Полный процесс биннинга для детекции концептов.
        
        Args:
            input_image: Входное изображение
            positive_images: Положительные примеры
            negative_images: Отрицательные примеры
            masks: Существующие маски (если есть)
            heatmap: Тепловая карта для генерации масок (если masks не предоставлены)
            use_pixel_level: Использовать биннинг на уровне пикселей (по умолчанию True)
            
        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: (выбранные_маски, все_маски)
        """
        if negative_images is None:
            negative_images = []
        
        # Если маски не предоставлены, генерируем их из хитмапы
        if masks is None:
            if heatmap is None:
                raise ValueError("Необходимо предоставить либо маски, либо хитмапу")
            # Получаем размер исходного изображения для правильного масштабирования
            input_image_np = np.array(input_image)
            image_height, image_width = input_image_np.shape[:2]
            masks = self.generate_masks_from_heatmap(
                heatmap=heatmap,
                target_size=(image_height, image_width)
            )
        
        if not masks:
            return [], []
        
        # Выбираем метод биннинга
        if use_pixel_level:
            # Новый подход: биннинг на уровне пикселей
            selected_masks, all_masks = self.process_with_pixel_binning(
                input_image, positive_images, negative_images, masks, heatmap
            )
            return selected_masks, all_masks
        else:
            # Старый подход: биннинг всей маски
            # Преобразуем входное изображение в numpy
            input_image_np = np.array(input_image)
            H, W = input_image_np.shape[:2]
            
            # Приводим размерность масок (14x14) к размеру исходного изображения и
            # пересчитываем bbox/area/point_coords/crop_box
            adjusted_masks = []
            for mask in masks:
                seg = mask.get('segmentation')
                if seg is None:
                    continue
                # Приведение к uint8 для корректного ресайза, затем обратно в bool
                if seg.dtype == bool:
                    seg_u8 = (seg.astype(np.uint8)) * 255
                else:
                    seg_u8 = (seg > 0).astype(np.uint8) * 255
                # Если размер не совпадает, масштабируем до размера входного изображения
                if seg_u8.shape[:2] != (H, W):
                    seg_u8_resized = cv2.resize(seg_u8, (W, H), interpolation=cv2.INTER_NEAREST)
                else:
                    seg_u8_resized = seg_u8
                seg_bool = seg_u8_resized > 0
                
                # Пересчёт bbox из маски
                ys, xs = np.where(seg_bool)
                if xs.size == 0 or ys.size == 0:
                    # Пустая маска после ресайза — пропускаем
                    continue
                x_min, x_max = int(xs.min()), int(xs.max())
                y_min, y_max = int(ys.min()), int(ys.max())
                bbox_w = x_max - x_min + 1
                bbox_h = y_max - y_min + 1
                bbox = [x_min, y_min, bbox_w, bbox_h]
                area = int(seg_bool.sum())
                cx = x_min + bbox_w // 2
                cy = y_min + bbox_h // 2
                
                # Обновляем поля маски под размер исходного изображения
                mask['segmentation'] = seg_bool
                mask['bbox'] = bbox
                mask['area'] = area
                mask['point_coords'] = [[cx, cy]]
                mask['crop_box'] = [0, 0, W, H]
                adjusted_masks.append(mask)
            masks = adjusted_masks
            
            if not masks:
                return [], []
            
            # Вычисляем скорректированные эмбеддинги
            adjusted_embeddings = self.compute_adjusted_embeddings(positive_images, negative_images)
            
            # Извлекаем признаки из масок
            mask_embeddings = self.extract_features_from_masks(input_image_np, masks)
            
            # Нормализуем эмбеддинги масок
            mask_embeddings = mask_embeddings / np.linalg.norm(mask_embeddings, axis=1, keepdims=True)
            
            # Вычисляем дистанции
            mask_distances = self.compute_distances(mask_embeddings, adjusted_embeddings)
            
            # Выполняем биннинг
            bins, distance_to_bin = self.perform_binning(mask_distances)
            
            # Детектируем концепты
            selected_mask_indices = self.detect_concepts(mask_distances, distance_to_bin)
            
            # Возвращаем выбранные маски вместо индексов для совместимости
            selected_masks = [masks[i] for i in selected_mask_indices if i < len(masks)]
            
            return selected_masks, masks
    
    def visualize_binning_results(self, bins: List[List[float]], 
                                 selected_masks: List[int],
                                 save_path: Optional[str] = None) -> None:
        """Визуализирует результаты биннинга.
        
        Args:
            bins: Список бинов с дистанциями
            selected_masks: Индексы выбранных масок
            save_path: Путь для сохранения графика (опционально)
        """
        # Вычисляем средние значения для каждого бина
        bin_averages = [np.mean(bin_distances) for bin_distances in bins]
        bin_numbers = np.arange(1, len(bins) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.bar(bin_numbers, bin_averages, color='skyblue')
        plt.title(f'Средние дистанции по бинам (Выбрано масок: {len(selected_masks)})')
        plt.xlabel('Номер бина')
        plt.ylabel('Средняя евклидова дистанция')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def visualize_selected_masks(self, input_image: Image.Image,
                               masks: List[Dict[str, Any]],
                               selected_indices: List[int],
                               save_path: Optional[str] = None) -> Image.Image:
        """Визуализирует выбранные маски на изображении.
        
        Args:
            input_image: Исходное изображение
            masks: Маски в формате SAM
            selected_indices: Индексы выбранных масок
            save_path: Путь для сохранения результата (опционально)
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.colors import ListedColormap
        import random
        
        # Преобразуем изображение в numpy
        img_array = np.array(input_image)
        
        # Создаем фигуру
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img_array)
        
        # Генерируем случайные цвета для масок
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
        
        # Отображаем выбранные маски
        for i, mask_idx in enumerate(selected_indices):
            if mask_idx < len(masks):
                mask = masks[mask_idx]
                segmentation = mask['segmentation']
                bbox = mask['bbox']
                
                # Создаем цветную маску
                color = colors[i % len(colors)]
                mask_overlay = np.zeros((*segmentation.shape, 4))
                mask_overlay[segmentation] = [*plt.colors.to_rgba(color, alpha=0.5)]
                
                # Накладываем маску
                ax.imshow(mask_overlay)
                
                # Добавляем bounding box
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2], bbox[3],
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                
                # Добавляем номер маски
                ax.text(bbox[0], bbox[1] - 5, f'Mask {mask_idx}', 
                       color=color, fontsize=12, fontweight='bold')
        
        ax.set_title(f'Выбранные маски после binning ({len(selected_indices)} из {len(masks)})')
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        plt.tight_layout()
        plt.show()
        
        return input_image