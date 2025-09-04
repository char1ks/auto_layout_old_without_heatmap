import argparse
import os
import sys
import cv2
import time
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Union, Any, Tuple
sys.path.append('./searchdet-main')
try:
    from mask_withsearch import initialize_sam as init_searchdet
    SEARCHDET_AVAILABLE = True
except Exception as e:
    print(f"⚠️ SearchDet недоступен: {e}")
    SEARCHDET_AVAILABLE = False
from .mask_generation import MaskGenerator
from .filtering import MaskFilter  
from .embeddings import EmbeddingExtractor
from .scoring import ScoreCalculator
from .step7_result_saving import ResultSaver
from .sam_predictor import SAMPredictor
from .utils import get_image_size, get_feature_map_size, upsample_feature_map
from .dinov3_encoder import DinoV3Encoder
from .heatmap_generator import HeatmapGenerator
from .binning_processor import BinningProcessor
from .enhanced_heatmap_processor import EnhancedHeatmapProcessor
from .models import DetectorConfig, ProcessingResult, MaskData, DetectionResult
from ..utils.validation import ImageValidator, DirectoryValidator, ValidationError, validate_processing_pipeline_inputs
import torch
from .models import MaskBackend, BackboneType


class SearchDetDetector:
    
    def __init__(self, config: Optional[DetectorConfig] = None, **kwargs: Any) -> None:
        if not SEARCHDET_AVAILABLE:
            raise RuntimeError("SearchDet не найден")
        
        if config is None:
            self.config = DetectorConfig.from_dict(kwargs)
        else:
            self.config = config.update(**kwargs) if kwargs else config
        
        self.params = self.config.to_dict()
        
        # Основные параметры
        self.sam_encoder = self.config.sam_encoder
        self.sam_model = self.config.sam_model
        self.device = self.config.device
        if self.device == "auto":
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.half = self.config.half_precision
        self.mask_backend = self.config.mask_backend
        # Поддержка Enum и нормализация значения
        if isinstance(self.mask_backend, MaskBackend):
            self.mask_backend = self.mask_backend.value
        if isinstance(self.mask_backend, str):
            self.mask_backend = self.mask_backend.replace('_', '-')
        self.backbone = self.config.backbone
        if isinstance(self.backbone, BackboneType):
            self.backbone = self.backbone.value
        self.dinov3_ckpt = self.config.dinov3_ckpt
        self.nms_iou = self.config.nms_iou
        if not self.backbone.startswith('dinov2'):
            feat_short = str(self.config.feat_short_side)
            os.environ['SEARCHDET_FEAT_SHORT_SIDE'] = feat_short
            print(f"🔧 Установлено SEARCHDET_FEAT_SHORT_SIDE={feat_short}")
        else:
            print(f"🔧 DINOv2 бэкенд: используется собственный размер модели")
        print(f"🔧 Выбран SAM энкодер: {self.sam_encoder}")
        self.searchdet_resnet, self.searchdet_layer, self.searchdet_transform, self.searchdet_sam = init_searchdet()
        if not self.backbone.startswith('dinov2'):
            import torchvision.transforms as transforms
            feat_short_side = int(os.getenv('SEARCHDET_FEAT_SHORT_SIDE', '384'))
            self.searchdet_transform = transforms.Compose([
                transforms.Resize(feat_short_side),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.searchdet_transform = None
        # Инициализируем MaskGenerator для FastSAM бэкенда
        # Фильтруем параметры, которые уже передаются явно, чтобы избежать дублирования ключей
        generator_params = {
            k: v for k, v in self.params.items()
            if k not in {"mask_backend", "device", "fastsam_model", "fastsam_device", "sam_generator"}
        }
        self.mask_generator = MaskGenerator(
            mask_backend=self.mask_backend,
            device=self.device,
            fastsam_model=self.config.fastsam_model,
            fastsam_device=self.config.fastsam_device or self.device,
            sam_generator=None,
            **generator_params
        )
        
        # Создаем универсальный SAMPredictor с поддержкой разных бэкендов
        # По умолчанию используем SAM, но можно переключиться на FastSAM или heatmap
        segmentation_backend = getattr(self.config, 'segmentation_backend', 'sam')
        
        if segmentation_backend == 'sam':
            self.sam_predictor = SAMPredictor(
                sam_model=self.searchdet_sam,
                backend_type="sam"
            )
        elif segmentation_backend == 'fastsam':
            self.sam_predictor = SAMPredictor(
                backend_type="fastsam",
                mask_generator=self.mask_generator
            )
        elif segmentation_backend == 'heatmap':
            heatmap_threshold = getattr(self.config, 'heatmap_threshold', 0.5)
            self.sam_predictor = SAMPredictor(
                backend_type="heatmap",
                threshold=heatmap_threshold
            )
        else:
            # Fallback к оригинальному SAM
            self.sam_predictor = SAMPredictor(
                sam_model=self.searchdet_sam,
                backend_type="sam"
            )
        self.mask_filter = MaskFilter(self.params)
        self.embedding_extractor = EmbeddingExtractor(
            backbone_name=self.config.dinov3_backbone,
            device=self.device,
            ckpt_path=self.config.dinov3_ckpt
        )
        self.score_calculator = ScoreCalculator(self.params)
        self.result_saver = ResultSaver(self.config.overlay_alpha)
        
        self.dinov3_encoder = DinoV3Encoder(
            backbone_name=self.config.dinov3_backbone,
            ckpt_path=self.config.dinov3_ckpt,
            device=self.device,
            half_precision=self.config.dino_half_precision,
            vit_pooling=self.config.vit_pooling,
            loader=self.config.loader,
            repo_dir=self.config.repo_dir
        )
        
        # Инициализация компонентов для heatmap и binning
        self.heatmap_generator = HeatmapGenerator(self.dinov3_encoder)
        self.binning_processor = BinningProcessor(self.dinov3_encoder)
        self.enhanced_heatmap_processor = EnhancedHeatmapProcessor(self.dinov3_encoder)
        

    def set_references(self, image_path: Union[str, Path], positive_dir: Union[str, Path], 
                      negative_dir: Optional[Union[str, Path]] = None) -> None:
        try:
            ImageValidator.validate_image_path(str(image_path))
            DirectoryValidator.validate_input_directory(str(positive_dir))
            if negative_dir:
                DirectoryValidator.validate_input_directory(str(negative_dir))
        except ValidationError as e:
            raise ValidationError(f"Ошибка валидации референсных данных: {e}")
            
        raise NotImplementedError("Метод set_references еще не реализован")
    
    def find_present_elements_with_heatmap(self, image_path: Union[str, Path], positive_dir: Union[str, Path], 
                                         negative_dir: Optional[Union[str, Path]] = None, 
                                         output_dir: str = "output",
                                         use_heatmap_masks: bool = True) -> Dict[str, Any]:
        """Поиск элементов с использованием heatmap и binning вместо SAM масок.
        
        Args:
            image_path: Путь к анализируемому изображению
            positive_dir: Директория с положительными примерами
            negative_dir: Директория с отрицательными примерами (опционально)
            output_dir: Директория для сохранения результатов
            use_heatmap_masks: Использовать маски из хитмапы вместо SAM
            
        Returns:
            Словарь с найденными элементами и масками
        """
        try:
            validate_processing_pipeline_inputs(
                image_path=str(image_path),
                positive_dir=str(positive_dir),
                negative_dir=str(negative_dir) if negative_dir else None,
                output_dir=output_dir
            )
        except ValidationError as e:
            raise ValidationError(f"Ошибка валидации входных данных: {e}")
        
        print(f"🔍 Анализ с heatmap: {image_path}" + "="*60)
        print("🔄 ПОСЛЕДОВАТЕЛЬНОСТЬ ВЫПОЛНЕНИЯ HEATMAP PIPELINE:")
        print("=" * 80)
        
        timing_info: Dict[str, float] = {}
        t_total = time.time()
        
        # Загрузка изображения
        t_loading = time.time()
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
        image_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_np.astype(np.uint8))
        timing_info['image_loading'] = time.time() - t_loading
        image_name = Path(image_path).name
        
        # Загрузка примеров
        print("1️⃣ Шаг 1: Загрузка positive/negative примеров")
        t_examples = time.time()   
        pos_by_class = self._load_positive_by_class(positive_dir)
        if len(pos_by_class) == 0:
            print("   ❌ Нет положительных примеров — прекращаем.")
            # Сохраним хотя бы heatmap, если она будет сгенерирована далее
            return {"found_elements": [], "masks": []}
        
        total_pos = sum(len(v) for v in pos_by_class.values())
        neg_imgs = self._load_example_images(negative_dir) if negative_dir else []
        timing_info['examples_loading'] = time.time() - t_examples
        print(f"   📁 Positive: {total_pos} в {len(pos_by_class)} классах\t📁 Negative: {len(neg_imgs)}")
        
        # Генерация хитмапы
        print("2️⃣ Шаг 2: Генерация heatmap с помощью DINOv3")
        t_heatmap = time.time()
        
        # Собираем все положительные изображения
        all_positive_images = []
        for class_images in pos_by_class.values():
            all_positive_images.extend(class_images)
        
        # Генерируем хитмапу
        heatmap = self.heatmap_generator.generate_heatmap(
            input_image=image_pil,
            positive_images=all_positive_images,
            negative_images=neg_imgs
        )
        timing_info['heatmap_generation'] = time.time() - t_heatmap
        print(f"   📊 Heatmap сгенерирована: {heatmap.shape}")
        
        if use_heatmap_masks:
            # Генерация масок из хитмапы
            print("3️⃣ Шаг 3: Генерация масок из heatmap")
            t_mask_gen = time.time()
            # Получаем размер исходного изображения
            image_height, image_width = image_np.shape[:2]
            print(f"   📐 Размер исходного изображения: {image_width}x{image_height}")
            
            masks = self.binning_processor.generate_masks_from_heatmap(
                heatmap=heatmap,
                threshold=0.3,  # Понижен порог для захвата оранжево-желтых зон
                min_area=50,    # Уменьшена минимальная площадь
                target_size=(image_height, image_width)
            )
            timing_info['mask_generation'] = time.time() - t_mask_gen
            print(f"   📊 Сгенерировано масок из heatmap: {len(masks)}")
        else:
            # Используем обычную генерацию масок SAM
            print("3️⃣ Шаг 3: Генерация масок через SAM/FastSAM")
            t_masks = time.time()
            masks = self.mask_generator.generate(image_np)
            timing_info['mask_generation'] = time.time() - t_masks
            print(f"   📊 Сгенерировано масок SAM: {len(masks)}")
        
        if not masks:
            print("   ❌ Нет масок для обработки.")
            # Всегда сохраняем heatmap, даже если масок нет
            _saved = self.result_saver.save_all_results(
                image_np,
                [],
                output_dir,
                image_name,
                pipeline_config={"backend": "heatmap_binning"},
                heatmap=heatmap
            )
            return {"found_elements": [], "masks": [], "saved_files": _saved, "heatmap": heatmap, "output_directory": output_dir}
        
        # Фильтрация масок
        print("4️⃣ Шаг 4: Фильтрация масок")
        t_filtering = time.time()
        masks = self.mask_filter.apply_all_filters(masks, image_np)
        timing_info['mask_filtering'] = time.time() - t_filtering
        
        if not masks:
            print("   ❌ Нет валидных масок после фильтров.")
            # Всегда сохраняем heatmap, даже если масок нет
            _saved = self.result_saver.save_all_results(
                image_np,
                [],
                output_dir,
                image_name,
                pipeline_config={"backend": "heatmap_binning"},
                heatmap=heatmap
            )
            return {"found_elements": [], "masks": [], "saved_files": _saved, "heatmap": heatmap, "output_directory": output_dir}
        
        # Binning процесс
        print("5️⃣ Шаг 5: Binning процесс для детекции концептов")
        t_binning = time.time()
        
        selected_mask_indices, all_masks = self.binning_processor.process_with_binning(
            input_image=image_pil,
            positive_images=all_positive_images,
            negative_images=neg_imgs,
            masks=masks
        )
        timing_info['binning_process'] = time.time() - t_binning
        
        # Определяем формат возвращаемых данных: индексы или маски
        is_index_list = len(selected_mask_indices) == 0 or isinstance(selected_mask_indices[0], (int, np.integer))
        total_candidates = len(masks)
        print(f"   📊 Выбрано масок после binning: {len(selected_mask_indices)} из {total_candidates}")
        
        if not selected_mask_indices:
            print("   ❌ Нет масок после binning процесса.")
            # Всегда сохраняем heatmap, даже если выбранных масок нет
            _saved = self.result_saver.save_all_results(
                image_np,
                [],
                output_dir,
                image_name,
                pipeline_config={"backend": "heatmap_binning"},
                heatmap=heatmap
            )
            return {"found_elements": [], "masks": [], "saved_files": _saved, "heatmap": heatmap, "output_directory": output_dir}
        
        # Формирование результатов
        print("6️⃣ Шаг 6: Формирование результатов")
        t_result = time.time()
        
        found = []
        result_masks = []
        H, W = image_np.shape[:2]
        
        if is_index_list:
            # Старый путь: получены индексы исходных масок
            for mask_idx in selected_mask_indices:
                if mask_idx >= len(masks):
                    continue
                mask_dict = masks[mask_idx].copy()
                class_name = list(pos_by_class.keys())[0] if pos_by_class else "detected"
                mask_dict['confidence'] = 0.8
                mask_dict['class'] = class_name
                if 'area' not in mask_dict and 'segmentation' in mask_dict:
                    mask_dict['area'] = int(np.sum(mask_dict['segmentation']))
                bx = mask_dict.get('bbox', [0,0,0,0])
                if len(bx) == 4 and (bx[2] <= W and bx[3] <= H):
                    x1, y1, w, h = bx
                    bbox_xyxy = [int(x1), int(y1), int(x1 + w), int(y1 + h)]
                else:
                    bbox_xyxy = [int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])]
                found.append({
                    'mask': mask_dict,
                    'confidence': float(mask_dict['confidence']),
                    'bbox': mask_dict['bbox'],
                    'class': class_name
                })
                result_masks.append(mask_dict)
        else:
            # Новый путь: получены готовые маски, созданные пиксельным биннингом
            for mask_dict in selected_mask_indices:
                if not isinstance(mask_dict, dict) or 'segmentation' not in mask_dict:
                    continue
                class_name = list(pos_by_class.keys())[0] if pos_by_class else "detected"
                mask_dict = mask_dict.copy()
                mask_dict['confidence'] = 0.8
                mask_dict['class'] = class_name
                if 'area' not in mask_dict:
                    mask_dict['area'] = int(np.sum(mask_dict['segmentation']))
                if 'bbox' not in mask_dict:
                    ys, xs = np.where(mask_dict['segmentation'])
                    if xs.size and ys.size:
                        x_min, x_max = int(xs.min()), int(xs.max())
                        y_min, y_max = int(ys.min()), int(ys.max())
                        mask_dict['bbox'] = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
                    else:
                        mask_dict['bbox'] = [0, 0, 0, 0]
                found.append({
                    'mask': mask_dict,
                    'confidence': float(mask_dict['confidence']),
                    'bbox': mask_dict['bbox'],
                    'class': class_name
                })
                result_masks.append(mask_dict)
        
        timing_info['result_formatting'] = time.time() - t_result
        
        # Сохранение результатов
        print("7️⃣ Шаг 7: Сохранение результатов")
        t_saving = time.time()
        saved_files = self.result_saver.save_all_results(
            image_np,
            result_masks,
            output_dir,
            image_name,
            pipeline_config={"backend": "heatmap_binning"},
            heatmap=heatmap
        )
        timing_info['result_saving'] = time.time() - t_saving
        
        total_time = time.time() - t_total
        timing_info['total_time'] = total_time
        
        print(f"🎯 Найдено элементов: {len(found)}")
        print(f"⏱️ Общее время: {total_time:.2f} сек")
        print(f"💾 Результаты сохранены в: {output_dir}")
        print(f"📁 Сохранено файлов: {len(saved_files)}")
        self._print_timing_statistics(timing_info)
        
        return {
            "found_elements": found, 
            "masks": result_masks,
            "timing_info": timing_info,
            "output_directory": output_dir,
            "saved_files": saved_files,
            "heatmap": heatmap
        }
    
    def find_present_elements_with_enhanced_heatmap(self, image_path: Union[str, Path], positive_dir: Union[str, Path], 
                                                   negative_dir: Optional[Union[str, Path]] = None, 
                                                   output_dir: str = "output") -> Dict[str, Any]:
        """Поиск элементов с улучшенной обработкой heatmap и пиксельным биннингом."""
        try:
            validate_processing_pipeline_inputs(
                image_path=str(image_path),
                positive_dir=str(positive_dir),
                negative_dir=str(negative_dir) if negative_dir else None,
                output_dir=output_dir
            )
        except ValidationError as e:
            raise ValidationError(f"Ошибка валидации входных данных: {e}")
        
        print(f"🔍 Улучшенный анализ с heatmap: {image_path}" + "="*60)
        print("🔄 ДЕТАЛЬНАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ ВЫПОЛНЕНИЯ ENHANCED HEATMAP PIPELINE:")
        print("=" * 80)
        
        timing_info: Dict[str, float] = {}
        t_total = time.time()
        
        # Шаг 1: Загрузка изображения
        print("1️⃣ Шаг 1: Загрузка изображения")
        t_loading = time.time()
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
        image_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        image_name = Path(image_path).stem
        timing_info['image_loading'] = time.time() - t_loading
        
        # Шаг 2: Загрузка примеров
        print("2️⃣ Шаг 2: Загрузка positive/negative примеров")
        t_examples = time.time()
        pos_by_class = self._load_positive_by_class(positive_dir)
        if len(pos_by_class) == 0:
            print("   ❌ Нет положительных примеров — прекращаем.")
            return {"found_elements": [], "masks": []}
        
        total_pos = sum(len(v) for v in pos_by_class.values())
        neg_imgs = self._load_example_images(negative_dir) if negative_dir else []
        timing_info['examples_loading'] = time.time() - t_examples
        print(f"   📁 Positive: {total_pos} в {len(pos_by_class)} классах\t📁 Negative: {len(neg_imgs)}")
        
        # Шаг 3: Генерация heatmap
        print("3️⃣ Шаг 3: Генерация heatmap с DINOv3")
        t_heatmap = time.time()
        
        # Преобразуем изображение в PIL формат
        image_pil = Image.fromarray(image_np.astype(np.uint8))
        
        # Собираем все положительные изображения
        all_positive_images = []
        for class_images in pos_by_class.values():
            all_positive_images.extend(class_images)
        
        heatmap = self.heatmap_generator.generate_heatmap(
            input_image=image_pil,
            positive_images=all_positive_images,
            negative_images=neg_imgs
        )
        timing_info['heatmap_generation'] = time.time() - t_heatmap
        print(f"   🔥 Heatmap размер: {heatmap.shape}, min: {heatmap.min():.3f}, max: {heatmap.max():.3f}")
        
        # Шаг 4: Улучшенная обработка heatmap
        print("4️⃣ Шаг 4: Улучшенная обработка heatmap с пиксельным биннингом")
        t_enhanced = time.time()
        
        # Преобразуем положительные примеры в PIL формат
        positive_images = []
        for class_name, class_images in pos_by_class.items():
            for img in class_images:
                if isinstance(img, np.ndarray):
                    positive_images.append(Image.fromarray(img.astype(np.uint8)))
                else:
                    positive_images.append(img)
        
        # Преобразуем отрицательные примеры в PIL формат
        negative_images = []
        for img in neg_imgs:
            if isinstance(img, np.ndarray):
                negative_images.append(Image.fromarray(img.astype(np.uint8)))
            else:
                negative_images.append(img)
        
        cleaned_heatmap, zone_masks = self.enhanced_heatmap_processor.process_enhanced_heatmap(
            image_pil, positive_images, negative_images, 
            save_cleaned_heatmap=True, output_dir=output_dir
        )
        timing_info['enhanced_processing'] = time.time() - t_enhanced
        
        print(f"   ✨ Получено {len(zone_masks)} зон после улучшенной обработки")
        
        # Шаг 6: Обработка масок и скоринг
        print("6️⃣ Шаг 6: Обработка масок и скоринг")
        t_scoring = time.time()
        
        found = []
        result_masks = []
        
        for i, mask_dict in enumerate(zone_masks):
            # zone_masks уже в формате SAM, используем их напрямую
            
            # Извлекаем эмбеддинги для маски
            image_pil = Image.fromarray(image_np.astype(np.uint8))
            try:
                mask_vecs = self.embedding_extractor.extract_mask_embeddings(image_pil, [mask_dict])
                if mask_vecs.shape[0] > 0:
                    # Вычисляем скоры для всех классов
                    class_scores = self.score_calculator.calculate_scores(
                        mask_vecs, pos_by_class, neg_imgs
                    )
                    
                    # Находим лучший класс
                    best_class = None
                    best_score = -1
                    for class_name, scores in class_scores.items():
                        if len(scores) > 0 and scores[0] > best_score:
                            best_score = scores[0]
                            best_class = class_name
                    
                    if best_class and best_score > 0.5:  # Порог для принятия
                        found.append({
                            'class': best_class,
                            'score': best_score,
                            'mask_index': i
                        })
                        
                        mask_dict.update({
                            'class': best_class,
                            'score': best_score,
                            'mask_index': i
                        })
                        result_masks.append(mask_dict)
                        
            except Exception as e:
                print(f"   ⚠️ Ошибка обработки маски {i}: {e}")
                continue
        
        timing_info['scoring'] = time.time() - t_scoring
        
        # Шаг 7: Сохранение результатов
        print("7️⃣ Шаг 7: Сохранение результатов")
        t_saving = time.time()
        saved_files = self.result_saver.save_all_results(
            image_np,
            result_masks,
            output_dir,
            image_name,
            pipeline_config={"backend": "enhanced_heatmap"},
            heatmap=cleaned_heatmap
        )
        timing_info['result_saving'] = time.time() - t_saving
        
        total_time = time.time() - t_total
        timing_info['total_time'] = total_time
        
        print(f"🎯 Найдено элементов: {len(found)}")
        print(f"⏱️ Общее время: {total_time:.2f} сек")
        print(f"💾 Результаты сохранены в: {output_dir}")
        print(f"📁 Сохранено файлов: {len(saved_files)}")
        self._print_timing_statistics(timing_info)
        
        return {
            "found_elements": found, 
            "masks": result_masks,
            "timing_info": timing_info,
            "output_directory": output_dir,
            "saved_files": saved_files,
            "heatmap": cleaned_heatmap,
            "zone_masks": zone_masks
        }
    
    def find_present_elements(self, image_path: Union[str, Path], positive_dir: Union[str, Path], 
                             negative_dir: Optional[Union[str, Path]] = None, 
                             output_dir: str = "output") -> Dict[str, Any]:
        try:
            validate_processing_pipeline_inputs(
                image_path=str(image_path),
                positive_dir=str(positive_dir),
                negative_dir=str(negative_dir) if negative_dir else None,
                output_dir=output_dir
            )
        except ValidationError as e:
            raise ValidationError(f"Ошибка валидации входных данных: {e}")
        
        print(f"🔍 Модульный анализ: {image_path}" + "="*60)
        print("🔄 ДЕТАЛЬНАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ ВЫПОЛНЕНИЯ МОДУЛЬНОГО PIPELINE:")
        print("=" * 80)
        print("8️⃣ searchdet_pipeline/core/detector.py → find_present_elements()")
        timing_info: Dict[str, float] = {}
        t_total = time.time()
        t_loading = time.time()
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
        image_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        timing_info['image_loading'] = time.time() - t_loading
        print("9️⃣ Шаг 1: _load_example_images() - загрузка positive/negative примеров")
        t_examples = time.time()   
        pos_by_class = self._load_positive_by_class(positive_dir)
        if len(pos_by_class) == 0:
            print("   ❌ Нет положительных примеров — прекращаем.")
            return {"found_elements": [], "masks": []}
        total_pos = sum(len(v) for v in pos_by_class.values())
        neg_imgs = self._load_example_images(negative_dir) if negative_dir else []
        timing_info['examples_loading'] = time.time() - t_examples
        print(f"   📁 Positive: {total_pos} в {len(pos_by_class)} классах	📁 Negative: {len(neg_imgs)}")
        print("🔟 Шаг 2: MaskGenerator.generate() - генерация масок через SAM/FastSAM")
        t_masks = time.time()
        masks = self.mask_generator.generate(image_np)
        timing_info['mask_generation'] = time.time() - t_masks
        print("1️⃣1️⃣ Шаг 3-7: MaskFilter.apply_all_filters() - все фильтры масок")
        t_filtering = time.time()
        masks = self.mask_filter.apply_all_filters(masks, image_np)
        timing_info['mask_filtering'] = time.time() - t_filtering
        if not masks:
            print("   ❌ Нет валидных масок после фильтров.")
            return {"found_elements": [], "masks": []}
        print("1️⃣2️⃣ Шаг 8: EmbeddingExtractor.extract_mask_embeddings() - эмбеддинги масок")
        t_embeddings = time.time()
        image_pil = Image.fromarray(image_np.astype(np.uint8))
        
        print(f"   🔍 ДИАГНОСТИКА: Начинаем обработку {len(masks)} масок")
        try:
            mask_vecs = self.embedding_extractor.extract_mask_embeddings(image_pil, masks)
            print(f"   🔍 ДИАГНОСТИКА: Получено {mask_vecs.shape[0]} валидных векторов из {len(masks)} масок")
            
            if mask_vecs.shape[0] == 0:
                print("   ❌ Не удалось получить эмбеддинги масок.")
                print("   📍 ПРИЧИНА: Все маски были отброшены как невалидные (NaN/Inf/нулевая норма)")
                
                print("   🔍 ДЕТАЛЬНАЯ ДИАГНОСТИКА МАСОК:")
                for i, mask in enumerate(masks):
                    try:
                        print(f"     Маска {i+1}: тип={type(mask)}, размер={getattr(mask, 'shape', 'неизвестно')}")
                        if hasattr(mask, 'segmentation'):
                            seg = mask['segmentation']
                            if isinstance(seg, np.ndarray):
                                print(f"       segmentation: shape={seg.shape}, dtype={seg.dtype}, sum={seg.sum()}")
                            else:
                                print(f"       segmentation: тип={type(seg)} (не numpy array)")
                    except Exception as e:
                        print(f"     Маска {i+1}: ошибка анализа - {e}")
                
                return {"found_elements": [], "masks": []}
                
        except Exception as e:
            import traceback
            print(f"   ❌ КРИТИЧЕСКАЯ ОШИБКА в extract_mask_embeddings: {e}")
            print("   📍 STACK TRACE:")
            traceback.print_exc()
            return {"found_elements": [], "masks": []}
        print(f"   📊 Масок с валидными векторами: {mask_vecs.shape[0]}")
        print("1️⃣3️⃣ Шаг 9: EmbeddingExtractor.build_queries_multiclass() - эмбеддинги примеров по классам")
        class_pos, q_neg = self.embedding_extractor.build_queries_multiclass(pos_by_class, neg_imgs, pos_as_query_masks=False)
        timing_info['embedding_extraction'] = time.time() - t_embeddings

        online_negatives = None
        # Если нет явных негативных примеров, используем онлайн-негативы
        if q_neg is None or q_neg.shape[0] == 0:
            print("   ⚠️ Нет явных негативных примеров, генерируем онлайн-негативы...")
            
            # 1. Собрать все позитивные запросы в один тензор
            pos_queries_tensors = [torch.from_numpy(v) for v in class_pos.values() if v.shape[0] > 0]

            if not pos_queries_tensors:
                print("   ❌ Нет эмбеддингов для positive-классов, невозможно сгенерировать онлайн-негативы.")
            else:
                all_pos_queries = torch.cat(pos_queries_tensors, dim=0)

                if all_pos_queries.shape[0] > 0 and mask_vecs.shape[0] > 0:
                    # 2. Рассчитать косинусное сходство между масками и всеми позитивными запросами
                    mask_vecs_torch = torch.from_numpy(mask_vecs)
                    
                    # Используем torch для расчета косинусной близости
                    sim_matrix = torch.nn.functional.cosine_similarity(mask_vecs_torch.unsqueeze(1), all_pos_queries.unsqueeze(0), dim=2)

                    # 3. Найти лучший позитивный скор для каждой маски
                    best_pos_scores, _ = torch.max(sim_matrix, dim=1)
                    
                    # 4. Определить количество для онлайн-негативов (нижние 40%)
                    num_online_negatives = int(mask_vecs.shape[0] * 0.4)
                    
                    if num_online_negatives > 0:
                        # 5. Найти индексы масок с наименьшими скорами
                        k = min(num_online_negatives, len(best_pos_scores))
                        if k > 0:
                            _, bottom_indices = torch.topk(best_pos_scores, k=k, largest=False)
                            
                            # 6. Собрать эмбеддинги для онлайн-негативов
                            online_negatives = mask_vecs[bottom_indices.numpy()]
                            print(f"   💡 Создано {online_negatives.shape[0]} онлайн-негативов из масок с наихудшими positive-скорами.")

        print("1️⃣4️⃣ Шаг 10: ScoreCalculator.score_multiclass() - скоринг и принятие решений")
        print("🔍 ЭТАП 3: Сопоставление с positive/negative по классам...")
        t_scoring = time.time()
        decisions, _ = self.score_calculator.score_multiclass(
            mask_vecs, 
            class_pos, 
            q_neg,
            online_negatives=online_negatives
        )
        timing_info['scoring_and_decisions'] = time.time() - t_scoring
        t_result = time.time()
        found = []
        result_masks = []
        candidates = []
        H, W = image_np.shape[:2]
        
        # Создаем idx_map для связи индексов решений с оригинальными индексами масок
        idx_map = list(range(len(masks)))
        
        print(f"\n🔍 Processing {len(decisions)} decisions...")
        for i, dec in enumerate(decisions):
            print(f"  - Decision {i}: accepted={dec.get('accepted')}, class='{dec.get('class')}', confidence={dec.get('confidence', 0.0):.3f}")
            if not dec.get('accepted'):
                print(f"    -> SKIPPED (not accepted)")
                continue
            original_idx = idx_map[i]
            print(f"    -> ACCEPTED. Original mask index: {original_idx}")
            mask_dict = masks[original_idx].copy()
            confidence = float(np.clip(dec.get('confidence', 0.0), 0.0, 1.0))
            mask_dict['confidence'] = confidence
            mask_dict['class'] = dec.get('class')
            if 'area' not in mask_dict and 'segmentation' in mask_dict:
                mask_dict['area'] = int(np.sum(mask_dict['segmentation']))
            bx = mask_dict.get('bbox', [0,0,0,0])
            if len(bx) == 4 and (bx[2] <= W and bx[3] <= H):
                x1, y1, w, h = bx
                bbox_xyxy = [int(x1), int(y1), int(x1 + w), int(y1 + h)]
            else:
                bbox_xyxy = [int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])]
            cls_label = dec.get('class')
            try:
                cls_label = str(cls_label) if cls_label is not None else "__unknown__"
            except Exception:
                cls_label = "__unknown__"
            print(f"    -> Appending candidate: class='{cls_label}', confidence={confidence:.3f}")
            candidates.append({
                'mask': mask_dict['segmentation'].astype(bool),
                'bbox_xyxy': bbox_xyxy,
                'confidence': confidence,
                'area': int(mask_dict['area']),
                'class': cls_label,
            })
        from collections import Counter
        print("NMS candidates by class:", Counter([c.get('class') for c in candidates]))
        kept = self._nms(candidates, class_aware=True)
        for e in kept:
            seg = e['mask']
            x1, y1, x2, y2 = e['bbox_xyxy']
            bbox_xywh = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            mask_dict = {
                'segmentation': seg,
                'bbox': bbox_xywh,
                'area': int(seg.sum()),
                'confidence': float(e['confidence']),
                'class': e.get('class')
            }
            found.append({
                'mask': mask_dict,
                'confidence': float(e['confidence']),
                'bbox': bbox_xywh,
                'class': e.get('class')
            })
            result_masks.append(mask_dict)
        timing_info['result_formatting'] = time.time() - t_result
        print("1️⃣5️⃣ Шаг 11: ResultSaver.save_all_results() - сохранение файлов")
        t_saving = time.time()
        image_name = Path(image_path).name
        # NOTE: (@gas) later will be moved from here - store operations will be happening in parallel without blocking inference
        saved_files = self.result_saver.save_all_results(
            image_np, 
            result_masks, 
            output_dir, 
            image_name,
            pipeline_config={"backend": self.mask_backend}
        )
        timing_info['result_saving'] = time.time() - t_saving
        total_time = time.time() - t_total
        timing_info['total_time'] = total_time
        print(f"🎯 Принято масок: {len(found)} (после правил и NMS)")
        print(f"⏱️ Общее время: {total_time:.2f} сек")
        print(f"💾 Результаты сохранены в: {output_dir}")
        print(f"📁 Сохранено файлов: {len(saved_files)}")
        self._print_timing_statistics(timing_info)
        return {
            "found_elements": found, 
            "masks": result_masks,
            "timing_info": timing_info,
            "output_directory": output_dir,
            "saved_files": saved_files
        }

    def _mask_iou(self, mask_a, mask_b):
        """Быстрое вычисление IoU для масок."""
        intersection = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        return intersection / union if union > 0 else 0.0

    def _nms(self, elements, class_aware=True, class_thresholds=None):
        """Эффективная реализация NMS с использованием векторизованных операций."""
        if not elements:
            return []
        
        try:
            import torch
            import torchvision.ops as ops
            use_torch = True
        except ImportError:
            use_torch = False
            
        from collections import defaultdict
        
        def _class_key(v):
            if v is None:
                return "__unknown__"
            try:
                return str(v)
            except Exception:
                return repr(v)
        
        groups = defaultdict(list)
        if class_aware:
            for el in elements:
                groups[_class_key(el.get('class'))].append(el)
        else:
            groups["__all__"] = list(elements)
        
        kept_all = []
        for cls_key, group in groups.items():
            if not group:
                continue
                
            iou_thr = (class_thresholds or {}).get(cls_key, self.nms_iou)
            
            if use_torch and len(group) > 10:  # Используем torch для больших групп
                kept_cls = self._nms_torch(group, iou_thr)
            else:
                kept_cls = self._nms_numpy(group, iou_thr)
                
            kept_all.extend(kept_cls)
        
        return kept_all
    
    def _nms_torch(self, elements, iou_threshold):
        """NMS с использованием torchvision для bbox + маски."""
        import torch
        import torchvision.ops as ops
        
        # Извлекаем bbox и scores
        boxes = []
        scores = []
        for el in elements:
            x1, y1, x2, y2 = el['bbox_xyxy']
            boxes.append([x1, y1, x2, y2])
            scores.append(el.get('confidence', 0.0))
        
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        
        # Применяем bbox NMS
        keep_indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold)
        bbox_kept = [elements[i] for i in keep_indices.tolist()]
        
        # Дополнительная фильтрация по маскам для оставшихся элементов
        if len(bbox_kept) <= 1:
            return bbox_kept
            
        final_kept = []
        for i, current in enumerate(bbox_kept):
            should_keep = True
            current_mask = current['mask']
            
            for j in range(i):
                if j < len(final_kept):
                    other_mask = final_kept[j]['mask']
                    if self._mask_iou(current_mask, other_mask) >= iou_threshold:
                        should_keep = False
                        break
            
            if should_keep:
                final_kept.append(current)
                
        return final_kept
    
    def _nms_numpy(self, elements: List[Dict[str, Any]], iou_threshold: float) -> List[Dict[str, Any]]:
        """Оптимизированная numpy реализация Non-Maximum Suppression.
        
        Args:
            elements: Список элементов с масками и confidence
            iou_threshold: Порог IoU для подавления
            
        Returns:
            Отфильтрованный список элементов
        """
        if len(elements) <= 1:
            return elements
            
        # Сортируем по confidence
        sorted_elements = sorted(elements, key=lambda e: float(e.get('confidence', 0.0)), reverse=True)
        
        kept = []
        masks_kept = []
        
        for current in sorted_elements:
            current_mask = current['mask']
            should_keep = True
            
            # Векторизованная проверка IoU с уже принятыми масками
            for kept_mask in masks_kept:
                if self._mask_iou(current_mask, kept_mask) >= iou_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                kept.append(current)
                masks_kept.append(current_mask)
                
        return kept

    def _get_bbox_from_mask(self, mask: np.ndarray) -> List[int]:
        """Получает bounding box из маски в формате [x, y, width, height]."""
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return [0, 0, 0, 0]
        
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        
        return [x_min, y_min, width, height]

    def _load_example_images(self, dir_path: Optional[Union[str, Path]]) -> List[Image.Image]:
        """Рекурсивно загружает все изображения из директории.
        
        Args:
            dir_path: Путь к директории с изображениями
            
        Returns:
            Список загруженных изображений PIL
            
        Raises:
            ValidationError: При некорректном пути к директории
        """
        from pathlib import Path
        from PIL import Image

        images = []
        if not dir_path:
            return images
            
        try:
            # Валидация директории
            DirectoryValidator.validate_input_directory(str(dir_path))
        except ValidationError as e:
            print(f"   ⚠️ Ошибка валидации директории {dir_path}: {e}")
            return images
            
        p_dir = Path(dir_path)
        if p_dir.name.startswith('.'):
            return images

        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for item in p_dir.iterdir():
            if item.name.startswith('.'):
                continue
            
            if item.is_dir():
                images.extend(self._load_example_images(item))  # Рекурсивный вызов
            elif item.is_file() and item.suffix.lower() in valid_extensions:
                try:
                    # Дополнительная валидация изображения
                    ImageValidator.validate_image_path(str(item))
                    img = Image.open(item).convert('RGB')
                    # Валидация содержимого изображения
                    ImageValidator.validate_image_content(str(item))
                    images.append(img)
                except ValidationError as e:
                    print(f"   ⚠️ Ошибка валидации изображения {item}: {e}")
                except Exception as e:
                    print(f"   ⚠️ Не удалось загрузить {item}: {e}")
        return images

    def _load_positive_by_class(self, dir_path: Optional[Union[str, Path]]) -> Dict[str, List[Image.Image]]:
        """Загружает позитивные примеры, распределяя их по классам.
        
        Args:
            dir_path: Путь к директории с положительными примерами
            
        Returns:
            Словарь, где ключи - названия классов, значения - списки изображений
        """
        from pathlib import Path
        result = {}
        if not dir_path:
            return result
            
        p_dir = Path(dir_path)
        if not p_dir.exists() or not p_dir.is_dir() or p_dir.name.startswith('.'):
            return result

        # Ищем нескрытые поддиректории
        subdirs = [p for p in p_dir.iterdir() if p.is_dir() and not p.name.startswith('.')]
        
        if subdirs:
            # Режим 1: Поддиректории являются классами
            print(f"   📂 Режим мульти-класса: подпапки в '{p_dir.name}' считаются классами.")
            for class_dir in subdirs:
                class_name = class_dir.name
                loaded_images = self._load_example_images(class_dir)
                if loaded_images:
                    result[class_name] = loaded_images
                    print(f"     -> Класс '{class_name}': найдено {len(loaded_images)} изображений.")
                else:
                    print(f"     -> Класс '{class_name}': 0 изображений.")
        else:
            # Режим 2: Нет поддиректорий, вся папка - один класс
            print(f"   📂 Режим одного класса: все изображения в '{p_dir.name}' будут принадлежать классу '{p_dir.name}'.")
            class_name = p_dir.name
            loaded_images = self._load_example_images(p_dir)
            if loaded_images:
                result[class_name] = loaded_images
                print(f"     -> Класс '{class_name}': найдено {len(loaded_images)} изображений.")
            else:
                print(f"     -> Класс '{class_name}': 0 изображений.")

        return result
    
    def switch_segmentation_backend(self, backend_type: str, **kwargs) -> None:
        """Переключает бэкенд сегментации.
        
        Args:
            backend_type: Тип бэкенда ('sam', 'fastsam', 'heatmap')
            **kwargs: Дополнительные параметры для бэкенда
        """
        print(f"🔄 Переключение бэкенда сегментации на: {backend_type}")
        
        if backend_type == 'sam':
            self.sam_predictor.switch_backend(
                backend_type="sam",
                sam_model=self.searchdet_sam
            )
        elif backend_type == 'fastsam':
            self.sam_predictor.switch_backend(
                backend_type="fastsam",
                mask_generator=self.mask_generator
            )
        elif backend_type == 'heatmap':
            threshold = kwargs.get('threshold', 0.5)
            self.sam_predictor.switch_backend(
                backend_type="heatmap",
                threshold=threshold
            )
        else:
            raise ValueError(f"Неподдерживаемый тип бэкенда: {backend_type}")
        
        print(f"✅ Бэкенд сегментации переключен на: {backend_type}")
    
    def get_current_segmentation_backend(self) -> str:
        """Возвращает текущий тип бэкенда сегментации.
        
        Returns:
            Строка с типом текущего бэкенда
        """
        return self.sam_predictor.get_backend_type()
    
    def set_heatmap_for_segmentation(self, heatmap: np.ndarray) -> None:
        """Устанавливает heatmap для сегментации (только для heatmap бэкенда).
        
        Args:
            heatmap: Тепловая карта для генерации масок
        """
        if self.get_current_segmentation_backend() == 'heatmap':
            self.sam_predictor.set_heatmap(heatmap)
        else:
            print(f"⚠️ Предупреждение: heatmap можно устанавливать только для heatmap бэкенда. "
                  f"Текущий бэкенд: {self.get_current_segmentation_backend()}")
    
    def _print_timing_statistics(self, timing_info: Dict[str, float]) -> None:
        """Выводит детальную статистику времени выполнения.
        
        Args:
            timing_info: Словарь с временными метриками
        """
        print("\n" + "="*60)
        print("⏱️ ДЕТАЛЬНАЯ СТАТИСТИКА ВРЕМЕНИ ВЫПОЛНЕНИЯ:")
        print("="*60)
        total_time = timing_info['total_time']
        stages = [
            ('image_loading', '📁 Загрузка изображения'),
            ('examples_loading', '🖼️ Загрузка примеров'),
            ('mask_generation', '🎯 Генерация масок (SAM/FastSAM)'),
            ('mask_filtering', '🔍 Фильтрация масок'),
            ('embedding_extraction', '🧠 Извлечение эмбеддингов'),
            ('scoring_and_decisions', '📊 Скоринг и решения'),
            ('result_formatting', '📋 Формирование результата'),
            ('result_saving', '💾 Сохранение файлов')
        ]
        
        for stage_key, stage_name in stages:
            if stage_key in timing_info:
                stage_time = timing_info[stage_key]
                percentage = (stage_time / total_time * 100) if total_time > 0 else 0
                print(f"{stage_name:<40}: {stage_time:>6.3f}с ({percentage:>5.1f}%)")
        print("-" * 60)
        print(f"{'🚀 ОБЩЕЕ ВРЕМЯ':<40}: {total_time:>6.3f}с (100.0%)")
        print("="*60)
        stage_times = [(name, timing_info.get(key, 0)) for key, name in stages if key in timing_info]
        stage_times.sort(key=lambda x: x[1], reverse=True)
        if len(stage_times) > 1:
            print("\n🐌 САМЫЕ МЕДЛЕННЫЕ ЭТАПЫ:")
            for i, (name, stage_time) in enumerate(stage_times[:3]):
                percentage = (stage_time / total_time * 100) if total_time > 0 else 0
                print(f"   {i+1}. {name}: {stage_time:.3f}с ({percentage:.1f}%)")
        if 'mask_generation' in timing_info and timing_info['mask_generation'] > total_time * 0.5:
            print("\n💡 РЕКОМЕНДАЦИИ ПО ОПТИМИЗАЦИИ:")
            print("   • Генерация масок занимает >50% времени")
            print("   • Попробуйте FastSAM вместо SAM-HQ для ускорения")
            print("   • Или уменьшите параметры SAM (points_per_side, imgsz)")
        if 'embedding_extraction' in timing_info and timing_info['embedding_extraction'] > total_time * 0.3:
            print("\n💡 РЕКОМЕНДАЦИИ ПО ОПТИМИЗАЦИИ:")
            print("   • Извлечение эмбеддингов занимает >30% времени")
            print("   • Проверьте размер feature map (SEARCHDET_FEAT_SHORT_SIDE)")
            print("   • Убедитесь что используется быстрый метод извлечения")
        print()