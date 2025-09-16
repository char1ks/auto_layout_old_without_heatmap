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
from mask_withsearch import initialize_sam as init_searchdet
from .mask_generation import MaskGenerator
from .filtering import MaskFilter  
from .embeddings import EmbeddingExtractor
from .scoring import ScoreCalculator
from .step7_result_saving import ResultSaver
from .utils import get_image_size, get_feature_map_size, upsample_feature_map
from .encoding import DinoV3Encoder
from .heatmap_generator import HeatmapGenerator
from .binning_processor import BinningProcessor
from .enhanced_heatmap_processor import EnhancedHeatmapProcessor
from .segmentation import FastSAMHeatmapProcessor, load_fastsam_model, load_sam_model, load_sam_predictor
from .models import DetectorConfig, ProcessingResult, MaskData, DetectionResult
from ..utils.validation import ImageValidator, DirectoryValidator, ValidationError, validate_processing_pipeline_inputs
import torch
from .models import MaskBackend, BackboneType

from flashbone.detector_base import DetectorBase
from flashbone.core.binning_processor import bin_filter_heatmap


class SearchDetDetector(DetectorBase):
    def __init__(self, config: Optional[DetectorConfig] = None, **kwargs: Any) -> None:
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

        # NOTE: (@gas) general functionality
        self.mask_filter = MaskFilter(self.params)
        self.embedding_extractor = EmbeddingExtractor(
            backbone_name=self.config.dinov3_backbone,
            device=self.device,
            ckpt_path=self.config.dinov3_ckpt
        )
        self.dinov3_encoder = DinoV3Encoder(
            backbone_name=self.config.dinov3_backbone,
            ckpt_path=self.config.dinov3_ckpt,
            device=self.device,
            half_precision=self.config.dino_half_precision,
            vit_pooling=self.config.vit_pooling,
            loader=self.config.loader,
            repo_dir=self.config.repo_dir
        )

        # # NOTE: (@gas) for sam-only approach
        # self.mask_generator = MaskGenerator(
        #     mask_backend=self.mask_backend,
        #     device=self.device,
        #     fastsam_model=self.config.fastsam_model,
        #     fastsam_device=self.config.fastsam_device or self.device, 
        #     sam_generator=None,
        #     **generator_params,
        # )
        # self.score_calculator = ScoreCalculator(self.params) 
        
        # NOTE: (@gas) for heatmaps
        optimal_size = 512 # TODO: (@gas) move to config
        self.heatmap_generator = HeatmapGenerator(self.dinov3_encoder, resize_size=optimal_size, crop_images=False)
        self.binning_processor = BinningProcessor(self.dinov3_encoder, concept_threshold=1)
        self.enhanced_heatmap_processor = EnhancedHeatmapProcessor(self.dinov3_encoder, resize_size=optimal_size)

        # NOTE: (@gas) use sam or fastsam
        sam_model_instance = load_fastsam_model()
        # sam_model_instance = load_sam_model()
        # sam_model_instance = load_sam_predictor()
        self.fastsam_processor = FastSAMHeatmapProcessor(
            fastsam_model=sam_model_instance,
            embedding_extractor=self.embedding_extractor,
            decision_threshold=self.config.decision_threshold,
        )

        self._performance_mode = False
        self._last_processing_time = 0.0
        self._target_processing_time = 0.1

        self.q_pos, self.q_neg = None, None

        # NOTE: (@gas) only for cli usage
        self.result_saver = ResultSaver(self.config.overlay_alpha)

    def read_reference_images(self, positive_dir: Union[str, Path], negative_dir: Optional[Union[str, Path]] = None) -> Tuple[Dict[str, List[Image.Image]], List[Image.Image]]:
        timing_info: Dict[str, float] = {}
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
        return pos_by_class, neg_imgs

    def set_references(self, pos_by_class: Dict[str, List[Image.Image]], neg_imgs: List[Image.Image]) -> None:
        timing_info: Dict[str, float] = {}
        t_embeddings = time.time()
        print("1️⃣3️⃣ Шаг 9: EmbeddingExtractor.build_queries_multiclass() - эмбеддинги примеров по классам")
        # TODO: (@gas) remove from here after figuring out heatmaps integration (images needed by the `heatmap_generator`)
        self.heatmap_generator.init_pooled_features_train(pos_by_class, neg_imgs)
        # 
        self.q_pos, self.q_neg = self.embedding_extractor.build_queries_multiclass(pos_by_class, neg_imgs)
        timing_info['embedding_extraction'] = time.time() - t_embeddings 

    def find_present_elements(self, image_np: np.ndarray) -> Dict[str, Any]:
        if self.config.use_heatmap_sam_hybrid:
            return self._find_present_elements_with_fastsam_integration(image_np)
        return dict()
         
    def _find_present_elements_with_fastsam_integration(self, image_np: np.ndarray)-> Dict[str, Any]:
        timing_info: Dict[str, float] = {}
        t_total = time.time()
        image_pil = Image.fromarray(image_np)

        t_heatmap =time.time()

        heatmap = self.heatmap_generator.generate_heatmap(input_image=image_pil)
        heatmap = heatmap.detach().cpu().numpy()
        timing_info['heatmap_generation'] = time.time()-t_heatmap
        print (f"   📊 Heatmap сгенерирована: {heatmap.shape}")

        # TODO: (@gas) add heatmap binning here
        # heatmap = bin_filter_heatmap(
        #     heatmap, max_bins=5, first_k_bins=2, fill_value=0, ascending=False)

        print ("3️⃣ Шаг 3: FastSAM интеграция с heatmap")
        t_fastsam =time.time()

        fastsam_masks = self.fastsam_processor.process_image(
            image=image_pil, 
            heatmap=heatmap, 
            min_overlap_ratio=0.5,
        )
        timing_info['fastsam_integration'] = time.time()-t_fastsam

        if fastsam_masks is None:
            fastsam_masks = []
            print ("   ⚠️ FastSAM вернул None, используем пустой список")

        print (f"   🎯 FastSAM сгенерировал {len(fastsam_masks)} финальных масок")

        if not fastsam_masks :
            print ("   ❌ Нет FastSAM масок для обработки.")
            return {"masks": [], "timing_info": timing_info }

        print ("4️⃣ Шаг 4: Формирование результатов")
        t_result =time.time()

        result_masks = []

        for mask_item in fastsam_masks:
            if isinstance(mask_item, np.ndarray):
                seg = (mask_item > 0.5).astype(bool)
                ys, xs = np.where(seg)
                if xs.size and ys.size :
                    x_min , x_max = int(xs.min()), int(xs.max())
                    y_min , y_max = int(ys.min()), int(ys.max())
                    bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
                else :
                    bbox = [0, 0, 0, 0]
                # TODO: (@gas) should be solved by the multiclass mask handler earlier
                class_name = list(self.q_pos.keys())[0] if self.q_pos else "detected" 
                # TODO: (@gas) compute confidence from mask values (like a mean or smth OR take from multiclass classifier distance)
                md = {'segmentation': seg, 'bbox': bbox, 'area': int(seg.sum()), 'confidence': 0.9, 'class': class_name} 
                result_masks.append(md)
            if isinstance(mask_item, torch.Tensor):
                seg = (mask_item.detach().cpu().numpy()>0.5).astype(bool)
                ys , xs = np.where(seg)
                if xs.size and ys.size :
                    x_min , x_max = int(xs.min()), int(xs.max())
                    y_min , y_max = int(ys.min()), int(ys.max())
                    bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
                else :
                    bbox = [0, 0, 0, 0]
                class_name = list(self.q_pos.keys())[0] if self.q_pos else "detected"
                md = {'segmentation': seg, 'bbox': bbox, 'area': int(seg.sum()), 'confidence': 0.9, 'class': class_name}
                result_masks.append(md)
            elif isinstance(mask_item, dict) and 'segmentation' in mask_item:
                class_name = list(self.q_pos.keys())[0] if self.q_pos else "detected"
                md = mask_item.copy()
                md['confidence'] =float(md.get('confidence', 0.9))
                md['class'] = md.get('class', class_name)
                if 'area' not in md:
                    md['area'] = int(np.sum(md['segmentation']))
                if 'bbox' not in md:
                    seg = md['segmentation']
                    ys , xs = np.where(seg)
                    if xs.size and ys.size:
                        x_min, x_max = int(xs.min()), int(xs.max())
                        y_min, y_max = int(ys.min()), int(ys.max())
                        md['bbox'] = [x_min, y_min, x_max-x_min+1, y_max-y_min+1]
                    else:
                        md['bbox'] = [0, 0, 0, 0]
                result_masks.append(md)
            else:
                continue

        timing_info['result_formatting'] = time.time()-t_result

        total_time = time.time()-t_total
        timing_info['total_time'] = total_time

        # print (f"🎯 Найдено FastSAM элементов: {len(md)}")
        print (f"⏱️ Общее время: {total_time:.2f} сек")
        self._print_timing_statistics(timing_info)

        return {
            "masks": result_masks,
            "timing_info": timing_info, 
            "heatmap": heatmap,
        }

    def save_results(self, image_np: np.ndarray, result_masks: List, image_path: str | Path, output_dir: str = "output") -> Dict[str, str]:
        print("1️⃣5️⃣ Шаг 11: ResultSaver.save_all_results() - сохранение файлов")
        timing_info: Dict[str, float] = {}
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
        print(f"💾 Результаты сохранены в: {output_dir}")
        print(f"📁 Сохранено файлов: {len(saved_files)}")

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