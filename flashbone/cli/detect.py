import argparse
import sys
import json
from pathlib import Path
from typing import Optional, List
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from ..utils.config import Config, DEFAULT_CONFIG
except ImportError:
    try:
        from flashbone.utils.config import Config, DEFAULT_CONFIG
    except ImportError:
        print("⚠️ Модули конфигурации недоступны, используем упрощенный режим")
        Config = None
        DEFAULT_CONFIG = None

# TODO: (@gas)


def _add_detect_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('image', help='Путь к изображению для обработки')
    
    parser.add_argument('--positive', '-p', help='Директория с положительными примерами')
    parser.add_argument('--negative', '-n', help='Директория с отрицательными примерами')
    
    parser.add_argument('--output', '-o', help='Директория для сохранения результатов')
    parser.add_argument('--no-save', action='store_true', help='Не сохранять результаты на диск')
    
    parser.add_argument('--config', '-c', help='Путь к файлу конфигурации JSON')
    parser.add_argument('--backend', choices=['sam-hq', 'sam2', 'fastsam'], 
                       help='Бэкенд для генерации масок')
    parser.add_argument('--confidence', type=float, help='Минимальный порог уверенности (0-1)')
    parser.add_argument('--max-masks', type=int, help='Максимальное количество детекций')
    parser.add_argument('--min-area', type=float, help='Минимальная площадь маски в процентах (0-100)')
    parser.add_argument('--max-area', type=float, help='Максимальная площадь маски в процентах (0-100)')
    parser.add_argument('--nested-iou', type=float, help='IoU порог для фильтра вложенных масок (0-1)')
    
    parser.add_argument('--score-margin', type=float, help='Зазор между positive и negative скором')
    parser.add_argument('--score-ratio', type=float, help='Соотношение positive/negative скора')
    parser.add_argument('--score-confidence', type=float, help='Минимальная уверенность для скоринга')
    parser.add_argument('--min-pos-score', type=float, help='Минимальный positive скор для принятия решения')
    parser.add_argument('--decision-threshold', type=float, help='Порог разности positive-negative для принятия решения')
    parser.add_argument('--adaptive-ratio', type=float, help='Коэффициент для адаптивного порога (0-1)')
    parser.add_argument('--adaptive-diff-floor', type=float, help='Минимальная разность для адаптивного режима')
    parser.add_argument('--topk', type=int, help='Количество топ-K примеров для агрегации')
    parser.add_argument('--pos-agg', choices=['max', 'mean_topk', 'mean'], default='max',
                       help='Режим агрегации positive-скоров: max (по умолчанию), mean_topk, mean')
    parser.add_argument(
        "--backbone",
        choices=[
            "resnet101",
            "dinov2_s","dinov2_b","dinov2_l","dinov2_g",
            "dinov3_vitb16","dinov3_vitl16","dinov3_vith14",
            "dinov3_convnext_tiny","dinov3_convnext_small","dinov3_convnext_base","dinov3_convnext_large",
            "vitb16","vitl16","vith14",
            "convnext_tiny","convnext_small","convnext_base","convnext_large",
        ],
        default="dinov2_b",
    )
    parser.add_argument("--dinov3-backbone", 
                       default=None, 
                       choices=["vitb16", "vitl16", "vith14", "vit7b16",
                                "convnext_tiny", "convnext_small", 
                                "convnext_base", "convnext_large"], 
                       help="Какой бэкенд DINOv3 использовать (ViT или ConvNeXt).")

    parser.add_argument('--layer', help='Слой для извлечения эмбеддингов (например, layer3)', default='layer3')
    parser.add_argument('--feat-short-side', type=int, help='Короткая сторона входа фич (например, 384/512/576)')
    parser.add_argument('--dinov3-ckpt', help='Путь к весам DINOv3 ConvNeXt-B (.pth)')
    parser.add_argument('--pos-as-query-masks', action='store_true', default=True,
                         help='Строить q_pos из масок positive (по умолчанию: True для DINOv3)')
    parser.add_argument('--no-pos-as-query-masks', dest='pos_as_query_masks', action='store_false')
    parser.add_argument('--sam-checkpoint', help='Путь к checkpoint SAM-HQ')
    parser.add_argument('--sam-encoder', choices=['vit_b','vit_l','vit_h'], help='Энкодер SAM-HQ/SAM2 (vit_b/vit_l/vit_h)')
    parser.add_argument('--sam2-checkpoint', help='Путь к checkpoint SAM2')
    parser.add_argument('--sam2-config', help='Путь к конфигурации SAM2')
    parser.add_argument('--fastsam-checkpoint', help='Путь к checkpoint FastSAM')
    parser.add_argument("--vit-pooling", type=str, choices=["cls","mean"], default="cls", help="Pooling for ViT (DINOv3)")
    parser.add_argument("--loader", type=str, default="timm", choices=["timm", "hub"],
                         help="Model loader: timm or torch.hub")
    parser.add_argument("--repo-dir", type=str, default=None,
                         help="Path to local DINOv3 repository for hub loader")

    parser.add_argument('--consensus-k', type=int, help='Минимум positive-попаданий для консенсуса')
    parser.add_argument('--consensus-thr', type=float, help='Порог сходства для консенсуса [0-1]')
    parser.add_argument('--nms-iou', type=float, help='IoU порог для NMS по боксам')

    parser.add_argument('--sam-long-side', type=int, help='Даунскейл длинной стороны перед SAM/FastSAM')
    parser.add_argument('--fastsam-imgsz', type=int, help='Размер входа для FastSAM')
    parser.add_argument('--fastsam-conf', type=float, help='Порог уверенности FastSAM')
    parser.add_argument('--fastsam-iou', type=float, help='Порог IoU FastSAM')
    parser.add_argument('--fastsam-retina', dest='fastsam_retina', action='store_true', help='Включить ретина-маски в FastSAM')
    parser.add_argument('--no-fastsam-retina', dest='fastsam_retina', action='store_false', help='Выключить ретина-маски в FastSAM')
    parser.set_defaults(fastsam_retina=True)
    
    parser.add_argument('--max-embedding-size', type=int, default=1024, 
                       help='Максимальный размер изображения для извлечения эмбеддингов (по умолчанию: 1024)')
    parser.add_argument('--dino-half-precision', action='store_true', 
                       help='Использовать половинную точность (float16) для DINO модели для ускорения')

    parser.add_argument('--ban-border-masks', dest='ban_border_masks', action='store_true', help='Удалять маски, касающиеся рамки')
    parser.add_argument('--no-ban-border-masks', dest='ban_border_masks', action='store_false', help='Разрешить маски, касающиеся рамки')
    parser.set_defaults(ban_border_masks=True)
    parser.add_argument('--border-width', type=int, help='Толщина рамки для фильтра границ (px)')
    
    parser.add_argument('--device', default='cuda', help='Устройство для выполнения (например, cuda или cpu)')
    parser.add_argument('--half', action='store_true', help='Использовать половинную точность (float16)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Подробный вывод')
    parser.add_argument('--quiet', '-q', action='store_true', help='Минимальный вывод')
    parser.add_argument('--defect', action='store_true', help='Включить режим поиска дефектов (beta)')
    parser.add_argument('--use-heatmap', action='store_true', help='Использовать heatmap режим вместо SAM/FastSAM для генерации масок')
    parser.add_argument('--use-heatmap-masks', action='store_true', default=True, help='Генерировать маски из heatmap (по умолчанию: True)')
    parser.add_argument('--no-heatmap-masks', dest='use_heatmap_masks', action='store_false', help='Использовать SAM даже в heatmap режиме')

def execute_detect(args) -> int:
    print("\n" + "="*70)
    print("📋 ПОРЯДОК ВЫПОЛНЕНИЯ МОДУЛЬНОГО ПАЙПЛАЙНА:")
    print("="*70)
    print("1️⃣ main.py → searchdet_pipeline.cli.main.main()")
    print("2️⃣ searchdet_pipeline/cli/main.py → _execute_detect()")
    
    import sys
    import os
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    try:
        print("3️⃣ Импорт: searchdet_pipeline.core.detector → SearchDetDetector (автономная версия)")
        from ..core.detector import SearchDetDetector
        
        print("4️⃣ Инициализация: SearchDetDetector.__init__()")
        print("🔧 Инициализация SearchDet детектора...")
        
        detector_params = {
            'mask_backend': args.backend or 'fastsam',
            'device': args.device,
            'half': args.half,
        }
        
        if hasattr(args, 'sam_checkpoint') and args.sam_checkpoint:
            detector_params['sam_model'] = args.sam_checkpoint
        if hasattr(args, 'sam_encoder') and args.sam_encoder:
            detector_params['sam_encoder'] = args.sam_encoder
        if hasattr(args, 'sam2_checkpoint') and args.sam2_checkpoint:
            detector_params['sam2_weights'] = args.sam2_checkpoint
        if hasattr(args, 'fastsam_checkpoint') and args.fastsam_checkpoint:
            detector_params['fastsam_model'] = args.fastsam_checkpoint
            
        if hasattr(args, 'confidence') and args.confidence is not None:
            detector_params['min_confidence'] = args.confidence
        if hasattr(args, 'max_masks') and args.max_masks is not None:
            detector_params['max_masks'] = args.max_masks
            
        if hasattr(args, 'min_area') and args.min_area is not None:
            detector_params['min_area_frac'] = args.min_area / 100.0
        if hasattr(args, 'max_area') and args.max_area is not None:
            detector_params['max_area_frac'] = args.max_area / 100.0
        if hasattr(args, 'nested_iou') and args.nested_iou is not None:
            detector_params['containment_iou'] = args.nested_iou
        
        if hasattr(args, 'score_margin') and args.score_margin is not None:
            detector_params['score_margin'] = args.score_margin
        if hasattr(args, 'score_ratio') and args.score_ratio is not None:
            detector_params['score_ratio'] = args.score_ratio
        if hasattr(args, 'score_confidence') and args.score_confidence is not None:
            detector_params['score_confidence'] = args.score_confidence
        if hasattr(args, 'min_pos_score') and getattr(args, 'min_pos_score', None) is not None:
            detector_params['min_positive_score'] = args.min_pos_score
        if hasattr(args, 'decision_threshold') and getattr(args, 'decision_threshold', None) is not None:
            detector_params['decision_threshold'] = args.decision_threshold
        if hasattr(args, 'adaptive_ratio') and getattr(args, 'adaptive_ratio', None) is not None:
            detector_params['adaptive_ratio'] = args.adaptive_ratio
        if hasattr(args, 'adaptive_diff_floor') and getattr(args, 'adaptive_diff_floor', None) is not None:
            detector_params['adaptive_diff_floor'] = args.adaptive_diff_floor
        if hasattr(args, 'topk') and getattr(args, 'topk', None) is not None:
            detector_params['topk'] = args.topk
        if hasattr(args, 'pos_agg') and getattr(args, 'pos_agg', None) is not None:
            detector_params['positive_aggregation'] = args.pos_agg
        
        # DINOv3 параметры
        if hasattr(args, 'dinov3_backbone') and args.dinov3_backbone:
            detector_params['dinov3_backbone'] = args.dinov3_backbone
        if hasattr(args, 'dinov3_ckpt') and args.dinov3_ckpt:
            detector_params['dinov3_ckpt'] = args.dinov3_ckpt
        if hasattr(args, 'vit_pooling') and args.vit_pooling:
            detector_params['vit_pooling'] = args.vit_pooling

        if hasattr(args, 'layer') and args.layer:
            detector_params['layer'] = args.layer
        if hasattr(args, 'feat_short_side') and args.feat_short_side is not None:
            detector_params['feat_short_side'] = args.feat_short_side
        if hasattr(args, 'backbone') and args.backbone:
            detector_params['backbone'] = args.backbone
        
        if hasattr(args, 'dinov3_ckpt') and args.dinov3_ckpt:
            detector_params['dinov3_ckpt'] = args.dinov3_ckpt
        
        if hasattr(args, 'loader') and args.loader:
            detector_params['loader'] = args.loader
        
        if hasattr(args, 'repo_dir') and args.repo_dir:
            detector_params['repo_dir'] = args.repo_dir
        
        if hasattr(args, 'layer') and args.layer:
            detector_params['layer'] = args.layer
        
        if hasattr(args, 'defect') and args.defect:
            detector_params['defect_mode'] = True

        if hasattr(args, 'consensus_k') and args.consensus_k is not None:
            detector_params['consensus_k'] = args.consensus_k
        if hasattr(args, 'consensus_thr') and args.consensus_thr is not None:
            detector_params['consensus_thr'] = args.consensus_thr
        if hasattr(args, 'nms_iou') and args.nms_iou is not None:
            detector_params['nms_iou'] = args.nms_iou

        if hasattr(args, 'sam_long_side') and args.sam_long_side is not None:
            detector_params['sam_long_side'] = args.sam_long_side
        if hasattr(args, 'fastsam_imgsz') and args.fastsam_imgsz is not None:
            detector_params['fastsam_imgsz'] = args.fastsam_imgsz
        if hasattr(args, 'fastsam_conf') and args.fastsam_conf is not None:
            detector_params['fastsam_conf'] = args.fastsam_conf
        if hasattr(args, 'fastsam_iou') and args.fastsam_iou is not None:
            detector_params['fastsam_iou'] = args.fastsam_iou
            
        if hasattr(args, 'max_embedding_size') and args.max_embedding_size is not None:
            detector_params['max_embedding_size'] = args.max_embedding_size
        if hasattr(args, 'dino_half_precision') and args.dino_half_precision:
            detector_params['dino_half_precision'] = True
        if hasattr(args, 'fastsam_retina'):
            detector_params['fastsam_retina'] = args.fastsam_retina

        if hasattr(args, 'ban_border_masks'):
            detector_params['border_ban'] = args.ban_border_masks
        if hasattr(args, 'border_width') and args.border_width is not None:
            detector_params['border_width'] = args.border_width

        detector_params.update({
            'smart_rectangle_filter': True,
            'rectangle_bbox_iou_threshold': 0.95, 
            'rectangle_straight_line_ratio': 0.8,  
            'rectangle_area_ratio_threshold': 0.95, 
            'rectangle_angle_tolerance': 10.0, 
            'rectangle_side_ratio_threshold': 0.9, 
            'perfect_rectangle_iou_threshold': 0.99,  
            'rectangle_similarity_iou_threshold': 0.94,
            'square_similarity_iou_threshold': 0.94,   
            'rectangle_use_silhouette': True,          
            'hole_area_ratio_threshold': 0.03          
        })

        import torch
        detector_params.update({
            'dinov3_backbone': getattr(args, 'dinov3_backbone', None),
            'dinov3_ckpt': getattr(args, 'dinov3_ckpt', None),
            'vit_pooling': getattr(args, 'vit_pooling', 'cls'),
            'encoder_device': "cuda" if torch.cuda.is_available() else "cpu",
            'dino_half_precision': getattr(args, 'dino_half_precision', False),
            'backbone': getattr(args, 'backbone', 'dinov2_b'),
            'pos_as_query_masks': getattr(args, 'pos_as_query_masks', True),
            'min_mask_area': 100
        })
        
        print("5️⃣ Создание: detector = SearchDetDetector(**params)")
        # ---
        detector = SearchDetDetector(**detector_params)
        pos_by_class, neg_imgs = detector.read_reference_images(args.positive, args.negative)
        input_img = detector.read_input_img(args.image)
        # ---

        detector.set_references(pos_by_class, neg_imgs)
        
        # # Проверяем режим работы
        # use_heatmap = getattr(args, 'use_heatmap', False)
        # use_heatmap_masks = getattr(args, 'use_heatmap_masks', True)
        
        print("6️⃣ Вызов: detector.find_present_elements() [СТАНДАРТНЫЙ SAM РЕЖИМ]")
        print("   ↳ Это запустит весь пайплайн из hybrid_searchdet_pipeline.py:")
        print("   ↳ _load_example_images() → _generate_sam_masks() → _filter_*() → _extract_mask_embeddings() → _score_masks()")
        # --- CORE METHOD TO CALL
        result = detector.find_present_elements(input_img)
        # ---
        _print_result(result, args)
        detector.save_results(
            input_img,
            result["masks"], 
            args.image,
            args.output if hasattr(args, 'output') and args.output else "output",
        ) 
        if 'success' in result:
            return 0 if result['success'] else 1
        else:
            return 0 if 'found_elements' in result else 1
        
    except ImportError as e:
        print(f"❌ Ошибка импорта модульного пайплайна: {e}")
        print("   Убедитесь что модули searchdet_pipeline/core/ доступны")
        return 1
    except Exception as e:
        print(f"❌ Ошибка выполнения детекции: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def _print_result(result: dict, args):
    """Выводит результат обработки."""
    # Проверяем формат результата (новый или старый)
    if 'success' in result:
        # Новый формат
        if result['success']:
            print(f"\n✅ Детекция завершена успешно!")
            print(f"⏱️ Время обработки: {result['processing_time']:.2f} сек")
            print(f"🔍 Найдено объектов: {len(result['detections'])}")
            
            if result['detections']:
                confidences = [d['confidence'] for d in result['detections']]
                print(f"📊 Средняя уверенность: {sum(confidences)/len(confidences):.3f}")
            
            if result.get('saved_files'):
                print(f"💾 Сохранено файлов: {len(result['saved_files'])}")
                if hasattr(args, 'verbose') and args.verbose:
                    for file_type, path in result['saved_files'].items():
                        print(f"   • {file_type}: {path}")
        else:
            print(f"\n❌ Ошибка детекции: {result.get('error', 'Неизвестная ошибка')}")
    else:
        # Старый формат (из detector.py)
        if 'found_elements' in result:
            found_elements = result['found_elements']
            print(f"\n✅ Детекция завершена успешно!")
            print(f"🔍 Найдено объектов: {len(found_elements)}")
            
            if found_elements:
                confidences = [elem.get('confidence', 0.0) for elem in found_elements]
                if any(c > 0 for c in confidences):
                    print(f"📊 Средняя уверенность: {sum(confidences)/len(confidences):.3f}")
            
            # Выводим информацию о папке сохранения
            output_dir = result.get('output_directory', 'output')
            print(f"💾 Результаты сохранены в: {output_dir}")
            
            # Показываем сохранённые файлы
            if 'saved_files' in result and result['saved_files']:
                saved_files = result['saved_files']
                print(f"📁 Сохранено файлов: {len(saved_files)}")
                if hasattr(args, 'verbose') and args.verbose:
                    print("   📋 Список файлов:")
                    for file_type, file_path in saved_files.items():
                        print(f"     • {file_type}: {Path(file_path).name}")
            
            # Если есть детальная статистика времени, выводим краткую сводку
            if 'timing_info' in result and (hasattr(args, 'verbose') and args.verbose):
                timing_info = result['timing_info']
                print(f"\n⏱️ КРАТКАЯ СТАТИСТИКА ВРЕМЕНИ:")
                if 'mask_generation' in timing_info:
                    print(f"   🎯 Генерация масок: {timing_info['mask_generation']:.3f}с")
                if 'embedding_extraction' in timing_info:
                    print(f"   🧠 Извлечение эмбеддингов: {timing_info['embedding_extraction']:.3f}с")
                if 'scoring_and_decisions' in timing_info:
                    print(f"   📊 Скоринг: {timing_info['scoring_and_decisions']:.3f}с")
                if 'result_saving' in timing_info:
                    print(f"   💾 Сохранение: {timing_info['result_saving']:.3f}с")
        else:
            print(f"\n❌ Неожиданный формат результата: {result}")

