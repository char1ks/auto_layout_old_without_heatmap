#!/usr/bin/env python3
"""
Тестовый скрипт для отладки проблемы mAP = 0.0
Использует новые отладочные методы ReportGenerator
"""

import sys
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent))

from searchdet_pipeline.eval_classes.ReportGenerator import ReportGenerator
from searchdet_pipeline.eval_classes.COCOAnnotations import COCOAnnotation
from searchdet_pipeline.eval_classes.DatasetModel import DatasetModel
from searchdet_pipeline.eval_classes.DatasetMeta import DatasetMeta
from searchdet_pipeline.eval_classes.metrics import MeanAveragePrecision


def parse_xml_annotation(xml_path):
    """Парсим XML аннотацию и возвращаем список объектов"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Получаем информацию об изображении
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        # Конвертируем в формат [x, y, width, height]
        bbox_coco = [xmin, ymin, xmax - xmin, ymax - ymin]
        area = (xmax - xmin) * (ymax - ymin)
        
        objects.append({
            'name': name,
            'bbox': bbox_coco,
            'area': area,
            'filename': filename,
            'image_size': (width, height),
            'width': width,
            'height': height
        })
    
    return objects

def create_test_gt_data():
    """Создаем ground truth данные из реального XML файла"""
    # Создаем фиктивные изображения и маски для совместимости
    dummy_img = np.zeros((400, 500, 3), dtype=np.uint8)  # Размер из XML
    dummy_mask = np.zeros((400, 500), dtype=np.uint8)
    
    # Реальные данные из XML
    xml_objects = [
        {'name': 'tomato', 'bbox': [189, 356, 62, 44], 'area': 2728, 'filename': 'tomato3.png', 'image_size': (500, 400), 'width': 500, 'height': 400},
        {'name': 'tomato', 'bbox': [372, 55, 31, 30], 'area': 930, 'filename': 'tomato3.png', 'image_size': (500, 400), 'width': 500, 'height': 400},
        {'name': 'tomato', 'bbox': [353, 63, 34, 35], 'area': 1190, 'filename': 'tomato3.png', 'image_size': (500, 400), 'width': 500, 'height': 400},
        {'name': 'tomato', 'bbox': [351, 26, 40, 36], 'area': 1440, 'filename': 'tomato3.png', 'image_size': (500, 400), 'width': 500, 'height': 400},
        {'name': 'tomato', 'bbox': [350, 1, 52, 19], 'area': 988, 'filename': 'tomato3.png', 'image_size': (500, 400), 'width': 500, 'height': 400},
        {'name': 'tomato', 'bbox': [1, 185, 15, 34], 'area': 510, 'filename': 'tomato3.png', 'image_size': (500, 400), 'width': 500, 'height': 400}
    ]
    
    annotations = []
    for i, obj in enumerate(xml_objects):
        ann = COCOAnnotation(
            img=dummy_img,
            mask=dummy_mask,
            label=obj['name'],
            image_size=obj['image_size'],
            width=obj['width'],
            height=obj['height'],
            area=obj['area'],
            file_name=obj['filename'],
            bbox=obj['bbox'],
            score=1.0,
            confidence=1.0,
            uid=f"gt_{i+1}"
        )
        annotations.append(ann)
    
    meta = DatasetMeta(name="tomato_dataset")
    gt_dataset = DatasetModel(data_points=annotations, uid="tomato_gt", meta=meta)
    
    return gt_dataset, annotations


def create_test_predictions():
    """Создаем тестовые предсказания, похожие на ground truth, но с небольшими отклонениями"""
    dummy_img = np.zeros((400, 500, 3), dtype=np.uint8)
    dummy_mask = np.zeros((400, 500), dtype=np.uint8)
    
    # Предсказания с небольшими отклонениями от GT
    predictions_data = [
        {'name': 'tomato', 'bbox': [190, 358, 60, 42], 'area': 2520, 'score': 0.95},  # Близко к первому GT
        {'name': 'tomato', 'bbox': [374, 57, 29, 28], 'area': 812, 'score': 0.88},   # Близко ко второму GT
        {'name': 'tomato', 'bbox': [355, 65, 32, 33], 'area': 1056, 'score': 0.92}, # Близко к третьему GT
        {'name': 'tomato', 'bbox': [353, 28, 38, 34], 'area': 1292, 'score': 0.87}, # Близко к четвертому GT
        {'name': 'tomato', 'bbox': [352, 3, 50, 17], 'area': 850, 'score': 0.83},   # Близко к пятому GT
        # Пропускаем шестой объект для тестирования recall
    ]
    
    predictions = []
    for i, pred in enumerate(predictions_data):
        ann = COCOAnnotation(
            img=dummy_img,
            mask=dummy_mask,
            label=pred['name'],
            image_size=(500, 400),
            width=500,
            height=400,
            area=pred['area'],
            file_name="tomato3.png",
            bbox=pred['bbox'],
            score=pred['score'],
            confidence=pred['score'],
            uid=f"pred_{i+1}"
        )
        predictions.append(ann)
    
    return predictions


def calculate_iou(box1, box2):
    """Вычисляем IoU между двумя bbox в формате [x, y, width, height]"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Координаты углов
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2
    
    # Пересечение
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def analyze_data_manually():
    """Анализируем данные вручную для диагностики проблем с mAP"""
    print("=== Анализ реальных данных из XML для диагностики mAP = 0.0 ===\n")
    
    # Создаем тестовые данные на основе реального XML
    gt_dataset, gt_annotations = create_test_gt_data()
    predictions = create_test_predictions()
    
    print(f"Ground Truth аннотаций: {len(gt_annotations)}")
    print(f"Предсказаний: {len(predictions)}")
    
    # Анализируем файлы
    gt_files = set(ann.file_name for ann in gt_annotations)
    pred_files = set(pred.file_name for pred in predictions)
    
    print(f"\nФайлы в GT: {gt_files}")
    print(f"Файлы в предсказаниях: {pred_files}")
    print(f"Пересечение файлов: {gt_files.intersection(pred_files)}")
    
    # Анализируем классы
    gt_classes = set(ann.label for ann in gt_annotations)
    pred_classes = set(pred.label for pred in predictions)
    
    print(f"\nКлассы в GT: {gt_classes}")
    print(f"Классы в предсказаниях: {pred_classes}")
    print(f"Пересечение классов: {gt_classes.intersection(pred_classes)}")
    
    # Детальный анализ bbox и IoU
    for file_name in gt_files.intersection(pred_files):
        print(f"\n--- Детальный анализ файла: {file_name} ---")
        
        file_gt = [ann for ann in gt_annotations if ann.file_name == file_name]
        file_pred = [pred for pred in predictions if pred.file_name == file_name]
        
        print(f"GT объектов в файле: {len(file_gt)}")
        print(f"Предсказаний в файле: {len(file_pred)}")
        
        print("\nGround Truth объекты:")
        for i, gt_ann in enumerate(file_gt):
            print(f"  GT {i+1}: bbox={gt_ann.bbox}, label={gt_ann.label}, area={gt_ann.area}")
            
        print("\nПредсказания:")
        for i, pred_ann in enumerate(file_pred):
            print(f"  Pred {i+1}: bbox={pred_ann.bbox}, label={pred_ann.label}, score={pred_ann.confidence:.2f}, area={pred_ann.area}")
        
        # Вычисляем IoU между всеми парами GT-Prediction
        print("\nМатрица IoU (GT vs Predictions):")
        print("     ", end="")
        for j in range(len(file_pred)):
            print(f"Pred{j+1:2d}", end="  ")
        print()
        
        for i, gt_ann in enumerate(file_gt):
            print(f"GT{i+1:2d}: ", end="")
            for j, pred_ann in enumerate(file_pred):
                if gt_ann.label == pred_ann.label:  # Только для одинаковых классов
                    iou = calculate_iou(gt_ann.bbox, pred_ann.bbox)
                    print(f"{iou:5.2f}", end="  ")
                else:
                    print("  ---", end="  ")
            print()
    
    # Проверяем, есть ли хорошие совпадения
    good_matches = 0
    total_gt = len(gt_annotations)
    
    for gt_ann in gt_annotations:
        best_iou = 0.0
        for pred_ann in predictions:
            if (gt_ann.file_name == pred_ann.file_name and 
                gt_ann.label == pred_ann.label):
                iou = calculate_iou(gt_ann.bbox, pred_ann.bbox)
                best_iou = max(best_iou, iou)
        
        if best_iou > 0.5:  # Стандартный порог IoU
            good_matches += 1
    
    print(f"\n=== СТАТИСТИКА СОВПАДЕНИЙ ===")
    print(f"Хороших совпадений (IoU > 0.5): {good_matches}/{total_gt}")
    print(f"Recall потенциальный: {good_matches/total_gt:.2f}")
    
    # Диагноз
    print("\n=== ДИАГНОЗ ===")
    if not gt_files.intersection(pred_files):
        print("❌ ПРОБЛЕМА: Нет пересечения файлов между GT и предсказаниями")
    elif not gt_classes.intersection(pred_classes):
        print("❌ ПРОБЛЕМА: Нет пересечения классов между GT и предсказаниями")
    elif good_matches == 0:
        print("❌ ПРОБЛЕМА: Нет хороших совпадений (IoU > 0.5)")
        print("   Возможные причины:")
        print("   - Координаты bbox сильно отличаются")
        print("   - Неправильный формат bbox (возможно, нужен другой формат)")
    else:
        print("✅ Есть хорошие совпадения, но mAP = 0.0")
        print("🔍 Возможные причины:")
        print("   - Ошибки в алгоритме расчета mAP")
        print("   - Неправильная обработка confidence scores")
        print("   - Проблемы с сортировкой предсказаний")


def debug_map_data_processing():
    """Отладочная функция для анализа обработки данных в MeanAveragePrecision"""
    print("\n" + "="*60)
    print("ОТЛАДКА ОБРАБОТКИ ДАННЫХ В MeanAveragePrecision")
    print("="*60)
    
    # Создаем тестовые данные
    gt_dataset, gt_annotations = create_test_gt_data()
    predictions = create_test_predictions()
    
    print(f"Создано GT аннотаций: {len(gt_annotations)}")
    print(f"Создано предсказаний: {len(predictions)}")
    
    # Инициализируем метрику mAP
    map_metric = MeanAveragePrecision()
    
    print("\n--- Анализ GT dataset ---")
    print(f"GT dataset type: {type(gt_dataset)}")
    print(f"GT dataset.data_points length: {len(gt_dataset.data_points)}")
    
    # Анализируем первый data_point
    if gt_dataset.data_points:
        first_dp = gt_dataset.data_points[0]
        print(f"\nПервый data_point:")
        print(f"  Type: {type(first_dp)}")
        print(f"  file_name: {getattr(first_dp, 'file_name', 'НЕТ')}")
        print(f"  label: {getattr(first_dp, 'label', 'НЕТ')}")
        print(f"  bbox: {getattr(first_dp, 'bbox', 'НЕТ')}")
        print(f"  annotations attr: {hasattr(first_dp, 'annotations')}")
        if hasattr(first_dp, 'annotations'):
            print(f"  annotations: {getattr(first_dp, 'annotations', [])}")
    
    # Тестируем _prepare_gt_data
    print("\n--- Тестирование _prepare_gt_data ---")
    try:
        gt_data = map_metric._prepare_gt_data(gt_dataset)
        print(f"GT data подготовлены:")
        print(f"  images: {len(gt_data['images'])}")
        print(f"  annotations: {len(gt_data['annotations'])}")
        print(f"  categories: {gt_data['categories']}")
        
        if gt_data['annotations']:
            print(f"\nПервая аннотация:")
            print(f"  {gt_data['annotations'][0]}")
        else:
            print("\n❌ ПРОБЛЕМА: Нет аннотаций в gt_data!")
            
    except Exception as e:
        print(f"❌ Ошибка в _prepare_gt_data: {e}")
        import traceback
        traceback.print_exc()
    
    # Тестируем _prepare_prediction_data
    print("\n--- Тестирование _prepare_prediction_data ---")
    try:
        if 'gt_data' in locals():
            pred_data = map_metric._prepare_prediction_data(predictions, gt_data)
            print(f"Prediction data подготовлены:")
            print(f"  Количество предсказаний: {len(pred_data)}")
            
            if pred_data:
                print(f"\nПервое предсказание:")
                print(f"  {pred_data[0]}")
            else:
                print("\n❌ ПРОБЛЕМА: Нет предсказаний в pred_data!")
        else:
            print("Пропускаем из-за ошибки в gt_data")
            
    except Exception as e:
        print(f"❌ Ошибка в _prepare_prediction_data: {e}")
        import traceback
        traceback.print_exc()


def test_map_calculation():
    """Тестируем реальный расчет mAP с использованием MeanAveragePrecision"""
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ РАСЧЕТА mAP")
    print("="*60)
    
    # Создаем тестовые данные
    gt_dataset, gt_annotations = create_test_gt_data()
    predictions = create_test_predictions()
    
    print(f"Создано GT аннотаций: {len(gt_annotations)}")
    print(f"Создано предсказаний: {len(predictions)}")
    
    # Инициализируем метрику mAP
    map_metric = MeanAveragePrecision()
    
    try:
        # Вычисляем mAP
        print("\nВычисляем mAP...")
        result = map_metric.compute(gt_dataset, predictions)
        
        print(f"\n✅ mAP успешно вычислен!")
        print(f"Метрика: {result.metric_name}")
        print(f"Значение: {result.score:.4f}")
        print(f"Статистика: {result.stats}")
        
        if result.score == 0.0:
            print("\n❌ mAP = 0.0 - проблема подтверждена!")
            print("Возможные причины:")
            print("- Низкие значения IoU между GT и предсказаниями")
            print("- Неправильный формат данных")
            print("- Проблемы в алгоритме расчета")
        else:
            print(f"\n✅ mAP = {result.score:.4f} - система работает корректно!")
            
    except Exception as e:
        print(f"\n❌ Ошибка при вычислении mAP: {e}")
        print(f"Тип ошибки: {type(e).__name__}")
        import traceback
        traceback.print_exc()

def main():
    analyze_data_manually()
    debug_map_data_processing()
    test_map_calculation()


if __name__ == "__main__":
    main()