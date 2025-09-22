#!/usr/bin/env python3
"""
Тестовый файл для вывода информации о SheepDataset
"""

from pathlib import Path
from searchdet_pipeline.eval_classes.Example_datasets.SheepDataset import SheepDataset

def test_sheep_dataset_info():
    """Выводит информацию о SheepDataset"""
    print("🐑 Информация о SheepDataset")
    print("=" * 50)
    
    dataset_path = Path("/Users/aodnoral/Desktop/sheep-detection")
    
    try:
        print(f"📂 Путь к датасету: {dataset_path}")
        print(f"📁 Существует ли путь: {dataset_path.exists()}")
        
        if not dataset_path.exists():
            print(f"❌ Путь {dataset_path} не существует!")
            return
        
        # Создаем экземпляр SheepDataset
        print(f"\n🔄 Создаем экземпляр SheepDataset...")
        dataset = SheepDataset.from_path(dataset_path)
        
        print(f"✅ SheepDataset успешно создан!")
        
        # Выводим информацию о датасете
        print(f"\n📊 Информация о датасете:")
        print(f"   Тип объекта: {type(dataset)}")
        
        # Проверяем атрибуты датасета
        if hasattr(dataset, 'images'):
            print(f"   Количество изображений: {len(dataset.images)}")
        
        if hasattr(dataset, 'annotations'):
            print(f"   Количество аннотаций: {len(dataset.annotations)}")
        
        if hasattr(dataset, 'categories'):
            print(f"   Количество категорий: {len(dataset.categories)}")
            
        # Выводим доступные методы
        print(f"\n🔧 Доступные методы:")
        methods = [method for method in dir(dataset) if not method.startswith('_')]
        for method in methods:
            print(f"   - {method}")
        
        # Если есть изображения, выводим первые несколько
        if hasattr(dataset, 'images') and dataset.images:
            print(f"\n🖼️  Первые изображения:")
            for i, image in enumerate(dataset.images[:5]):  # Показываем первые 5
                print(f"   {i+1}. ID: {image.id}, Файл: {image.file_name}")
        
        # Если есть категории, выводим их
        if hasattr(dataset, 'categories') and dataset.categories:
            print(f"\n🏷️  Категории:")
            for category in dataset.categories:
                print(f"   ID: {category.id}, Название: {category.name}")
        
        print(f"\n✅ Тестирование завершено!")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sheep_dataset_info()