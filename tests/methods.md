# Методы + классы,которые требуют тестов

 -`COCOAnnotations.py`:Создание сущностей ,тест валидации полей,тест сериализации `DONE`
 -`DatasetModel.py`:Тест создания сущности,post_init(),работа с meta данными `DONE`
 -`Context.py`:Тест методов finish(),тест расчета dutration,тест работы со spans `DONE`
 -`metrics.py`:Намокать данные ,потестировать расчет метрик для всех 4-ех основных классов,тест edge cases `NOT DONE`
 -`DatasetPoint.py`:Тест методов detect_all(ключевая точка входа для старта evaluate),evaluate(высчитывание метрик),set_references(установка референсных изображений) `NOT DONE`
 -`DetectorBase.py`: Тест методов detect() и read_input_img() — корректность создания Context, обработка file_name, успешное/ошибочное завершение; через наследника-мока проверить find_present_elements() и _convert_to_annotations() на формирование COCOAnnotation. `NOT DONE`
 -`Dataset.py`: Контракт абстрактных методов: smoke-тесты для from_json()/from_path() на примерах, проверка вызова _build_annotations() и корректной сборки DatasetModel. `NOT DONE`
 -`DatasetMeta.py`: Тест from_dict() — нормализация типов (uid, total_images/annotations → int), приведение categories/color_channels к строкам, сбор extra неизвестных ключей. `NOT DONE`
 -`Example_datasets/ArchiveVOCDataset.py`: Тест _parse_single_voc_xml() и from_path(): корректность bbox (VOC xyxy → xywh), генерации mask из bbox, обработка «swapped» размеров, наполнение meta (counts, categories, paths). `NOT DONE`
 -`Example_datasets/COCODataset.py`: Тест _build_annotations(), from_json()/from_path(): генерация mask из bbox/segmentation (poly/RLE-заглушка), расчет area, корректность image_size/label и meta. `NOT DONE`