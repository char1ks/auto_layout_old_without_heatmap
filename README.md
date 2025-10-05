!pip install poetry
!git clone https://github.com/facebookresearch/dinov3.git
!poetry install
!cp /content/drive/MyDrive/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth ~/.cache/torch/hub/checkpoints/
!MPLBACKEND=Agg LOG_LEVEL=INFO poetry run python -m cli config.yaml

РЕОМЕНДУЮ:ознакомьтесь с Makefile и запустите команды  make checks(для mypy и ruff),а также не забывайте
менять config.yaml файл в папке evaluate для изменения параметров запуска тестирования


# Методы + классы,которые требуют тестов
 -`coco_annotation.py`:Создание сущностей ,тест валидации полей,тест сериализации `DONE`
 -`DatasetModel.py`:Тест создания сущности,post_init(),работа с meta данными `DONE`
 -`Context.py`:Тест методов finish(),тест расчета dutration,тест работы со spans `DONE`
 -`metrics.py`:Намокать данные ,потестировать расчет метрик для всех 4-ех основных классов,тест edge cases `DONE`
 -`Example_datasets/voc_dataset.py`: Тест _parse_single_voc_xml() и from_path(): корректность bbox (VOC xyxy → xywh), генерации mask из bbox, обработка «swapped» размеров, наполнение meta (counts, categories, paths). `DONE`
 -`Example_datasets/coco_dataset.py`: Тест _build_annotations(), from_json()/from_path(): генерация mask из bbox/segmentation (poly/RLE-заглушка), расчет area, корректность image_size/label и meta. `DONE`