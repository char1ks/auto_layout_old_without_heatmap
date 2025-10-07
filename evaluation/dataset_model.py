from dataclasses import dataclass, field
from typing import List
import uuid

from evaluation.coco_annotation import CocoAnnotation
from evaluation.dataset_meta import DatasetMeta

#Грубо говоря это модель датасета, которая содержит в себе аннотации к изображениям и простое описание датасета:Имя, дата, источник датасета, ссылки, и тд
@dataclass
class DatasetModel:
    data_points: List[CocoAnnotation]
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    meta: DatasetMeta = field(default_factory=DatasetMeta)