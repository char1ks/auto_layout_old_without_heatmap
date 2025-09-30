from dataclasses import dataclass, field
from typing import List
import uuid
import sys
from pathlib import Path

EVAL_CLASSES_PATH = Path(__file__).parent
sys.path.insert(0, str(EVAL_CLASSES_PATH))

from COCOAnnotations import COCOAnnotation
from DatasetMeta import DatasetMeta

#Грубо говоря это модель датасета, которая содержит в себе аннотации к изображениям и простое описание датасета:Имя, дата, источник датасета, ссылки, и тд
@dataclass
class DatasetModel:
    data_points: List[COCOAnnotation]
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    meta: DatasetMeta = field(default_factory=DatasetMeta) 

    def __post_init__(self) -> None:
        if isinstance(self.meta, dict):
            self.meta = DatasetMeta.from_dict(self.meta)