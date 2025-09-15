from dataclasses import dataclass, field
from typing import List, Any
import uuid
from .COCOAnnotations import COCOAnnotation

#Грубо говоря это модель датасета, которая содержит в себе аннотации к изображениям и простое описание датасета:Имя, дата, источник датасета, ссылки, и тд
@dataclass
class DatasetModel:
    data_points: List[COCOAnnotation]
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    meta: dict[str, Any] = field(default_factory=dict)

    #Метаданные датасета:
    #name-имя датасета
    #categories-массив категорий
    #total_images-количество фото в датасете
    #total_annotations-количество аннотаций в датасете
    #url-ссылка до источника датасета 
    #color_channels-список доступных каналов

    #Сколько аннотаций всего у нас
    def annotations_len(self) -> int:
        return len(self.data_points)

    #Какие категории есть в датасете(аннотациях)
    def get_all_category_names(self) -> List[str]:
        return {ann.label for ann in self.data_points if isinstance(ann.label, str)}

    #Получаем все аннотации по категории
    def filter_by_category(self,category_name:str) -> List[COCOAnnotation]:
        return [ann for ann in self.data_points if ann.label == category_name]