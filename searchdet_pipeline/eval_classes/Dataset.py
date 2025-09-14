import abc
import json
from pathlib import Path
from typing import Any
from .COCOAnnotations import COCOAnnotation
from .DatasetModel import DatasetModel

class Dataset(abc.ABC):
    def __init__(self,*args,**kwargs):
        self.name=self.__class__.__name__
        self.data:DatasetModel = data
    
    @classmethod
    @abc.abstractmethod
    def from_json(self,obj:dict[str,Any])->Dataset:
        annotations = []
        for item in obj,get('annotations',[]):
            annotation=COCOAnnotation(
                img=item['img'],
                mask=item['mask'],
                label=item['label'],
                width=item['width'],
                height=item['height'],
                area=item['area'],
                bbox=item['bbox'],
                image_resolution=item['image_resolution']
            )
            annotations.append(annotation)
        data = DatasetModel(
            data_points=annotations,
            meta=obj.get('meta', {})
        )
        return cls(dataset=data)
        
    @classmethod
    @abc.abstractmethod
    def from_path(self,path: Path)->Dataset:
        with open(path,'r') as f:
            obj = json.load(f)
        return self.from_json(obj)