import abc
import json
import os
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
from searchdet_pipeline.eval_classes.COCOAnnotations import COCOAnnotation
from searchdet_pipeline.eval_classes.DatasetModel import DatasetModel
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw
import numpy as np

class Dataset(abc.ABC):
    def __init__(self, dataset: DatasetModel, *args, **kwargs):
        self.name = self.__class__.__name__
        self.data: DatasetModel = dataset
    
    def __getattr__(self, name):
        if hasattr(self.data, name):
            return getattr(self.data, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    @abc.abstractmethod
    def from_json(cls, obj: dict[str, Any]) -> "Dataset":
        annotations = cls._build_annotations(obj, base_dir=None)
        data = DatasetModel(
            data_points=annotations,
            meta=obj.get('meta', {})
        )
        return cls(dataset=data)
    @abc.abstractmethod
    def from_path(cls, path: Path) -> "Dataset":
        with open(path, 'r') as f:
            obj = json.load(f)
        base_dir = Path(path).parent
        annotations = cls._build_annotations(obj, base_dir=base_dir)
        data = DatasetModel(
            data_points=annotations,
            meta=obj.get('meta', {})
        )
        return cls(dataset=data)
    @abc.abstractmethod
    def _build_annotations(cls, obj: dict[str, Any], base_dir: Path | None) -> list[COCOAnnotation]:
       pass