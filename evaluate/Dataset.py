import abc
import json
from pathlib import Path
import sys
from typing import  Any
EVAL_CLASSES_PATH = Path(__file__).parent
sys.path.insert(0, str(EVAL_CLASSES_PATH))

from COCOAnnotations import COCOAnnotation
from DatasetModel import DatasetModel

class Dataset(abc.ABC):
    def __init__(self, dataset: DatasetModel, *args, **kwargs):
        self.name = self.__class__.__name__
        self.data: DatasetModel = dataset
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