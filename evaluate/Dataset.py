import abc
import json
from pathlib import Path
import sys
from typing import Any
from COCOAnnotations import COCOAnnotation
from DatasetModel import DatasetModel
EVAL_CLASSES_PATH = Path(__file__).parent
sys.path.insert(0, str(EVAL_CLASSES_PATH))


class Dataset(abc.ABC):
    def __init__(self, dataset: DatasetModel, *args, **kwargs):
        self.name = self.__class__.__name__
        self.data: DatasetModel = dataset

    @classmethod
    def from_json(cls: type["Dataset"], obj: dict[str, Any]) -> "Dataset":
        annotations = cls._build_annotations(obj, base_dir=None)
        data = DatasetModel(
            data_points=annotations,
            meta=obj.get('meta', {})
        )
        return cls(dataset=data)

    @classmethod
    def from_path(cls: type["Dataset"], path: Path, **kwargs) -> "Dataset":
        with open(path, 'r') as f:
            obj = json.load(f)
        base_dir = Path(path).parent
        annotations = cls._build_annotations(obj, base_dir=base_dir)
        data = DatasetModel(
            data_points=annotations,
            meta=obj.get('meta', {})
        )
        return cls(dataset=data)

    @classmethod
    @abc.abstractmethod
    def _build_annotations(cls: type["Dataset"], obj: dict[str, Any], base_dir: Path | None) -> list[COCOAnnotation]:
        pass