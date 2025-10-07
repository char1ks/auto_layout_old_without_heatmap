import abc
import json
from pathlib import Path
from typing import Any
from evaluation.coco_annotation import CocoAnnotation
from evaluation.dataset_model import DatasetModel
from evaluation.dataset_meta import DatasetMeta


class Dataset(abc.ABC):
    def __init__(self, dataset: DatasetModel, *args, **kwargs):
        self.name = self.__class__.__name__
        self.data: DatasetModel = dataset

    @classmethod
    def from_json(cls: type["Dataset"], obj: dict[str, Any]) -> "Dataset":
        annotations = cls.build_annotations(obj, base_dir=None)
        existing_meta = obj.get('meta', {})
        meta = DatasetMeta(
            uid=existing_meta.get('uid'),
            name=existing_meta.get('name'),
            categories=obj.get('categories', existing_meta.get('categories', [])),
            total_images=len(obj.get('images', [])) if 'images' in obj else existing_meta.get('total_images'),
            total_annotations=len(obj.get('annotations', [])) if 'annotations' in obj else existing_meta.get('total_annotations'),
            url=existing_meta.get('url'),
            color_channels=existing_meta.get('color_channels', []),
            source_path=existing_meta.get('source_path'),
            base_directory=existing_meta.get('base_directory'),
            dataset_type=existing_meta.get('dataset_type'),
            extra=existing_meta.get('extra', {})
        )
        data = DatasetModel(data_points=annotations, meta=meta)
        return cls(dataset=data)

    @classmethod
    def from_path(cls: type["Dataset"], path: Path, **kwargs) -> "Dataset":
        with open(path, 'r') as f:
            obj = json.load(f)
        base_dir = Path(path).parent
        annotations = cls.build_annotations(obj, base_dir=base_dir)
        existing_meta = obj.get('meta', {})
        meta = DatasetMeta(
            uid=existing_meta.get('uid'),
            name=existing_meta.get('name'),
            categories=obj.get('categories', existing_meta.get('categories', [])),
            total_images=len(obj.get('images', [])) if 'images' in obj else existing_meta.get('total_images'),
            total_annotations=len(obj.get('annotations', [])) if 'annotations' in obj else existing_meta.get('total_annotations'),
            url=existing_meta.get('url'),
            color_channels=existing_meta.get('color_channels', []),
            source_path=str(path),
            base_directory=str(base_dir),
            dataset_type=existing_meta.get('dataset_type'),
            extra=existing_meta.get('extra', {})
        )
        data = DatasetModel(data_points=annotations, meta=meta)
        return cls(dataset=data)

    @classmethod
    @abc.abstractmethod
    def build_annotations(cls, obj: dict[str, Any], base_dir: Path | None) -> list[CocoAnnotation]:
        pass