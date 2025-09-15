import abc
import json
from pathlib import Path
from typing import Any
from PIL import Image
import numpy as np
from .COCOAnnotations import COCOAnnotation
from .DatasetModel import DatasetModel

class Dataset(abc.ABC):
    def __init__(self, dataset: DatasetModel, *args, **kwargs):
        self.name = self.__class__.__name__
        self.data: DatasetModel = dataset
    
    @classmethod
    @abc.abstractmethod
    def from_json(cls, obj: dict[str, Any]) -> "Dataset":
        annotations = cls._build_annotations(obj, base_dir=None)
        data = DatasetModel(
            data_points=annotations,
            meta=obj.get('meta', {})
        )
        return cls(dataset=data)

    @classmethod
    def _build_annotations(cls, obj: dict[str, Any], base_dir: Path | None) -> list[COCOAnnotation]:
        annotations: list[COCOAnnotation] = []

        images = {img.get('id'): img for img in obj.get('images', [])}
        categories = {cat.get('id'): cat.get('name', cat.get('supercategory', '')) for cat in obj.get('categories', [])}
        
        for ann in obj.get('annotations', []):
            img_info = images.get(ann.get('image_id'), {})
            width = img_info.get('width', 0)
            height = img_info.get('height', 0)
            bbox = ann.get('bbox', [])
            area = ann.get('area', 0.0)#Также тут можно высчитывать area путем перемножения высоты (bbox[2] на высоту bbox[3])
            label = categories.get(ann.get('category_id'), ann.get('category_id'))
            file_name = img_info.get('file_name', '')
            img_path = (Path(base_dir) / file_name) if (base_dir and file_name) else (Path(file_name) if file_name else None)
            img_array = None
            if img_path is not None and img_path.exists():
                try:
                    with Image.open(img_path) as im:
                        im = im.convert('RGB')
                        img_array = np.array(im)
                        if not width or not height:
                            height, width = img_array.shape[:2]
                except Exception:
                    pass
            if img_array is None:
                if width and height:
                    img_array = np.zeros((height, width, 3), dtype=np.uint8)
                else:
                    img_array = np.zeros((0, 0, 3), dtype=np.uint8)

            annotation = COCOAnnotation(
                img=img_array,
                mask=ann.get('segmentation'),
                label=label,
                width=width,
                height=height,
                area=area,
                bbox=bbox,
                image_resolution=(width, height),
                file_name=file_name,
            )
            annotations.append(annotation)
        return annotations
        
    @classmethod
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