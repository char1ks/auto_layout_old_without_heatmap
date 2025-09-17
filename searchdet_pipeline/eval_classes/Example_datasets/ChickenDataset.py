from pathlib import Path
from typing import Any
from searchdet_pipeline.eval_classes.Dataset import Dataset
from searchdet_pipeline.eval_classes.DatasetModel import DatasetModel

class ChickenDataset(Dataset):
    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "ChickenDataset":
        annotations = cls._build_annotations(obj, base_dir=None)
        meta = obj.get('meta', {})
        meta.update({
            'dataset_type': 'chicken_detection',
            'total_images': len(obj.get('images', [])),
            'total_annotations': len(obj.get('annotations', [])),
            'categories': obj.get('categories', [])
        })
        
        data = DatasetModel(
            data_points=annotations,
            meta=meta
        )
        return cls(dataset=data)
    
    @classmethod
    def from_path(cls, path: Path) -> "ChickenDataset":
        import json
        
        with open(path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        
        base_dir = Path(path).parent
        annotations = cls._build_annotations(obj, base_dir=base_dir)
        meta = obj.get('meta', {})
        meta.update({
            'dataset_type': 'chicken_detection',
            'source_path': str(path),
            'base_directory': str(base_dir),
            'total_images': len(obj.get('images', [])),
            'total_annotations': len(obj.get('annotations', [])),
            'categories': obj.get('categories', [])
        })
        
        data = DatasetModel(
            data_points=annotations,
            meta=meta
        )
        return cls(dataset=data)