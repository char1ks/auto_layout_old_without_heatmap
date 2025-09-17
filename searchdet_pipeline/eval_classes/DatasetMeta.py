from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class DatasetMeta:
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    total_images: Optional[int] = None
    total_annotations: Optional[int] = None
    url: Optional[str] = None
    color_channels: List[str] = field(default_factory=list)
    source_path: Optional[str] = None
    base_directory: Optional[str] = None
    dataset_type: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            value = getattr(self, key)
            return default if value is None else value
        return self.extra.get(key, default)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            val = getattr(self, key)
            if val is None:
                raise KeyError(key)
            return val
        if key in self.extra:
            return self.extra[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.extra[key] = value

    def to_dict(self) -> Dict[str, Any]:
        base = {
            'uid': self.uid,
            'name': self.name,
            'categories': self.categories,
            'total_images': self.total_images,
            'total_annotations': self.total_annotations,
            'url': self.url,
            'color_channels': self.color_channels,
            'source_path': self.source_path,
            'base_directory': self.base_directory,
            'dataset_type': self.dataset_type,
        }
        base = {k: v for k, v in base.items() if v is not None}
        for k, v in self.extra.items():
            if k not in base:
                base[k] = v
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetMeta':
        data = dict(data or {})
        known_keys = {
            'uid', 'name', 'categories', 'total_images', 'total_annotations',
            'url', 'color_channels', 'source_path', 'base_directory', 'dataset_type'
        }
        known = {k: data.pop(k) for k in list(data.keys()) if k in known_keys}
        if 'categories' in known and isinstance(known['categories'], list):
            known['categories'] = [str(x) for x in known['categories']]
        if 'color_channels' in known and isinstance(known['color_channels'], list):
            known['color_channels'] = [str(x) for x in known['color_channels']]
        if 'total_images' in known and known['total_images'] is not None:
            try:
                known['total_images'] = int(known['total_images'])
            except Exception:
                known['total_images'] = None
        if 'total_annotations' in known and known['total_annotations'] is not None:
            try:
                known['total_annotations'] = int(known['total_annotations'])
            except Exception:
                known['total_annotations'] = None
        if 'uid' in known and known['uid'] is not None:
            try:
                known['uid'] = str(known['uid'])
            except Exception:
                known['uid'] = str(uuid.uuid4())

        return cls(**known, extra=data)