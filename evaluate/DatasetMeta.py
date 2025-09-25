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

    #используется в DatasetModel.post_init()
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetMeta':
        if not data:
            return cls()
        data = dict(data)
        known_keys = {
            'uid', 'name', 'categories', 'total_images', 'total_annotations',
            'url', 'color_channels', 'source_path', 'base_directory', 'dataset_type'
        }
        known = {k: data.pop(k) for k in list(data.keys()) if k in known_keys}
        if 'categories' in known and known['categories']:
            known['categories'] = [str(x) for x in known['categories']]
        
        if 'color_channels' in known and known['color_channels']:
            known['color_channels'] = [str(x) for x in known['color_channels']]
        for field in ['total_images', 'total_annotations']:
            if field in known and known[field] is not None:
                try:
                    known[field] = int(known[field])
                except (ValueError, TypeError):
                    known[field] = None
        if 'uid' in known and known['uid'] is not None:
            known['uid'] = str(known['uid'])
        
        return cls(**known, extra=data)