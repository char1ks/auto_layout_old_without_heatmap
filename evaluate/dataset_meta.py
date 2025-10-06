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