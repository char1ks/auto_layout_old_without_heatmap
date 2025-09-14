from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from uuid import uuid4
from datetime import datetime


@dataclass
class Context:
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    detector_name: str = ""
    image_shape: Optional[tuple[int, int, int]] = None

    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    duration: Optional[float] = None

    success: bool = False
    error: Optional[str] = None
    
    metrics: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def finish(self, success: bool = True, error: Optional[str] = None):
        self.ended_at = datetime.utcnow()
        self.duration = (self.ended_at - self.started_at).total_seconds()
        self.success = success
        self.error = error
