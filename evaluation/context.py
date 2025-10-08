from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from uuid import uuid4
from datetime import datetime


@dataclass
class Context:
    uid: str = field(default_factory=lambda: str(uuid4()))
    detector_name: str = ""
    image_shape: Optional[tuple[int, int, int]] = None

    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    duration: Optional[float] = None

    success: bool = False
    error: Optional[str] = None

    metrics: Dict[str, Any] = field(default_factory=dict) #Сюда идут любые числовые/логические/строковые показатели по одному запуску детекции для конкретного изображения
    extra: Dict[str, Any] = field(default_factory=dict)
    spans: List[dict] = field(default_factory=list) #Сюда идут данные о времени выполнения различных частей пайплайна детекции (например, времени на предсказание, времени на постобработку, времени на запись в файл и тд)

    def finish(self, success: bool = True, error: Optional[str] = None):
        self.ended_at = datetime.utcnow()
        self.success = success
        self.error = error
        if self.started_at and self.ended_at:
            self.duration = (self.ended_at - self.started_at).total_seconds()


    def span(self, name: str, fn):
        start = datetime.utcnow()
        out = fn()
        end = datetime.utcnow()
        self.spans.append({
            "name": name,
            "started_at": start,
            "ended_at": end,
            "duration": (end - start).total_seconds(),
            "attributes": {},
        })
        return out