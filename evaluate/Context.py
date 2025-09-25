from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from uuid import uuid4
from datetime import datetime
from contextlib import contextmanager


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
        try:
            if self.started_at and self.ended_at:
                self.duration = (self.ended_at - self.started_at).total_seconds()
        except Exception:
            pass

    def add_metric(self, key: str, value: Any) -> None:
        self.metrics[key] = value

    def log_extra(self, key: str, value: Any) -> None:
        self.extra[key] = value

    #Добавил эту аннотацию для того,чтобы IDE не ругалась на то,что span может быть None
    @contextmanager
    def span(self, name: str, **attrs: Any):
        span: Dict[str, Any] = {
            "name": name,
            "started_at": datetime.utcnow(),
            "ended_at": None,
            "attributes": dict(attrs) if attrs else {},
        }
        self.spans.append(span)
        try:
            yield span
            span["ended_at"] = datetime.utcnow()
            # Добавляем длительность для удобного сбора статистики
            try:
                span["duration"] = (span["ended_at"] - span["started_at"]).total_seconds()
            except Exception:
                pass
        except Exception as e:
            span["ended_at"] = datetime.utcnow()
            span["attributes"]["error"] = str(e)
            try:
                span["duration"] = (span["ended_at"] - span["started_at"]).total_seconds()
            except Exception:
                pass
            raise

    def start_span(self, name: str, **attrs: Any) -> int:
        span: Dict[str, Any] = {
            "name": name,
            "started_at": datetime.utcnow(),
            "ended_at": None,
            "attributes": dict(attrs) if attrs else {},
        }
        self.spans.append(span)
        return len(self.spans) - 1

    def end_span(self, idx: int, **attrs: Any) -> None:
        if 0 <= idx < len(self.spans):
            span = self.spans[idx]
            span["ended_at"] = datetime.utcnow()
            if attrs:
                span["attributes"].update(attrs)
            # Вычисляем длительность span
            try:
                span["duration"] = (span["ended_at"] - span["started_at"]).total_seconds()
            except Exception:
                pass
