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
    duration: Optional[float] = None  # kept as raw field, not auto-computed

    success: bool = False
    error: Optional[str] = None

    metrics: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    # raw spans; no auto duration calculations here
    spans: List[dict] = field(default_factory=list)

    def finish(self, success: bool = True, error: Optional[str] = None):
        self.ended_at = datetime.utcnow()
        self.success = success
        self.error = error

    def add_metric(self, key: str, value: Any) -> None:
        self.metrics[key] = value

    def log_extra(self, key: str, value: Any) -> None:
        self.extra[key] = value

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
        except Exception as e:
            span["ended_at"] = datetime.utcnow()
            span["attributes"]["error"] = str(e)
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
