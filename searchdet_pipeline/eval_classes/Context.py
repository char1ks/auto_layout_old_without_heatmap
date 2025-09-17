from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable, List
from uuid import uuid4
from datetime import datetime
import json
import sys
import cProfile
import pstats
from contextlib import contextmanager
import subprocess
import tempfile
import os


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
    
    metrics: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    spans: List[dict] = field(default_factory=list)
    trace_sinks: List[Callable[[dict], None]] = field(default_factory=list)
    trace_to_stdout: bool = True
    trace_file: Optional[str] = None  

    _profiler: Optional[cProfile.Profile] = field(default=None, repr=False, compare=False)

    def finish(self, success: bool = True, error: Optional[str] = None):
        self.ended_at = datetime.utcnow()
        self.duration = (self.ended_at - self.started_at).total_seconds()
        self.success = success
        self.error = error
        self.emit()

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
            "duration": None,
            "attributes": dict(attrs) if attrs else {},
        }
        self.spans.append(span)
        try:
            yield span
            span["ended_at"] = datetime.utcnow()
            span["duration"] = (span["ended_at"] - span["started_at"]).total_seconds()
        except Exception as e:
            span["ended_at"] = datetime.utcnow()
            span["duration"] = (span["ended_at"] - span["started_at"]).total_seconds()
            span["attributes"]["error"] = str(e)
            raise

    def start_span(self, name: str, **attrs: Any) -> int:
        span: Dict[str, Any] = {
            "name": name,
            "started_at": datetime.utcnow(),
            "ended_at": None,
            "duration": None,
            "attributes": dict(attrs) if attrs else {},
        }
        self.spans.append(span)
        return len(self.spans) - 1

    def end_span(self, idx: int, **attrs: Any) -> None:
        if 0 <= idx < len(self.spans):
            span = self.spans[idx]
            span["ended_at"] = datetime.utcnow()
            span["duration"] = (span["ended_at"] - span["started_at"]).total_seconds()
            if attrs:
                span["attributes"].update(attrs)

    def _serialize(self) -> Dict[str, Any]:
        def ser_dt(dt: Optional[datetime]):
            return dt.isoformat() + "Z" if isinstance(dt, datetime) else None
        return {
            "uid": self.uid,
            "detector_name": self.detector_name,
            "image_shape": self.image_shape,
            "started_at": ser_dt(self.started_at),
            "ended_at": ser_dt(self.ended_at),
            "duration": self.duration,
            "success": self.success,
            "error": self.error,
            "metrics": self.metrics,
            "extra": self.extra,
            "spans": [
                {
                    "name": s.get("name"),
                    "started_at": ser_dt(s.get("started_at")),
                    "ended_at": ser_dt(s.get("ended_at")),
                    "duration": s.get("duration"),
                    "attributes": s.get("attributes", {}),
                }
                for s in self.spans
            ],
        }

    def emit(self) -> None:
        payload = self._serialize()
        if self.trace_to_stdout:
            try:
                print(json.dumps(payload, ensure_ascii=False, default=str), file=sys.stdout)
            except Exception:
                print(str(payload), file=sys.stdout)
        for sink in self.trace_sinks:
            try:
                sink(payload)
            except Exception:
                pass
        if self.trace_file:
            try:
                with open(self.trace_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            except Exception:
                pass

    def dump_chrome_trace(self, path: str) -> str:
        events: List[Dict[str, Any]] = []
        base = self.started_at
        def to_us(dt: datetime) -> int:
            return int((dt - base).total_seconds() * 1_000_000)
        events.append({"name": "Context.start", "ph": "B", "ts": 0, "pid": 1, "tid": 1})
        for s in self.spans:
            if not s.get("started_at") or not s.get("ended_at"):
                continue
            events.append({
                "name": s.get("name", "span"),
                "ph": "B",
                "ts": to_us(s["started_at"]),
                "pid": 1,
                "tid": 1,
                "args": s.get("attributes", {}),
            })
            events.append({
                "name": s.get("name", "span"),
                "ph": "E",
                "ts": to_us(s["ended_at"]),
                "pid": 1,
                "tid": 1,
            })
        end_ts = to_us(self.ended_at) if self.ended_at else to_us(datetime.utcnow())
        events.append({"name": "Context.end", "ph": "E", "ts": end_ts, "pid": 1, "tid": 1})
        trace = {"traceEvents": events}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(trace, f)
        return path

    def frame_graph(self, prof_path: str, svg_path: str) -> None:
        try:
            result = subprocess.run([sys.executable, "-m", "flameprof", prof_path], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout:
                with open(svg_path, "w", encoding="utf-8") as f:
                    f.write(result.stdout)
        except Exception:
            pass

    @contextmanager
    def profile(self, sort: str = "cumtime", top_k: int = 20, dump_path: Optional[str] = None, flamegraph_path: Optional[str] = None):
        self._profiler = cProfile.Profile()
        self._profiler.enable()
        tmp_prof_path: Optional[str] = None
        try:
            yield
        finally:
            self._profiler.disable()
            stats = pstats.Stats(self._profiler)
            try:
                stats.strip_dirs().sort_stats(sort).print_stats(top_k)
            except Exception:
                stats.print_stats(top_k)
            if flamegraph_path:
                if not dump_path:
                    try:
                        tmp = tempfile.NamedTemporaryFile(prefix="ctx_", suffix=".prof", delete=False)
                        tmp_prof_path = tmp.name
                        tmp.close()
                        dump_path = tmp_prof_path
                    except Exception:
                        tmp_prof_path = None
                try:
                    stats.dump_stats(dump_path)
                except Exception:
                    pass
                self.frame_graph(dump_path, flamegraph_path)
                if tmp_prof_path:
                    try:
                        os.unlink(tmp_prof_path)
                    except Exception:
                        pass
            else:
                if dump_path:
                    try:
                        stats.dump_stats(dump_path)
                    except Exception:
                        pass
            self._profiler = None
