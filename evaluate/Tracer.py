from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import sys
import cProfile
import pstats
from contextlib import contextmanager
import subprocess
import tempfile
import os
from evaluate.context import Context
from evaluate.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TraceSpan:
    name: Optional[str]
    started_at: Optional[str]
    ended_at: Optional[str]
    duration: Optional[float]
    attributes: Dict[str, Any]


@dataclass
class TraceRecord:
    uid: str
    detector_name: str
    image_shape: Optional[tuple[int, int, int]]
    started_at: Optional[str]
    ended_at: Optional[str]
    duration: Optional[float]
    success: bool
    error: Optional[str]
    metrics: Dict[str, Any]
    extra: Dict[str, Any]
    spans: List[TraceSpan]


class Tracer:
    def __init__(
        self,
        to_stdout: bool = True,
        trace_file: Optional[str] = None,
        sinks: Optional[List[Callable[[dict], None]]] = None,
    ) -> None:
        self.to_stdout = to_stdout
        self.trace_file = trace_file
        self.sinks = sinks or []

    def serialize(self, ctx: Context) -> TraceRecord:
        def ser_dt(dt: Optional[datetime]):
            return dt.isoformat() + "Z" if isinstance(dt, datetime) else None

        duration_val = (
            (ctx.ended_at - ctx.started_at).total_seconds()
            if ctx.started_at and ctx.ended_at
            else None
        )
        spans_out: List[TraceSpan] = []
        for s in ctx.spans:
            st = s.get("started_at")
            en = s.get("ended_at")
            d = (en - st).total_seconds() if isinstance(st, datetime) and isinstance(en, datetime) else None
            spans_out.append(
                TraceSpan(
                    name=s.get("name"),
                    started_at=ser_dt(st),
                    ended_at=ser_dt(en),
                    duration=d,
                    attributes=s.get("attributes", {}),
                )
            )

        return TraceRecord(
            uid=ctx.uid,
            detector_name=ctx.detector_name,
            image_shape=ctx.image_shape,
            started_at=ser_dt(ctx.started_at),
            ended_at=ser_dt(ctx.ended_at),
            duration=duration_val,
            success=ctx.success,
            error=ctx.error,
            metrics=ctx.metrics,
            extra=ctx.extra,
            spans=spans_out,
        )

    def emit(self, ctx: Context) -> TraceRecord:
        payload = self.serialize(ctx)
        payload_dict = asdict(payload)
        if self.to_stdout:
            logger.info(json.dumps(payload_dict, ensure_ascii=False, default=str))
        for sink in self.sinks:
            sink(payload_dict)
        if self.trace_file:
            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload_dict, ensure_ascii=False, default=str) + "\n")
        return payload
    def frame_graph(self, prof_path: str, svg_path: str) -> None:
        result = subprocess.run([sys.executable, "-m", "flameprof", prof_path], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout:
            with open(svg_path, "w", encoding="utf-8") as f:
                f.write(result.stdout)

    @contextmanager
    def profile(self, sort: str = "cumtime", top_k: int = 20, dump_path: Optional[str] = None, flamegraph_path: Optional[str] = None):
        profiler = cProfile.Profile()
        profiler.enable()
        tmp_prof_path: Optional[str] = None
        try:
            yield
        finally:
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.strip_dirs().sort_stats(sort).print_stats(top_k)
            if flamegraph_path:
                if not dump_path:
                    tmp = tempfile.NamedTemporaryFile(prefix="ctx_", suffix=".prof", delete=False)
                    tmp_prof_path = tmp.name
                    tmp.close()
                    dump_path = tmp_prof_path
                if dump_path:
                    stats.dump_stats(dump_path)
                    self.frame_graph(dump_path, flamegraph_path)
                if tmp_prof_path:
                    os.unlink(tmp_prof_path)
            elif dump_path:
                stats.dump_stats(dump_path)