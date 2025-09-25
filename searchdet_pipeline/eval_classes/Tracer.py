from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
import json
import sys
import cProfile
import pstats
from contextlib import contextmanager
import subprocess
import tempfile
import os
from pathlib import Path
EVAL_CLASSES_PATH = Path(__file__).parent
sys.path.insert(0, str(EVAL_CLASSES_PATH))
from Context import Context


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

    def serialize(self, ctx: Context) -> Dict[str, Any]:
        def ser_dt(dt: Optional[datetime]):
            return dt.isoformat() + "Z" if isinstance(dt, datetime) else None

        duration_val = (
            (ctx.ended_at - ctx.started_at).total_seconds()
            if ctx.started_at and ctx.ended_at
            else None
        )
        spans_out: List[Dict[str, Any]] = []
        for s in ctx.spans:
            st = s.get("started_at")
            en = s.get("ended_at")
            d = (en - st).total_seconds() if isinstance(st, datetime) and isinstance(en, datetime) else None
            spans_out.append(
                {
                    "name": s.get("name"),
                    "started_at": ser_dt(st),
                    "ended_at": ser_dt(en),
                    "duration": d,
                    "attributes": s.get("attributes", {}),
                }
            )

        return {
            "uid": ctx.uid,
            "detector_name": ctx.detector_name,
            "image_shape": ctx.image_shape,
            "started_at": ser_dt(ctx.started_at),
            "ended_at": ser_dt(ctx.ended_at),
            "duration": duration_val,
            "success": ctx.success,
            "error": ctx.error,
            "metrics": ctx.metrics,
            "extra": ctx.extra,
            "spans": spans_out,
        }

    def emit(self, ctx: Context) -> Dict[str, Any]:
        payload = self.serialize(ctx)
        if self.to_stdout:
            print(json.dumps(payload, ensure_ascii=False, default=str), file=sys.stdout)
        for sink in self.sinks:
            sink(payload)
        if self.trace_file:
            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        return payload

    def dump_chrome_trace(self, ctx: Context, path: str) -> str:
        events: List[Dict[str, Any]] = []
        base = ctx.started_at

        def to_us(dt: datetime) -> int:
            return int((dt - base).total_seconds() * 1_000_000)

        events.append({"name": "Context.start", "ph": "B", "ts": 0, "pid": 1, "tid": 1})
        for s in ctx.spans:
            st = s.get("started_at")
            en = s.get("ended_at")
            if not isinstance(st, datetime) or not isinstance(en, datetime):
                continue
            events.append(
                {
                    "name": s.get("name", "span"),
                    "ph": "B",
                    "ts": to_us(st),
                    "pid": 1,
                    "tid": 1,
                    "args": s.get("attributes", {}),
                }
            )
            events.append(
                {
                    "name": s.get("name", "span"),
                    "ph": "E",
                    "ts": to_us(en),
                    "pid": 1,
                    "tid": 1,
                }
            )
        end_ts = to_us(ctx.ended_at) if isinstance(ctx.ended_at, datetime) else to_us(datetime.utcnow())
        events.append({"name": "Context.end", "ph": "E", "ts": end_ts, "pid": 1, "tid": 1})
        trace = {"traceEvents": events}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(trace, f)
        return path

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