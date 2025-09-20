from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple, Union
from pathlib import Path
from datetime import datetime
import os
import json
import statistics

import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich import box

from searchdet_pipeline.eval_classes.Context import Context
from searchdet_pipeline.eval_classes.metrics import MetricOutputModel
from searchdet_pipeline.eval_classes.ReportConfig import ReportConfig


class ReportGenerator(ReportConfig):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.console = Console()

    def generate_report(
        self,
        contexts: List[Context],
        metrics: List[MetricOutputModel],
        dump_report: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Path]:
        timing_stats = self._collect_timing_stats(contexts)
        if not dump_report:
            self._generate_terminal_report(metrics, timing_stats)
            return None
        report_dir = self._prepare_report_dir(output_dir)
        images = self._save_graphs(report_dir, timing_stats)
        self._write_markdown(report_dir, metrics, timing_stats, images)
        self._write_json(report_dir, metrics, timing_stats)
        self.console.print(f"[green]Отчёт сохранён в[/green] {report_dir}")
        return report_dir

    def _generate_terminal_report(self, metrics: List[MetricOutputModel], timing_stats: Dict[str, Any]) -> None:
        self.console.rule("МЕТРИКИ (сводная таблица)")
        t = Table(box=box.SIMPLE_HEAVY)
        t.add_column("Metric", style="cyan", no_wrap=True)
        t.add_column("Score", justify="right")
        for m in metrics or []:
            score = f"{m.score:.4f}" if isinstance(m.score, (int, float)) else str(m.score)
            t.add_row(m.metric_name, score)
        self.console.print(t)

        self.console.rule("СТАТИСТИКА ВРЕМЕНИ ДЕТЕКТОРА")
        tt = Table(box=box.SIMPLE_HEAVY)
        tt.add_column("Показатель", style="magenta")
        tt.add_column("Значение", justify="right")
        for k in [
            ("Запусков", timing_stats.get("total_runs", 0)),
            ("Успешных", timing_stats.get("success", 0)),
            ("Ошибок", timing_stats.get("errors", 0)),
            ("Среднее, сек", _fmt_float(timing_stats.get("mean", 0.0))),
            ("Медиана, сек", _fmt_float(timing_stats.get("median", 0.0))),
            ("Std, сек", _fmt_float(timing_stats.get("std", 0.0))),
            ("Мин, сек", _fmt_float(timing_stats.get("min", 0.0))),
            ("Макс, сек", _fmt_float(timing_stats.get("max", 0.0))),
        ]:
            tt.add_row(str(k[0]), str(k[1]))
        self.console.print(tt)

        spans_avg: Dict[str, float] = timing_stats.get("spans_avg", {})
        if spans_avg and self.include_spans:
            st = Table(title="Среднее время по этапам", box=box.SIMPLE_HEAVY)
            st.add_column("Этап")
            st.add_column("Среднее, сек", justify="right")
            for name, val in sorted(spans_avg.items(), key=lambda x: x[1], reverse=True):
                st.add_row(name, _fmt_float(val))
            self.console.print(st)

    def _collect_timing_stats(self, contexts: List[Context]) -> Dict[str, Any]:
        durations: List[float] = []
        success = 0
        errors = 0
        spans: Dict[str, List[float]] = {}
        for c in contexts or []:
            if c.duration is not None:
                durations.append(float(c.duration))
            if c.success:
                success += 1
            if c.error:
                errors += 1
            if self.include_spans and getattr(c, "spans", None):
                for s in c.spans:
                    name = str(s.get("name", s.get("stage", "span")))
                    val = s.get("duration") or s.get("time") or 0
                    try:
                        val = float(val)
                        spans.setdefault(name, []).append(val)
                    except Exception:
                        pass
        stats: Dict[str, Any] = {
            "total_runs": len(contexts or []),
            "success": success,
            "errors": errors,
            "durations": durations,
        }
        if durations:
            stats.update(
                {
                    "mean": float(statistics.mean(durations)),
                    "median": float(statistics.median(durations)),
                    "std": float(statistics.pstdev(durations)) if len(durations) > 1 else 0.0,
                    "min": float(min(durations)),
                    "max": float(max(durations)),
                }
            )
        if spans:
            stats["spans_avg"] = {k: float(statistics.mean(v)) for k, v in spans.items() if v}
        else:
            stats["spans_avg"] = {}
        return stats

    def _prepare_report_dir(self, output_dir: Optional[Union[str, Path]]) -> Path:
        base = Path(output_dir) if output_dir else Path.cwd()
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_dir = base / f"eval_report_{ts}"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "images").mkdir(exist_ok=True)
        return report_dir

    def _save_graphs(self, report_dir: Path, timing_stats: Dict[str, Any]) -> Dict[str, str]:
        images_dir = report_dir / "images"
        saved: Dict[str, str] = {}
        durations = timing_stats.get("durations", [])
        if durations:
            plt.figure(figsize=(6, 4))
            plt.hist(durations, bins=max(5, min(30, len(durations)//2)), color="#4C78A8")
            plt.xlabel("Duration, sec")
            plt.ylabel("Count")
            plt.tight_layout()
            p = images_dir / "durations_hist.png"
            plt.savefig(p)
            plt.close()
            saved["durations_hist"] = str(p.relative_to(report_dir))

            plt.figure(figsize=(6, 3))
            plt.plot(range(1, len(durations) + 1), durations, marker="o", linestyle="-", color="#F58518")
            plt.xlabel("Run #")
            plt.ylabel("Duration, sec")
            plt.tight_layout()
            p2 = images_dir / "durations_series.png"
            plt.savefig(p2)
            plt.close()
            saved["durations_series"] = str(p2.relative_to(report_dir))
        spans_avg: Dict[str, float] = timing_stats.get("spans_avg", {})
        if spans_avg:
            items = sorted(spans_avg.items(), key=lambda x: x[1], reverse=True)
            names = [k for k, _ in items]
            vals = [v for _, v in items]
            plt.figure(figsize=(7, 4))
            plt.barh(names, vals, color="#54A24B")
            plt.xlabel("Avg duration, sec")
            plt.tight_layout()
            p3 = images_dir / "spans_avg.png"
            plt.savefig(p3)
            plt.close()
            saved["spans_avg"] = str(p3.relative_to(report_dir))
        return saved

    def _write_markdown(
        self,
        report_dir: Path,
        metrics: List[MetricOutputModel],
        timing_stats: Dict[str, Any],
        images: Dict[str, str],
    ) -> None:
        lines: List[str] = []
        lines.append(f"# Evaluation Report\n")
        lines.append(f"Generated: {datetime.utcnow().isoformat()} UTC\n")
        lines.append("\n## Summary\n")
        lines.append(f"- Runs: {timing_stats.get('total_runs', 0)}\n")
        lines.append(f"- Success: {timing_stats.get('success', 0)}\n")
        lines.append(f"- Errors: {timing_stats.get('errors', 0)}\n")
        if timing_stats.get("durations"):
            lines.append(
                f"- Mean/Median/Std: {timing_stats.get('mean', 0.0):.3f} / {timing_stats.get('median', 0.0):.3f} / {timing_stats.get('std', 0.0):.3f} sec\n"
            )
        lines.append("\n## Metrics\n")
        lines.append("| Metric | Score |\n|---|---:|\n")
        for m in metrics or []:
            score = f"{m.score:.4f}" if isinstance(m.score, (int, float)) else str(m.score)
            lines.append(f"| {m.metric_name} | {score} |\n")
        lines.append("\n## Timing\n")
        if "durations_hist" in images:
            lines.append(f"![Durations histogram]({images['durations_hist']})\n")
        if "durations_series" in images:
            lines.append(f"![Durations series]({images['durations_series']})\n")
        if "spans_avg" in images:
            lines.append(f"![Stage avg time]({images['spans_avg']})\n")
        (report_dir / "report.md").write_text("".join(lines), encoding="utf-8")

    def _write_json(self, report_dir: Path, metrics: List[MetricOutputModel], timing_stats: Dict[str, Any]) -> None:
        data = {
            "metrics": [
                {"metric_name": m.metric_name, "score": m.score, "stats": m.stats} for m in (metrics or [])
            ],
            "timing": timing_stats,
        }
        (report_dir / "report.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _fmt_float(v: Any, nd: int = 4) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)