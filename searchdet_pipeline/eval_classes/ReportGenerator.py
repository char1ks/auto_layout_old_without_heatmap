from __future__ import annotations

from typing import List, Optional, Dict, Any, Union, Iterable
from pathlib import Path
from datetime import datetime
import json
import statistics

import matplotlib.pyplot as plt
import matplotlib as mpl

# Console output
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
except Exception:  # pragma: no cover
    Console = None
    Table = None
    box = None

# Config + datamodels (light coupling; tolerate import differences)
try:
    from ReportConfig import ReportConfig
except Exception:  # pragma: no cover
    class ReportConfig:  # minimal stub if imported standalone
        include_spans: bool = True
        include_contexts_table: bool = True
        include_errors: bool = True
        top_k_errors: int = 5
        include_detector_breakdown: bool = True
        include_class_distributions: bool = True
        include_ap_graphs: bool = True
        top_k_lowest_map: int = 5

try:
    from metrics import MetricOutputModel  # local
except Exception:  # pragma: no cover
    try:
        # alternative path used in some projects
        from searchdet_pipeline.eval_classes.metrics import MetricOutputModel
    except Exception:
        from dataclasses import dataclass
        @dataclass
        class MetricOutputModel:  # very small fallback
            metric_name: str
            score: float
            stats: Dict[str, Any]

# Matplotlib safe defaults for SVG outputs
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["savefig.facecolor"] = "white"
mpl.rcParams["figure.facecolor"] = "white"


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if isinstance(x, (int, float)):
            return float(x)
        return float(str(x))
    except Exception:
        return float(default)


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


class ReportGenerator(ReportConfig):
    """
    Lightweight, dependency-friendly report generator.

    * Keeps the public surface compatible with the previous implementation:
      - generate_report(contexts, metrics, dump_report=True, output_dir=None) -> Optional[Path]
    * Leans on ready-made libraries instead of hand-rolled code:
      - matplotlib for plots
      - rich for terminal pretty-print (if available)
    * All inputs are handled defensively to avoid schema breakages.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.console = Console() if Console else None

    # ---------------------- Public API ---------------------- #
    def generate_report(
        self,
        contexts: List[Dict[str, Any]],
        metrics: List[MetricOutputModel],
        dump_report: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Path]:
        """
        Build terminal + markdown + json report. Returns the report directory path if dump_report=True.
        """
        timing_stats = self._collect_timing_stats(contexts or [])
        self._generate_terminal_report(metrics or [], timing_stats, verbose=dump_report)
        if not dump_report:
            return None
        report_dir = self._prepare_report_dir(output_dir)
        images = self._save_graphs(report_dir, timing_stats, metrics or [])
        self._write_markdown(report_dir, metrics or [], timing_stats, images, contexts or [])
        self._write_json(report_dir, metrics or [], timing_stats)
        if self.console:
            self.console.print(f"[green]Отчёт сохранён в[/green] {report_dir}")
        return report_dir

    # ---------------------- Helpers ---------------------- #
    def _generate_terminal_report(
        self, metrics: List[MetricOutputModel], timing_stats: Dict[str, Any], verbose: bool = False
    ) -> None:
        if not verbose or not self.console or not Table:
            return

        from rich import box  # local import to avoid hard dep when not needed

        self.console.rule("Метрики")
        t = Table(box=box.SIMPLE_HEAVY)
        t.add_column("Метрика", style="cyan")
        t.add_column("Значение", justify="right")
        for m in metrics:
            try:
                score = f"{float(m.score):.4f}"
            except Exception:
                score = str(m.score)
            t.add_row(str(m.metric_name), score)
        self.console.print(t)

        self.console.rule("Статистика времени")
        tt = Table(box=box.SIMPLE_HEAVY)
        tt.add_column("Показатель", style="magenta")
        tt.add_column("Значение", justify="right")
        for k, v in [
            ("Запусков", timing_stats.get("total_runs", 0)),
            ("Успешных", timing_stats.get("success", 0)),
            ("Ошибок", timing_stats.get("errors", 0)),
            ("Среднее, сек", f"{_as_float(timing_stats.get('mean', 0.0)):.4f}"),
            ("Медиана, сек", f"{_as_float(timing_stats.get('median', 0.0)):.4f}"),
            ("Std, сек", f"{_as_float(timing_stats.get('std', 0.0)):.4f}"),
            ("Мин, сек", f"{_as_float(timing_stats.get('min', 0.0)):.4f}"),
            ("Макс, сек", f"{_as_float(timing_stats.get('max', 0.0)):.4f}"),
        ]:
            tt.add_row(k, str(v))
        self.console.print(tt)

    def _prepare_report_dir(self, output_dir: Optional[Union[str, Path]]) -> Path:
        base = Path(output_dir) if output_dir else Path("report_" + _now_stamp())
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _save_graphs(
        self, report_dir: Path, timing_stats: Dict[str, Any], metrics: List[MetricOutputModel]
    ) -> Dict[str, str]:
        images: Dict[str, str] = {}

        # 1) Durations histogram / series
        durations: List[float] = [float(d) for d in (timing_stats.get("durations") or []) if isinstance(d, (int, float))]
        if durations:
            plt.figure(figsize=(6, 3))
            plt.hist(durations, bins=20)
            plt.title("Durations (sec)")
            hist_path = report_dir / "durations_hist.svg"
            plt.tight_layout()
            plt.savefig(hist_path, format="svg", bbox_inches="tight")
            plt.close()
            images["durations_hist"] = str(hist_path)

            plt.figure(figsize=(6, 3))
            plt.plot(durations)
            plt.title("Durations by run")
            ser_path = report_dir / "durations_series.svg"
            plt.tight_layout()
            plt.savefig(ser_path, format="svg", bbox_inches="tight")
            plt.close()
            images["durations_series"] = str(ser_path)

        # 2) Span averages
        spans_avg: Dict[str, float] = timing_stats.get("spans_avg", {}) or {}
        if spans_avg:
            names = list(spans_avg.keys())
            values = [spans_avg[k] for k in names]
            plt.figure(figsize=(6, 3))
            plt.bar(names, values)
            plt.xticks(rotation=45, ha="right")
            plt.title("Average span durations (sec)")
            span_path = report_dir / "spans_avg.svg"
            plt.tight_layout()
            plt.savefig(span_path, format="svg", bbox_inches="tight")
            plt.close()
            images["spans_avg"] = str(span_path)

        # 3) mAP breakdown graph if present in metrics stats (torchmetrics style)
        try:
            for m in metrics or []:
                if str(m.metric_name).lower() in {"map", "meanaverageprecision", "mean_average_precision"}:
                    st = m.stats or {}
                    # torchmetrics returns per IoU and per area metrics
                    if isinstance(st, dict) and "map_per_iou" in st:
                        xs = [str(k) for k in st.get("iou_thresholds", [])] if "iou_thresholds" in st else [str(i) for i in range(len(st["map_per_iou"]))]
                        ys = [float(v) for v in (st.get("map_per_iou") or [])]
                        if ys:
                            plt.figure(figsize=(6, 3))
                            plt.plot(xs, ys, marker="o")
                            plt.title("mAP per IoU threshold")
                            plt.xlabel("IoU threshold")
                            plt.ylabel("mAP")
                            mp_path = report_dir / "map_per_iou.svg"
                            plt.tight_layout()
                            plt.savefig(mp_path, format="svg", bbox_inches="tight")
                            plt.close()
                            images["map_per_iou"] = str(mp_path)
                    break
        except Exception:
            pass

        return images

    def _write_markdown(
        self,
        report_dir: Path,
        metrics: List[MetricOutputModel],
        timing_stats: Dict[str, Any],
        images: Dict[str, str],
        contexts: List[Dict[str, Any]],
    ) -> None:
        lines: List[str] = []
        lines.append(f"# Отчёт по запуску — {datetime.now().isoformat(timespec='seconds')}\n\n")

        # Metrics
        lines.append("## Метрики\n\n")
        for m in metrics or []:
            try:
                score = f"{float(m.score):.4f}"
            except Exception:
                score = str(m.score)
            lines.append(f"- **{m.metric_name}**: `{score}`\n")
        lines.append("\n")

        # Timing summary
        lines.append("## Время выполнения\n\n")
        for k in ["total_runs", "success", "errors", "mean", "median", "std", "min", "max"]:
            if k in timing_stats:
                lines.append(f"- **{k}**: `{timing_stats[k]}`\n")
        lines.append("\n")

        # Images
        if images:
            lines.append("## Графики\n\n")
            for name, path in images.items():
                rel = Path(path).name
                lines.append(f"![{name}]({rel})\n\n")

        # Contexts table (optional)
        if getattr(self, "include_contexts_table", True) and contexts:
            lines.append("## Прогоны\n\n")
            lines.append("| # | ok | duration(s) | error |\n")
            lines.append("|---:|:--:|-----------:|-------|\n")
            for i, c in enumerate(contexts, 1):
                ok = c.get("ok")
                duration = c.get("duration") or c.get("time") or 0
                err = (c.get("error") or "")[:120] if isinstance(c.get("error"), str) else ""
                lines.append(f"| {i} | {ok} | {duration:.4f} | {err} |\n")
            lines.append("\n")

        # Raw stats for debugging
        lines.append("## Сырые данные метрик\n\n")
        for m in metrics or []:
            lines.append(f"### {m.metric_name}\n\n")
            try:
                text = json.dumps(m.stats or {}, ensure_ascii=False, indent=2)
            except Exception:
                text = str(m.stats)
            lines.append("```json\n")
            lines.append(text)
            if not text.endswith("\n"):
                lines.append("\n")
            lines.append("```\n\n")

        (report_dir / "report.md").write_text("".join(lines), encoding="utf-8")

    def _write_json(self, report_dir: Path, metrics: List[MetricOutputModel], timing_stats: Dict[str, Any]) -> None:
        data = {
            "metrics": [
                {"metric_name": m.metric_name, "score": m.score, "stats": m.stats} for m in (metrics or [])
            ],
            "timing": timing_stats,
        }
        (report_dir / "report.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------------- Stats extraction ---------------------- #
    def _collect_timing_stats(self, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        durations: List[float] = []
        success = 0
        errors = 0
        spans: Dict[str, List[float]] = {}

        for c in contexts or []:
            # duration
            duration = c.get("duration", c.get("time", None))
            if isinstance(duration, (int, float)):
                durations.append(float(duration))

            # ok flag + error counting
            ok = c.get("ok")
            if isinstance(ok, bool):
                success += int(ok)
                errors += int(not ok)

            # spans
            if getattr(self, "include_spans", True):
                for s in c.get("spans", []) or []:
                    name = str(s.get("name", "span"))
                    val = s.get("duration") or s.get("time") or 0
                    if isinstance(val, (int, float)):
                        spans.setdefault(name, []).append(float(val))

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
            stats["spans_avg"] = {k: float(statistics.mean(v)) for k, v in spans.items()}
        return stats
