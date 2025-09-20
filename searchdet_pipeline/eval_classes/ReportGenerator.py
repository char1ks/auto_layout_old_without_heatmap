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
        # Всегда печатаем в терминал: кратко или подробно в зависимости от dump_report
        self._generate_terminal_report(metrics, timing_stats, verbose=dump_report)
        if not dump_report:
            return None
        # Подробный режим: сохраняем полный отчёт с графиками и JSON
        report_dir = self._prepare_report_dir(output_dir)
        images = self._save_graphs(report_dir, timing_stats)
        self._write_markdown(report_dir, metrics, timing_stats, images)
        self._write_json(report_dir, metrics, timing_stats)
        self.console.print(f"[green]Отчёт сохранён в[/green] {report_dir}")
        return report_dir

    def _generate_terminal_report(self, metrics: List[MetricOutputModel], timing_stats: Dict[str, Any], verbose: bool = False) -> None:
        # Сводная таблица метрик — всегда
        self.console.rule("МЕТРИКИ (сводная таблица)")
        t = Table(box=box.SIMPLE_HEAVY)
        t.add_column("Metric", style="cyan", no_wrap=True)
        t.add_column("Score", justify="right")
        for m in metrics or []:
            score = f"{m.score:.4f}" if isinstance(m.score, (int, float)) else str(m.score)
            t.add_row(m.metric_name, score)
        self.console.print(t)

        # Статистика времени — всегда
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

        # Подробный режим — детализация метрик
        if not verbose:
            return

        # Детализация mAP, если присутствует соответствующая метрика
        for m in metrics or []:
            if m.metric_name.lower() in {"map", "meanaverageprecision"} and isinstance(m.stats, dict):
                self.console.rule("Детализация mAP")
                mt = Table(box=box.SIMPLE_HEAVY)
                mt.add_column("Metric", style="cyan")
                mt.add_column("Value", justify="right")
                for key in [
                    "mAP", "mAP@0.5", "mAP@0.75", "mAP_small", "mAP_medium", "mAP_large"
                ]:
                    if key in m.stats:
                        try:
                            mt.add_row(key, f"{float(m.stats[key]):.4f}")
                        except Exception:
                            mt.add_row(key, str(m.stats[key]))
                self.console.print(mt)
                break

        # Сквозной отчёт в стиле sklearn.metrics.classification_report
        for m in metrics or []:
            if m.metric_name == "classification_report" and isinstance(m.stats, dict):
                self.console.rule("Classification report (sklearn)")
                rep_dict = m.stats.get("dict")
                rep_text = m.stats.get("text")
                if isinstance(rep_dict, dict):
                    # Таблица: label | precision | recall | f1-score | support
                    crt = Table(box=box.SIMPLE_HEAVY)
                    crt.add_column("label", style="cyan")
                    crt.add_column("precision", justify="right")
                    crt.add_column("recall", justify="right")
                    crt.add_column("f1-score", justify="right")
                    crt.add_column("support", justify="right")
                    # сначала классы (не агрегаты), затем агрегаты
                    def _is_agg(k: str) -> bool:
                        lk = str(k).lower()
                        return lk in {"accuracy", "macro avg", "weighted avg", "micro avg", "samples avg"}
                    keys = [k for k in rep_dict.keys() if not _is_agg(k)]
                    for k in keys:
                        row = rep_dict.get(k, {}) or {}
                        crt.add_row(
                            str(k),
                            _fmt_float(row.get("precision", 0.0), 2),
                            _fmt_float(row.get("recall", 0.0), 2),
                            _fmt_float(row.get("f1-score", 0.0), 2),
                            str(row.get("support", 0)),
                        )
                    # Разделитель и агрегаты
                    crt.add_section()
                    for agg in ["accuracy", "macro avg", "weighted avg", "micro avg", "samples avg"]:
                        if agg in rep_dict:
                            row = rep_dict[agg]
                            if isinstance(row, dict):
                                prec = _fmt_float(row.get("precision", 0.0), 2)
                                rec = _fmt_float(row.get("recall", 0.0), 2)
                                f1 = _fmt_float(row.get("f1-score", 0.0), 2)
                                sup = str(row.get("support", 0))
                            else:
                                # accuracy может быть скаляром
                                prec = rec = f1 = _fmt_float(row, 2)
                                sup = "-"
                            crt.add_row(agg, prec, rec, f1, sup)
                    self.console.print(crt)
                elif rep_text:
                    # Фолбэк — печать текстового отчёта sklearn
                    self.console.print(rep_text)
                break

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
        out = Path(output_dir) if output_dir else Path(os.getcwd()) / "report"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _save_graphs(self, report_dir: Path, timing_stats: Dict[str, Any]) -> Dict[str, str]:
        images: Dict[str, str] = {}
        durations = timing_stats.get("durations", [])
        if durations:
            plt.figure(figsize=(6, 3))
            plt.hist(durations, bins=20, color="#4C78A8"); plt.title("Durations (sec)")
            hist_path = report_dir / "durations_hist.png"
            plt.tight_layout(); plt.savefig(hist_path); plt.close()
            images["durations_hist"] = str(hist_path)

            plt.figure(figsize=(6, 3))
            plt.plot(durations, color="#F58518"); plt.title("Durations by run")
            ser_path = report_dir / "durations_series.png"
            plt.tight_layout(); plt.savefig(ser_path); plt.close()
            images["durations_series"] = str(ser_path)

        spans_avg: Dict[str, float] = timing_stats.get("spans_avg", {})
        if spans_avg:
            names = list(spans_avg.keys())
            values = [spans_avg[k] for k in names]
            plt.figure(figsize=(6, 3))
            plt.barh(names, values, color="#54A24B"); plt.title("Avg span time (sec)")
            plt.tight_layout()
            p = report_dir / "spans_avg.png"
            plt.savefig(p); plt.close()
            images["spans_avg"] = str(p)
        return images

    def _write_markdown(
        self,
        report_dir: Path,
        metrics: List[MetricOutputModel],
        timing_stats: Dict[str, Any],
        images: Dict[str, str],
    ) -> None:
        lines: List[str] = []
        lines.append(f"# Detection Report\n\n")
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


# Утилита форматирования чисел для таблиц
def _fmt_float(v: Any, nd: int = 4) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)