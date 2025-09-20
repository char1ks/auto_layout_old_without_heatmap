from __future__ import annotations

# from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from datetime import datetime
import os
import json
import csv
import statistics

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
        dump_report: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Path]:
        timing_stats = self._collect_timing_stats(contexts)
        self._generate_terminal_report(metrics, timing_stats, verbose=dump_report)
        if not dump_report:
            return None
        # Подробный режим: сохраняем полный отчёт с графиками и JSON
        report_dir = self._prepare_report_dir(output_dir)
        images = self._save_graphs(report_dir, timing_stats, metrics)
        self._write_markdown(report_dir, metrics, timing_stats, images, contexts)
        self._write_json(report_dir, metrics, timing_stats)
        self.console.print(f"[green]Отчёт сохранён в[/green] {report_dir}")
        return report_dir

    def _generate_terminal_report(self, metrics: List[MetricOutputModel], timing_stats: Dict[str, Any], verbose: bool = False) -> None:
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
        if not verbose:
            return
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
        for m in metrics or []:
            if m.metric_name == "classification_report" and isinstance(m.stats, dict):
                self.console.rule("Classification report (sklearn)")
                rep_dict = m.stats.get("dict")
                rep_text = m.stats.get("text")
                if isinstance(rep_dict, dict):
                    crt = Table(box=box.SIMPLE_HEAVY)
                    crt.add_column("label", style="cyan")
                    crt.add_column("precision", justify="right")
                    crt.add_column("recall", justify="right")
                    crt.add_column("f1-score", justify="right")
                    crt.add_column("support", justify="right")
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

    def _save_graphs(self, report_dir: Path, timing_stats: Dict[str, Any], metrics: List[MetricOutputModel]) -> Dict[str, str]:
        images: Dict[str, str] = {}
        # Время
        durations = timing_stats.get("durations", [])
        if durations:
            plt.figure(figsize=(6, 3))
            plt.hist(durations, bins=20, color="#4C78A8"); plt.title("Durations (sec)")
            hist_path = report_dir / "durations_hist.svg"
            plt.tight_layout(); plt.savefig(hist_path, format="svg"); plt.close()
            images["durations_hist"] = str(hist_path)

            plt.figure(figsize=(6, 3))
            plt.plot(durations, color="#F58518"); plt.title("Durations by run")
            ser_path = report_dir / "durations_series.svg"
            plt.tight_layout(); plt.savefig(ser_path, format="svg"); plt.close()
            images["durations_series"] = str(ser_path)

        spans_avg: Dict[str, float] = timing_stats.get("spans_avg", {})
        if spans_avg:
            names = list(spans_avg.keys())
            values = [spans_avg[k] for k in names]
            plt.figure(figsize=(6, 3))
            plt.barh(names, values, color="#54A24B"); plt.title("Avg span time (sec)")
            plt.tight_layout()
            p = report_dir / "spans_avg.svg"
            plt.savefig(p, format="svg"); plt.close()
            images["spans_avg"] = str(p)

        # По метрикам: распределения и mAP-графики
        map_metric: Optional[MetricOutputModel] = None
        for m in metrics or []:
            if str(m.metric_name).lower() in {"map", "meanaverageprecision"}:
                map_metric = m
                break
        if map_metric and isinstance(map_metric.stats, dict):
            st = map_metric.stats
            categories = st.get("categories") or []
            gt_counts = st.get("gt_counts") or []
            pred_counts = st.get("pred_counts") or []
            # Распределения классов
            if self.include_class_distributions and categories and (gt_counts or pred_counts):
                try:
                    x = list(range(len(categories)))
                    labels = [str(c) for c in categories]
                    if gt_counts:
                        plt.figure(figsize=(max(6, len(labels) * 0.5), 3))
                        plt.bar(x, gt_counts, color="#4C78A8"); plt.title("GT class distribution"); plt.xticks(x, labels, rotation=45, ha="right")
                        p1 = report_dir / "gt_class_distribution.svg"
                        plt.tight_layout(); plt.savefig(p1, format="svg"); plt.close()
                        images["gt_class_distribution"] = str(p1)
                    if pred_counts:
                        plt.figure(figsize=(max(6, len(labels) * 0.5), 3))
                        plt.bar(x, pred_counts, color="#F58518"); plt.title("Predicted class distribution"); plt.xticks(x, labels, rotation=45, ha="right")
                        p2 = report_dir / "pred_class_distribution.svg"
                        plt.tight_layout(); plt.savefig(p2, format="svg"); plt.close()
                        images["pred_class_distribution"] = str(p2)
                except Exception:
                    pass
            # AP vs IoU (macro/micro)
            ap_iou_macro = st.get("ap_iou_macro") or []
            ap_iou_micro = st.get("ap_iou_micro") or []
            ious = st.get("iou_thresholds") or []
            if self.include_ap_graphs and ious and (ap_iou_macro or ap_iou_micro):
                try:
                    plt.figure(figsize=(6, 3))
                    if ap_iou_macro:
                        plt.plot(ious, ap_iou_macro, label="macro", color="#4C78A8")
                    if ap_iou_micro:
                        plt.plot(ious, ap_iou_micro, label="micro", color="#F58518")
                    plt.xlabel("IoU threshold"); plt.ylabel("AP"); plt.title("AP vs IoU"); plt.legend()
                    p = report_dir / "ap_vs_iou.svg"
                    plt.tight_layout(); plt.savefig(p, format="svg"); plt.close()
                    images["ap_vs_iou"] = str(p)
                except Exception:
                    pass
            # PR curves (macro/micro) для 0.50 и 0.75
            if self.include_ap_graphs:
                for key, tag in [("0.50", "050"), ("0.75", "075")]:
                    try:
                        prM = st.get("pr_macro", {}).get(key)
                        prm = st.get("pr_micro", {}).get(key)
                        if prM and prM.get("recall") and prM.get("precision"):
                            plt.figure(figsize=(6, 3))
                            plt.plot(prM["recall"], prM["precision"], color="#4C78A8")
                            plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR macro @IoU={key}")
                            pM = report_dir / f"pr_macro_{tag}.svg"
                            plt.tight_layout(); plt.savefig(pM, format="svg"); plt.close()
                            images[f"pr_macro_{tag}"] = str(pM)
                        if prm and prm.get("recall") and prm.get("precision"):
                            plt.figure(figsize=(6, 3))
                            plt.plot(prm["recall"], prm["precision"], color="#F58518")
                            plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR micro @IoU={key}")
                            pm = report_dir / f"pr_micro_{tag}.svg"
                            plt.tight_layout(); plt.savefig(pm, format="svg"); plt.close()
                            images[f"pr_micro_{tag}"] = str(pm)
                    except Exception:
                        pass
            # per-class AP и топ-K
            per_class_ap = st.get("per_class_ap") or []
            if categories and per_class_ap:
                try:
                    # пары с поддержкой
                    pairs_full = [
                        (
                            categories[i],
                            float(per_class_ap[i]),
                            int(gt_counts[i]) if i < len(gt_counts) else 0,
                        )
                        for i in range(min(len(categories), len(per_class_ap)))
                    ]
                    # Барчарт по всем классам (по убыванию AP)
                    pairs_sorted = sorted(pairs_full, key=lambda x: x[1], reverse=True)
                    names_all = [p[0] for p in pairs_sorted]
                    vals_all = [p[1] for p in pairs_sorted]
                    plt.figure(figsize=(max(6, len(names_all) * 0.5), 3))
                    plt.bar(range(len(names_all)), vals_all, color="#54A24B"); plt.title("Per-class AP (sorted)")
                    plt.xticks(range(len(names_all)), names_all, rotation=45, ha="right")
                    p_all = report_dir / "per_class_ap.svg"
                    plt.tight_layout(); plt.savefig(p_all, format="svg"); plt.close()
                    images["per_class_ap"] = str(p_all)
                    # CSV: полная таблица per-class AP с support
                    csv_all = report_dir / "per_class_ap.csv"
                    try:
                        with open(csv_all, "w", newline="", encoding="utf-8") as fcsv:
                            w = csv.writer(fcsv)
                            w.writerow(["class", "AP", "support"])
                            for name, ap, sup in pairs_sorted:
                                w.writerow([name, f"{ap:.6f}", sup])
                        images["per_class_ap_csv"] = str(csv_all)
                    except Exception:
                        pass
                    # Топ-K худших
                    k = max(1, int(self.top_k_lowest_map or 5))
                    pairs_low = sorted(pairs_full, key=lambda x: x[1])[:k]
                    names_k = [p[0] for p in pairs_low]
                    vals_k = [p[1] for p in pairs_low]
                    plt.figure(figsize=(max(6, len(names_k) * 0.6), 3))
                    plt.bar(range(len(names_k)), vals_k, color="#E45756"); plt.title(f"Lowest {k} AP classes")
                    plt.xticks(range(len(names_k)), names_k, rotation=45, ha="right")
                    p_k = report_dir / "per_class_ap_lowest.svg"
                    plt.tight_layout(); plt.savefig(p_k, format="svg"); plt.close()
                    images["per_class_ap_lowest"] = str(p_k)
                    # CSV: lowest-K с support
                    csv_k = report_dir / f"per_class_ap_lowest_{k}.csv"
                    try:
                        with open(csv_k, "w", newline="", encoding="utf-8") as fcsv:
                            w = csv.writer(fcsv)
                            w.writerow(["class", "AP", "support"])
                            for name, ap, sup in pairs_low:
                                w.writerow([name, f"{ap:.6f}", sup])
                        images["per_class_ap_lowest_csv"] = str(csv_k)
                    except Exception:
                        pass
                except Exception:
                    pass

        return images

    def _write_markdown(
        self,
        report_dir: Path,
        metrics: List[MetricOutputModel],
        timing_stats: Dict[str, Any],
        images: Dict[str, str],
        contexts: List[Context],
    ) -> None:
        lines: List[str] = []
        lines.append(f"# Detection Report\n\n")
        lines.append(f"Generated: {datetime.utcnow().isoformat()} UTC\n")
        # Описание модели (из Context.extra)
        if contexts:
            extra = getattr(contexts[0], "extra", {}) or {}
            cls_name = extra.get("detector_cls")
            module = extra.get("detector_module")
            doc = extra.get("detector_doc")
            if cls_name or module or doc:
                lines.append("\n## Model\n")
                if cls_name or module:
                    lines.append(f"- Class: {cls_name or '-'}\n")
                    lines.append(f"- Module: {module or '-'}\n")
                if doc:
                    lines.append("\n**Description (from docstring):**\n\n")
                    lines.append(doc.strip() + "\n")
        lines.append("\n## Summary\n")
        lines.append(f"- Runs: {timing_stats.get('total_runs', 0)}\n")
        lines.append(f"- Success: {timing_stats.get('success', 0)}\n")
        lines.append(f"- Errors: {timing_stats.get('errors', 0)}\n")
        if timing_stats.get("durations"):
            lines.append(
                f"- Mean/Median/Std: {timing_stats.get('mean', 0.0):.3f} / {timing_stats.get('median', 0.0):.3f} / {timing_stats.get('std', 0.0):.3f} sec\n"
            )
        # Метрики: таблица + описание из __doc__
        lines.append("\n## Metrics\n")
        lines.append("| Metric | Score |\n|---|---:|\n")
        for m in metrics or []:
            score = f"{m.score:.4f}" if isinstance(m.score, (int, float)) else str(m.score)
            lines.append(f"| {m.metric_name} | {score} |\n")
        # Описание
        for m in metrics or []:
            desc = None
            if isinstance(m.stats, dict):
                desc = m.stats.get("description")
            if not desc:
                desc = ""
            if desc:
                lines.append(f"\n### {m.metric_name}: описание\n\n")
                lines.append(desc.strip() + "\n")
        # Время
        lines.append("\n## Timing\n")
        if "durations_hist" in images:
            lines.append(f"![Durations histogram]({images['durations_hist']})\n")
        if "durations_series" in images:
            lines.append(f"![Durations series]({images['durations_series']})\n")
        if "spans_avg" in images:
            lines.append(f"![Stage avg time]({images['spans_avg']})\n")
        # Данные: распределения классов
        if self.include_class_distributions and ("gt_class_distribution" in images or "pred_class_distribution" in images):
            lines.append("\n## Data distributions\n")
            if "gt_class_distribution" in images:
                lines.append(f"![GT class distribution]({images['gt_class_distribution']})\n")
            if "pred_class_distribution" in images:
                lines.append(f"![Predicted class distribution]({images['pred_class_distribution']})\n")
        # Детализация mAP
        map_metric: Optional[MetricOutputModel] = None
        for m in metrics or []:
            if str(m.metric_name).lower() in {"map", "meanaverageprecision"}:
                map_metric = m; break
        if map_metric and isinstance(map_metric.stats, dict):
            st = map_metric.stats
            lines.append("\n## mAP details\n")
            # Краткие сводки
            for key in ["mAP", "mAP@0.5", "mAP@0.75", "mAP_small", "mAP_medium", "mAP_large"]:
                if key in st:
                    try:
                        lines.append(f"- {key}: {float(st[key]):.4f}\n")
                    except Exception:
                        lines.append(f"- {key}: {st[key]}\n")
            # AP vs IoU
            if "ap_vs_iou" in images:
                lines.append(f"\n![AP vs IoU]({images['ap_vs_iou']})\n")
            # PR curves
            for tag, title in [("050", "PR curves @IoU=0.50"), ("075", "PR curves @IoU=0.75")]:
                if self.include_ap_graphs and (f"pr_macro_{tag}" in images or f"pr_micro_{tag}" in images):
                    lines.append(f"\n### {title}\n")
                    if f"pr_macro_{tag}" in images:
                        lines.append(f"![PR macro {tag}]({images[f'pr_macro_{tag}']})\n")
                    if f"pr_micro_{tag}" in images:
                        lines.append(f"![PR micro {tag}]({images[f'pr_micro_{tag}']})\n")
            # Per-class AP
            if "per_class_ap" in images:
                lines.append(f"\n![Per-class AP]({images['per_class_ap']})\n")
                if "per_class_ap_csv" in images:
                    lines.append(f"[CSV per-class AP]({images['per_class_ap_csv']})\n")
            # TOP-K lowest AP (таблица)
            cats = st.get("categories") or []
            per_ap = st.get("per_class_ap") or []
            gt_counts = st.get("gt_counts") or []
            if cats and per_ap:
                k = max(1, int(self.top_k_lowest_map or 5))
                pairs = [(cats[i], float(per_ap[i]), int(gt_counts[i]) if i < len(gt_counts) else 0) for i in range(min(len(cats), len(per_ap)))]
                pairs_low = sorted(pairs, key=lambda x: x[1])[:k]
                if pairs_low:
                    lines.append("\n### Lowest AP classes\n")
                    lines.append("| class | AP | support |\n|---|---:|---:|\n")
                    for name, ap, sup in pairs_low:
                        lines.append(f"| {name} | {ap:.4f} | {sup} |\n")
                    if "per_class_ap_lowest" in images:
                        lines.append(f"\n![Lowest AP classes]({images['per_class_ap_lowest']})\n")
                    if "per_class_ap_lowest_csv" in images:
                        lines.append(f"[CSV lowest {k}]({images['per_class_ap_lowest_csv']})\n")
        # Classification report таблица, если есть
        for m in metrics or []:
            if m.metric_name == "classification_report" and isinstance(m.stats, dict) and isinstance(m.stats.get("dict"), dict):
                rep = m.stats["dict"]
                lines.append("\n## Classification report (sklearn)\n")
                # Пер-класс
                lines.append("| label | precision | recall | f1-score | support |\n|---|---:|---:|---:|---:|\n")
                agg_keys = {"accuracy", "macro avg", "weighted avg", "micro avg", "samples avg"}
                for k, row in rep.items():
                    if str(k) in agg_keys:
                        continue
                    if isinstance(row, dict):
                        lines.append(f"| {k} | {_fmt_float(row.get('precision', 0.0), 2)} | {_fmt_float(row.get('recall', 0.0), 2)} | {_fmt_float(row.get('f1-score', 0.0), 2)} | {row.get('support', 0)} |\n")
                # Агрегаты
                lines.append("\n| aggregate | precision | recall | f1-score | support |\n|---|---:|---:|---:|---:|\n")
                for agg in ["accuracy", "macro avg", "weighted avg", "micro avg", "samples avg"]:
                    if agg in rep:
                        row = rep[agg]
                        if isinstance(row, dict):
                            lines.append(f"| {agg} | {_fmt_float(row.get('precision', 0.0), 2)} | {_fmt_float(row.get('recall', 0.0), 2)} | {_fmt_float(row.get('f1-score', 0.0), 2)} | {row.get('support', 0)} |\n")
                        else:
                            lines.append(f"| {agg} | {_fmt_float(row, 2)} | {_fmt_float(row, 2)} | {_fmt_float(row, 2)} | - |\n")
                break

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