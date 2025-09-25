from __future__ import annotations

# from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union, Tuple
from pathlib import Path
from datetime import datetime
import os
import json
import csv
import statistics
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"  
mpl.rcParams["savefig.facecolor"] = "white"
mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["axes.facecolor"] = "white"
mpl.rcParams["savefig.transparent"] = False
from rich.console import Console
from rich.table import Table
from rich import box

from searchdet_pipeline.eval_classes.Context import Context
from searchdet_pipeline.eval_classes.metrics import MetricOutputModel
from searchdet_pipeline.eval_classes.ReportConfig import ReportConfig
from searchdet_pipeline.eval_classes.DatasetModel import DatasetModel
from searchdet_pipeline.eval_classes.COCOAnnotations import COCOAnnotation


class ReportGenerator(ReportConfig):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.console = Console()
        self.fig_w: float = 6.0
        self.fig_h: float = 3.0
        self.tick_fontsize: int = 8
        self.max_label_len: int = 14
    def _shorten_label(self, s: Any) -> str:
        t = str(s)
        return t if len(t) <= self.max_label_len else (t[: self.max_label_len - 1] + "…")
    def _shorten_labels(self, labels: List[Any]) -> List[str]:
        return [self._shorten_label(l) for l in labels]
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
        report_dir = self._prepare_report_dir(output_dir)
        images = self._save_graphs(report_dir, metrics, timing_stats, contexts)
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

        self.console.rule("Статистика времени ")
        tt = Table(box=box.SIMPLE_HEAVY)
        tt.add_column("Показатель", style="magenta")
        tt.add_column("Значение", justify="right")
        for k in [
            ("Запусков", timing_stats.get("total_runs", 0)),
            ("Успешных", timing_stats.get("success", 0)),
            ("Ошибок", timing_stats.get("errors", 0)),
            ("Среднее, сек", f"{float(timing_stats.get('mean', 0.0)):.4f}"),
            ("Медиана, сек", f"{float(timing_stats.get('median', 0.0)):.4f}"),
            ("Std, сек", f"{float(timing_stats.get('std', 0.0)):.4f}"),
            ("Мин, сек", f"{float(timing_stats.get('min', 0.0)):.4f}"),
            ("Макс, сек", f"{float(timing_stats.get('max', 0.0)):.4f}"),
        ]:
            tt.add_row(str(k[0]), str(k[1]))
        self.console.print(tt)

        spans_avg: Dict[str, float] = timing_stats.get("spans_avg", {})
        if spans_avg and self.include_spans:
            st = Table(title="Среднее время по спанам", box=box.SIMPLE_HEAVY)
            st.add_column("Этап")
            st.add_column("Среднее, сек", justify="right")
            for name, val in sorted(spans_avg.items(), key=lambda x: x[1], reverse=True):
                st.add_row(name, f"{float(val):.4f}")
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
            stats["spans_avg"] = {k: float(statistics.mean(v)) for k, v in spans.items() if v}
        else:
            stats["spans_avg"] = {}
        return stats

    def _prepare_report_dir(self, output_dir: Optional[Union[str, Path]]) -> Path:
        out = Path(output_dir) if output_dir else Path(os.getcwd()) / "report"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _save_graphs(self, report_dir: Path, metrics: List[MetricOutputModel], timing_stats: Dict[str, Any], contexts: List[Context]) -> Dict[str, str]:
        images: Dict[str, str] = {}

        def save(filename: str, key: str) -> None:
            p = report_dir / filename
            plt.tight_layout()
            plt.savefig(p, format="svg", facecolor="white", bbox_inches="tight", transparent=False, dpi=150)
            plt.close()
            images[key] = str(p)

        durations = timing_stats.get("durations", [])
        if durations:
            plt.figure(figsize=(self.fig_w, self.fig_h))
            plt.hist(durations, bins=20, color="#4C78A8")
            plt.title("Durations (sec)")
            plt.tick_params(labelsize=self.tick_fontsize)
            save("durations_hist.svg", "durations_hist")

            plt.figure(figsize=(self.fig_w, self.fig_h))
            plt.plot(durations, color="#F58518")
            plt.title("Durations by run")
            plt.tick_params(labelsize=self.tick_fontsize)
            save("durations_series.svg", "durations_series")

        spans_avg: Dict[str, float] = timing_stats.get("spans_avg", {})
        if spans_avg:
            names = list(spans_avg.keys())
            values = [spans_avg[k] for k in names]
            short_names = self._shorten_labels(names)
            plt.figure(figsize=(self.fig_w, self.fig_h))
            plt.barh(short_names, values, color="#54A24B")
            plt.title("Avg span time (sec)")
            plt.tick_params(axis="y", labelsize=self.tick_fontsize)
            plt.tick_params(axis="x", labelsize=self.tick_fontsize)
            save("spans_avg.svg", "spans_avg")

        map_metric: Optional[MetricOutputModel] = next((m for m in (metrics or []) if str(m.metric_name).lower() in {"map", "meanaverageprecision"}), None)
        if map_metric and isinstance(map_metric.stats, dict):
            st = map_metric.stats
            categories = st.get("categories") or []
            gt_counts = st.get("gt_counts") or []
            pred_counts = st.get("pred_counts") or []

            if self.include_class_distributions and categories:
                for title, counts, color, key, fname in [
                    ("GT class distribution", gt_counts, "#4C78A8", "gt_class_distribution", "gt_class_distribution.svg"),
                    ("Predicted class distribution", pred_counts, "#F58518", "pred_class_distribution", "pred_class_distribution.svg"),
                ]:
                    if counts:
                        x = list(range(len(categories)))
                        labels = [str(c) for c in categories]
                        labels = self._shorten_labels(labels)
                        plt.figure(figsize=(self.fig_w, self.fig_h))
                        plt.bar(x, counts, color=color)
                        plt.title(title)
                        plt.xticks(x, labels, rotation=45, ha="right", fontsize=self.tick_fontsize)
                        plt.tick_params(axis="y", labelsize=self.tick_fontsize)
                        save(fname, key)

            ap_macro = st.get("ap_iou_macro") or []
            ap_micro = st.get("ap_iou_micro") or []
            ious = st.get("iou_thresholds") or []
            if self.include_ap_graphs and ious:
                plt.figure(figsize=(self.fig_w, self.fig_h))
                plotted = False
                if ap_macro and len(ap_macro) == len(ious):
                    plt.plot(ious, ap_macro, label="macro", color="#4C78A8", linewidth=2, linestyle='-', marker='o', markersize=3)
                    plotted = True
                if ap_micro and len(ap_micro) == len(ious):
                    plt.plot(ious, ap_micro, label="micro", color="#F58518", linewidth=2, linestyle='--', marker='s', markersize=3)
                    plotted = True
                if plotted:
                    plt.xlabel("IoU threshold")
                    plt.ylabel("AP")
                    plt.title("AP vs IoU")
                    plt.legend(fontsize=self.tick_fontsize)
                    plt.grid(True, alpha=0.3)
                    plt.tick_params(labelsize=self.tick_fontsize)
                    save("ap_vs_iou.svg", "ap_vs_iou")
                else:
                    plt.close()

            def get_pr(d: Dict[str, Any], k: str) -> Optional[Dict[str, Any]]:
                v = d.get(k)
                if v:
                    return v
                try:
                    kf = float(k)
                    for mk in list(d.keys()):
                        try:
                            if abs(float(mk) - kf) < 1e-6:
                                return d[mk]
                        except Exception:
                            continue
                except Exception:
                    pass
                return None

            if self.include_ap_graphs:
                for key, tag in [("0.50", "050"), ("0.75", "075")]:
                    macro_dict = st.get("pr_macro", {}) or {}
                    micro_dict = st.get("pr_micro", {}) or {}
                    prM = get_pr(macro_dict, key)
                    prm = get_pr(micro_dict, key)
                    for pr, color, label, fname_key, marker in [
                        (prM, "#4C78A8", f"PR Curve (Macro) @IoU={key}", f"pr_macro_{tag}", 'o'),
                        (prm, "#F58518", f"PR Curve (Micro) @IoU={key}", f"pr_micro_{tag}", 's'),
                    ]:
                        if pr and pr.get("recall") and pr.get("precision") and len(pr["recall"]) > 1 and len(pr["precision"]) > 1:
                            plt.figure(figsize=(self.fig_w, self.fig_h))
                            plt.plot(pr["recall"], pr["precision"], color=color, linewidth=2, marker=marker, markersize=3, alpha=0.8)
                            plt.xlabel("Recall")
                            plt.ylabel("Precision")
                            plt.title(label)
                            plt.grid(True, alpha=0.3)
                            plt.xlim(0, 1)
                            plt.ylim(0, 1)
                            if key == "0.50" and st.get("mAP@0.5"):
                                ap_score = st["mAP@0.5"]
                                plt.text(0.6, 0.2, f'AP@0.5: {ap_score:.3f}', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                            if key == "0.75" and st.get("mAP@0.75"):
                                ap_score = st["mAP@0.75"]
                                plt.text(0.6, 0.2, f'AP@0.75: {ap_score:.3f}', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                            plt.tick_params(labelsize=self.tick_fontsize)
                            save(f"{fname_key}.svg", fname_key)

            # Scatter of best PR points per IoU threshold (each point is a thr)
            if self.include_ap_graphs:
                macro_pts: List[Tuple[float, float, str]] = []  # (recall, precision, label)
                micro_pts: List[Tuple[float, float, str]] = []
                prM_all = st.get("pr_macro", {}) or {}
                prm_all = st.get("pr_micro", {}) or {}
                for thr in ious:
                    key = f"{float(thr):.2f}"
                    for pr_dict, store in [(prM_all, macro_pts), (prm_all, micro_pts)]:
                        pr = pr_dict.get(key)
                        if pr and pr.get("recall") and pr.get("precision") and len(pr["recall"]) == len(pr["precision"]) and len(pr["recall"]) > 0:
                            r = np.array(pr["recall"], dtype=float)
                            p = np.array(pr["precision"], dtype=float)
                            # choose point with max F1
                            f1 = (2 * p * r) / (p + r + 1e-12)
                            idx = int(np.nanargmax(f1))
                            store.append((float(r[idx]), float(p[idx]), key))
                if macro_pts:
                    plt.figure(figsize=(self.fig_w, self.fig_h))
                    plt.scatter([x for x, _, _ in macro_pts], [y for _, y, _ in macro_pts], c="#4C78A8")
                    for x, y, label in macro_pts:
                        plt.annotate(label, (x, y), textcoords="offset points", xytext=(4, 2), fontsize=7)
                    plt.xlabel("Recall")
                    plt.ylabel("Precision")
                    plt.title("Macro PR best-points per IoU threshold")
                    plt.xlim(0, 1)
                    plt.ylim(0, 1)
                    plt.grid(True, alpha=0.3)
                    plt.tick_params(labelsize=self.tick_fontsize)
                    save("ap_pr_points_macro.svg", "ap_pr_points_macro")
                if micro_pts:
                    plt.figure(figsize=(self.fig_w, self.fig_h))
                    plt.scatter([x for x, _, _ in micro_pts], [y for _, y, _ in micro_pts], c="#F58518")
                    for x, y, label in micro_pts:
                        plt.annotate(label, (x, y), textcoords="offset points", xytext=(4, 2), fontsize=7)
                    plt.xlabel("Recall")
                    plt.ylabel("Precision")
                    plt.title("Micro PR best-points per IoU threshold")
                    plt.xlim(0, 1)
                    plt.ylim(0, 1)
                    plt.grid(True, alpha=0.3)
                    plt.tick_params(labelsize=self.tick_fontsize)
                    save("ap_pr_points_micro.svg", "ap_pr_points_micro")

            # Generate per-class AP graphs and CSVs
            per_ap = st.get("per_class_ap_avg") or st.get("per_class_ap") or []
            cats = st.get("categories") or []
            if self.include_ap_graphs and cats and per_ap:
                # Sorted per-class AP
                pairs = [(cats[i], float(per_ap[i])) for i in range(min(len(cats), len(per_ap)))]
                pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
                if pairs_sorted:
                    names, values = zip(*pairs_sorted)
                    names_short = self._shorten_labels(list(names))
                    plt.figure(figsize=(self.fig_w, self.fig_h))
                    plt.bar(range(len(names_short)), values, color="#4C78A8")
                    plt.title("Per-class AP (sorted)")
                    plt.xlabel("Class")
                    plt.ylabel("AP")
                    plt.xticks(range(len(names_short)), names_short, rotation=45, ha="right", fontsize=self.tick_fontsize)
                    plt.grid(True, alpha=0.3)
                    plt.tick_params(axis="y", labelsize=self.tick_fontsize)
                    save("per_class_ap.svg", "per_class_ap")
                    
                    # Save CSV
                    csv_path = report_dir / "per_class_ap.csv"
                    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(["class", "AP"])
                        writer.writerows(pairs_sorted)
                    images["per_class_ap_csv"] = str(csv_path)

                # Lowest k AP classes
                k = max(1, int(self.top_k_lowest_map or 5))
                pairs_low = sorted(pairs, key=lambda x: x[1])[:k]
                if pairs_low:
                    names_low, values_low = zip(*pairs_low)
                    names_low_short = self._shorten_labels(list(names_low))
                    plt.figure(figsize=(self.fig_w, self.fig_h))
                    plt.bar(range(len(names_low_short)), values_low, color="#E45756")
                    plt.title(f"Lowest {k} AP classes")
                    plt.xlabel("Class")
                    plt.ylabel("AP")
                    plt.xticks(range(len(names_low_short)), names_low_short, rotation=45, ha="right", fontsize=self.tick_fontsize)
                    plt.grid(True, alpha=0.3)
                    plt.tick_params(axis="y", labelsize=self.tick_fontsize)
                    save("per_class_ap_lowest.svg", "per_class_ap_lowest")
                    
                    # Save CSV
                    csv_path_low = report_dir / "per_class_ap_lowest.csv"
                    with open(csv_path_low, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(["class", "AP"])
                        writer.writerows(pairs_low)
                    images["per_class_ap_lowest_csv"] = str(csv_path_low)

        return images

    def _write_markdown(self, report_dir: Path, metrics: List[MetricOutputModel], timing_stats: Dict[str, Any], images: Dict[str, str], contexts: List[Context]) -> None:
        lines: List[str] = []
        
        # Header
        lines.append("# Отчёт по оценке модели\n\n")
        lines.append(f"Сгенерирован: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Model info from context.extra
        if contexts:
            for ctx in contexts:
                if hasattr(ctx, 'extra') and ctx.extra:
                    model_class = ctx.extra.get('model_class')
                    model_module = ctx.extra.get('model_module') 
                    model_doc = ctx.extra.get('model_doc')
                    if model_class:
                        lines.append(f"**Модель:** {model_class}\n")
                    if model_module:
                        lines.append(f"**Модуль:** {model_module}\n")
                    if model_doc:
                        lines.append(f"**Описание:** {model_doc}\n")
                    lines.append("\n")
                    break

        # Timing stats
        lines.append("## Статистика времени выполнения\n\n")
        lines.append(f"- Всего запусков: {timing_stats.get('total_runs', 0)}\n")
        lines.append(f"- Успешных: {timing_stats.get('success', 0)}\n")
        lines.append(f"- Ошибок: {timing_stats.get('errors', 0)}\n")
        if timing_stats.get('mean'):
            lines.append(f"- Среднее время: {timing_stats['mean']:.4f} сек\n")
        if timing_stats.get('median'):
            lines.append(f"- Медиана: {timing_stats['median']:.4f} сек\n")
        lines.append("\n")

        # Metrics summary
        lines.append("## Метрики\n\n")
        for m in metrics or []:
            # Auto-generate description from metric docstring if available
            desc = ""
            if hasattr(m, 'stats') and isinstance(m.stats, dict):
                desc = m.stats.get('doc', '')
            
            lines.append(f"### {m.metric_name}\n")
            if desc:
                lines.append(f"{desc}\n\n")
            
            score = f"{m.score:.4f}" if isinstance(m.score, (int, float)) else str(m.score)
            lines.append(f"**Значение:** {score}\n\n")

        # Graphs section
        if images:
            lines.append("## Графики\n\n")
            
            # Duration graphs
            if images.get("durations_hist"):
                lines.append("Гистограмма времени: показывает распределение длительности запусков 'Histogram(duration_values, bins=20)'\n\n")
                lines.append(f"![Гистограмма времени]({images['durations_hist']})\n\n")
            if images.get("durations_series"):
                lines.append("Время по запускам: показывает изменение времени выполнения 'plot(run_index, duration)'\n\n")
                lines.append(f"![Время по запускам]({images['durations_series']})\n\n")
            if images.get("spans_avg"):
                lines.append("Среднее время этапов: показывает среднюю длительность каждого этапа 'mean(span_durations)'\n\n")
                lines.append(f"![Среднее время этапов]({images['spans_avg']})\n\n")
            
            # Class distributions
            if images.get("gt_class_distribution"):
                lines.append("Распределение GT: показывает количество объектов по классам 'bar(classes, gt_counts)'\n\n")
                lines.append(f"![Распределение GT]({images['gt_class_distribution']})\n\n")
            if images.get("pred_class_distribution"):
                lines.append("Распределение предсказаний: показывает количество предсказанных объектов 'bar(classes, pred_counts)'\n\n")
                lines.append(f"![Распределение предсказаний]({images['pred_class_distribution']})\n\n")
            
            if images.get("ap_vs_iou"):
                lines.append("AP vs IoU: показывает зависимость точности от порога пересечения 'plot(iou_thresholds, ap_values)'\n\n")
                lines.append(f"![AP vs IoU]({images['ap_vs_iou']})\n\n")
            
            # PR curves formulas only
            for tag in ["050", "075"]:
                if images.get(f"pr_macro_{tag}"):
                    lines.append(f"PR макро @{tag[0]}.{tag[1:]}: показывает точность и полноту 'Precision = TP/(TP+FP), Recall = TP/(TP+FN)'\n\n")
                    lines.append(f"![PR макро @{tag[0]}.{tag[1:]}]({images[f'pr_macro_{tag}']})\n\n")
                if images.get(f"pr_micro_{tag}"):
                    lines.append(f"PR микро @{tag[0]}.{tag[1:]}: показывает агрегированную точность 'Precision_micro = Σ(TP_i)/Σ(TP_i+FP_i)'\n\n")
                    lines.append(f"![PR микро @{tag[0]}.{tag[1:]}]({images[f'pr_micro_{tag}']})\n\n")
            
            # PR best-points formulas only
            if images.get("ap_pr_points_macro"):
                lines.append("Лучшие точки PR макро: показывает оптимальные точки по F1 'F1 = 2·P·R/(P+R)'\n\n")
                lines.append(f"![Лучшие точки PR макро]({images['ap_pr_points_macro']})\n\n")
            if images.get("ap_pr_points_micro"):
                lines.append("Лучшие точки PR микро: показывает оптимальные микро точки 'F1_micro = 2·P_micro·R_micro/(P_micro+R_micro)'\n\n")
                lines.append(f"![Лучшие точки PR микро]({images['ap_pr_points_micro']})\n\n")
            
            # Per-class AP formula only
            if images.get("per_class_ap"):
                lines.append("AP по классам: показывает точность для каждого класса 'AP_class = ∫₀¹ P(R) dR'\n\n")
                lines.append(f"![AP по классам]({images['per_class_ap']})\n\n")

        # mAP numeric details
        map_metric = next((m for m in (metrics or []) if str(m.metric_name).lower() in {"map", "meanaverageprecision"}), None)
        if map_metric and isinstance(map_metric.stats, dict):
            st = map_metric.stats
            lines.append("## Детали mAP\n\n")
            for key in ["mAP", "mAP@0.5", "mAP@0.75", "mAP_small", "mAP_medium", "mAP_large"]:
                if key in st:
                    try:
                        lines.append(f"- {key}: {float(st[key]):.4f}\n")
                    except Exception:
                        lines.append(f"- {key}: {st[key]}\n")
            
            # Lowest AP classes table
            cats = st.get("categories") or []
            per_ap = st.get("per_class_ap_avg") or st.get("per_class_ap") or []
            gt_counts = st.get("gt_counts") or []
            if cats and per_ap:
                k = max(1, int(self.top_k_lowest_map or 5))
                pairs = [(cats[i], float(per_ap[i]), int(gt_counts[i]) if i < len(gt_counts) else 0) for i in range(min(len(cats), len(per_ap)))]
                pairs_low = sorted(pairs, key=lambda x: x[1])[:k]
                if pairs_low:
                    lines.append("\n### Низшие по AP классы\n\n")
                    lines.append("| class | AP | support |\n|---|---:|---:|\n")
                    for name, ap, sup in pairs_low:
                        lines.append(f"| {name} | {ap:.4f} | {sup} |\n")
            lines.append("\n")
        cr = next((m for m in (metrics or []) if m.metric_name == "classification_report" and isinstance(m.stats, dict)), None)
        if cr:
            text = cr.stats.get("text")
            if isinstance(text, str) and text.strip():
                lines.append("## Classification report (sklearn)\n\n")
                lines.append("```\n")
                lines.append(text)
                if not text.endswith("\n"):
                    lines.append("\n")
                lines.append("```\n")

        (report_dir / "report.md").write_text("".join(lines), encoding="utf-8")

    def _write_json(self, report_dir: Path, metrics: List[MetricOutputModel], timing_stats: Dict[str, Any]) -> None:
        data = {
            "metrics": [
                {"metric_name": m.metric_name, "score": m.score, "stats": m.stats} for m in (metrics or [])
            ],
            "timing": timing_stats,
        }
        (report_dir / "report.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")