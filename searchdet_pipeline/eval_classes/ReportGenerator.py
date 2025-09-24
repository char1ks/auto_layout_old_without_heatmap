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

    def _save_graphs(self, report_dir: Path, timing_stats: Dict[str, Any], metrics: List[MetricOutputModel]) -> Dict[str, str]:
        images: Dict[str, str] = {}

        def save(filename: str, key: str) -> None:
            p = report_dir / filename
            plt.tight_layout()
            plt.savefig(p, format="svg", facecolor="white", bbox_inches="tight", transparent=False, dpi=150)
            plt.close()
            images[key] = str(p)

        durations = timing_stats.get("durations", [])
        if durations:
            plt.figure(figsize=(6, 3))
            plt.hist(durations, bins=20, color="#4C78A8")
            plt.title("Durations (sec)")
            save("durations_hist.svg", "durations_hist")

            plt.figure(figsize=(6, 3))
            plt.plot(durations, color="#F58518")
            plt.title("Durations by run")
            save("durations_series.svg", "durations_series")

        spans_avg: Dict[str, float] = timing_stats.get("spans_avg", {})
        if spans_avg:
            names = list(spans_avg.keys())
            values = [spans_avg[k] for k in names]
            plt.figure(figsize=(6, 3))
            plt.barh(names, values, color="#54A24B")
            plt.title("Avg span time (sec)")
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
                        plt.figure(figsize=(max(6, len(labels) * 0.5), 3))
                        plt.bar(x, counts, color=color)
                        plt.title(title)
                        plt.xticks(x, labels, rotation=45, ha="right")
                        save(fname, key)

            ap_macro = st.get("ap_iou_macro") or []
            ap_micro = st.get("ap_iou_micro") or []
            ious = st.get("iou_thresholds") or []
            if self.include_ap_graphs and ious:
                plt.figure(figsize=(6, 3))
                plotted = False
                if ap_macro and len(ap_macro) == len(ious):
                    plt.plot(ious, ap_macro, label="macro", color="#4C78A8", linewidth=2, linestyle='-', marker='o', markersize=4)
                    plotted = True
                if ap_micro and len(ap_micro) == len(ious):
                    plt.plot(ious, ap_micro, label="micro", color="#F58518", linewidth=2, linestyle='--', marker='s', markersize=4)
                    plotted = True
                if plotted:
                    plt.xlabel("IoU threshold")
                    plt.ylabel("AP")
                    plt.title("AP vs IoU")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
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
                            plt.figure(figsize=(8, 6))
                            plt.plot(pr["recall"], pr["precision"], color=color, linewidth=2, marker=marker, markersize=3, alpha=0.8)
                            plt.xlabel("Recall")
                            plt.ylabel("Precision")
                            plt.title(label)
                            plt.grid(True, alpha=0.3)
                            plt.xlim(0, 1)
                            plt.ylim(0, 1)
                            if key == "0.50" and st.get("mAP@0.5"):
                                ap_score = st["mAP@0.5"]
                                plt.text(0.6, 0.2, f'AP@0.5: {ap_score:.3f}', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                            if key == "0.75" and st.get("mAP@0.75"):
                                ap_score = st["mAP@0.75"]
                                plt.text(0.6, 0.2, f'AP@0.75: {ap_score:.3f}', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                            save(f"{fname_key}.svg", fname_key)

            per_class_ap = st.get("per_class_ap") or []
            if categories and per_class_ap:
                pairs = [
                    (categories[i], float(per_class_ap[i]), int(gt_counts[i]) if i < len(gt_counts) else 0)
                    for i in range(min(len(categories), len(per_class_ap)))
                ]
                pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
                names_all = [p[0] for p in pairs_sorted]
                vals_all = [p[1] for p in pairs_sorted]
                plt.figure(figsize=(max(6, len(names_all) * 0.5), 3))
                plt.bar(range(len(names_all)), vals_all, color="#54A24B")
                plt.title("Per-class AP (sorted)")
                plt.xticks(range(len(names_all)), names_all, rotation=45, ha="right")
                save("per_class_ap.svg", "per_class_ap")
                csv_all = report_dir / "per_class_ap.csv"
                try:
                    with open(csv_all, "w", newline="", encoding="utf-8") as fcsv:
                        w = csv.writer(fcsv)
                        w.writerow(["class", "AP", "support"])
                        [w.writerow([n, f"{a:.6f}", s]) for n, a, s in pairs_sorted]
                    images["per_class_ap_csv"] = str(csv_all)
                except Exception:
                    pass

                k = max(1, int(self.top_k_lowest_map or 5))
                pairs_low = sorted(pairs, key=lambda x: x[1])[:k]
                names_k = [p[0] for p in pairs_low]
                vals_k = [p[1] for p in pairs_low]
                plt.figure(figsize=(max(6, len(names_k) * 0.6), 3))
                plt.bar(range(len(names_k)), vals_k, color="#E45756")
                plt.title(f"Lowest {k} AP classes")
                plt.xticks(range(len(names_k)), names_k, rotation=45, ha="right")
                save("per_class_ap_lowest.svg", "per_class_ap_lowest")
                csv_k = report_dir / f"per_class_ap_lowest_{k}.csv"
                try:
                    with open(csv_k, "w", newline="", encoding="utf-8") as fcsv:
                        w = csv.writer(fcsv)
                        w.writerow(["class", "AP", "support"])
                        [w.writerow([n, f"{a:.6f}", s]) for n, a, s in pairs_low]
                    images["per_class_ap_lowest_csv"] = str(csv_k)
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
        lines.append("\n## Metrics\n")
        lines.append("| Metric | Score |\n|---|---:|\n")
        for m in metrics or []:
            score = f"{m.score:.4f}" if isinstance(m.score, (int, float)) else str(m.score)
            lines.append(f"| {m.metric_name} | {score} |\n")
        for m in metrics or []:
            desc = None
            if isinstance(m.stats, dict):
                desc = m.stats.get("description")
            if not desc:
                desc = ""
            if desc:
                lines.append(f"\n### {m.metric_name}: описание\n\n")
                lines.append(desc.strip() + "\n")
        lines.append("\n## Timing\n")
        if "durations_hist" in images:
            lines.append(f"![Durations histogram]({images['durations_hist']})\n")
        if "durations_series" in images:
            lines.append(f"![Durations series]({images['durations_series']})\n")
        if "spans_avg" in images:
            lines.append(f"![Stage avg time]({images['spans_avg']})\n")
        if self.include_class_distributions:
            dist_imgs = [("GT class distribution", images.get("gt_class_distribution")), ("Predicted class distribution", images.get("pred_class_distribution"))]
            dist_imgs = [(title, url) for title, url in dist_imgs if url]
            if dist_imgs:
                lines.append("\n## Data distributions\n")
                for title, url in dist_imgs:
                    lines.append(f"![{title}]({url})\n")
        map_metric: Optional[MetricOutputModel] = next((m for m in (metrics or []) if str(m.metric_name).lower() in {"map", "meanaverageprecision"}), None)
        if map_metric and isinstance(map_metric.stats, dict):
            st = map_metric.stats
            lines.append("\n## mAP details\n")
            for key in ["mAP", "mAP@0.5", "mAP@0.75", "mAP_small", "mAP_medium", "mAP_large"]:
                if key in st:
                    lines.append(f"- {key}: {float(st[key]):.4f}\n")
            if "ap_vs_iou" in images:
                lines.append(f"\n![AP vs IoU]({images['ap_vs_iou']})\n")
            for tag, title in [("050", "PR curves @IoU=0.50"), ("075", "PR curves @IoU=0.75")]:
                if not self.include_ap_graphs:
                    continue
                prs = [(f"PR macro {tag}", images.get(f"pr_macro_{tag}")), (f"PR micro {tag}", images.get(f"pr_micro_{tag}"))]
                prs = [(t, url) for t, url in prs if url]
                if not prs:
                    continue
                lines.append(f"\n### {title}\n")
                for t, url in prs:
                    lines.append(f"![{t}]({url})\n")
            if "per_class_ap" in images:
                lines.append(f"\n![Per-class AP]({images['per_class_ap']})\n")
                if "per_class_ap_csv" in images:
                    lines.append(f"[CSV per-class AP]({images['per_class_ap_csv']})\n")
    
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
        cr = next((m for m in (metrics or []) if m.metric_name == "classification_report" and isinstance(m.stats, dict)), None)
        if cr:
            text = cr.stats.get("text")
            if isinstance(text, str) and text.strip():
                lines.append("\n## Classification report (sklearn)\n")
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