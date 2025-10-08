from __future__ import annotations
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
import json
import math
import os
from statistics import mean, median, pstdev
import matplotlib as mpl
import matplotlib.pyplot as plt
from rich import box
from rich.console import Console
from rich.table import Table
from evaluation.report_config import ReportConfig

def _savefig(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


class ReportGenerator(ReportConfig):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.console = Console()
        self.figure_width, self.figure_height = 6.0, 3.2
        self.tick_font_size, self.max_label_length = 8, 14
        mpl.rcParams["svg.fonttype"] = self.svg_fonttype
        mpl.rcParams["savefig.facecolor"] = self.savefig_facecolor
        mpl.rcParams["figure.facecolor"] = self.figure_facecolor
        mpl.rcParams["axes.facecolor"] = self.axes_facecolor
        mpl.rcParams["savefig.transparent"] = self.savefig_transparent

    def generate_report(
        self,
        contexts: List[Any],
        metrics: List[Any],
        dump_report: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Path]:
        timing = self._collect_timing_stats(contexts)
        if dump_report:
            self._terminal_report(metrics, timing, contexts)

        report_dir = Path(output_dir) if output_dir else Path(os.getcwd()) / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        images = self._save_graphs(report_dir, metrics, timing, contexts)
        self._write_markdown(report_dir, metrics, timing, images, contexts)
        self._write_json(report_dir, metrics, timing)
        if dump_report:
            self.console.print(f"[green]Отчёт сохранён в {report_dir.resolve()}[/green]")
        return report_dir if dump_report else None

    def _collect_timing_stats(self, contexts: List[Any]) -> Dict[str, Any]:
        durations: List[float] = [float(c.duration) for c in (contexts or [])]
        errors: List[Any] = [c.error for c in (contexts or []) if c.error]
        return {
            "total_runs": len(contexts or []),
            "success": len(durations),
            "errors": len(errors),
            "mean": mean(durations) if durations else 0.0,
            "median": median(durations) if durations else 0.0,
            "std": pstdev(durations) if len(durations) > 1 else 0.0,
            "min": min(durations) if durations else 0.0,
            "max": max(durations) if durations else 0.0,
            "p95": _percentile(durations, 95) if durations else 0.0,
            "p99": _percentile(durations, 99) if durations else 0.0,
            "durations": durations,
        }

    def _terminal_report(self, metrics: List[Any], timing: Dict[str, Any], contexts: List[Any]) -> None:
        self.console.rule("[bold]Timing[/bold]")
        t = Table(title="Время исполнения", box=box.SIMPLE, show_lines=False)
        t.add_column("Показатель"); t.add_column("Значение", justify="right")
        for k, v in [
            ("Запусков", timing.get("total_runs", 0)),
            ("Успешных", timing.get("success", 0)),
            ("Ошибок", timing.get("errors", 0)),
            ("Среднее, сек", f"{float(timing.get('mean', 0.0)):.4f}"),
            ("Медиана, сек", f"{float(timing.get('median', 0.0)):.4f}"),
            ("Std, сек", f"{float(timing.get('std', 0.0)):.4f}"),
            ("Мин, сек", f"{float(timing.get('min', 0.0)):.4f}"),
            ("Макс, сек", f"{float(timing.get('max', 0.0)):.4f}"),
            ("p95, сек", f"{float(timing.get('p95', 0.0)):.4f}"),
            ("p99, сек", f"{float(timing.get('p99', 0.0)):.4f}"),
        ]:
            t.add_row(str(k), str(v))
        self.console.print(t)

        if metrics:
            self.console.rule("[bold]Metrics[/bold]")
            mtab = Table(box=box.SIMPLE)
            mtab.add_column("Metric"); mtab.add_column("Score", justify="right")
            for m in metrics or []:
                name = m.metric_name
                score = float(m.score)
                mtab.add_row(str(name), f"{score:.4f}")
            self.console.print(mtab)

        if self.include_errors:
            errs = [c.error for c in (contexts or []) if c.error]
            if errs:
                self.console.rule("[bold]Ошибки[/bold]")
                et = Table(box=box.SIMPLE)
                et.add_column("Ошибка")
                for err in errs[: int(self.top_k_errors)]:
                    et.add_row(str(err))
                self.console.print(et)

    def _save_graphs(self, out: Path, metrics: List[Any], timing: Dict[str, Any], contexts: List[Any]) -> Dict[str, Path]:
        images: Dict[str, Path] = {}

        if timing.get("durations"):
            plt.figure(figsize=(self.figure_width, self.figure_height))
            plt.hist(timing["durations"], bins=min(30, max(5, int(math.sqrt(len(timing["durations"]))))), alpha=0.8)
            plt.xlabel("Seconds"); plt.ylabel("Count"); plt.title("Durations")
            images["durations_hist"] = _savefig(out / "durations_hist.svg")
            plt.figure(figsize=(self.figure_width, self.figure_height))
            xs = list(range(1, len(timing["durations"]) + 1))
            plt.plot(xs, timing["durations"], marker="o", linewidth=1)
            plt.xlabel("Run #"); plt.ylabel("Seconds"); plt.title("Durations (series)")
            images["durations_series"] = _savefig(out / "durations_series.svg")

        if self.include_spans:
            spans: Dict[str, List[float]] = {}
            for c in contexts or []:
                sd = c.spans or []
                for s in sd:
                    # s ожидается как dict из Context.span
                    name = str(s.get("name", "-"))
                    dur = s.get("duration", None)
                    if dur is None:
                        continue
                    try:
                        spans.setdefault(name, []).append(float(dur))
                    except (TypeError, ValueError):
                        continue
            if spans:
                keys = list(spans.keys())
                vals = [mean(spans[k]) for k in keys]
                labels = [k if len(k) <= self.max_label_length else (k[: self.max_label_length - 1] + "…") for k in keys]
                plt.figure(figsize=(self.figure_width, self.figure_height))
                plt.barh(labels, vals)
                plt.title("Avg span time (sec)"); plt.tick_params(axis="both", labelsize=self.tick_font_size)
                images["spans_avg"] = _savefig(out / "spans_avg.svg")

        mm = _find_map_metric(metrics)
        if mm:
            st_obj = mm.stats
            cats = st_obj.categories
            for title, counts, stem in [
                ("GT class distribution", (st_obj.gt_class_counts), "gt_class_distribution"),
                ("Predicted class distribution", (st_obj.pred_class_counts), "pred_class_distribution"),
            ]:
                counts = counts if counts is not None else []
                if cats and counts and len(cats) == len(counts):
                    xs = list(range(len(cats)))
                    labels = [str(c) for c in cats]
                    labels = [l if len(l) <= self.max_label_length else l[: self.max_label_length - 1] + "…" for l in labels]
                    plt.figure(figsize=(self.figure_width, self.figure_height))
                    plt.bar(xs, counts)
                    plt.xticks(xs, labels, rotation=45, ha="right", fontsize=self.tick_font_size)
                    plt.title(title)
                    images[stem] = _savefig(out / f"{stem}.svg")

            if self.include_ap_graphs:
                for iou_target, tag in [(0.50, "050"), (0.75, "075")]:
                    for curves_attr, stem in [("pr_macro_curves", "pr_macro"), ("pr_micro_curves", "pr_micro")]:
                        curves = (st_obj.pr_macro_curves if curves_attr == "pr_macro_curves" else st_obj.pr_micro_curves) or []
                        if curves:
                            c = min(curves, key=lambda x: abs(float(x.iou_threshold) - float(iou_target)))
                            rec = c.recall
                            prec = c.precision
                            if rec and prec and len(rec) > 1:
                                plt.figure(figsize=(self.figure_width, self.figure_height))
                                plt.plot(rec, prec)
                                plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR Curve ({'Macro' if curves_attr=='pr_macro_curves' else 'Micro'}) @IoU={iou_target:.2f}")
                                images[f"{stem}_{tag}"] = _savefig(out / f"{stem}_{tag}.svg")

                ious = st_obj.iou_thresholds
                apM = st_obj.ap_iou_macro
                apm = st_obj.ap_iou_micro 
                if ious and (apM or apm):
                    plt.figure(figsize=(self.figure_width, self.figure_height))
                    if apM: plt.plot(ious, apM, label="Macro")
                    if apm: plt.plot(ious, apm, label="Micro")
                    plt.xlabel("IoU"); plt.ylabel("AP"); plt.title("AP vs IoU"); plt.legend()
                    images["ap_vs_iou"] = _savefig(out / "ap_vs_iou.svg")

                def _best_f1_points(curves_list: List[Any]) -> List[tuple]:
                    pts: List[tuple] = []
                    for c in curves_list or []:
                        rec = c.recall or []
                        prec = c.precision or []
                        if not rec or not prec or len(rec) != len(prec):
                            continue
                        best_i = 0; best_f1 = -1.0
                        for i in range(len(rec)):
                            r = float(rec[i]); p = float(prec[i])
                            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
                            if f1 > best_f1:
                                best_f1 = f1; best_i = i
                        pts.append((float(rec[best_i]), float(prec[best_i]), float(c.iou_threshold)))
                    return pts

                macro_points = _best_f1_points(st_obj.pr_macro_curves or [])
                micro_points = _best_f1_points(st_obj.pr_micro_curves or [])
                if macro_points:
                    plt.figure(figsize=(self.figure_width, self.figure_height))
                    plt.scatter([r for r, p, _ in macro_points], [p for r, p, _ in macro_points], s=20)
                    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Best F1 points (Macro)")
                    images["ap_pr_points_macro"] = _savefig(out / "ap_pr_points_macro.svg")
                if micro_points:
                    plt.figure(figsize=(self.figure_width, self.figure_height))
                    plt.scatter([r for r, p, _ in micro_points], [p for r, p, _ in micro_points], s=20)
                    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Best F1 points (Micro)")
                    images["ap_pr_points_micro"] = _savefig(out / "ap_pr_points_micro.svg")

                per_class = st_obj.per_class or []
                cats_map = {str(c): i for i, c in enumerate(cats)} if cats else {}
                gt_counts = st_obj.gt_class_counts or []

                items: List[tuple] = []  
                if per_class:
                    for pc in per_class:
                        label = str(pc.label)
                        ap = float(pc.ap)
                        support = int(pc.support)
                        items.append((label, ap, support))

                if items:
                    items_desc = sorted(items, key=lambda x: x[1], reverse=True)
                    labels = [i[0] for i in items_desc]
                    aps = [i[1] for i in items_desc]
                    show_labels = [l if len(l) <= self.max_label_length else (l[: self.max_label_length - 1] + "…") for l in labels]
                    plt.figure(figsize=(self.figure_width, self.figure_height))
                    xs = list(range(len(items_desc)))
                    plt.bar(xs, aps)
                    plt.xticks(xs, show_labels, rotation=45, ha="right", fontsize=self.tick_font_size)
                    plt.ylabel("AP"); plt.title("Per-class AP")
                    images["per_class_ap"] = _savefig(out / "per_class_ap.svg")

                    csv_lines = ["class,ap,support"] + [f"{labels[i]},{aps[i]:.6f},{items_desc[i][2]}" for i in range(len(items_desc))]
                    (out / "per_class_ap.csv").write_text("\n".join(csv_lines), encoding="utf-8")
                    items_asc = sorted(items, key=lambda x: x[1])[: int(self.top_k_lowest_map)]
                    if items_asc:
                        labels_low = [i[0] for i in items_asc]
                        aps_low = [i[1] for i in items_asc]
                        show_labels_low = [l if len(l) <= self.max_label_length else (l[: self.max_label_length - 1] + "…") for l in labels_low]
                        plt.figure(figsize=(self.figure_width, self.figure_height))
                        xs = list(range(len(items_asc)))
                        plt.bar(xs, aps_low, color="#d9534f")
                        plt.xticks(xs, show_labels_low, rotation=45, ha="right", fontsize=self.tick_font_size)
                        plt.ylabel("AP"); plt.title("Lowest AP classes")
                        images["per_class_ap_lowest"] = _savefig(out / "per_class_ap_lowest.svg")

                        csv_lines_low = ["class,ap,support"] + [f"{items_asc[i][0]},{items_asc[i][1]:.6f},{items_asc[i][2]}" for i in range(len(items_asc))]
                        (out / "per_class_ap_lowest.csv").write_text("\n".join(csv_lines_low), encoding="utf-8")
        miou_metric = next((m for m in (metrics or []) if str(m.metric_name) == "mIoU"), None)
        if miou_metric is not None:
            st_obj = miou_metric.stats
            per_class_iou = st_obj.per_class or []
            if per_class_iou:
                items_iou = [(str(pc.label), float(pc.iou)) for pc in per_class_iou]
                items_iou_desc = sorted(items_iou, key=lambda x: x[1], reverse=True)
                labels_iou = [i[0] for i in items_iou_desc]
                vals_iou = [i[1] for i in items_iou_desc]
                show_labels = [l if len(l) <= self.max_label_length else (l[: self.max_label_length - 1] + "…") for l in labels_iou]
                plt.figure(figsize=(self.figure_width, self.figure_height))
                xs = list(range(len(items_iou_desc)))
                plt.bar(xs, vals_iou)
                plt.xticks(xs, show_labels, rotation=45, ha="right", fontsize=self.tick_font_size)
                plt.ylabel("IoU"); plt.title("Per-class IoU")
                images["per_class_iou"] = _savefig(out / "per_class_iou.svg")

                csv_lines_iou = ["class,iou"] + [f"{labels_iou[i]},{vals_iou[i]:.6f}" for i in range(len(items_iou_desc))]
                (out / "per_class_iou.csv").write_text("\n".join(csv_lines_iou), encoding="utf-8")

        return images

    def _write_markdown(self, out: Path, metrics: List[Any], timing: Dict[str, Any], images: Dict[str, Path], contexts: List[Any]) -> None:
        lines: List[str] = []
        add = lines.append

        add(f"# Report ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n\n")

        add("## Timing\n\n")
        add("| Metric | Value |\n|---|---:|\n")
        for k in ["total_runs", "success", "errors", "mean", "median", "std", "min", "max", "p95", "p99"]:
            add(f"| {k} | {timing.get(k, 0)} |\n")

        if images:
            add("\n## Plots\n\n")
            for name, p in images.items():
                add(f"**{name}**\n\n![]({p.name})\n\n")

        if metrics:
            add("\n## Metrics\n\n")
            add("| Metric | Score |\n|---|---:|\n")
            for m in metrics or []:
                add(f"| {m.metric_name} | {float(m.score):.4f} |\n")

            if self.include_detector_breakdown:
                mm = _find_map_metric(metrics)
                if mm:
                    st_obj = mm.stats
                    per_class = st_obj.per_class or []
                    if per_class:
                        items = sorted(((
                            pc.label,
                            float(pc.ap),
                            int(pc.support),
                        ) for pc in per_class), key=lambda x: x[1])
                        low = items[: int(self.top_k_lowest_map)]
                        if low:
                            add("\n### Низшие по AP классы\n\n| class | AP | support |\n|---|---:|---:|\n")
                            for name, ap, sup in low:
                                add(f"| {name} | {ap:.4f} | {sup} |\n")
        (out / "report.md").write_text("".join(lines), encoding="utf-8")

    def _write_json(self, out: Path, metrics: List[Any], timing: Dict[str, Any]) -> None:
        payload = {
            "metrics": [
                {
                    "metric_name": m.metric_name,
                    "score": float(m.score),
                    "stats": (asdict(m.stats) if is_dataclass(m.stats) else (m.stats if isinstance(m.stats, dict) else {"value": m.stats})),
                }
                for m in (metrics or [])
            ],
            "timing": timing,
        }
        (out / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _find_map_metric(metrics: Iterable[Any]) -> Optional[Any]:
    for m in metrics or []:
        try:
            name = str(m.metric_name).lower()
        except Exception:
            continue
        if name in {"map", "meanaverageprecision", "mean_average_precision", "mean-average-precision"}:
            return m
    return None



def _percentile(data: List[float], p: int | float) -> float:
    if not data:
        return 0.0
    xs = sorted(float(x) for x in data)
    k = (len(xs) - 1) * (float(p) / 100.0)
    f = math.floor(k); c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    d0 = xs[f] * (c - k); d1 = xs[c] * (k - f)
    return float(d0 + d1)
