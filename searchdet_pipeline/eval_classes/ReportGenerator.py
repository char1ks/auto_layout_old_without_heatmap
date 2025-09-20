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
# Matplotlib SVG export settings to avoid black/transparent areas
mpl.rcParams["svg.fonttype"] = "none"  # keep text as text (no path conversion)
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
        # Подробный режим: сохраняем полный отчёт с графиками и JSON
        report_dir = self._prepare_report_dir(output_dir)
        images = self._save_graphs(report_dir, timing_stats, metrics)
        self._write_markdown(report_dir, metrics, timing_stats, images, contexts)
        self._write_json(report_dir, metrics, timing_stats)
        self.console.print(f"[green]Отчёт сохранён в[/green] {report_dir}")
        return report_dir

    def debug_map_calculation(
        self, 
        gt: DatasetModel, 
        predictions: List[COCOAnnotation], 
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Детальная отладка вычисления mAP с выводом всех промежуточных данных
        """
        self.console.rule("[bold red]DEBUG: mAP CALCULATION ANALYSIS")
        
        debug_info = {
            "gt_analysis": {},
            "prediction_analysis": {},
            "file_matching": {},
            "class_matching": {},
            "bbox_analysis": {},
            "errors": []
        }
        
        try:
            # Анализ GT данных
            self.console.print("[bold cyan]1. АНАЛИЗ GT ДАННЫХ")
            gt_files = set()
            gt_classes = set()
            gt_annotations_count = 0
            
            for i, dp in enumerate(gt.data_points or []):
                file_name = getattr(dp, 'file_name', None)
                if file_name:
                    normalized_file = os.path.basename(str(file_name))
                    gt_files.add(normalized_file)
                    self.console.print(f"  GT[{i}]: file='{file_name}' -> normalized='{normalized_file}'")
                    
                    annotations = getattr(dp, 'annotations', [])
                    for j, ann in enumerate(annotations):
                        label = getattr(ann, 'label', getattr(dp, 'label', None))
                        if label:
                            gt_classes.add(str(label))
                            gt_annotations_count += 1
                            bbox = getattr(ann, 'bbox', None)
                            self.console.print(f"    Ann[{j}]: label='{label}', bbox={bbox}")
                else:
                    self.console.print(f"  GT[{i}]: [red]NO FILE_NAME[/red]")
            
            debug_info["gt_analysis"] = {
                "total_datapoints": len(gt.data_points or []),
                "unique_files": list(gt_files),
                "unique_classes": list(gt_classes),
                "total_annotations": gt_annotations_count
            }
            
            self.console.print(f"[green]GT Summary: {len(gt_files)} files, {len(gt_classes)} classes, {gt_annotations_count} annotations[/green]")
            
            # Анализ предсказаний
            self.console.print("\n[bold cyan]2. АНАЛИЗ ПРЕДСКАЗАНИЙ")
            pred_files = set()
            pred_classes = set()
            pred_annotations_count = len(predictions or [])
            
            for i, pred in enumerate(predictions or []):
                file_name = getattr(pred, 'file_name', None)
                label = getattr(pred, 'label', None)
                score = getattr(pred, 'score', getattr(pred, 'confidence', None))
                bbox = getattr(pred, 'bbox', None)
                
                if file_name:
                    normalized_file = os.path.basename(str(file_name))
                    pred_files.add(normalized_file)
                else:
                    normalized_file = "NO_FILE"
                
                if label:
                    pred_classes.add(str(label))
                
                self.console.print(f"  Pred[{i}]: file='{file_name}' -> '{normalized_file}', label='{label}', score={score}, bbox={bbox}")
            
            debug_info["prediction_analysis"] = {
                "total_predictions": pred_annotations_count,
                "unique_files": list(pred_files),
                "unique_classes": list(pred_classes)
            }
            
            self.console.print(f"[green]Predictions Summary: {len(pred_files)} files, {len(pred_classes)} classes, {pred_annotations_count} predictions[/green]")
            
            # Анализ совпадения файлов
            self.console.print("\n[bold cyan]3. АНАЛИЗ СОВПАДЕНИЯ ФАЙЛОВ")
            common_files = gt_files & pred_files
            gt_only_files = gt_files - pred_files
            pred_only_files = pred_files - gt_files
            
            self.console.print(f"[green]Общие файлы ({len(common_files)}): {sorted(common_files)}[/green]")
            if gt_only_files:
                self.console.print(f"[yellow]Только в GT ({len(gt_only_files)}): {sorted(gt_only_files)}[/yellow]")
            if pred_only_files:
                self.console.print(f"[red]Только в предсказаниях ({len(pred_only_files)}): {sorted(pred_only_files)}[/red]")
            
            debug_info["file_matching"] = {
                "common_files": list(common_files),
                "gt_only_files": list(gt_only_files),
                "pred_only_files": list(pred_only_files),
                "file_match_ratio": len(common_files) / max(len(gt_files), 1)
            }
            
            # Анализ совпадения классов
            self.console.print("\n[bold cyan]4. АНАЛИЗ СОВПАДЕНИЯ КЛАССОВ")
            common_classes = gt_classes & pred_classes
            gt_only_classes = gt_classes - pred_classes
            pred_only_classes = pred_classes - gt_classes
            
            self.console.print(f"[green]Общие классы ({len(common_classes)}): {sorted(common_classes)}[/green]")
            if gt_only_classes:
                self.console.print(f"[yellow]Только в GT ({len(gt_only_classes)}): {sorted(gt_only_classes)}[/yellow]")
            if pred_only_classes:
                self.console.print(f"[red]Только в предсказаниях ({len(pred_only_classes)}): {sorted(pred_only_classes)}[/red]")
            
            debug_info["class_matching"] = {
                "common_classes": list(common_classes),
                "gt_only_classes": list(gt_only_classes),
                "pred_only_classes": list(pred_only_classes),
                "class_match_ratio": len(common_classes) / max(len(gt_classes), 1)
            }
            
            # Детальный анализ по файлам и классам
            self.console.print("\n[bold cyan]5. ДЕТАЛЬНЫЙ АНАЛИЗ ПО ФАЙЛАМ")
            file_analysis = {}
            
            for file_name in sorted(common_files):
                self.console.print(f"\n[bold]Файл: {file_name}[/bold]")
                
                # GT для этого файла
                gt_for_file = []
                for dp in gt.data_points or []:
                    if os.path.basename(str(getattr(dp, 'file_name', ''))) == file_name:
                        for ann in getattr(dp, 'annotations', []):
                            label = getattr(ann, 'label', getattr(dp, 'label', None))
                            bbox = getattr(ann, 'bbox', None)
                            if label:
                                gt_for_file.append({"label": str(label), "bbox": bbox})
                
                # Предсказания для этого файла
                pred_for_file = []
                for pred in predictions or []:
                    if os.path.basename(str(getattr(pred, 'file_name', ''))) == file_name:
                        label = getattr(pred, 'label', None)
                        bbox = getattr(pred, 'bbox', None)
                        score = getattr(pred, 'score', getattr(pred, 'confidence', 1.0))
                        if label:
                            pred_for_file.append({"label": str(label), "bbox": bbox, "score": score})
                
                self.console.print(f"  GT: {len(gt_for_file)} аннотаций")
                for i, gt_ann in enumerate(gt_for_file):
                    self.console.print(f"    GT[{i}]: {gt_ann}")
                
                self.console.print(f"  Pred: {len(pred_for_file)} предсказаний")
                for i, pred_ann in enumerate(pred_for_file):
                    self.console.print(f"    Pred[{i}]: {pred_ann}")
                
                file_analysis[file_name] = {
                    "gt_count": len(gt_for_file),
                    "pred_count": len(pred_for_file),
                    "gt_annotations": gt_for_file,
                    "predictions": pred_for_file
                }
            
            debug_info["bbox_analysis"] = file_analysis
            
            # Проверка потенциальных проблем
            self.console.print("\n[bold cyan]6. ПОТЕНЦИАЛЬНЫЕ ПРОБЛЕМЫ")
            issues = []
            
            if len(common_files) == 0:
                issues.append("КРИТИЧНО: Нет общих файлов между GT и предсказаниями!")
                self.console.print("[bold red]❌ КРИТИЧНО: Нет общих файлов между GT и предсказаниями![/bold red]")
            
            if len(common_classes) == 0:
                issues.append("КРИТИЧНО: Нет общих классов между GT и предсказаниями!")
                self.console.print("[bold red]❌ КРИТИЧНО: Нет общих классов между GT и предсказаниями![/bold red]")
            
            if debug_info["file_matching"]["file_match_ratio"] < 0.5:
                issues.append(f"ВНИМАНИЕ: Низкое совпадение файлов ({debug_info['file_matching']['file_match_ratio']:.2%})")
                self.console.print(f"[yellow]⚠️  ВНИМАНИЕ: Низкое совпадение файлов ({debug_info['file_matching']['file_match_ratio']:.2%})[/yellow]")
            
            if debug_info["class_matching"]["class_match_ratio"] < 0.5:
                issues.append(f"ВНИМАНИЕ: Низкое совпадение классов ({debug_info['class_matching']['class_match_ratio']:.2%})")
                self.console.print(f"[yellow]⚠️  ВНИМАНИЕ: Низкое совпадение классов ({debug_info['class_matching']['class_match_ratio']:.2%})[/yellow]")
            
            if not issues:
                self.console.print("[green]✅ Основные проблемы не обнаружены[/green]")
            
            debug_info["errors"] = issues
            
        except Exception as e:
            error_msg = f"Ошибка при отладке mAP: {str(e)}"
            debug_info["errors"].append(error_msg)
            self.console.print(f"[bold red]ОШИБКА: {error_msg}[/bold red]")
        
        # Сохранение отладочной информации в файл
        if output_dir:
            debug_dir = Path(output_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_file = debug_dir / "map_debug_analysis.json"
            
            try:
                with open(debug_file, 'w', encoding='utf-8') as f:
                    json.dump(debug_info, f, ensure_ascii=False, indent=2)
                self.console.print(f"[green]Отладочная информация сохранена в: {debug_file}[/green]")
            except Exception as e:
                self.console.print(f"[red]Ошибка сохранения отладочной информации: {e}[/red]")
        
        self.console.rule("[bold red]END DEBUG ANALYSIS")
        return debug_info

    def _generate_terminal_report(self, metrics: List[MetricOutputModel], timing_stats: Dict[str, Any], verbose: bool = False) -> None:
        self.console.rule("МЕТРИКИ (сводная таблица)")
        t = Table(box=box.SIMPLE_HEAVY)
        t.add_column("Metric", style="cyan", no_wrap=True)
        t.add_column("Score", justify="right")
        
        # Проверяем наличие проблемных метрик
        map_metric = None
        for m in metrics or []:
            score = f"{m.score:.4f}" if isinstance(m.score, (int, float)) else str(m.score)
            t.add_row(m.metric_name, score)
            
            # Ищем mAP метрику для отладки
            if str(m.metric_name).lower() in {"map", "meanaverageprecision"}:
                map_metric = m
        
        self.console.print(t)
        
        # Автоматическая отладка mAP если он равен 0
        if map_metric and isinstance(map_metric.score, (int, float)) and map_metric.score == 0.0:
            self.console.print("\n[bold red]⚠️  ОБНАРУЖЕН mAP = 0.0! Запускаем автоматическую отладку...[/bold red]")
            
            # Пытаемся извлечь GT и predictions из stats метрики
            if isinstance(map_metric.stats, dict):
                gt_data = map_metric.stats.get('gt_data')
                pred_data = map_metric.stats.get('pred_data')
                
                if gt_data and pred_data:
                    self.console.print("[yellow]Найдены данные GT и predictions в stats метрики, запускаем детальный анализ...[/yellow]")
                    # Здесь можно было бы запустить отладку, но нам нужны исходные объекты
                else:
                    self.console.print("[yellow]Для полной отладки mAP используйте метод debug_map_calculation() с исходными данными GT и predictions[/yellow]")
                    
                # Выводим доступную информацию из stats
                self.console.print("\n[bold cyan]Доступная информация из mAP stats:[/bold cyan]")
                for key, value in map_metric.stats.items():
                    if key in ['error', 'fallback']:
                        self.console.print(f"  [red]{key}: {value}[/red]")
                    elif isinstance(value, (list, dict)) and len(str(value)) < 200:
                        self.console.print(f"  {key}: {value}")
                    elif isinstance(value, (int, float)):
                        self.console.print(f"  {key}: {value}")
                    else:
                        self.console.print(f"  {key}: {type(value).__name__} (length: {len(str(value))})")
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
            plt.tight_layout(); plt.savefig(hist_path, format="svg", facecolor="white", bbox_inches="tight", transparent=False); plt.close()
            images["durations_hist"] = str(hist_path)

            plt.figure(figsize=(6, 3))
            plt.plot(durations, color="#F58518"); plt.title("Durations by run")
            ser_path = report_dir / "durations_series.svg"
            plt.tight_layout(); plt.savefig(ser_path, format="svg", facecolor="white", bbox_inches="tight", transparent=False); plt.close()
            images["durations_series"] = str(ser_path)

        spans_avg: Dict[str, float] = timing_stats.get("spans_avg", {})
        if spans_avg:
            names = list(spans_avg.keys())
            values = [spans_avg[k] for k in names]
            plt.figure(figsize=(6, 3))
            plt.barh(names, values, color="#54A24B"); plt.title("Avg span time (sec)")
            plt.tight_layout()
            p = report_dir / "spans_avg.svg"
            plt.savefig(p, format="svg", facecolor="white", bbox_inches="tight", transparent=False); plt.close()
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
                        plt.tight_layout(); plt.savefig(p1, format="svg", facecolor="white", bbox_inches="tight", transparent=False); plt.close()
                        images["gt_class_distribution"] = str(p1)
                    if pred_counts:
                        plt.figure(figsize=(max(6, len(labels) * 0.5), 3))
                        plt.bar(x, pred_counts, color="#F58518"); plt.title("Predicted class distribution"); plt.xticks(x, labels, rotation=45, ha="right")
                        p2 = report_dir / "pred_class_distribution.svg"
                        plt.tight_layout(); plt.savefig(p2, format="svg", facecolor="white", bbox_inches="tight", transparent=False); plt.close()
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
                    plt.tight_layout(); plt.savefig(p, format="svg", facecolor="white", bbox_inches="tight", transparent=False); plt.close()
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
                            plt.tight_layout(); plt.savefig(pM, format="svg", facecolor="white", bbox_inches="tight", transparent=False); plt.close()
                            images[f"pr_macro_{tag}"] = str(pM)
                        if prm and prm.get("recall") and prm.get("precision"):
                            plt.figure(figsize=(6, 3))
                            plt.plot(prm["recall"], prm["precision"], color="#F58518")
                            plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR micro @IoU={key}")
                            pm = report_dir / f"pr_micro_{tag}.svg"
                            plt.tight_layout(); plt.savefig(pm, format="svg", facecolor="white", bbox_inches="tight", transparent=False); plt.close()
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
                    plt.tight_layout(); plt.savefig(p_all, format="svg", facecolor="white", bbox_inches="tight", transparent=False); plt.close()
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
                    plt.tight_layout(); plt.savefig(p_k, format="svg", facecolor="white", bbox_inches="tight", transparent=False); plt.close()
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

    def debug_file_matching(self, gt_annotations: List[COCOAnnotation], predictions: List[COCOAnnotation], 
                           output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Отладка сопоставления имен файлов между GT и predictions
        
        Args:
            gt_annotations: Список аннотаций ground truth
            predictions: Список предсказаний
            output_dir: Директория для сохранения отладочной информации
            
        Returns:
            Словарь с результатами анализа сопоставления файлов
        """
        self.console.print("\n[bold cyan]🔍 АНАЛИЗ СОПОСТАВЛЕНИЯ ИМЕН ФАЙЛОВ[/bold cyan]")
        
        # Извлекаем уникальные имена файлов
        gt_files = set()
        pred_files = set()
        
        for ann in gt_annotations:
            if hasattr(ann, 'file_name') and ann.file_name:
                gt_files.add(ann.file_name)
        
        for pred in predictions:
            if hasattr(pred, 'file_name') and pred.file_name:
                pred_files.add(pred.file_name)
        
        # Анализ совпадений
        matching_files = gt_files.intersection(pred_files)
        gt_only_files = gt_files - pred_files
        pred_only_files = pred_files - gt_files
        
        # Статистика
        stats = {
            'total_gt_files': len(gt_files),
            'total_pred_files': len(pred_files),
            'matching_files': len(matching_files),
            'gt_only_files': len(gt_only_files),
            'pred_only_files': len(pred_only_files),
            'match_percentage': (len(matching_files) / max(len(gt_files), 1)) * 100
        }
        
        # Вывод статистики
        self.console.print(f"📊 Всего файлов в GT: [bold]{stats['total_gt_files']}[/bold]")
        self.console.print(f"📊 Всего файлов в predictions: [bold]{stats['total_pred_files']}[/bold]")
        self.console.print(f"✅ Совпадающих файлов: [bold green]{stats['matching_files']}[/bold green]")
        self.console.print(f"❌ Только в GT: [bold red]{stats['gt_only_files']}[/bold red]")
        self.console.print(f"❌ Только в predictions: [bold red]{stats['pred_only_files']}[/bold red]")
        self.console.print(f"📈 Процент совпадений: [bold]{stats['match_percentage']:.1f}%[/bold]")
        
        # Детальный анализ несовпадений
        if gt_only_files:
            self.console.print(f"\n[bold red]Файлы только в GT (первые 10):[/bold red]")
            for i, filename in enumerate(sorted(gt_only_files)[:10]):
                self.console.print(f"  {i+1}. {filename}")
            if len(gt_only_files) > 10:
                self.console.print(f"  ... и ещё {len(gt_only_files) - 10} файлов")
        
        if pred_only_files:
            self.console.print(f"\n[bold red]Файлы только в predictions (первые 10):[/bold red]")
            for i, filename in enumerate(sorted(pred_only_files)[:10]):
                self.console.print(f"  {i+1}. {filename}")
            if len(pred_only_files) > 10:
                self.console.print(f"  ... и ещё {len(pred_only_files) - 10} файлов")
        
        # Анализ паттернов имен файлов
        self.console.print(f"\n[bold cyan]📋 АНАЛИЗ ПАТТЕРНОВ ИМЕН ФАЙЛОВ[/bold cyan]")
        
        # Анализ расширений
        gt_extensions = {}
        pred_extensions = {}
        
        for filename in gt_files:
            ext = filename.split('.')[-1].lower() if '.' in filename else 'no_ext'
            gt_extensions[ext] = gt_extensions.get(ext, 0) + 1
            
        for filename in pred_files:
            ext = filename.split('.')[-1].lower() if '.' in filename else 'no_ext'
            pred_extensions[ext] = pred_extensions.get(ext, 0) + 1
        
        self.console.print("Расширения в GT:", dict(gt_extensions))
        self.console.print("Расширения в predictions:", dict(pred_extensions))
        
        # Анализ префиксов (первые 5 символов)
        gt_prefixes = {}
        pred_prefixes = {}
        
        for filename in gt_files:
            prefix = filename[:5]
            gt_prefixes[prefix] = gt_prefixes.get(prefix, 0) + 1
            
        for filename in pred_files:
            prefix = filename[:5]
            pred_prefixes[prefix] = pred_prefixes.get(prefix, 0) + 1
        
        # Показываем топ-5 префиксов
        top_gt_prefixes = sorted(gt_prefixes.items(), key=lambda x: x[1], reverse=True)[:5]
        top_pred_prefixes = sorted(pred_prefixes.items(), key=lambda x: x[1], reverse=True)[:5]
        
        self.console.print("Топ-5 префиксов в GT:", top_gt_prefixes)
        self.console.print("Топ-5 префиксов в predictions:", top_pred_prefixes)
        
        # Сохранение детальной информации
        debug_info = {
            'statistics': stats,
            'gt_files': sorted(list(gt_files)),
            'pred_files': sorted(list(pred_files)),
            'matching_files': sorted(list(matching_files)),
            'gt_only_files': sorted(list(gt_only_files)),
            'pred_only_files': sorted(list(pred_only_files)),
            'gt_extensions': gt_extensions,
            'pred_extensions': pred_extensions,
            'gt_prefixes': dict(top_gt_prefixes),
            'pred_prefixes': dict(top_pred_prefixes)
        }
        
        if output_dir:
            debug_file = Path(output_dir) / "file_matching_debug.json"
            debug_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(debug_file, 'w', encoding='utf-8') as f:
                json.dump(debug_info, f, indent=2, ensure_ascii=False)
            
            self.console.print(f"\n💾 Детальная отладочная информация сохранена в: {debug_file}")
        
        return debug_info