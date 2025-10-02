from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import os, json, statistics, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from rich.console import Console
from rich.table import Table
from rich import box
from evaluate.ReportConfig import ReportConfig
from evaluate.Context import Context
from evaluate.metrics import MetricOutputModel

mpl.rcParams.update({"figure.facecolor":"white","axes.facecolor":"white","savefig.transparent":False,"svg.fonttype":"none","lines.markersize":3.0,"lines.linewidth":1.5})

class ReportGenerator(ReportConfig):
    def __init__(self, **kwargs: Any)->None:
        super().__init__(**kwargs); self.console=Console(); self.fw=6.4; self.fh=4.0; self.fs=9; self.maxlen=18

    def generate_report(self, contexts: List[Context], metrics: List[MetricOutputModel], dump_report: bool=False, output_dir: Optional[str]=None)->Optional[Path]:
        timing=self._timing(contexts); self._terminal(metrics,timing,verbose=dump_report)
        if not dump_report: return None
        od=Path(output_dir) if output_dir else Path(os.getcwd())/"report"; od.mkdir(parents=True,exist_ok=True)
        imgs=self._save_graphs(od,metrics,timing,contexts); self._write_md(od,metrics,timing,imgs,contexts); self._write_json(od,metrics,timing); self.console.print(f"[green]Отчёт сохранён в[/green] {od}"); return od

    def _timing(self, contexts: List[Context])->Dict[str,Any]:
        d=[float(c.duration) for c in (contexts or []) if getattr(c,"duration",None) is not None]
        s=sum(1 for c in contexts if not getattr(c,"error",None)); e=sum(1 for c in contexts if getattr(c,"error",None))
        out={"total_runs":len(contexts or []),"success":s,"errors":e,"durations":d}
        if d: out.update({"mean":float(statistics.mean(d)),"median":float(statistics.median(d)),"std":float(statistics.pstdev(d)) if len(d)>1 else 0.0})
        return out

    def _build_step(self, r, p):
        r=np.asarray(r,float); p=np.asarray(p,float)
        m=np.isfinite(r)&np.isfinite(p); r,p=r[m],p[m]
        if r.size<1 or p.size<1: return np.array([0.0]), np.array([0.0])
        o=np.argsort(r); r,p=r[o],p[o]
        ur, idx, cnts = np.unique(r, return_index=True, return_counts=True)
        up=np.zeros_like(ur); s=0
        for i,c in enumerate(cnts): up[i]=float(np.max(p[s:s+c])); s+=c
        return ur, up

    def _pad01(self, ur: np.ndarray, up: np.ndarray)->Tuple[np.ndarray,np.ndarray]:
        if ur.size==0: return np.array([0.0,1.0]), np.array([0.0,0.0])
        u0=float(up[0]); ul=float(up[-1])
        R=np.concatenate(([0.0], ur, [1.0])) if (ur[0]>0.0 or ur[-1]<1.0) else ur
        P=np.concatenate(([u0], up, [ul])) if (ur[0]>0.0 or ur[-1]<1.0) else up
        return R, P

    def _plot_pr(self, curve: Dict[str, Any], title: str, path: Path):
        ur, up = self._build_step(curve.get("recall", []), curve.get("precision", []))
        ur, up = self._pad01(ur, up)
        fig=plt.figure(figsize=(self.fw,self.fh), dpi=120); ax=fig.add_subplot(111)
        ax.step(ur, up, where="post"); ax.scatter(ur, up, s=8)
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.grid(True, alpha=0.25)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(title)
        fig.tight_layout(); fig.savefig(path, format="svg", facecolor="white", bbox_inches="tight", dpi=150); plt.close(fig)

    def _save_graphs(self, od: Path, metrics: List[MetricOutputModel], timing: Dict[str,Any], contexts: List[Context])->Dict[str,str]:
        imgs={}
        def save(fn,k, fig=None):
            if fig is not None:
                fig.tight_layout(); p=od/fn; fig.savefig(p,format="svg",facecolor="white",bbox_inches="tight",dpi=150); plt.close(fig); imgs[k]=str(p)
            else:
                p=od/fn; plt.tight_layout(); plt.savefig(p,format="svg",facecolor="white",bbox_inches="tight",dpi=150); plt.close(); imgs[k]=str(p)
        if timing.get("durations"):
            fig=plt.figure(figsize=(self.fw,self.fh), dpi=120); ax=fig.add_subplot(111)
            ax.hist(timing["durations"],bins=20); ax.set_title("Durations (sec)"); save("durations_hist.svg","durations_hist",fig)
            fig=plt.figure(figsize=(self.fw,self.fh), dpi=120); ax=fig.add_subplot(111)
            ax.plot(timing["durations"]); ax.set_title("Durations by run"); save("durations_series.svg","durations_series",fig)

        m_map=next((m for m in metrics if m.metric_name=="map"),None)
        if m_map and isinstance(m_map.stats,dict):
            st=m_map.stats; cats=list(st.get("categories") or []); gt=list(st.get("gt_counts") or []); pr=list(st.get("pred_counts") or [])
            if cats and gt:
                fig=plt.figure(figsize=(self.fw,self.fh), dpi=120); ax=fig.add_subplot(111)
                ax.bar(range(len(cats)),gt); ax.set_xticks(range(len(cats))); ax.set_xticklabels([str(s)[:self.maxlen] for s in cats], rotation=45, ha="right", fontsize=self.fs)
                ax.set_title("GT class distribution"); save("gt_class_distribution.svg","gt_class_distribution",fig)
            if cats and pr:
                fig=plt.figure(figsize=(self.fw,self.fh), dpi=120); ax=fig.add_subplot(111)
                ax.bar(range(len(cats)),pr); ax.set_xticks(range(len(cats))); ax.set_xticklabels([str(s)[:self.maxlen] for s in cats], rotation=45, ha="right", fontsize=self.fs)
                ax.set_title("Pred class distribution"); save("pred_class_distribution.svg","pred_class_distribution",fig)

            thr=st.get("iou_thresholds") or []; apm=st.get("ap_iou_macro") or []
            if thr and apm:
                xs=[float(t) for t in thr if isinstance(t,(int,float))]
                ys=[float(v) for v in apm if isinstance(v,(int,float))]
                n=min(len(xs),len(ys)); xs,ys=xs[:n],ys[:n]
                pairs=sorted([(x,y) for x,y in zip(xs,ys) if 0.0<=x<=1.0 and 0.0<=y<=1.0], key=lambda t:t[0])
                if pairs:
                    xx,yy=zip(*pairs); fig=plt.figure(figsize=(self.fw,self.fh), dpi=120); ax=fig.add_subplot(111)
                    ax.plot(xx,yy, marker='o'); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.grid(True, alpha=0.25)
                    ax.set_xticks(sorted(set(xx))); ax.set_title("AP vs IoU"); ax.set_xlabel("IoU threshold"); ax.set_ylabel("AP (macro)")
                    for x_i,y_i in pairs: ax.annotate(f"{y_i:.3f}", (x_i,y_i), textcoords="offset points", xytext=(0,6), ha="center", fontsize=self.fs)
                    save("ap_vs_iou.svg","ap_vs_iou",fig)

            for k_raw in thr or []:
                k=f"{float(k_raw):.2f}"
                pm=(st.get("pr_macro") or {}).get(k); mi=(st.get("pr_micro") or {}).get(k)
                if pm: self._plot_pr(pm, f"PR Curve (Macro) @IoU={k}", od/f"pr_macro_{k.replace('.','')}.svg"); imgs[f"pr_macro_{k.replace('.','')}"]=str(od/f"pr_macro_{k.replace('.','')}.svg")
                if mi: self._plot_pr(mi, f"PR Curve (Micro) @IoU={k}", od/f"pr_micro_{k.replace('.','')}.svg"); imgs[f"pr_micro_{k.replace('.','')}"]=str(od/f"pr_micro_{k.replace('.','')}.svg")

            per=st.get("per_class_ap_avg") or st.get("per_class_ap") or []
            if cats and per:
                pairs=sorted([(cats[i],float(per[i])) for i in range(min(len(cats),len(per)))], key=lambda x:x[1], reverse=True)
                fig=plt.figure(figsize=(self.fw,self.fh), dpi=120); ax=fig.add_subplot(111)
                ax.bar(range(len(pairs)),[p[1] for p in pairs]); ax.set_xticks(range(len(pairs))); ax.set_xticklabels([str(p[0])[:self.maxlen] for p in pairs],rotation=45,ha="right",fontsize=self.fs)
                ax.set_title("AP per class"); save("per_class_ap.svg","per_class_ap",fig)

        return imgs

    def _terminal(self, metrics: List[MetricOutputModel], timing: Dict[str,Any], verbose: bool=False)->None:
        self.console.rule("МЕТРИКИ"); t=Table(box=box.SIMPLE_HEAVY); t.add_column("Metric",style="cyan"); t.add_column("Score",justify="right")
        for m in metrics or []: t.add_row(m.metric_name, f"{float(m.score):.4f}")
        self.console.print(t)
        if verbose:
            self.console.rule("СТАТИСТИКА ВРЕМЕНИ"); st=Table(box=box.SIMPLE_HEAVY); st.add_column("Всего"); st.add_column("Успех"); st.add_column("Ошибки"); st.add_column("Среднее"); st.add_column("Медиана")
            st.add_row(str(timing.get("total_runs",0)),str(timing.get("success",0)),str(timing.get("errors",0)),f"{float(timing.get('mean',0.0)):.4f}",f"{float(timing.get('median',0.0)):.4f}")
            self.console.print(st)

    def _write_md(self, od: Path, metrics: List[MetricOutputModel], timing: Dict[str,Any], imgs: Dict[str,str], contexts: List[Context])->None:
        lines=[f"# Отчёт по оценке модели\n\nСгенерирован: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n## Статистика времени выполнения\n\n- Всего запусков: {timing.get('total_runs',0)}\n- Успешных: {timing.get('success',0)}\n- Ошибок: {timing.get('errors',0)}\n- Среднее время: {float(timing.get('mean',0.0)):.4f} сек\n- Медиана: {float(timing.get('median',0.0)):.4f} сек\n\n## Метрики\n"]
        for m in metrics or []:
            lines.append(f"### {m.metric_name}\n**Значение:** {float(m.score):.4f}\n\n")
            if m.metric_name=="classification_report" and isinstance(m.stats,dict) and m.stats.get("text"):
                lines.append("```\n"+str(m.stats["text"]).strip()+"\n```\n\n")
        def addimg(k,t):
            p=imgs.get(k)
            if p: lines.append(f"![{t}]({Path(p).name})\n\n")
        lines+=["## Графики\n\n"]; addimg("durations_hist","Гистограмма времени"); addimg("durations_series","Время по запускам")
        addimg("gt_class_distribution","Распределение GT"); addimg("pred_class_distribution","Распределение предсказаний"); addimg("ap_vs_iou","AP vs IoU")
        for k in ["050","075"]:
            addimg(f"pr_macro_{k}",f"PR макро @{k.replace('0','0.')}"); addimg(f"pr_micro_{k}",f"PR микро @{k.replace('0','0.')}")
        addimg("per_class_ap","AP по классам")
        (od/"report.md").write_text("".join(lines),encoding="utf-8")

    def _write_json(self, od: Path, metrics: List[MetricOutputModel], timing: Dict[str,Any])->None:
        data={"metrics":[{"metric_name":m.metric_name,"score":m.score,"stats":m.stats} for m in (metrics or [])],"timing":timing}
        (od/"report.json").write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
