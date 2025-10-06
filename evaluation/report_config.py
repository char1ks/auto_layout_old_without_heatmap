from dataclasses import dataclass

@dataclass
class ReportConfig:
    include_spans: bool=True
    include_contexts_table: bool=True
    include_errors: bool=True
    top_k_errors: int=5
    include_detector_breakdown: bool=True
    include_class_distributions: bool=True
    include_ap_graphs: bool=True
    top_k_lowest_map: int=5
    svg_fonttype: str = "none"
    savefig_facecolor: str = "white"
    figure_facecolor: str = "white"
    axes_facecolor: str = "white"
    savefig_transparent: bool = False