from dataclasses import dataclass
import sys
from pathlib import Path
EVAL_CLASSES_PATH = Path(__file__).parent
sys.path.insert(0, str(EVAL_CLASSES_PATH))

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