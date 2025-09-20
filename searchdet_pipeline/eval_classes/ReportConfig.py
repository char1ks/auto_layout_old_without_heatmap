from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from collections import Counter
import statistics
import json

from searchdet_pipeline.eval_classes.Context import Context
from searchdet_pipeline.eval_classes.metrics import MetricOutputModel

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