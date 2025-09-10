
__version__ ="1.0.0"
__author__ ="SearchDet Team"

from .core .detector import SearchDetDetector
from .core .pipeline import PipelineProcessor

__all__ =[
'SearchDetDetector',
'PipelineProcessor',
]