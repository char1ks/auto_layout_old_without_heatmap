
__version__ = "1.0.0"
__author__ = "SearchDet Team"
def __getattr__(name: str):
    if name == "SearchDetDetector":
        from .core.detector import SearchDetDetector  # импорт только при обращении
        return SearchDetDetector
    raise AttributeError(f"module 'searchdet_pipeline' has no attribute {name!r}")

__all__ = [
    "SearchDetDetector",
]