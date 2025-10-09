import time
import numpy as np
from evaluation.context import Context
def test_searchdet_pipeline(detector, image_right):
    start = time.perf_counter()
    ctx = Context(detector_name=detector.__class__.__name__, image_shape=tuple(np.array(image_right).shape))
    results = detector.detect(image_right, heatmap_threshold=0.3, class_threshold=0.4, ctx=ctx)
    end = time.perf_counter()

    print(f"{int((end-start)*1000)} ms.") 
    assert isinstance(results, list)
    assert ctx is not None
    assert results, "Empty from serchdet"
    if results:
        det = results[0]
        assert hasattr(det, "class_id")
        assert hasattr(det, "score")
        assert hasattr(det, "bbox")
        assert hasattr(det, "area")