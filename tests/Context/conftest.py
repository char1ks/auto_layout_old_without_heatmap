import pytest
from datetime import datetime, timedelta
from evaluate.Context import Context

@pytest.fixture
def context_now() -> Context:
    return Context(detector_name="dummy-detector", image_shape=(300, 400, 3))

@pytest.fixture
def context_with_past_start() -> Context:
    c = Context(detector_name="dummy-detector", image_shape=(300, 400, 3))
    c.started_at = datetime.utcnow() - timedelta(seconds=1.2)
    return c

@pytest.fixture
def spans_sample():
    now = datetime.utcnow()
    return [
        {"name": "predict", "started_at": now, "ended_at": now, "duration": 0.01},
        {"name": "postprocess", "started_at": now, "ended_at": now, "duration": 0.02},
    ]