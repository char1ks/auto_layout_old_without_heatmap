from datetime import datetime
from evaluate.Context import Context


def test_finish_sets_flags_and_times(context_now):
    c: Context = context_now
    assert c.success is False
    assert c.ended_at is None
    assert c.duration is None

    c.finish(success=True)

    assert c.success is True
    assert c.ended_at is not None
    assert isinstance(c.ended_at, datetime)
    assert isinstance(c.started_at, datetime)
    assert c.duration is not None
    assert c.duration >= 0.0


def test_duration_calculation(context_with_past_start):
    c: Context = context_with_past_start
    c.finish()
    assert c.duration is not None
    assert c.duration >= 1.0


def test_spans_workflow(context_now, spans_sample):
    c: Context = context_now
    assert c.spans == []
    c.spans.extend(spans_sample)

    assert len(c.spans) == 2
    names = [s.get("name") for s in c.spans]
    assert names == ["predict", "postprocess"]
    for s in c.spans:
        assert s.get("duration") is not None
        assert s["duration"] >= 0.0