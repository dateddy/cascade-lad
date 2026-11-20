import itertools

import pytest

from cascadelad.evaluation import cost
from cascadelad.evaluation.cost import Timer, ms_per_item


def test_timer_measures_elapsed(monkeypatch):
    ticks = itertools.chain([10.0, 10.25])
    monkeypatch.setattr(cost.time, "perf_counter", lambda: next(ticks))
    with Timer() as t:
        pass
    assert t.elapsed_ms == pytest.approx(250.0)


def test_timer_records_elapsed_when_body_raises(monkeypatch):
    ticks = itertools.chain([1.0, 1.5])
    monkeypatch.setattr(cost.time, "perf_counter", lambda: next(ticks))
    with pytest.raises(RuntimeError):
        with Timer() as t:
            raise RuntimeError("boom")
    assert t.elapsed_ms == pytest.approx(500.0)


def test_ms_per_item():
    assert ms_per_item(500.0, 100) == pytest.approx(5.0)


@pytest.mark.parametrize("n_items", [0, -3])
def test_ms_per_item_zero_raises(n_items):
    with pytest.raises(ValueError):
        ms_per_item(10.0, n_items)
