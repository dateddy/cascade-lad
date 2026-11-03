import numpy as np
import pandas as pd
import pytest

from cascadelad.detectors.base import Detector


class LengthDetector(Detector):
    """Does not call check_windows itself: the base class must enforce it."""

    def fit(self, train, val=None):
        self.fitted_on = len(train)

    def score(self, windows):
        return windows["event_ids"].map(len).to_numpy(dtype=float)


class WrongShapeDetector(LengthDetector):
    def score(self, windows):
        return np.zeros(len(windows) + 1)


def make_windows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group_id": ["a", "b", "c"],
            "event_ids": [[1], [1, 2, 3], [1, 2]],
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        }
    )


def test_detector_is_abstract():
    with pytest.raises(TypeError):
        Detector()


def test_score_shape_dtype_and_order():
    windows = make_windows()
    det = LengthDetector()
    det.fit(windows)
    scores = det.score(windows)
    assert scores.dtype == np.float64
    assert scores.shape == (3,)
    assert scores.tolist() == [1.0, 3.0, 2.0]


def test_score_aligned_by_group_id():
    windows = make_windows().iloc[::-1]
    scores = LengthDetector().score(windows)
    by_group = dict(zip(windows["group_id"], scores, strict=True))
    assert by_group == {"a": 1.0, "b": 3.0, "c": 2.0}


def test_check_windows_missing_column():
    with pytest.raises(ValueError, match="event_ids"):
        Detector.check_windows(make_windows().drop(columns=["event_ids"]))


def test_fit_rejects_label_column():
    labeled = make_windows().assign(label=[0, 0, 1])
    with pytest.raises(ValueError, match="label"):
        LengthDetector().fit(labeled)
    with pytest.raises(ValueError, match="label"):
        LengthDetector().fit(make_windows(), val=labeled)


def test_score_rejects_label_column():
    with pytest.raises(ValueError, match="label"):
        LengthDetector().score(make_windows().assign(label=[0, 0, 1]))


def test_rejects_duplicate_group_id():
    dup = make_windows().assign(group_id=["a", "a", "c"])
    with pytest.raises(ValueError, match="duplicate"):
        LengthDetector().score(dup)


def test_score_wrong_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        WrongShapeDetector().score(make_windows())
