import numpy as np
import pandas as pd
import pytest

from cascadelad.detectors.base import Detector


class LengthDetector(Detector):
    def fit(self, train, val=None):
        self.check_windows(train)

    def score(self, windows):
        self.check_windows(windows)
        return windows["event_ids"].map(len).to_numpy(dtype=float)


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


def test_check_windows_missing_column():
    with pytest.raises(ValueError, match="event_ids"):
        Detector.check_windows(make_windows().drop(columns=["event_ids"]))
