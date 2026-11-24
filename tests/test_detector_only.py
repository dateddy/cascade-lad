import numpy as np
import pandas as pd
import pytest

from cascadelad.detectors import get_detector
from cascadelad.detectors.base import Detector
from cascadelad.methods.detector_only import MethodResult, run_detector_only, strip_labels

EVAL_CFG = {
    "bootstrap": {
        "n_resamples": 20,
        "seed": 0,
        "ci_level": 0.95,
        "block_length": None,
        "metrics": ["f1", "recall"],
    },
    "flag_rates": [0.1, 0.2],
}
POLICY = {"method": "val_percentile", "percentile": 90}


def split(n, labels, start=0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group_id": [f"g{start + i}" for i in range(n)],
            "event_ids": [[2, 3]] * n,
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="min"),
            "label": labels,
            "line_labels": [[0, 1]] * n,  # extra processed columns must not reach the detector
            "line_pos": [[0, 1]] * n,
        }
    )


def make_splits(test_labels=None):
    n_test = 40
    test_labels = [0, 1] * (n_test // 2) if test_labels is None else test_labels
    return {
        "train": split(30, [0] * 30),
        "val": split(50, [0] * 50, start=100),
        "test": split(n_test, test_labels, start=200),
    }


class ScoreByPosition(Detector):
    """Scores 0..n-1 by row, so thresholds are easy to predict; records what it was given."""

    decision = "threshold"

    def __init__(self):
        self.seen_columns: list[list[str]] = []

    def fit(self, train, val=None):
        self.seen_columns.append(list(train.columns))
        if val is not None:
            self.seen_columns.append(list(val.columns))

    def score(self, windows):
        self.seen_columns.append(list(windows.columns))
        return np.arange(len(windows), dtype=float)


def test_strip_labels_keeps_only_window_columns():
    out = strip_labels(split(3, [0, 1, 0]))
    assert list(out.columns) == ["group_id", "event_ids", "timestamp"]


def test_detector_only_never_passes_labels():
    det = ScoreByPosition()
    run_detector_only("spy", det, make_splits(), POLICY, None, EVAL_CFG)
    assert det.seen_columns, "detector was never called"
    for cols in det.seen_columns:
        assert cols == ["group_id", "event_ids", "timestamp"]


def test_rule_decision_skips_threshold():
    cfg = {"name": "always_anomaly", "decision": "rule", "rule_threshold": 0.5}
    det = get_detector(cfg, n_features=4)
    res = run_detector_only("always_anomaly", det, make_splits(), POLICY, 0.5, EVAL_CFG)
    assert isinstance(res, MethodResult)
    assert res.threshold_method == "rule"
    assert res.threshold == 0.5
    assert res.test_pred.tolist() == [1] * 40
    assert res.metrics["recall"] == 1.0


def test_threshold_decision_uses_policy_on_val_scores():
    res = run_detector_only("spy", ScoreByPosition(), make_splits(), POLICY, None, EVAL_CFG)
    assert res.threshold_method == "val_percentile"
    assert res.threshold == pytest.approx(np.percentile(np.arange(50.0), 90))
    assert res.val_flag_rate == pytest.approx(float((np.arange(50.0) > res.threshold).mean()))
    assert res.test_pred.tolist() == (np.arange(40.0) > res.threshold).astype(int).tolist()


def test_threshold_decision_uses_val_scores_only():
    # Changing the test labels must not move tau or the predictions, only the metrics.
    base = run_detector_only("spy", ScoreByPosition(), make_splits(), POLICY, None, EVAL_CFG)
    relabeled = make_splits(test_labels=[0] * 40)
    other = run_detector_only("spy", ScoreByPosition(), relabeled, POLICY, None, EVAL_CFG)
    assert other.threshold == base.threshold
    assert other.test_pred.tolist() == base.test_pred.tolist()
    assert other.metrics["fn"] != base.metrics["fn"]


def test_result_fields_and_timing():
    res = run_detector_only("spy", ScoreByPosition(), make_splits(), POLICY, None, EVAL_CFG)
    assert res.name == "spy"
    assert res.val_scores.shape == (50,)
    assert res.test_scores.shape == (40,)
    assert res.infer_ms_per_window >= 0.0
    assert {"f1", "recall", "au_pr", "recall_at_10", "recall_at_20", "f1_ci_low"} <= set(
        res.metrics
    )


def test_rule_decision_requires_rule_threshold():
    cfg = {"name": "always_anomaly", "decision": "rule", "rule_threshold": 0.5}
    det = get_detector(cfg, n_features=4)
    with pytest.raises(ValueError, match="rule_threshold"):
        run_detector_only("always_anomaly", det, make_splits(), POLICY, None, EVAL_CFG)
