import numpy as np
import pandas as pd
import pytest

from cascadelad.detectors import get_detector
from cascadelad.evaluation.report import evaluate

ALWAYS = {"name": "always_anomaly", "decision": "rule", "rule_threshold": 0.5}
UNSEEN = {"name": "unseen_sequence", "decision": "rule", "rule_threshold": 0.5}


def windows(sequences, start=0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group_id": [f"g{start + i}" for i in range(len(sequences))],
            "event_ids": sequences,
            "timestamp": pd.date_range("2026-01-01", periods=len(sequences), freq="min"),
        }
    )


def eval_kwargs():
    return {
        "ci_metrics": ["f1", "recall"],
        "n_resamples": 20,
        "seed": 0,
        "ci_level": 0.95,
        "block_length": None,
    }


def test_get_detector_unknown_raises():
    with pytest.raises(KeyError, match="nope"):
        get_detector({"name": "nope", "decision": "rule"}, n_features=4)


@pytest.mark.parametrize("seed", [0, 1])
def test_always_anomaly_recall_one_spec_zero(seed):
    # S5-3: flag everything -> recall 1, specificity 0, on any test set with both classes.
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=200)
    labels[:2] = [0, 1]
    test = windows([[1, 2]] * 200)
    det = get_detector(ALWAYS, n_features=4)
    det.fit(windows([[1, 2]] * 5))
    scores = det.score(test)
    pred = (scores > ALWAYS["rule_threshold"]).astype(int)
    m = evaluate(labels, pred, scores, [0.1], **eval_kwargs())
    assert m["recall"] == 1.0
    assert m["specificity"] == 0.0


def test_unseen_sequence_seen_scores_zero():
    # S5-4: a window identical to a train window scores 0.
    train = windows([[1, 2, 3], [4, 5]])
    det = get_detector(UNSEEN, n_features=8)
    det.fit(train)
    assert det.score(windows([[4, 5], [1, 2, 3]], start=10)).tolist() == [0.0, 0.0]


def test_unseen_sequence_unseen_scores_one():
    # S5-4: unseen sequence scores 1: new order, new length, and a sequence with <UNK> (id 1).
    train = windows([[2, 3, 4], [5, 6]])
    det = get_detector(UNSEEN, n_features=8)
    det.fit(train)
    test = windows([[4, 3, 2], [2, 3], [2, 1, 4], [2, 3, 4]], start=10)
    assert det.score(test).tolist() == [1.0, 1.0, 1.0, 0.0]


def test_detectors_expose_decision():
    assert get_detector(ALWAYS, n_features=4).decision == "rule"
    assert get_detector(UNSEEN, n_features=4).decision == "rule"


@pytest.mark.parametrize("cfg", [ALWAYS, UNSEEN])
def test_detectors_reject_label_column(cfg):
    det = get_detector(cfg, n_features=4)
    labeled = windows([[1, 2]]).assign(label=[1])
    with pytest.raises(ValueError, match="label"):
        det.fit(labeled)
    det.fit(windows([[1, 2]]))
    with pytest.raises(ValueError, match="label"):
        det.score(labeled)
