import numpy as np
import pytest
from sklearn import metrics as skm

from cascadelad.evaluation.metrics import (
    average_precision,
    classification_metrics,
    confusion,
    recall_at_flag_rate,
    top_k_mask,
)

Y_TRUE = np.array([1, 1, 1, 0, 0, 0, 0, 1, 0, 0])
Y_PRED = np.array([1, 0, 1, 1, 0, 0, 0, 1, 0, 1])
# Counted by hand: tp = idx {0,2,7}, fn = idx {1}, fp = idx {3,9}, tn = idx {4,5,6,8}.
TP, FN, FP, TN = 3, 1, 2, 4


def test_confusion_hand_values():
    assert confusion(Y_TRUE, Y_PRED) == {"tp": TP, "fp": FP, "fn": FN, "tn": TN}


def test_metrics_hand_values_1e9():
    m = classification_metrics(Y_TRUE, Y_PRED)
    precision = 3 / 5
    recall = 3 / 4
    f1 = 2 * precision * recall / (precision + recall)
    specificity = 4 / 6
    mcc = (3 * 4 - 2 * 1) / np.sqrt((3 + 2) * (3 + 1) * (4 + 2) * (4 + 1))
    assert m["precision"] == pytest.approx(precision, abs=1e-9)
    assert m["recall"] == pytest.approx(recall, abs=1e-9)
    assert m["f1"] == pytest.approx(f1, abs=1e-9)
    assert m["specificity"] == pytest.approx(specificity, abs=1e-9)
    assert m["mcc"] == pytest.approx(mcc, abs=1e-9)


def test_counts_sum_to_n():
    m = classification_metrics(Y_TRUE, Y_PRED)
    assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == m["n"] == len(Y_TRUE)


def test_no_positive_predictions():
    m = classification_metrics(Y_TRUE, np.zeros(10, dtype=int))
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
    assert m["flag_rate"] == 0.0


def test_all_positive_predictions():
    m = classification_metrics(Y_TRUE, np.ones(10, dtype=int))
    assert m["specificity"] == 0.0
    assert m["recall"] == 1.0
    assert m["flag_rate"] == 1.0


def test_mcc_zero_denominator():
    m = classification_metrics(np.zeros(5, dtype=int), np.zeros(5, dtype=int))
    assert m["mcc"] == 0.0


def test_flag_rate():
    assert classification_metrics(Y_TRUE, Y_PRED)["flag_rate"] == pytest.approx(0.5)


def test_rejects_length_mismatch():
    with pytest.raises(ValueError):
        confusion(np.array([0, 1]), np.array([0, 1, 1]))


def test_rejects_non_binary():
    with pytest.raises(ValueError):
        confusion(np.array([0, 2]), np.array([0, 1]))


def test_rejects_empty():
    with pytest.raises(ValueError):
        confusion(np.array([], dtype=int), np.array([], dtype=int))


def test_matches_sklearn_on_random():
    rng = np.random.default_rng(0)
    y_true = (rng.random(1000) < 0.2).astype(int)
    y_pred = (rng.random(1000) < 0.3).astype(int)
    m = classification_metrics(y_true, y_pred)
    assert m["precision"] == pytest.approx(skm.precision_score(y_true, y_pred), abs=1e-9)
    assert m["recall"] == pytest.approx(skm.recall_score(y_true, y_pred), abs=1e-9)
    assert m["f1"] == pytest.approx(skm.f1_score(y_true, y_pred), abs=1e-9)
    assert m["mcc"] == pytest.approx(skm.matthews_corrcoef(y_true, y_pred), abs=1e-9)


@pytest.mark.parametrize("seed", range(5))
def test_au_pr_matches_sklearn_random(seed):
    rng = np.random.default_rng(seed)
    y_true = (rng.random(500) < 0.15).astype(int)
    scores = rng.random(500)
    assert average_precision(y_true, scores) == pytest.approx(
        skm.average_precision_score(y_true, scores), rel=1e-9
    )


def test_au_pr_matches_sklearn_with_ties():
    rng = np.random.default_rng(7)
    y_true = (rng.random(400) < 0.2).astype(int)
    scores = rng.integers(0, 4, size=400).astype(float)  # many tied scores
    assert average_precision(y_true, scores) == pytest.approx(
        skm.average_precision_score(y_true, scores), rel=1e-9
    )


def test_au_pr_no_positives_nan():
    assert np.isnan(average_precision(np.zeros(5, dtype=int), np.arange(5.0)))


def test_au_pr_rejects_non_finite_scores():
    with pytest.raises(ValueError):
        average_precision(np.array([0, 1]), np.array([0.1, np.nan]))


def test_top_k_exact_count():
    assert top_k_mask(np.random.default_rng(0).random(100), 0.1).sum() == 10
    assert top_k_mask(np.random.default_rng(0).random(95), 0.1).sum() == 10  # ceil(9.5)


def test_top_k_flags_highest_scores():
    scores = np.array([0.1, 0.9, 0.5, 0.7, 0.3])
    assert top_k_mask(scores, 0.4).tolist() == [False, True, False, True, False]


def test_top_k_ties_stable():
    mask = top_k_mask(np.zeros(10), 0.3)
    assert mask.sum() == 3
    assert mask.tolist() == [True] * 3 + [False] * 7  # earlier rows win ties


def test_recall_at_flag_rate_hand_value():
    y_true = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.3, 0.1, 0.2, 0.4, 0.5, 0.6])
    # top 20% = 2 windows (idx 0, 2) -> both anomalous; 3 anomalies in total
    assert recall_at_flag_rate(y_true, scores, 0.2) == pytest.approx(2 / 3)


@pytest.mark.parametrize("rate", [0.0, -0.1, 1.5])
def test_rate_out_of_range_raises(rate):
    with pytest.raises(ValueError):
        top_k_mask(np.arange(10.0), rate)
