import numpy as np
import pytest
from sklearn import metrics as skm

from cascadelad.evaluation.metrics import classification_metrics, confusion

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
