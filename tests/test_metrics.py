import numpy as np
import pytest
from sklearn import metrics as skm

from cascadelad.evaluation.metrics import (
    average_precision,
    bootstrap_ci,
    classification_metrics,
    confusion,
    paired_bootstrap_diff,
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


def test_recall_at_flag_rate_constant_scores_equals_rate():
    # every window tied: the expected share of anomalies in any k flagged windows is k/n
    y_true = np.array([1] * 10 + [0] * 90)  # all anomalies in the first rows (early burst)
    for rate in (0.1, 0.2, 0.5):
        assert recall_at_flag_rate(y_true, np.zeros(100), rate) == pytest.approx(rate, abs=1e-12)


def test_recall_at_flag_rate_row_order_invariant():
    rng = np.random.default_rng(4)
    y_true = (rng.random(300) < 0.1).astype(int)
    scores = rng.integers(0, 3, size=300).astype(float)  # heavy ties
    perm = rng.permutation(300)
    for rate in (0.1, 0.2):
        assert recall_at_flag_rate(y_true, scores, rate) == pytest.approx(
            recall_at_flag_rate(y_true[perm], scores[perm], rate), abs=1e-12
        )


def test_recall_at_flag_rate_partial_tie_hand_value():
    # n=10, rate 0.3 -> k=3. One window above the tie (0.9, anomalous); 2 slots left for the
    # tied group at 0.5 (4 windows, 1 anomalous) -> expected tp = 1 + 2 * 1/4 = 1.5
    y_true = np.array([1, 0, 1, 0, 0, 0, 0, 0, 1, 0])
    scores = np.array([0.9, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert recall_at_flag_rate(y_true, scores, 0.3) == pytest.approx(1.5 / 3, abs=1e-12)


def test_recall_at_flag_rate_no_positives_zero():
    assert recall_at_flag_rate(np.zeros(10, dtype=int), np.arange(10.0), 0.2) == 0.0


@pytest.mark.parametrize("rate", [0.0, -0.1, 1.5])
def test_rate_out_of_range_raises(rate):
    with pytest.raises(ValueError):
        top_k_mask(np.arange(10.0), rate)


def _fixture(n=500, pos=0.1, seed=3):
    rng = np.random.default_rng(seed)
    y_true = (rng.random(n) < pos).astype(int)
    # a noisy detector: finds ~70% of positives, ~5% false alarms
    y_pred = np.where(y_true == 1, rng.random(n) < 0.7, rng.random(n) < 0.05).astype(int)
    return y_true, y_pred


KW = {"n_resamples": 200, "seed": 0, "ci_level": 0.95}


def test_bootstrap_deterministic_with_seed():
    y_true, y_pred = _fixture()
    a = bootstrap_ci(y_true, y_pred, ("f1", "recall"), **KW)
    b = bootstrap_ci(y_true, y_pred, ("f1", "recall"), **KW)
    assert a == b


def test_bootstrap_different_seed_differs():
    y_true, y_pred = _fixture()
    a = bootstrap_ci(y_true, y_pred, ("f1",), **KW)
    b = bootstrap_ci(y_true, y_pred, ("f1",), **{**KW, "seed": 1})
    assert a["f1"][1:] != b["f1"][1:]


def test_bootstrap_lower_le_point_le_upper():
    y_true, y_pred = _fixture()
    out = bootstrap_ci(y_true, y_pred, ("f1", "recall"), **{**KW, "n_resamples": 1000})
    for point, lower, upper in out.values():
        assert lower <= point <= upper


def test_bootstrap_point_matches_classification_metrics():
    y_true, y_pred = _fixture()
    m = classification_metrics(y_true, y_pred)
    out = bootstrap_ci(y_true, y_pred, ("f1", "recall", "precision", "mcc"), **KW)
    for name in out:
        assert out[name][0] == pytest.approx(m[name], abs=1e-12)


def test_bootstrap_resample_metrics_match_scalar_formulas():
    # the vectorized formulas must agree with classification_metrics on every resample
    y_true, y_pred = _fixture(n=60)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 60, size=(50, 60))
    expected = [classification_metrics(y_true[i], y_pred[i])["f1"] for i in idx]
    lo, hi = np.percentile(expected, [2.5, 97.5])
    out = bootstrap_ci(y_true, y_pred, ("f1",), n_resamples=50, seed=0, ci_level=0.95)
    assert out["f1"][1] == pytest.approx(lo, abs=1e-12)
    assert out["f1"][2] == pytest.approx(hi, abs=1e-12)


def test_bootstrap_perfect_predictor_zero_width():
    y_true, _ = _fixture()
    out = bootstrap_ci(y_true, y_true, ("f1", "recall"), **KW)
    assert out["f1"] == (1.0, 1.0, 1.0)
    assert out["recall"] == (1.0, 1.0, 1.0)


def test_bootstrap_rejects_unknown_metric():
    y_true, y_pred = _fixture()
    with pytest.raises(ValueError):
        bootstrap_ci(y_true, y_pred, ("auc",), **KW)


def test_paired_diff_identical_systems_is_zero():
    y_true, y_pred = _fixture()
    assert paired_bootstrap_diff(y_true, y_pred, y_pred, "f1", **KW) == (0.0, 0.0, 0.0)


def test_paired_diff_better_system_positive_interval():
    y_true, y_pred = _fixture(n=2000)
    rng = np.random.default_rng(9)
    worse = np.where(rng.random(2000) < 0.5, 1 - y_true, y_true)  # flips half the labels
    point, lower, upper = paired_bootstrap_diff(y_true, y_pred, worse, "f1", **KW)
    assert point > 0
    assert lower > 0  # interval excludes 0
    assert lower <= point <= upper


def test_paired_diff_rejects_length_mismatch():
    y_true, y_pred = _fixture()
    with pytest.raises(ValueError):
        paired_bootstrap_diff(y_true, y_pred, y_pred[:-1], "f1", **KW)
