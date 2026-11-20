"""Detection metrics shared by every method (PCA, DeepLog, LLM-only, cascade).

All functions take plain 1-D arrays and never see anything but labels and predictions/scores:
thresholds are chosen elsewhere, without labels. Any ratio with a zero denominator is 0.0.
"""

import numpy as np


def _as_binary(name: str, values) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isin(arr, (0, 1)).all():
        raise ValueError(f"{name} must contain only 0 and 1")
    return arr.astype(np.int64)


def confusion(y_true, y_pred) -> dict[str, int]:
    """Return tp, fp, fn, tn for binary labels (1 = anomalous)."""
    t = _as_binary("y_true", y_true)
    p = _as_binary("y_pred", y_pred)
    if t.shape != p.shape:
        raise ValueError(f"length mismatch: y_true {t.shape} vs y_pred {p.shape}")
    return {
        "tp": int(np.sum((t == 1) & (p == 1))),
        "fp": int(np.sum((t == 0) & (p == 1))),
        "fn": int(np.sum((t == 1) & (p == 0))),
        "tn": int(np.sum((t == 0) & (p == 0))),
    }


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def classification_metrics(y_true, y_pred) -> dict[str, float | int]:
    """Counts, precision, recall, F1, specificity, MCC and flag rate from hard predictions."""
    c = confusion(y_true, y_pred)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    n = tp + fp + fn + tn
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    # float64 products: HDFS has ~1e5 windows, int64 would still fit but float is explicit
    denom = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    return {
        **c,
        "n": n,
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * precision * recall, precision + recall),
        "specificity": _ratio(tn, tn + fp),
        "mcc": _ratio(float(tp) * tn - float(fp) * fn, denom),
        "flag_rate": _ratio(tp + fp, n),
    }


def _check_scores(y_true, scores) -> tuple[np.ndarray, np.ndarray]:
    t = _as_binary("y_true", y_true)
    s = np.asarray(scores, dtype=float)
    if s.shape != t.shape:
        raise ValueError(f"length mismatch: y_true {t.shape} vs scores {s.shape}")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    return t, s


def average_precision(y_true, scores) -> float:
    """AU-PR as step-wise average precision: sum over thresholds of (R_k - R_{k-1}) * P_k.

    Windows with equal scores form one threshold (same convention as sklearn's
    average_precision_score). Returns nan when there is no positive window.
    """
    t, s = _check_scores(y_true, scores)
    n_pos = int(t.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    t_sorted, s_sorted = t[order], s[order]
    last_of_group = np.r_[np.flatnonzero(np.diff(s_sorted)), len(s_sorted) - 1]
    tp = np.cumsum(t_sorted)[last_of_group]
    flagged = last_of_group + 1
    precision = tp / flagged
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _k_for_rate(n: int, rate: float) -> int:
    """Number of windows flagged at `rate`: k = ceil(rate * n)."""
    if not 0 < rate <= 1:
        raise ValueError(f"rate must be in (0, 1], got {rate}")
    return int(np.ceil(round(rate * n, 9)))  # round: 0.07 * 100 must give 7, not 8


def top_k_mask(scores, rate: float) -> np.ndarray:
    """Boolean mask of the k = ceil(rate * n) highest scores; ties go to the earlier row."""
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1 or s.size == 0 or not np.isfinite(s).all():
        raise ValueError("scores must be a non-empty finite 1-D array")
    k = _k_for_rate(s.size, rate)
    mask = np.zeros(s.size, dtype=bool)
    mask[np.argsort(-s, kind="stable")[:k]] = True
    return mask


def recall_at_flag_rate(y_true, scores, rate: float) -> float:
    """Expected share of anomalous windows among the top k = ceil(rate * n) scores (D-09bd).

    Windows scoring above the k-th highest score are flagged; the remaining slots are filled
    from the windows tied at that score, each counting at the tied group's anomaly share
    (the mean over random tie orders). The result does not depend on row order.
    """
    t, s = _check_scores(y_true, scores)
    k = _k_for_rate(s.size, rate)
    n_pos = int(t.sum())
    if n_pos == 0:
        return 0.0
    cut = np.sort(s)[::-1][k - 1]  # k-th highest score
    above = s > cut
    tied = s == cut
    slots = k - int(above.sum())
    expected_tp = t[above].sum() + slots * t[tied].sum() / tied.sum()
    return float(expected_tp / n_pos)


# Memory bound for one block of resample indices (rows x n), not a research parameter.
_MAX_INDEX_CELLS = 10_000_000
_BOOTSTRAP_METRICS = ("precision", "recall", "f1", "specificity", "mcc")


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.zeros(np.shape(num), dtype=float), where=den != 0)


def _metric_from_counts(name: str, tp, fp, fn, tn) -> np.ndarray:
    """Vectorized twin of classification_metrics for many resamples; 0/0 gives 0.0."""
    tp, fp, fn, tn = (np.asarray(x, dtype=float) for x in (tp, fp, fn, tn))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    if name == "precision":
        return precision
    if name == "recall":
        return recall
    if name == "f1":
        return _safe_div(2 * precision * recall, precision + recall)
    if name == "specificity":
        return _safe_div(tn, tn + fp)
    if name == "mcc":
        return _safe_div(tp * tn - fp * fn, np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    raise ValueError(f"unknown metric {name!r}; choose from {_BOOTSTRAP_METRICS}")


def _check_bootstrap_args(n_resamples: int, ci_level: float, names) -> None:
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if not 0 < ci_level < 1:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    for name in names:
        if name not in _BOOTSTRAP_METRICS:
            raise ValueError(f"unknown metric {name!r}; choose from {_BOOTSTRAP_METRICS}")


def _resampled_counts(t: np.ndarray, preds: list[np.ndarray], n_resamples: int, seed: int):
    """Per prediction vector, a (4, n_resamples) array of tp, fp, fn, tn on bootstrap resamples.

    One index matrix is drawn per block and applied to every prediction vector, so systems
    compared in a paired bootstrap see exactly the same resampled windows.
    """
    rng = np.random.default_rng(seed)
    n = t.size
    block = max(1, _MAX_INDEX_CELLS // n)
    out = [np.empty((4, n_resamples)) for _ in preds]
    for start in range(0, n_resamples, block):
        stop = min(start + block, n_resamples)
        idx = rng.integers(0, n, size=(stop - start, n))
        tb = t[idx] == 1
        for counts, p in zip(out, preds, strict=True):
            pb = p[idx] == 1
            counts[0, start:stop] = (tb & pb).sum(axis=1)
            counts[1, start:stop] = (~tb & pb).sum(axis=1)
            counts[2, start:stop] = (tb & ~pb).sum(axis=1)
            counts[3, start:stop] = (~tb & ~pb).sum(axis=1)
    return out


def _percentile_interval(values: np.ndarray, ci_level: float) -> tuple[float, float]:
    alpha = 1.0 - ci_level
    lower, upper = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lower), float(upper)


def bootstrap_ci(
    y_true, y_pred, metrics: tuple[str, ...], n_resamples: int, seed: int, ci_level: float
) -> dict[str, tuple[float, float, float]]:
    """Percentile bootstrap CI (D-bd43): name -> (point, lower, upper).

    Test windows are resampled with replacement n_resamples times; the point estimate is
    computed on the full data and the interval is not clamped to contain it.
    """
    _check_bootstrap_args(n_resamples, ci_level, metrics)
    t = _as_binary("y_true", y_true)
    p = _as_binary("y_pred", y_pred)
    if t.shape != p.shape:
        raise ValueError(f"length mismatch: y_true {t.shape} vs y_pred {p.shape}")
    point = classification_metrics(t, p)
    (counts,) = _resampled_counts(t, [p], n_resamples, seed)
    out = {}
    for name in metrics:
        lower, upper = _percentile_interval(_metric_from_counts(name, *counts), ci_level)
        out[name] = (float(point[name]), lower, upper)
    return out


def paired_bootstrap_diff(
    y_true,
    y_pred_a,
    y_pred_b,
    metric: str,
    n_resamples: int,
    seed: int,
    ci_level: float,
) -> tuple[float, float, float]:
    """Paired bootstrap of metric(a) - metric(b) on the same resampled windows (D-e726).

    Returns (point, lower, upper). A difference counts as real when the interval excludes 0.
    """
    _check_bootstrap_args(n_resamples, ci_level, (metric,))
    t = _as_binary("y_true", y_true)
    a = _as_binary("y_pred_a", y_pred_a)
    b = _as_binary("y_pred_b", y_pred_b)
    if not t.shape == a.shape == b.shape:
        raise ValueError(f"length mismatch: {t.shape}, {a.shape}, {b.shape}")
    point = classification_metrics(t, a)[metric] - classification_metrics(t, b)[metric]
    counts_a, counts_b = _resampled_counts(t, [a, b], n_resamples, seed)
    diffs = _metric_from_counts(metric, *counts_a) - _metric_from_counts(metric, *counts_b)
    lower, upper = _percentile_interval(diffs, ci_level)
    return float(point), lower, upper
