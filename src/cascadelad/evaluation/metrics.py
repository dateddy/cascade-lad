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
