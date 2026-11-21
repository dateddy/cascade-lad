"""Label-free threshold policies (D-cbb3, spec 3.5).

A policy turns the scores of normal-only validation windows into one threshold tau. Policies see
scores and config only, never labels; a test asserts that no signature has a label argument.
A window is flagged when its score is strictly greater than tau.
"""

from collections.abc import Callable, Mapping

import numpy as np

THRESHOLDS: dict[str, Callable[..., float]] = {}


def register_threshold(name: str) -> Callable[[Callable[..., float]], Callable[..., float]]:
    def decorator(fn: Callable[..., float]) -> Callable[..., float]:
        if name in THRESHOLDS:
            raise ValueError(f"threshold '{name}' is already registered")
        THRESHOLDS[name] = fn
        return fn

    return decorator


def check_scores(scores) -> np.ndarray:
    """Return scores as a non-empty 1-D float array of finite values, or raise ValueError."""
    arr = np.asarray(scores, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"scores must be a non-empty 1-D array, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("scores contain nan or inf")
    return arr


@register_threshold("val_percentile")
def val_percentile(val_scores, *, percentile: float) -> float:
    """The `percentile`-th percentile of val scores (default 99)."""
    return float(np.percentile(check_scores(val_scores), percentile))


@register_threshold("flag_rate")
def flag_rate(val_scores, *, rate: float) -> float:
    """The (1 - rate) quantile of val scores, so about `rate` of val is flagged (spec 3.5)."""
    return float(np.quantile(check_scores(val_scores), 1.0 - rate))


def compute_threshold(val_scores, cfg: Mapping) -> float:
    """Dispatch on cfg["method"]; the other cfg keys are the policy's keyword arguments."""
    method = cfg["method"]
    if method not in THRESHOLDS:
        raise KeyError(f"unknown threshold '{method}'; known: {sorted(THRESHOLDS)}")
    kwargs = {k: v for k, v in cfg.items() if k != "method"}
    return THRESHOLDS[method](val_scores, **kwargs)


def apply_threshold(scores, tau: float) -> np.ndarray:
    """1 where score > tau, else 0 (int8)."""
    return (check_scores(scores) > tau).astype(np.int8)
