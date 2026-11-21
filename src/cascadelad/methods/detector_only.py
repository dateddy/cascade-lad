"""Detector-only method: score every window, threshold, evaluate (invariants 3-5, 7).

This is the only place in a Stage 5 run where labels meet scores. The detector and the threshold
policy never see them: labels are stripped before fit/score, thresholds come from val scores,
and the test labels are read once, after predictions exist, by `evaluation.report.evaluate`.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cascadelad.detectors.base import WINDOW_COLUMNS, Detector
from cascadelad.evaluation.cost import Timer, ms_per_item
from cascadelad.evaluation.report import evaluate
from cascadelad.evaluation.thresholds import apply_threshold, compute_threshold

RULE = "rule"


@dataclass
class MethodResult:
    name: str
    threshold_method: str
    threshold: float
    val_flag_rate: float
    val_scores: np.ndarray
    test_scores: np.ndarray
    test_pred: np.ndarray
    infer_ms_per_window: float
    metrics: dict


def strip_labels(windows: pd.DataFrame) -> pd.DataFrame:
    """Only the columns a detector may see (labels and per-line extras are dropped)."""
    return windows[list(WINDOW_COLUMNS)]


def run_detector_only(
    name: str,
    detector: Detector,
    splits: Mapping[str, pd.DataFrame],
    threshold_cfg: Mapping,
    rule_threshold: float | None,
    eval_cfg: Mapping,
) -> MethodResult:
    """Fit on train (+ val for early stopping), score val and test, threshold, evaluate on test.

    A detector with `decision == "rule"` flags `score > rule_threshold` (trivial baselines,
    D-47fc); every other detector gets tau from the threshold policy applied to val scores.
    """
    train, val, test = (strip_labels(splits[s]) for s in ("train", "val", "test"))
    detector.fit(train, val)
    val_scores = detector.score(val)
    with Timer() as timer:
        test_scores = detector.score(test)

    if detector.decision == RULE:
        if rule_threshold is None:
            raise ValueError(f"{name}: decision 'rule' needs a rule_threshold")
        method, tau = RULE, float(rule_threshold)
    else:
        method, tau = threshold_cfg["method"], compute_threshold(val_scores, threshold_cfg)
    pred = apply_threshold(test_scores, tau)

    boot = eval_cfg["bootstrap"]
    metrics = evaluate(
        splits["test"]["label"].to_numpy(),
        pred,
        test_scores,
        eval_cfg["flag_rates"],
        ci_metrics=boot["metrics"],
        n_resamples=boot["n_resamples"],
        seed=boot["seed"],
        ci_level=boot["ci_level"],
        block_length=boot["block_length"],
    )
    return MethodResult(
        name=name,
        threshold_method=method,
        threshold=tau,
        val_flag_rate=float(apply_threshold(val_scores, tau).mean()),
        val_scores=val_scores,
        test_scores=test_scores,
        test_pred=pred,
        infer_ms_per_window=ms_per_item(timer.elapsed_ms, len(test)),
        metrics=metrics,
    )
