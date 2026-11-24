"""Trivial baseline: flag every window (spec 4.3)."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from cascadelad.detectors.base import Detector
from cascadelad.detectors.registry import register_detector


@register_detector("always_anomaly")
class AlwaysAnomaly(Detector):
    def __init__(self, cfg: Mapping, n_features: int) -> None:
        self.decision = cfg["decision"]  # "rule": decided by cfg["rule_threshold"], not a policy

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> None:
        """Nothing to learn."""

    def score(self, windows: pd.DataFrame) -> np.ndarray:
        return np.ones(len(windows))
