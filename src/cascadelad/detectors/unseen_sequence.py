"""Trivial baseline: flag a window whose event sequence never occurs in train (spec 4.3).

Measures how much of a benchmark novelty alone solves. The match is exact: same ids, same order,
same length. A sequence containing `<UNK>` was never seen in train, so it scores 1 as well.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from cascadelad.detectors.base import Detector
from cascadelad.detectors.registry import register_detector


@register_detector("unseen_sequence")
class UnseenSequence(Detector):
    def __init__(self, cfg: Mapping, n_features: int) -> None:
        self.decision = cfg["decision"]
        self.seen: set[tuple[int, ...]] = set()

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> None:
        self.seen = {tuple(ids) for ids in train["event_ids"]}

    def score(self, windows: pd.DataFrame) -> np.ndarray:
        return np.array([0.0 if tuple(ids) in self.seen else 1.0 for ids in windows["event_ids"]])
