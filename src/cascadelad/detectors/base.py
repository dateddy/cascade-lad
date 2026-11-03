"""Detector interface shared by PCA, DeepLog and LogBERT.

A window DataFrame has the columns:
  group_id   (str)        block id / time-window id
  event_ids  (list[int])  vocab ids of the window's events, unseen templates already mapped to <UNK>
  timestamp               time of the window's first event
Detectors must not read a `label` column: anomaly labels never reach fit or score.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

WINDOW_COLUMNS: tuple[str, ...] = ("group_id", "event_ids", "timestamp")


class Detector(ABC):
    @abstractmethod
    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> None:
        """Fit on normal train windows. `val` (normal only) is for early stopping on val loss."""

    @abstractmethod
    def score(self, windows: pd.DataFrame) -> np.ndarray:
        """Return a 1-D float array, one value per row in the same order; higher = more anomalous.

        Never threshold here: thresholding (incl. top-g) is applied after scoring.
        """

    @staticmethod
    def check_windows(windows: pd.DataFrame) -> None:
        missing = [c for c in WINDOW_COLUMNS if c not in windows.columns]
        if missing:
            raise ValueError(f"windows is missing required columns: {missing}")
