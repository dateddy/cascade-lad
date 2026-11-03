"""Detector interface shared by PCA, DeepLog and LogBERT.

A window DataFrame has the columns:
  group_id   (str)        block id / time-window id, unique within one DataFrame
  event_ids  (list[int])  vocab ids of the window's events, unseen templates already mapped to <UNK>
  timestamp               time of the window's first event
It must not have a `label` column: anomaly labels never reach fit or score, and the caller
strips them. The base class enforces this on every subclass (see `__init_subclass__`).
"""

import functools
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

WINDOW_COLUMNS: tuple[str, ...] = ("group_id", "event_ids", "timestamp")
LABEL_COLUMN = "label"


class Detector(ABC):
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "fit" in cls.__dict__:
            cls.fit = _checked_fit(cls.__dict__["fit"])
        if "score" in cls.__dict__:
            cls.score = _checked_score(cls.__dict__["score"])

    @abstractmethod
    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> None:
        """Fit on normal train windows. `val` (normal only) is for early stopping on val loss."""

    @abstractmethod
    def score(self, windows: pd.DataFrame) -> np.ndarray:
        """Score many windows at once.

        Returns a 1-D float array with one value per row, in row order: `scores[i]` belongs to
        `windows["group_id"].iloc[i]` (group_id is unique, so this is aligned by group_id).
        Higher = more anomalous. Never threshold here: thresholding (incl. top-g) is applied
        after scoring.
        """

    @staticmethod
    def check_windows(windows: pd.DataFrame) -> None:
        missing = [c for c in WINDOW_COLUMNS if c not in windows.columns]
        if missing:
            raise ValueError(f"windows is missing required columns: {missing}")
        if LABEL_COLUMN in windows.columns:
            raise ValueError(
                f"windows has a '{LABEL_COLUMN}' column; strip labels before fit/score"
            )
        if windows["group_id"].duplicated().any():
            raise ValueError("windows has duplicate group_id values")


def _checked_fit(fit):
    @functools.wraps(fit)
    def wrapper(self, train, val=None):
        Detector.check_windows(train)
        if val is not None:
            Detector.check_windows(val)
        return fit(self, train, val)

    return wrapper


def _checked_score(score):
    @functools.wraps(score)
    def wrapper(self, windows):
        Detector.check_windows(windows)
        out = np.asarray(score(self, windows), dtype=float)
        if out.shape != (len(windows),):
            raise ValueError(f"score must return shape ({len(windows)},), got {out.shape}")
        return out

    return wrapper
