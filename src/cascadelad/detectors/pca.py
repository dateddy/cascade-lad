"""PCA detector (Xu et al. 2009, as in loglizer; D-58a7).

Input is the per-window template count vector (length = vocab size). Counts are weighted by
TF-IDF (idf = log((N + 1) / (df + 1)) from train, so idf >= 0 and `<UNK>`, which has df = 0,
gets the largest weight), centered on the train mean, and projected on the top principal
components that explain `n_components` of the train variance. The score is the squared
prediction error (SPE): the squared norm of what the projection leaves out. Higher = more
anomalous; thresholding happens after scoring.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from cascadelad.detectors.base import Detector
from cascadelad.detectors.registry import register_detector

WEIGHTINGS = ("tfidf", "none")
EXPLAINED_TOL = 1e-12  # float slack when asking for 100% of the variance


@register_detector("pca")
class PCADetector(Detector):
    def __init__(self, cfg: Mapping, n_features: int) -> None:
        if cfg["term_weighting"] not in WEIGHTINGS:
            raise ValueError(
                f"term_weighting must be one of {WEIGHTINGS}, got {cfg['term_weighting']!r}"
            )
        if cfg["normalization"] != "zero_mean":
            raise ValueError(f"unsupported normalization {cfg['normalization']!r}, use 'zero_mean'")
        self.decision = cfg["decision"]
        self.term_weighting = cfg["term_weighting"]
        self.n_components = cfg["n_components"]
        self.n_features = n_features
        self.n_components_: int | None = None
        self._idf: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._proj: np.ndarray | None = None

    def _counts(self, windows: pd.DataFrame) -> np.ndarray:
        x = np.zeros((len(windows), self.n_features))
        for row, ids in enumerate(windows["event_ids"]):
            ids = np.asarray(ids, dtype=int)
            if ids.size and (ids.min() < 0 or ids.max() >= self.n_features):
                raise ValueError(f"event id outside [0, {self.n_features}) in window {row}")
            x[row] = np.bincount(ids, minlength=self.n_features)
        return x

    def _components(self, singular: np.ndarray) -> int:
        """Number of components: an int is taken as is; a float in (0, 1] is a variance share."""
        rank = singular.size
        if isinstance(self.n_components, int) and not isinstance(self.n_components, bool):
            return min(self.n_components, rank)
        energy = singular**2
        if energy.sum() == 0.0:
            return 0
        explained = np.cumsum(energy) / energy.sum()
        return int(min(np.searchsorted(explained, self.n_components - EXPLAINED_TOL) + 1, rank))

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> None:
        if len(train) == 0:
            raise ValueError("cannot fit PCA on an empty train set")
        x = self._counts(train)
        if self.term_weighting == "tfidf":
            df = (x > 0).sum(axis=0)
            self._idf = np.log((len(x) + 1) / (df + 1))
            x = x * self._idf
        self._mean = x.mean(axis=0)
        _, singular, vt = np.linalg.svd(x - self._mean, full_matrices=False)
        self.n_components_ = self._components(singular)
        self._proj = vt[: self.n_components_].T

    def score(self, windows: pd.DataFrame) -> np.ndarray:
        if self._proj is None:
            raise RuntimeError("PCADetector.score called before fit")
        x = self._counts(windows)
        if self._idf is not None:
            x = x * self._idf
        y = x - self._mean
        residual = y - y @ self._proj @ self._proj.T
        return (residual**2).sum(axis=1)
