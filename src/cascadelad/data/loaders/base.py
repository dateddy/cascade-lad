"""Loader interface and registry.

Every dataset loader returns a DataFrame with exactly the STANDARD_COLUMNS:
  timestamp  (datetime64)  time of the log line
  content    (str)         message text, header fields stripped
  group_id   (str)         block id / session id the line belongs to
  label      (int)         0 = normal, 1 = anomalous (label of the line's group)
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping

import pandas as pd

STANDARD_COLUMNS: tuple[str, ...] = ("timestamp", "content", "group_id", "label")
VALID_LABELS = frozenset({0, 1})


def check_standard(df: pd.DataFrame) -> None:
    if tuple(df.columns) != STANDARD_COLUMNS:
        raise ValueError(f"expected columns {list(STANDARD_COLUMNS)}, got {list(df.columns)}")
    bad = set(df["label"].unique()) - VALID_LABELS
    if bad:
        raise ValueError(f"label must be 0 or 1, found {sorted(bad)}")


class LogLoader(ABC):
    def __init__(self, cfg: Mapping) -> None:
        self.cfg = cfg

    @abstractmethod
    def _load(self) -> pd.DataFrame:
        """Read the raw files and return a frame with the standard columns."""

    def load(self) -> pd.DataFrame:
        df = self._load()
        check_standard(df)
        return df


_REGISTRY: dict[str, type[LogLoader]] = {}


def register_loader(name: str) -> Callable[[type[LogLoader]], type[LogLoader]]:
    def decorator(cls: type[LogLoader]) -> type[LogLoader]:
        if name in _REGISTRY:
            raise ValueError(f"loader '{name}' is already registered")
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_loader(name: str, cfg: Mapping) -> LogLoader:
    if name not in _REGISTRY:
        raise KeyError(f"unknown loader '{name}'; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](cfg)
