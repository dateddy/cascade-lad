"""Loader interface and registry.

Every dataset loader returns a DataFrame with exactly the STANDARD_COLUMNS:
  timestamp  (datetime64)  time of the log line
  content   (str)         raw message text with parameter values, header fields stripped
  group_id   (str)         block id / session id the line belongs to
  label      (int8)        0 = normal, 1 = anomalous. The per-line label when the loader's
                           label_granularity is "line"; the group's label broadcast to every
                           line of the group when it is "group".
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import ClassVar

import pandas as pd

STANDARD_COLUMNS: tuple[str, ...] = ("timestamp", "content", "group_id", "label")
VALID_LABELS = frozenset({0, 1})
LABEL_GRANULARITIES: tuple[str, ...] = ("line", "group")
LABEL_DTYPE = "int8"


def check_standard(df: pd.DataFrame) -> None:
    if tuple(df.columns) != STANDARD_COLUMNS:
        raise ValueError(f"expected columns {list(STANDARD_COLUMNS)}, got {list(df.columns)}")
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise ValueError(f"timestamp must be datetime64, got {df['timestamp'].dtype}")
    for col in ("content", "group_id"):
        if not pd.api.types.is_string_dtype(df[col]):
            raise ValueError(f"{col} must be a string dtype, got {df[col].dtype}")
    if str(df["label"].dtype) != LABEL_DTYPE:
        raise ValueError(f"label must be {LABEL_DTYPE}, got {df['label'].dtype}")
    bad = set(df["label"].unique()) - VALID_LABELS
    if bad:
        raise ValueError(f"label must be 0 or 1, found {sorted(bad)}")


class LogLoader(ABC):
    label_granularity: ClassVar[str]  # "line" or "group"; required on every loader

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
        granularity = getattr(cls, "label_granularity", None)
        if granularity not in LABEL_GRANULARITIES:
            raise ValueError(
                f"loader '{name}': label_granularity must be one of "
                f"{list(LABEL_GRANULARITIES)}, got {granularity!r}"
            )
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_loader(name: str, cfg: Mapping) -> LogLoader:
    if name not in _REGISTRY:
        raise KeyError(f"unknown loader '{name}'; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](cfg)
