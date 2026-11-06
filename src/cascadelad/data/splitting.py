"""Chronological train/val/test split of sessions (research invariants 1 and 3).

Sessions are ordered by start time and cut by count. HDFS blocks overlap in time, so a block
can start in one part and end in the next; such blocks are purged so that every kept train
block ends before every val block starts, and every kept val block ends before every test
block starts. Train additionally keeps normal blocks only.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

RATIO_KEYS = ("train_ratio", "val_ratio", "test_ratio")


@dataclass(frozen=True)
class SplitResult:
    train: list[str]
    val: list[str]
    test: list[str]
    n_purged: dict[str, int]
    n_train_anomalous_dropped: int


def chronological_split(bounds: pd.DataFrame, cfg: Mapping) -> SplitResult:
    """Split `bounds` (columns group_id, start, end, label) by start time.

    Returns group ids per split. Purged blocks are dropped from every split; anomalous
    blocks are additionally dropped from train (counted separately, not as purged).
    """
    train_r, val_r, test_r = (cfg[k] for k in RATIO_KEYS)
    if not math.isclose(train_r + val_r + test_r, 1.0):
        raise ValueError(f"split ratios must sum to 1, got {train_r + val_r + test_r}")

    ordered = bounds.sort_values(["start", "group_id"], kind="stable").reset_index(drop=True)
    n = len(ordered)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    if n_train < 1 or n_val < 1 or n - n_train - n_val < 1:
        raise ValueError(f"too few blocks ({n}) for ratios {train_r}/{val_r}/{test_r}")

    train_c = ordered.iloc[:n_train]
    val_c = ordered.iloc[n_train : n_train + n_val]
    test_c = ordered.iloc[n_train + n_val :]
    t_val = val_c["start"].iloc[0]  # first val block start
    t_test = test_c["start"].iloc[0]  # first test block start

    train_ok = train_c[train_c["end"] < t_val]
    val_ok = val_c[val_c["end"] < t_test]
    train_normal = train_ok[train_ok["label"] == 0]

    return SplitResult(
        train=train_normal["group_id"].tolist(),
        val=val_ok["group_id"].tolist(),
        test=test_c["group_id"].tolist(),
        n_purged={
            "train": len(train_c) - len(train_ok),
            "val": len(val_c) - len(val_ok),
            "test": 0,
        },
        n_train_anomalous_dropped=len(train_ok) - len(train_normal),
    )
