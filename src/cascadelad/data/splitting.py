"""Train/val/test split of groups (sessions or windows); research invariants 1 and 3.

`chronological` (default): groups are ordered by start time and cut by count. HDFS blocks
overlap in time, so a block can start in one part and end in the next; with `purge` such
blocks are dropped so that every kept train block ends before every val block starts, and
every kept val block ends before every test block starts (D-7d73). Fixed windows never
overlap, so they are split without purging.

`random`: groups are shuffled with a fixed seed and cut by the same counts. It exists only to
compare HDFS with published numbers that use a random split; it is never the main protocol.

In both modes anomalous groups are dropped from train when `drop_train_anomalies` is true
(default) and from val when `drop_val_anomalies` is true (default). `drop_train_anomalies:
false` is the dirty-train ablation, a labeled exception to invariant 3 (D-fb5d). Test always
keeps its natural anomaly ratio.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

RATIO_KEYS = ("train_ratio", "val_ratio", "test_ratio")
MODES = ("chronological", "random")


@dataclass(frozen=True)
class SplitResult:
    train: list[str]
    val: list[str]
    test: list[str]
    n_purged: dict[str, int]
    n_train_anomalous_dropped: int
    n_val_anomalous_dropped: int
    n_train_anomalous_kept: int


def _cut_counts(n: int, cfg: Mapping) -> tuple[int, int]:
    """Number of train and val groups; test gets the rest. Raises on bad ratios or too few."""
    train_r, val_r, test_r = (cfg[k] for k in RATIO_KEYS)
    if not math.isclose(train_r + val_r + test_r, 1.0):
        raise ValueError(f"split ratios must sum to 1, got {train_r + val_r + test_r}")
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    if n_train < 1 or n_val < 1 or n - n_train - n_val < 1:
        raise ValueError(f"too few groups ({n}) for ratios {train_r}/{val_r}/{test_r}")
    return n_train, n_val


def _finish(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, n_purged: dict, cfg: Mapping
) -> SplitResult:
    """Apply the anomaly-drop flags to train and val and return group ids in split order."""
    train_kept = train[train["label"] == 0] if cfg["drop_train_anomalies"] else train
    val_kept = val[val["label"] == 0] if cfg["drop_val_anomalies"] else val
    return SplitResult(
        train=train_kept["group_id"].tolist(),
        val=val_kept["group_id"].tolist(),
        test=test["group_id"].tolist(),
        n_purged=n_purged,
        n_train_anomalous_dropped=len(train) - len(train_kept),
        n_val_anomalous_dropped=len(val) - len(val_kept),
        n_train_anomalous_kept=int((train_kept["label"] != 0).sum()),
    )


def chronological_split(bounds: pd.DataFrame, cfg: Mapping) -> SplitResult:
    """Split `bounds` (columns group_id, start, end, label) by start time.

    With cfg["purge"], groups that straddle a split boundary are dropped from train and val
    and counted in `n_purged`; anomalous groups dropped by the flags are counted separately.
    """
    ordered = bounds.sort_values(["start", "group_id"], kind="stable").reset_index(drop=True)
    n_train, n_val = _cut_counts(len(ordered), cfg)
    train = ordered.iloc[:n_train]
    val = ordered.iloc[n_train : n_train + n_val]
    test = ordered.iloc[n_train + n_val :]
    n_purged = {"train": 0, "val": 0, "test": 0}
    if cfg["purge"]:
        t_val = val["start"].iloc[0]  # first val group start
        t_test = test["start"].iloc[0]  # first test group start
        train_ok = train[train["end"] < t_val]
        val_ok = val[val["end"] < t_test]
        n_purged = {"train": len(train) - len(train_ok), "val": len(val) - len(val_ok), "test": 0}
        train, val = train_ok, val_ok
    return _finish(train, val, test, n_purged, cfg)


def random_split(bounds: pd.DataFrame, cfg: Mapping) -> SplitResult:
    """Shuffle groups with cfg["seed"] and cut by count; no purge.

    Groups are sorted by id before shuffling, so the result does not depend on input order.
    Within each split, ids are ordered by (start, group_id).
    """
    base = bounds.sort_values("group_id", kind="stable").reset_index(drop=True)
    n_train, n_val = _cut_counts(len(base), cfg)
    perm = np.random.default_rng(cfg["seed"]).permutation(len(base))
    shuffled = base.iloc[perm]

    def part(rows: pd.DataFrame) -> pd.DataFrame:
        return rows.sort_values(["start", "group_id"], kind="stable")

    return _finish(
        part(shuffled.iloc[:n_train]),
        part(shuffled.iloc[n_train : n_train + n_val]),
        part(shuffled.iloc[n_train + n_val :]),
        {"train": 0, "val": 0, "test": 0},
        cfg,
    )


def split_groups(bounds: pd.DataFrame, cfg: Mapping) -> SplitResult:
    """Split by cfg["mode"]: "chronological" or "random"."""
    if cfg["mode"] == "chronological":
        return chronological_split(bounds, cfg)
    if cfg["mode"] == "random":
        return random_split(bounds, cfg)
    raise ValueError(f"unknown split mode {cfg['mode']!r}, expected one of {MODES}")
