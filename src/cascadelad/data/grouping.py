"""Group log lines into sessions (HDFS: one block id = one session)."""

import pandas as pd


def _check_nonempty(lines: pd.DataFrame) -> None:
    if lines.empty:
        raise ValueError("empty input: no log lines to group")


def session_bounds(lines: pd.DataFrame) -> pd.DataFrame:
    """One row per group: first/last line timestamp and the group's label."""
    _check_nonempty(lines)
    grouped = lines.groupby("group_id", sort=True)
    n_labels = grouped["label"].nunique()
    if (n_labels > 1).any():
        bad = sorted(n_labels[n_labels > 1].index)
        raise ValueError(f"{len(bad)} groups have more than one label, e.g. {bad[:3]}")
    return grouped.agg(
        start=("timestamp", "min"), end=("timestamp", "max"), label=("label", "first")
    ).reset_index()


def build_windows(lines: pd.DataFrame, token_col: str) -> pd.DataFrame:
    """One row per group with its events in row order, ordered by (timestamp, group_id).

    `timestamp` is the group's first line. `lines` must already be in log order.
    """
    _check_nonempty(lines)
    bounds = session_bounds(lines)
    events = lines.groupby("group_id", sort=True)[token_col].agg(list)
    windows = pd.DataFrame(
        {
            "group_id": bounds["group_id"],
            "templates": bounds["group_id"].map(events),
            "timestamp": bounds["start"],
            "label": bounds["label"],
        }
    )
    return windows.sort_values(["timestamp", "group_id"], kind="stable").reset_index(drop=True)
