"""Group log lines into sequences.

Session mode (HDFS): one group id = one session, all lines share the session label.
Fixed mode (BGL, Thunderbird): consecutive lines form non-overlapping windows of `size` lines;
a window is anomalous if any of its lines is (D-cfbf). Both modes keep per-line labels.
"""

from collections.abc import Sequence

import pandas as pd

LABEL_RULES = ("same", "max")
WINDOW_ID_FORMAT = "w{:09d}"  # zero-padded so id order equals window order


def _check_nonempty(lines: pd.DataFrame) -> None:
    if lines.empty:
        raise ValueError("empty input: no log lines to group")


def session_bounds(lines: pd.DataFrame, label_rule: str = "same") -> pd.DataFrame:
    """One row per group: first/last line timestamp and the group's label.

    label_rule "same": every line of a group must carry the same label (else ValueError).
    label_rule "max": the group label is the max of its line labels (fixed windows).
    """
    _check_nonempty(lines)
    if label_rule not in LABEL_RULES:
        raise ValueError(f"unknown label_rule {label_rule!r}, expected one of {LABEL_RULES}")
    grouped = lines.groupby("group_id", sort=True)
    if label_rule == "same":
        n_labels = grouped["label"].nunique()
        if (n_labels > 1).any():
            bad = sorted(n_labels[n_labels > 1].index)
            raise ValueError(f"{len(bad)} groups have more than one label, e.g. {bad[:3]}")
    return grouped.agg(
        start=("timestamp", "min"), end=("timestamp", "max"), label=("label", "max")
    ).reset_index()


def assign_fixed_windows(lines: pd.DataFrame, size: int, step: int) -> tuple[pd.DataFrame, int]:
    """Replace `group_id` with a fixed-window id: row i goes to window i // size.

    `lines` must be in log order. Only non-overlapping windows are supported (step == size),
    so no window can cross a split boundary. The incomplete last window is dropped; returns
    the windowed lines and the number of dropped tail lines.
    """
    _check_nonempty(lines)
    if size < 1:
        raise ValueError(f"window size must be >= 1, got {size}")
    if step != size:
        raise ValueError(f"only non-overlapping windows are supported: step {step} != size {size}")
    n_full = len(lines) // size * size
    kept = lines.iloc[:n_full].copy()
    kept["group_id"] = [WINDOW_ID_FORMAT.format(i // size) for i in range(n_full)]
    return kept, len(lines) - n_full


def build_windows(
    lines: pd.DataFrame,
    token_col: str,
    extra_cols: Sequence[str] = (),
    label_rule: str = "same",
) -> pd.DataFrame:
    """One row per group with its events in row order, ordered by (timestamp, group_id).

    `timestamp` is the group's first line. `lines` must already be in log order. `line_labels`
    holds each line's label, aligned with `templates`; each name in `extra_cols` becomes a list
    column aligned the same way (e.g. each line's position). See `session_bounds` for
    `label_rule`.
    """
    _check_nonempty(lines)
    bounds = session_bounds(lines, label_rule)
    grouped = lines.groupby("group_id", sort=True)
    windows = pd.DataFrame(
        {
            "group_id": bounds["group_id"],
            "templates": bounds["group_id"].map(grouped[token_col].agg(list)),
            "timestamp": bounds["start"],
            "label": bounds["label"],
            "line_labels": bounds["group_id"].map(grouped["label"].agg(list)),
        }
    )
    for col in extra_cols:
        windows[col] = bounds["group_id"].map(grouped[col].agg(list))
    return windows.sort_values(["timestamp", "group_id"], kind="stable").reset_index(drop=True)
