import pandas as pd
import pytest

from cascadelad.data.grouping import build_windows, session_bounds


def make_lines(rows: list[tuple[str, str, int, str]]) -> pd.DataFrame:
    """rows: (timestamp, group_id, label, template)"""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime([r[0] for r in rows]),
            "group_id": [r[1] for r in rows],
            "label": [r[2] for r in rows],
            "template": [r[3] for r in rows],
        }
    )


LINES = make_lines(
    [
        ("2020-01-01 00:00:05", "b", 1, "E2"),
        ("2020-01-01 00:00:01", "a", 0, "E1"),
        ("2020-01-01 00:00:09", "a", 0, "E3"),
        ("2020-01-01 00:00:02", "a", 0, "E2"),
        ("2020-01-01 00:00:07", "b", 1, "E1"),
    ]
)


def test_session_bounds_start_end_label():
    bounds = session_bounds(LINES).set_index("group_id")
    assert bounds.loc["a", "start"] == pd.Timestamp("2020-01-01 00:00:01")
    assert bounds.loc["a", "end"] == pd.Timestamp("2020-01-01 00:00:09")
    assert bounds.loc["b", "start"] == pd.Timestamp("2020-01-01 00:00:05")
    assert bounds.loc["b", "end"] == pd.Timestamp("2020-01-01 00:00:07")
    assert bounds.loc["a", "label"] == 0
    assert bounds.loc["b", "label"] == 1
    assert list(session_bounds(LINES).columns) == ["group_id", "start", "end", "label"]


def test_mixed_label_group_raises():
    bad = make_lines([("2020-01-01", "a", 0, "E1"), ("2020-01-02", "a", 1, "E2")])
    with pytest.raises(ValueError, match="more than one label"):
        session_bounds(bad)


def test_windows_keep_row_order_within_group():
    # row order (not timestamp order) defines the event sequence
    windows = build_windows(LINES, "template").set_index("group_id")
    assert windows.loc["a", "templates"] == ["E1", "E3", "E2"]
    assert windows.loc["b", "templates"] == ["E2", "E1"]


def test_windows_columns_timestamp_and_order():
    windows = build_windows(LINES, "template")
    assert list(windows.columns) == ["group_id", "templates", "timestamp", "label"]
    assert list(windows["group_id"]) == ["a", "b"]  # ordered by start timestamp
    assert windows["timestamp"].iloc[0] == pd.Timestamp("2020-01-01 00:00:01")


def test_ties_broken_by_group_id_and_deterministic():
    tied = make_lines(
        [
            ("2020-01-01", "z", 0, "E1"),
            ("2020-01-01", "m", 0, "E1"),
            ("2020-01-01", "a", 0, "E1"),
        ]
    )
    first = build_windows(tied, "template")
    assert list(first["group_id"]) == ["a", "m", "z"]
    assert first.equals(build_windows(tied.iloc[::-1].reset_index(drop=True), "template"))


def test_empty_input_raises():
    empty = LINES.iloc[0:0]
    with pytest.raises(ValueError, match="empty"):
        session_bounds(empty)
    with pytest.raises(ValueError, match="empty"):
        build_windows(empty, "template")
