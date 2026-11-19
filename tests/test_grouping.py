import pandas as pd
import pytest

from cascadelad.data.grouping import assign_fixed_windows, build_windows, session_bounds


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
    assert list(windows.columns) == ["group_id", "templates", "timestamp", "label", "line_labels"]
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


def test_extra_cols_are_aligned_lists():
    lines = LINES.assign(line_pos=[10, 11, 12, 13, 14])
    windows = build_windows(lines, "template", extra_cols=("line_pos",)).set_index("group_id")
    assert windows.loc["a", "line_pos"] == [11, 12, 13]
    assert windows.loc["b", "line_pos"] == [10, 14]
    assert len(windows.loc["a", "line_pos"]) == len(windows.loc["a", "templates"])
    default = build_windows(LINES, "template")
    assert list(default.columns) == ["group_id", "templates", "timestamp", "label", "line_labels"]


def make_stream(labels: list[int]) -> pd.DataFrame:
    """One line per second in log order; group_id is the source node (not the window)."""
    n = len(labels)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=n, freq="s"),
            "group_id": ["node"] * n,
            "label": labels,
            "template": [f"E{i % 3}" for i in range(n)],
        }
    )


def test_fixed_windows_have_exactly_size_lines():
    """S3-5: every window has exactly `size` lines."""
    lines, _ = assign_fixed_windows(make_stream([0] * 12), size=4, step=4)
    windows = build_windows(lines, "template", label_rule="max")
    assert windows["group_id"].tolist() == ["w000000000", "w000000001", "w000000002"]
    assert windows["templates"].map(len).tolist() == [4, 4, 4]


def test_fixed_window_tail_dropped_and_counted():
    """S3-5: the incomplete last window is dropped and its line count returned."""
    lines, n_tail = assign_fixed_windows(make_stream([0] * 10), size=4, step=4)
    assert n_tail == 2
    assert len(lines) == 8
    assert lines["group_id"].nunique() == 2


def test_fixed_window_label_is_max_of_lines():
    """S3-5: window label = max of its line labels; per-line labels are kept."""
    lines, _ = assign_fixed_windows(make_stream([0, 0, 1, 0, 0, 0, 0, 0]), size=4, step=4)
    windows = build_windows(lines, "template", label_rule="max")
    assert windows["label"].tolist() == [1, 0]
    assert windows["line_labels"].tolist() == [[0, 0, 1, 0], [0, 0, 0, 0]]


def test_line_labels_aligned_length():
    lines, _ = assign_fixed_windows(make_stream([0, 1] * 6), size=3, step=3)
    windows = build_windows(lines, "template", label_rule="max")
    assert (windows["line_labels"].map(len) == windows["templates"].map(len)).all()


def test_fixed_windows_follow_row_order():
    lines, _ = assign_fixed_windows(make_stream([0] * 6), size=3, step=3)
    assert lines["group_id"].tolist() == ["w000000000"] * 3 + ["w000000001"] * 3


@pytest.mark.parametrize(("size", "step"), [(4, 2), (4, 8), (0, 0)])
def test_step_not_equal_size_raises(size, step):
    with pytest.raises(ValueError, match="non-overlapping|size must be"):
        assign_fixed_windows(make_stream([0] * 8), size=size, step=step)


def test_fixed_window_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        assign_fixed_windows(make_stream([]), size=4, step=4)


def test_mixed_labels_need_max_rule():
    lines, _ = assign_fixed_windows(make_stream([0, 1, 0, 0]), size=4, step=4)
    with pytest.raises(ValueError, match="more than one label"):
        session_bounds(lines)
    assert session_bounds(lines, label_rule="max")["label"].tolist() == [1]


def test_unknown_label_rule_raises():
    with pytest.raises(ValueError, match="unknown label_rule"):
        session_bounds(LINES, label_rule="min")


def test_session_line_labels_broadcast():
    """Session mode: every line of a block carries the block label."""
    windows = build_windows(LINES, "template").set_index("group_id")
    assert windows.loc["a", "line_labels"] == [0, 0, 0]
    assert windows.loc["b", "line_labels"] == [1, 1]
