from pathlib import Path

import numpy as np
import pytest

from cascadelad.evaluation.metrics import classification_metrics, recall_at_flag_rate
from cascadelad.evaluation.report import evaluate, format_table

SNAPSHOT = Path(__file__).parent / "snapshots" / "metric_table.md"
KW = {
    "dataset": "hdfs",
    "split": "chrono 70/10/20",
    "seeds": [0, 1, 2],
    "threshold": "val_percentile=99",
    "decimals": 4,
    "ddof": 1,
}


def _row(seed_shift: float = 0.0, au_pr: float = 0.5) -> dict:
    return {
        "tp": 8, "fp": 4, "fn": 2, "tn": 86, "n": 100,
        "precision": 0.6667 + seed_shift, "recall": 0.8, "f1": 0.7273, "specificity": 0.9556,
        "mcc": 0.6964, "au_pr": au_pr, "flag_rate": 0.12,
        "recall_at_10": 0.7, "recall_at_20": 0.9,
    }  # fmt: skip


ROWS = {
    "always-anomaly": [
        {
            **_row(),
            "tp": 10,
            "fp": 90,
            "fn": 0,
            "tn": 0,
            "precision": 0.1,
            "recall": 1.0,
            "f1": 0.1818,
            "specificity": 0.0,
            "mcc": 0.0,
            "au_pr": float("nan"),
            "flag_rate": 1.0,
            "recall_at_10": 0.1,
            "recall_at_20": 0.2,
        }  # fmt: skip
    ],
    "pca": [_row(-0.01), _row(0.0, au_pr=0.52), _row(0.01, au_pr=0.54)],
}


def test_table_matches_snapshot():
    table = format_table(ROWS, **KW)
    assert "\r" not in table
    assert table == SNAPSHOT.read_bytes().decode("utf-8")


def test_table_header_states_dataset_split_seeds_threshold():
    first = format_table(ROWS, **KW).splitlines()[0]
    for part in ("hdfs", "chrono 70/10/20", "0,1,2", "val_percentile=99"):
        assert part in first


def test_table_columns_in_agreed_order():
    header = format_table(ROWS, **KW).splitlines()[2]
    cols = [c.strip() for c in header.strip("|").split("|")]
    assert cols == [
        "method", "P", "R", "F1", "Spec", "MCC", "AU-PR", "flag", "R@10%", "R@20%",
        "TP", "FP", "FN", "TN",
    ]  # fmt: skip


def test_multi_seed_mean_std_ddof1():
    rows = {"m": [{**_row(), "f1": 0.5}, {**_row(), "f1": 0.7}, {**_row(), "f1": 0.9}]}
    line = format_table(rows, **KW).splitlines()[4]
    assert "0.7000 ± 0.2000" in line  # mean 0.7, sample std (ddof=1) 0.2


def test_ddof_is_configurable():
    rows = {"m": [{**_row(), "f1": 0.5}, {**_row(), "f1": 0.9}]}
    line = format_table(rows, **{**KW, "ddof": 0}).splitlines()[4]
    assert "0.7000 ± 0.2000" in line  # population std of [0.5, 0.9] is 0.2


def test_single_seed_no_plus_minus():
    line = format_table({"m": [_row()]}, **KW).splitlines()[4]
    assert "±" not in line
    assert "| 8 | 4 | 2 | 86 |" in line  # counts stay integers


def test_nan_prints_na():
    line = format_table({"m": [{**_row(), "au_pr": float("nan")}]}, **KW).splitlines()[4]
    assert "n/a" in line


def test_evaluate_keys_and_values():
    y_true = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 1])
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.3, 0.1, 0.2, 0.4, 0.5, 0.6])
    out = evaluate(y_true, y_pred, scores, flag_rates=[0.1, 0.2])
    expected = classification_metrics(y_true, y_pred)
    for key, value in expected.items():
        assert out[key] == value
    assert set(out) == set(expected) | {"au_pr", "recall_at_10", "recall_at_20"}
    assert out["recall_at_20"] == recall_at_flag_rate(y_true, scores, 0.2)


def test_evaluate_without_scores_gives_nan_score_metrics():
    out = evaluate(np.array([1, 0]), np.array([1, 0]), None, flag_rates=[0.5])
    assert np.isnan(out["au_pr"])
    assert np.isnan(out["recall_at_50"])
    assert out["f1"] == pytest.approx(1.0)
