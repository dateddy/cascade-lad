from pathlib import Path

import numpy as np
import pytest

from cascadelad.evaluation.metrics import (
    bootstrap_ci,
    classification_metrics,
    recall_at_flag_rate,
    resolve_block_length,
)
from cascadelad.evaluation.report import evaluate, format_ci_table, format_table

SNAPSHOT = Path(__file__).parent / "snapshots" / "metric_table.md"
CI_SNAPSHOT = Path(__file__).parent / "snapshots" / "ci_table.md"
BOOT = {
    "ci_metrics": ("f1", "recall"),
    "n_resamples": 100,
    "seed": 0,
    "ci_level": 0.95,
    "block_length": None,
}
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
    out = evaluate(y_true, y_pred, scores, flag_rates=[0.1, 0.2], **BOOT)
    expected = classification_metrics(y_true, y_pred)
    for key, value in expected.items():
        assert out[key] == value
    ci_keys = {"f1_ci_low", "f1_ci_high", "recall_ci_low", "recall_ci_high", "ci_block_length"}
    assert set(out) == set(expected) | {"au_pr", "recall_at_10", "recall_at_20"} | ci_keys
    assert out["recall_at_20"] == recall_at_flag_rate(y_true, scores, 0.2)


def test_evaluate_without_scores_gives_nan_score_metrics():
    out = evaluate(np.array([1, 0]), np.array([1, 0]), None, flag_rates=[0.5], **BOOT)
    assert np.isnan(out["au_pr"])
    assert np.isnan(out["recall_at_50"])
    assert out["f1"] == pytest.approx(1.0)


def test_evaluate_ci_keys_match_bootstrap_ci():
    rng = np.random.default_rng(2)
    y_true = (rng.random(300) < 0.2).astype(int)
    y_pred = np.where(y_true == 1, rng.random(300) < 0.7, rng.random(300) < 0.1).astype(int)
    out = evaluate(y_true, y_pred, None, flag_rates=[], **BOOT)
    ci = bootstrap_ci(
        y_true,
        y_pred,
        BOOT["ci_metrics"],
        n_resamples=BOOT["n_resamples"],
        seed=BOOT["seed"],
        ci_level=BOOT["ci_level"],
        block_length=BOOT["block_length"],
    )
    for name, (_, lower, upper) in ci.items():
        assert out[f"{name}_ci_low"] == lower
        assert out[f"{name}_ci_high"] == upper
    assert out["ci_block_length"] == resolve_block_length(300, None) == 7


def _ci_run(f1: float, lo: float, hi: float) -> dict:
    return {
        "f1": f1, "f1_ci_low": lo, "f1_ci_high": hi,
        "recall": 0.8, "recall_ci_low": 0.7, "recall_ci_high": 0.9,
        "ci_block_length": 49,
    }  # fmt: skip


CI_ROWS = {
    "always-anomaly": [_ci_run(0.1818, 0.17, 0.19)],
    "pca": [_ci_run(0.70, 0.65, 0.75), _ci_run(0.72, 0.66, 0.77), _ci_run(0.74, 0.69, 0.79)],
}
CI_KW = {"dataset": "hdfs", "split": "chrono 70/10/20", "seeds": [0, 1, 2], "ci_level": 0.95,
         "decimals": 4}  # fmt: skip


def test_ci_table_matches_snapshot():
    table = format_ci_table(CI_ROWS, **CI_KW)
    assert "\r" not in table
    assert table == CI_SNAPSHOT.read_bytes().decode("utf-8")


def test_ci_table_one_row_per_method_and_seed():
    lines = format_ci_table(CI_ROWS, **CI_KW).splitlines()
    assert lines[2].startswith("| method | seed | F1 | F1 CI | R | R CI | L |")
    assert len(lines) == 4 + 1 + 3  # header, blank, column names, rule, then rows
    assert "| pca | 2 | 0.7400 | [0.6900, 0.7900] |" in lines[-1]
    assert "| always-anomaly | - |" in lines[4]  # single deterministic run: no seed


def test_ci_table_rejects_seed_count_mismatch():
    with pytest.raises(ValueError):
        format_ci_table({"pca": [_ci_run(0.7, 0.6, 0.8)] * 2}, **CI_KW)
