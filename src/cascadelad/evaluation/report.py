"""One metric dict per run and the shared Markdown metric table (D-c59c).

Every method goes through `evaluate` and `format_table`, so tables in run.log and results/
have the same columns and number format.
"""

from collections.abc import Sequence

import numpy as np

from cascadelad.evaluation.metrics import (
    average_precision,
    classification_metrics,
    recall_at_flag_rate,
)

NA = "n/a"
STD_SIGN = "±"
_COUNT_KEYS = ("tp", "fp", "fn", "tn")
_FIXED_COLUMNS = (
    ("P", "precision"),
    ("R", "recall"),
    ("F1", "f1"),
    ("Spec", "specificity"),
    ("MCC", "mcc"),
    ("AU-PR", "au_pr"),
    ("flag", "flag_rate"),
)


def _rate_key(rate: float) -> str:
    return f"recall_at_{round(rate * 100)}"


def evaluate(y_true, y_pred, scores, flag_rates: Sequence[float]) -> dict[str, float | int]:
    """Classification metrics plus AU-PR and recall at each flag rate.

    `scores` may be None for methods without a continuous score; the score-based entries
    are then nan. This is the only function that turns labels into metrics for a run.
    """
    out = dict(classification_metrics(y_true, y_pred))
    if scores is None:
        out["au_pr"] = float("nan")
        out.update({_rate_key(r): float("nan") for r in flag_rates})
        return out
    out["au_pr"] = average_precision(y_true, scores)
    out.update({_rate_key(r): recall_at_flag_rate(y_true, scores, r) for r in flag_rates})
    return out


def _cell(values: list[float], decimals: int, ddof: int, integer: bool) -> str:
    arr = np.asarray(values, dtype=float)
    if np.isnan(arr).any():
        return NA
    if arr.size == 1:
        return str(int(arr[0])) if integer else f"{arr[0]:.{decimals}f}"
    if arr.size <= ddof:
        raise ValueError(f"need more than ddof={ddof} seeds to compute a std, got {arr.size}")
    return f"{arr.mean():.{decimals}f} {STD_SIGN} {arr.std(ddof=ddof):.{decimals}f}"


def _recall_columns(rows: dict[str, list[dict]]) -> list[tuple[str, str]]:
    keys = {k for runs in rows.values() for run in runs for k in run if k.startswith("recall_at_")}
    ordered = sorted(keys, key=lambda k: int(k.removeprefix("recall_at_")))
    return [(f"R@{k.removeprefix('recall_at_')}%", k) for k in ordered]


def format_table(
    rows: dict[str, list[dict]],
    *,
    dataset: str,
    split: str,
    seeds: Sequence[int],
    threshold: str,
    decimals: int,
    ddof: int,
) -> str:
    """Markdown table, one row per method. A method maps to one metric dict per seed.

    One seed prints plain values; several seeds print mean ± std (std uses `ddof`).
    """
    columns = [
        *_FIXED_COLUMNS,
        *_recall_columns(rows),
        *((k.upper(), k) for k in _COUNT_KEYS),
    ]
    header = [name for name, _ in columns]
    lines = [
        f"dataset: {dataset} | split: {split} | seeds: {','.join(map(str, seeds))}"
        f" | threshold: {threshold}",
        "",
        "| " + " | ".join(["method", *header]) + " |",
        "|" + "|".join(["---", *["---:"] * len(header)]) + "|",
    ]
    for method, runs in rows.items():
        cells = [
            _cell([run[key] for run in runs], decimals, ddof, integer=key in _COUNT_KEYS)
            for _, key in columns
        ]
        lines.append("| " + " | ".join([method, *cells]) + " |")
    return "\n".join(lines) + "\n"
