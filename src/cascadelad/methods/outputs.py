"""Serialize method results: the metrics.json dict and the scores.parquet frame.

`ex` flattens the nested metrics dict, so `test.f1` is the primary method's F1 and
`baselines.<name>.test.f1` a baseline's (D-d181).
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from cascadelad.methods.detector_only import RULE, MethodResult

SCORED_SPLITS = ("val", "test")
THRESHOLD_SPLIT_VAL = "val"
THRESHOLD_SPLIT_NONE = "none"  # a fixed rule reads no split


def _plain(value):
    """numpy scalars to Python numbers, so json.dumps accepts them."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _method_entry(res: MethodResult) -> dict:
    return {
        "test": {k: _plain(v) for k, v in res.metrics.items()},
        "threshold": float(res.threshold),
        "threshold_method": res.threshold_method,
        "threshold_split": (
            THRESHOLD_SPLIT_NONE if res.threshold_method == RULE else THRESHOLD_SPLIT_VAL
        ),
        "val_flag_rate": float(res.val_flag_rate),
        "infer_ms_per_window": float(res.infer_ms_per_window),
    }


def metrics_json(
    primary: MethodResult, baselines: Mapping[str, MethodResult], *, max_lines: int | None
) -> dict:
    """Nested metrics dict: the primary method at the top level, baselines under `baselines`."""
    out = _method_entry(primary)
    out["data"] = {"max_lines": max_lines}
    out["baselines"] = {name: _method_entry(res) for name, res in baselines.items()}
    return out


def scores_frame(
    results: Sequence[MethodResult], splits: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    """One row per val and test window with each method's score and test prediction.

    Columns: split, group_id, timestamp, label, then `score_<name>` and `pred_<name>` for each
    result. `pred_<name>` is empty (<NA>) on val rows, where no decision is made.
    """
    parts = []
    for split in SCORED_SPLITS:
        df = splits[split]
        part = pd.DataFrame(
            {
                "split": split,
                "group_id": df["group_id"].to_numpy(),
                "timestamp": df["timestamp"].to_numpy(),
                "label": df["label"].to_numpy(),
            }
        )
        for res in results:
            if split == "val":
                part[f"score_{res.name}"] = res.val_scores
                part[f"pred_{res.name}"] = pd.array([pd.NA] * len(df), dtype="Int8")
            else:
                part[f"score_{res.name}"] = res.test_scores
                part[f"pred_{res.name}"] = pd.array(res.test_pred, dtype="Int8")
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    # Reorder to: base columns, then score_/pred_ pairs in result order.
    ordered = ["split", "group_id", "timestamp", "label"]
    for res in results:
        ordered += [f"score_{res.name}", f"pred_{res.name}"]
    return frame[ordered]
