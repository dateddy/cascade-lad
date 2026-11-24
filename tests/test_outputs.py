import json

import numpy as np
import pandas as pd

from cascadelad.methods.detector_only import MethodResult
from cascadelad.methods.outputs import metrics_json, scores_frame


def result(name, method="val_percentile", tau=0.7, n_val=3, n_test=4) -> MethodResult:
    return MethodResult(
        name=name,
        threshold_method=method,
        threshold=tau,
        val_flag_rate=0.0,
        val_scores=np.linspace(0, 0.5, n_val),
        test_scores=np.linspace(0, 1, n_test),
        test_pred=np.array([0, 0, 1, 1], dtype=np.int8),
        infer_ms_per_window=0.25,
        metrics={"precision": 0.5, "recall": 1.0, "f1": 2 / 3, "fp": np.int64(1), "fn": 0, "tp": 1},
    )


def splits(n_val=3, n_test=4) -> dict[str, pd.DataFrame]:
    def frame(n, start, labels):
        return pd.DataFrame(
            {
                "group_id": [f"g{start + i}" for i in range(n)],
                "event_ids": [[1]] * n,
                "timestamp": pd.date_range("2026-01-01", periods=n, freq="min"),
                "label": labels,
            }
        )

    return {
        "train": frame(2, 0, [0, 0]),
        "val": frame(n_val, 10, [0] * n_val),
        "test": frame(n_test, 20, [0, 1, 0, 1]),
    }


def test_metrics_json_keys():
    pca = result("pca")
    base = result("always_anomaly", method="rule", tau=0.5)
    out = metrics_json(pca, {"always_anomaly": base}, max_lines=100000)
    assert out["test"]["f1"] == pca.metrics["f1"]
    assert out["test"]["fp"] == 1 and out["test"]["fn"] == 0
    assert out["threshold"] == 0.7
    assert out["threshold_method"] == "val_percentile"
    assert out["threshold_split"] == "val"
    assert out["infer_ms_per_window"] == 0.25
    assert out["data"]["max_lines"] == 100000
    b = out["baselines"]["always_anomaly"]
    assert b["threshold_method"] == "rule"
    assert b["threshold_split"] == "none"
    assert b["threshold"] == 0.5
    assert b["test"]["recall"] == 1.0


def test_metrics_json_is_json_serializable():
    out = metrics_json(result("pca"), {}, max_lines=None)
    again = json.loads(json.dumps(out))
    assert again["data"]["max_lines"] is None
    assert again["test"]["fp"] == 1


def test_scores_frame_columns():
    frame = scores_frame([result("pca"), result("unseen_sequence")], splits())
    assert list(frame.columns) == [
        "split",
        "group_id",
        "timestamp",
        "label",
        "score_pca",
        "pred_pca",
        "score_unseen_sequence",
        "pred_unseen_sequence",
    ]
    assert frame["split"].tolist() == ["val"] * 3 + ["test"] * 4
    assert frame["group_id"].tolist() == ["g10", "g11", "g12", "g20", "g21", "g22", "g23"]
    assert frame["label"].tolist() == [0, 0, 0, 0, 1, 0, 1]
    assert frame["pred_pca"].isna().tolist() == [True] * 3 + [False] * 4
    assert frame["pred_pca"].dropna().tolist() == [0, 0, 1, 1]
    assert np.allclose(frame["score_pca"].to_numpy()[3:], np.linspace(0, 1, 4))


def test_scores_frame_writes_parquet(tmp_path):
    frame = scores_frame([result("pca")], splits())
    path = tmp_path / "scores.parquet"
    frame.to_parquet(path, index=False)
    back = pd.read_parquet(path)
    assert back["pred_pca"].isna().sum() == 3
    assert len(back) == 7
