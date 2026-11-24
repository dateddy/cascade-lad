import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from cascadelad.methods.pipeline import run_pipeline
from cascadelad.utils.run import setup_logging

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
N_FEATURES = 8


def load_cfg(experiment: str) -> dict:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)):
        cfg = compose(config_name="config", overrides=[f"experiment={experiment}", "run_name=t"])
    return OmegaConf.to_container(cfg, resolve=True)


def split(sequences, labels, start) -> pd.DataFrame:
    n = len(sequences)
    return pd.DataFrame(
        {
            "group_id": [f"g{start + i}" for i in range(n)],
            "event_ids": sequences,
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="min"),
            "label": labels,
            "line_labels": [[0] * len(s) for s in sequences],
            "line_pos": [list(range(len(s))) for s in sequences],
        }
    )


def make_splits():
    rng = np.random.default_rng(0)

    def normal(n, start):
        seqs = []
        for _ in range(n):
            k = int(rng.integers(0, 5))
            seqs.append([2] * k + [3] * k + [4])
        return split(seqs, [0] * n, start)

    test_seqs = [[2, 3, 4]] * 30 + [[2, 2, 5], [6, 7], [2, 5, 5]] + [[2, 2, 3, 3, 4]] * 7
    test_labels = [0] * 30 + [1, 1, 1] + [0] * 7
    return {
        "train": normal(80, 0),
        "val": normal(40, 1000),
        "test": split(test_seqs, test_labels, 2000),
    }


@pytest.fixture
def run(tmp_path):
    cfg = load_cfg("baseline_hdfs_pca")
    logger = setup_logging(tmp_path, name="test_evaluate_script")
    out = run_pipeline(cfg, make_splits(), N_FEATURES, logger, tmp_path)
    yield tmp_path, out
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def test_run_pipeline_writes_outputs(run):
    run_dir, _ = run
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["threshold_method"] == "val_percentile"
    assert metrics["threshold_split"] == "val"
    assert np.isfinite(metrics["threshold"])
    assert {"precision", "recall", "f1", "fp", "fn"} <= set(metrics["test"])
    assert metrics["data"]["max_lines"] is None
    base = metrics["baselines"]
    assert set(base) == {"always_anomaly", "unseen_sequence"}
    assert base["always_anomaly"]["test"]["recall"] == 1.0
    assert base["always_anomaly"]["threshold_method"] == "rule"
    scores = pd.read_parquet(run_dir / "scores.parquet")
    assert len(scores) == 40 + 40
    for col in ("score_pca", "pred_pca", "score_always_anomaly", "score_unseen_sequence"):
        assert col in scores.columns


def test_run_log_has_model_eval_and_outputs_sections(run):
    run_dir, _ = run
    log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert log.index("[model]") < log.index("[eval]") < log.index("[outputs]")
    assert "| method |" in log
    assert "pca" in log and "always_anomaly" in log and "unseen_sequence" in log
    assert "threshold: val_percentile" in log
    assert "[1m" not in log  # plain text, no color codes


def test_run_pipeline_returns_metrics(run):
    _, out = run
    assert out["metrics"]["test"]["f1"] >= 0.0


@pytest.mark.parametrize("experiment", ["baseline_hdfs_pca", "baseline_hdfs_pca_dev"])
def test_experiment_configs_compose(experiment):
    cfg = load_cfg(experiment)
    assert cfg["model"]["name"] == "pca"
    assert cfg["threshold"]["method"] == "val_percentile"
    assert set(cfg["baselines"]) == {"always_anomaly", "unseen_sequence"}
    assert cfg["dataset"]["name"] == "hdfs"
    assert cfg["split"]["mode"] == "chronological"
    expected = 100000 if experiment.endswith("_dev") else None
    assert cfg["dataset"]["max_lines"] == expected
