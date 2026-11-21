import inspect
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from cascadelad.evaluation.thresholds import (
    THRESHOLDS,
    apply_threshold,
    compute_threshold,
    flag_rate,
    val_percentile,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
LABEL_NAMES = {"label", "labels", "y", "y_true", "y_label"}


def test_threshold_signatures_have_no_labels():
    # S5-1: thresholds see scores and config only.
    for name, fn in [*THRESHOLDS.items(), ("compute_threshold", compute_threshold)]:
        params = set(inspect.signature(fn).parameters)
        assert not params & LABEL_NAMES, f"{name} takes a label argument: {params}"


def test_val_percentile_flags_one_percent():
    # S5-2: 99th percentile of normal-only val scores flags 1% +- 0.2% of val.
    scores = np.random.default_rng(0).normal(size=10_000)
    tau = val_percentile(scores, percentile=99)
    share = float(apply_threshold(scores, tau).mean())
    assert 0.008 <= share <= 0.012


def test_flag_rate_on_fresh_test():
    # S5-2: tau from val, applied to a fresh draw of the same distribution.
    rng = np.random.default_rng(1)
    val, test = rng.normal(size=10_000), rng.normal(size=10_000)
    tau = flag_rate(val, rate=0.1)
    share = float(apply_threshold(test, tau).mean())
    assert 0.09 <= share <= 0.11


def test_apply_threshold_is_strictly_greater():
    pred = apply_threshold(np.array([0.5, 1.0, 1.5]), 1.0)
    assert pred.tolist() == [0, 0, 1]
    assert pred.dtype == np.int8


@pytest.mark.parametrize(
    "fn", [lambda s: val_percentile(s, percentile=99), lambda s: flag_rate(s, rate=0.1)]
)
@pytest.mark.parametrize("bad", [[], [1.0, np.nan], [1.0, np.inf]])
def test_threshold_rejects_empty_and_nan(fn, bad):
    with pytest.raises(ValueError):
        fn(np.array(bad, dtype=float))


def test_apply_threshold_rejects_nan():
    with pytest.raises(ValueError):
        apply_threshold(np.array([1.0, np.nan]), 0.5)


def test_compute_threshold_dispatch():
    scores = np.arange(101, dtype=float)
    assert compute_threshold(scores, {"method": "val_percentile", "percentile": 90}) == 90.0
    assert compute_threshold(scores, {"method": "flag_rate", "rate": 0.1}) == 90.0
    with pytest.raises(KeyError):
        compute_threshold(scores, {"method": "nope"})


@pytest.mark.parametrize("method", ["val_percentile", "flag_rate"])
def test_threshold_configs_compose(method):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)):
        cfg = compose(config_name="config", overrides=[f"threshold={method}", "run_name=x"])
    assert cfg.threshold.method == method
    assert method in THRESHOLDS


def test_default_threshold_is_val_percentile():
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)):
        cfg = compose(config_name="config", overrides=["run_name=x"])
    assert cfg.threshold.method == "val_percentile"
    assert cfg.threshold.percentile == 99
