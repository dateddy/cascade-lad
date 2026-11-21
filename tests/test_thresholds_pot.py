import numpy as np
import pytest

from cascadelad.evaluation.thresholds import THRESHOLDS, compute_threshold

POT = {"method": "pot", "init_level": 0.98, "risk": 0.001, "min_excesses": 10}


def _pot(scores, **over):
    return compute_threshold(scores, {**POT, **over})


def test_pot_finite_above_median():
    # S5-2: finite threshold above the val median.
    scores = np.random.default_rng(0).exponential(size=10_000)
    tau = _pot(scores)
    assert np.isfinite(tau)
    assert tau > np.median(scores)


def test_pot_matches_exponential_tail():
    # Exponential(1) has a GPD tail with gamma = 0, so z_q ~ -log(q): q = 0.001 -> ~6.9.
    scores = np.random.default_rng(1).exponential(size=200_000)
    assert _pot(scores) == pytest.approx(-np.log(0.001), rel=0.1)


def test_pot_raises_on_constant_scores():
    with pytest.raises(ValueError):
        _pot(np.ones(1000))


def test_pot_raises_on_zero_excess_variance():
    # 20 identical excesses above the init level: variance 0, the moments fit is undefined.
    scores = np.concatenate([np.zeros(980), np.full(20, 5.0)])
    with pytest.raises(ValueError, match="variance"):
        _pot(scores, init_level=0.97)


def test_pot_raises_on_too_few_excesses():
    scores = np.random.default_rng(2).exponential(size=100)  # ~2 excesses above the 98th pct
    with pytest.raises(ValueError, match="excesses"):
        _pot(scores)


def test_pot_raises_when_risk_too_large():
    scores = np.random.default_rng(3).exponential(size=1000)
    with pytest.raises(ValueError, match="risk"):
        _pot(scores, risk=0.5)


def test_pot_dispatch_via_compute_threshold():
    assert "pot" in THRESHOLDS
    scores = np.random.default_rng(4).exponential(size=5000)
    assert compute_threshold(scores, POT) == THRESHOLDS["pot"](
        scores, init_level=0.98, risk=0.001, min_excesses=10
    )
