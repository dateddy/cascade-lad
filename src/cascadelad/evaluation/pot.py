"""Peaks-over-threshold (POT) threshold, Siffer et al. 2017 (D-a3c4).

Fits a generalized Pareto distribution (GPD) to the excesses of normal-only val scores over a
high quantile, by the method of moments in numpy, and returns the score whose tail probability
equals `risk`. Reads scores only, never labels.
"""

import numpy as np

from cascadelad.evaluation.thresholds import check_scores, register_threshold


@register_threshold("pot")
def pot(val_scores, *, init_level: float, risk: float, min_excesses: int) -> float:
    """Threshold z with P(score > z) ~ `risk` under a GPD fitted to val excesses over t.

    t is the `init_level` quantile; y = s[s > t] - t are the excesses. The moments fit gives
    gamma = (1 - m^2/v) / 2 and sigma = m (m^2/v + 1) / 2 with m, v the mean and variance of y.
    Raises ValueError when the fit is not valid: too few excesses, zero variance, or a risk
    that the excesses cannot resolve (risk * n / n_t >= 1).
    """
    s = check_scores(val_scores)
    t = float(np.quantile(s, init_level))
    y = s[s > t] - t
    n, n_t = s.size, y.size
    if n_t < min_excesses:
        raise ValueError(f"POT needs at least {min_excesses} excesses over t={t:.6g}, got {n_t}")
    m, v = float(y.mean()), float(y.var())
    if v == 0.0:
        raise ValueError("POT excesses have zero variance; the moments fit is undefined")
    q = risk * n / n_t
    if q >= 1.0:
        raise ValueError(f"POT risk {risk} is not below the excess share {n_t / n:.6g}")
    ratio = m * m / v
    gamma, sigma = 0.5 * (1.0 - ratio), 0.5 * m * (ratio + 1.0)
    if np.isclose(gamma, 0.0):
        return t - sigma * float(np.log(q))
    return t + sigma / gamma * (q ** (-gamma) - 1.0)
