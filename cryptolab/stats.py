"""Statistical controls against backtest overfitting.

A Sharpe ratio on its own says little: it is noisy, inflated by fat tails being
ignored, and inflated again by every parameter combination that was tried and
discarded. These functions put error bars and multiple-testing corrections on it.

References
----------
Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier"
Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio"
Politis & Romano (1994), "The Stationary Bootstrap"
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from .config import PERIODS_PER_YEAR

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329


def _moments(returns) -> tuple[float, float, float, int]:
    r = pd.Series(returns).dropna()
    sr = r.mean() / r.std()
    return float(sr), float(r.skew()), float(r.kurt() + 3.0), len(r)  # kurt -> raw (non-excess)


def probabilistic_sharpe(returns, sr_benchmark_annual: float = 0.0,
                         periods: int = PERIODS_PER_YEAR) -> float:
    """P(true Sharpe > benchmark | observed returns), correcting for length, skew and kurtosis."""
    sr, skew, kurt, n = _moments(returns)
    sr_star = sr_benchmark_annual / np.sqrt(periods)
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return _N.cdf((sr - sr_star) * np.sqrt(n - 1) / denom)


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """Expected maximum of ``n_trials`` Sharpe estimates drawn under the null of zero skill."""
    if n_trials < 2:
        return 0.0
    return sr_std * ((1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n_trials)
                     + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n_trials * np.e)))


def deflated_sharpe(returns, trial_sharpes_per_period, periods: int = PERIODS_PER_YEAR) -> dict:
    """Deflated Sharpe Ratio: PSR against the Sharpe you'd expect from luck alone,
    given how many configurations were tried and how dispersed their results were.

    ``trial_sharpes_per_period`` are the (non-annualised) Sharpe ratios of every
    configuration evaluated, including the selected one.
    """
    trials = np.asarray(trial_sharpes_per_period, float)
    sr0 = expected_max_sharpe(len(trials), trials.std(ddof=1))
    dsr = probabilistic_sharpe(returns, sr0 * np.sqrt(periods), periods)
    return {"DSR": dsr, "SR0_annual": sr0 * np.sqrt(periods), "n_trials": len(trials)}


def min_track_record(returns, sr_benchmark_annual: float = 0.0, confidence: float = 0.95,
                     periods: int = PERIODS_PER_YEAR) -> float:
    """Minimum number of periods needed for PSR to reach ``confidence``."""
    sr, skew, kurt, _ = _moments(returns)
    sr_star = sr_benchmark_annual / np.sqrt(periods)
    if sr <= sr_star:
        return np.inf
    z = _N.inv_cdf(confidence)
    return float(1 + (1 - skew * sr + (kurt - 1) / 4 * sr ** 2) * (z / (sr - sr_star)) ** 2)


def stationary_bootstrap_sharpe(returns, n_boot: int = 2000, mean_block: int = 20,
                                periods: int = PERIODS_PER_YEAR, seed: int = 0,
                                ci: float = 0.95) -> dict:
    """Confidence interval for the annualised Sharpe using the stationary bootstrap,
    which resamples random-length blocks and so preserves volatility clustering."""
    r = pd.Series(returns).dropna().to_numpy()
    n = len(r)
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n)
        new_block = rng.random(n) < p
        new_block[0] = True
        # position t: if a new block starts, jump to a random start; else advance by one
        block_id = np.cumsum(new_block) - 1
        block_start = starts[new_block][block_id]
        offset = np.arange(n) - np.flatnonzero(new_block)[block_id]
        sample = r[(block_start + offset) % n]
        sd = sample.std(ddof=1)
        out[b] = sample.mean() / sd * np.sqrt(periods) if sd > 0 else np.nan
    a = (1 - ci) / 2
    return {"low": float(np.nanquantile(out, a)), "high": float(np.nanquantile(out, 1 - a)),
            "p_sharpe_gt_0": float(np.nanmean(out > 0))}
