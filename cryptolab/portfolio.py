"""Portfolio construction: covariance estimation and allocation rules."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ledoit_wolf(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Ledoit-Wolf (2004) shrinkage towards a scaled identity matrix.

    Sample covariance matrices of 14 noisy, highly correlated crypto assets are
    badly conditioned; shrinkage gives an optimal (in Frobenius loss) blend
    Sigma = delta*mu*I + (1-delta)*S and makes optimisers far more stable.
    Returns the shrunk covariance and the shrinkage intensity delta.
    """
    x = returns.dropna(how="any").to_numpy()
    n, p = x.shape
    x = x - x.mean(0)
    s = x.T @ x / n
    mu = np.trace(s) / p
    target = mu * np.eye(p)
    d2 = np.sum((s - target) ** 2) / p
    # sum_i ||x_i x_i' - S||^2 expands to sum_i ||x_i||^4 - n ||S||^2, which avoids
    # building an outer product for every day
    b2_bar = (np.sum(np.sum(x ** 2, axis=1) ** 2) - n * np.sum(s ** 2)) / (n ** 2 * p)
    b2 = min(b2_bar, d2)
    delta = b2 / d2 if d2 > 0 else 1.0
    shrunk = delta * target + (1 - delta) * s
    return pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns), float(delta)


def inverse_vol(cov: pd.DataFrame) -> pd.Series:
    w = 1.0 / np.sqrt(np.diag(cov))
    return pd.Series(w / w.sum(), index=cov.index)


def risk_contributions(w, cov) -> np.ndarray:
    """Fraction of total portfolio variance contributed by each asset (sums to 1)."""
    w, cov = np.asarray(w, float), np.asarray(cov, float)
    mrc = cov @ w
    return w * mrc / (w @ mrc)


def risk_parity(cov: pd.DataFrame, budget=None, tol: float = 1e-10, max_iter: int = 10_000) -> pd.Series:
    """Equal (or budgeted) risk contribution via cyclical coordinate descent
    (Griveau-Billion, Richard & Roncalli, 2013). Each step solves a 1-D quadratic exactly."""
    S = np.asarray(cov, float)
    p = len(S)
    b = np.full(p, 1.0 / p) if budget is None else np.asarray(budget, float) / np.sum(budget)
    w = 1.0 / np.sqrt(np.diag(S))
    w /= w.sum()
    for _ in range(max_iter):
        w_old = w.copy()
        for i in range(p):
            sigma_p = np.sqrt(w @ S @ w)
            a = S[i] @ w - S[i, i] * w[i]
            w[i] = (-a + np.sqrt(a * a + 4 * S[i, i] * b[i] * sigma_p)) / (2 * S[i, i])
        if np.max(np.abs(w - w_old)) < tol:
            break
    return pd.Series(w / w.sum(), index=cov.index)


def _project_simplex(v: np.ndarray, cap: float = 1.0) -> np.ndarray:
    """Euclidean projection onto {w >= 0, sum w = 1, w <= cap} (bisection on the shift)."""
    lo, hi = v.min() - 1.0, v.max()
    for _ in range(100):
        tau = (lo + hi) / 2
        s = np.clip(v - tau, 0, cap).sum()
        lo, hi = (tau, hi) if s > 1 else (lo, tau)
    return np.clip(v - (lo + hi) / 2, 0, cap)


def min_variance(cov: pd.DataFrame, max_weight: float = 0.35, iters: int = 5000) -> pd.Series:
    """Long-only minimum-variance portfolio with a per-asset cap, solved by
    projected gradient descent (step = 1/L, L = largest eigenvalue)."""
    S = np.asarray(cov, float)
    p = len(S)
    cap = max(max_weight, 1.0 / p)
    step = 1.0 / np.linalg.eigvalsh(S).max()
    w = np.full(p, 1.0 / p)
    for _ in range(iters):
        w_new = _project_simplex(w - step * 2 * S @ w, cap)
        if np.max(np.abs(w_new - w)) < 1e-12:
            break
        w = w_new
    return pd.Series(w, index=cov.index)


def diversification_ratio(w, cov) -> float:
    """Weighted average vol / portfolio vol (1 = no diversification benefit)."""
    w, S = np.asarray(w, float), np.asarray(cov, float)
    return float(w @ np.sqrt(np.diag(S)) / np.sqrt(w @ S @ w))
