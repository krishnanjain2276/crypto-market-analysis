"""Return, drawdown, tail-risk and dependence metrics."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from .config import PERIODS_PER_YEAR

_N = NormalDist()


def simple_returns(prices):
    return prices.pct_change(fill_method=None)


def log_returns(prices):
    return np.log(prices).diff()


def cagr(returns: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    r = returns.dropna()
    if r.empty:
        return np.nan
    growth = float((1 + r).prod())
    return growth ** (periods / len(r)) - 1 if growth > 0 else -1.0


def ann_vol(returns: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    return float(returns.dropna().std() * np.sqrt(periods))


def sharpe(returns: pd.Series, rf: float = 0.0, periods: int = PERIODS_PER_YEAR) -> float:
    r = returns.dropna() - rf / periods
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else np.nan


def sortino(returns: pd.Series, rf: float = 0.0, periods: int = PERIODS_PER_YEAR) -> float:
    r = returns.dropna() - rf / periods
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean())
    return float(r.mean() / downside * np.sqrt(periods)) if downside > 0 else np.nan


def drawdown(returns: pd.Series) -> pd.Series:
    wealth = (1 + returns.fillna(0)).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown(returns).min())


def max_drawdown_duration(returns: pd.Series) -> int:
    """Longest stretch (in periods) spent below a previous equity high."""
    under = (drawdown(returns) < 0).to_numpy()
    if not under.any():
        return 0
    # label each underwater stretch, then take the longest one
    stretch_id = np.cumsum(~under)
    return int(np.bincount(stretch_id[under]).max())


def calmar(returns: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    mdd = max_drawdown(returns)
    return cagr(returns, periods) / abs(mdd) if mdd < 0 else np.nan


def var_historical(returns: pd.Series, level: float = 0.95) -> float:
    """Value-at-Risk as a positive loss number."""
    return float(-np.nanquantile(returns.dropna(), 1 - level))


def cvar_historical(returns: pd.Series, level: float = 0.95) -> float:
    """Expected shortfall: mean loss beyond the VaR threshold."""
    r = returns.dropna()
    cutoff = np.quantile(r, 1 - level)
    return float(-r[r <= cutoff].mean())


def var_gaussian(returns: pd.Series, level: float = 0.95) -> float:
    r = returns.dropna()
    return float(-(r.mean() + _N.inv_cdf(1 - level) * r.std()))


def var_cornish_fisher(returns: pd.Series, level: float = 0.95) -> float:
    """Modified VaR adjusting the normal quantile for skewness and excess kurtosis."""
    r = returns.dropna()
    z = _N.inv_cdf(1 - level)
    s, k = r.skew(), r.kurt()
    z_cf = (z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24
            - (2 * z ** 3 - 5 * z) * s ** 2 / 36)
    return float(-(r.mean() + z_cf * r.std()))


def rolling_beta(asset: pd.Series, market: pd.Series, n: int = 90) -> pd.Series:
    cov = asset.rolling(n, min_periods=n).cov(market)
    var = market.rolling(n, min_periods=n).var()
    return cov / var


def downside_correlation(returns: pd.DataFrame, market: str = "BTC", q: float = 0.1) -> pd.Series:
    """Correlation with the market on the market's worst q-fraction of days.

    Crypto correlations rise sharply in sell-offs; this shows how much
    diversification actually survives when it is needed.
    """
    m = returns[market]
    stress = m <= m.quantile(q)
    return returns[stress].corr()[market].drop(market)


def ewma_cov(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    r = returns.dropna(how="any").to_numpy()
    w = (1 - lam) * lam ** np.arange(len(r))[::-1]
    w /= w.sum()
    rc = r - (w[:, None] * r).sum(0)
    return pd.DataFrame((w[:, None] * rc).T @ rc, index=returns.columns, columns=returns.columns)


def calendar_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Compounded return per calendar year (rows) for each column. Partial years are included."""
    yearly = (1 + returns.fillna(0)).groupby(returns.index.year).prod() - 1
    yearly.index.name = "Year"
    return yearly


def performance_summary(returns: pd.Series | pd.DataFrame,
                        periods: int = PERIODS_PER_YEAR) -> pd.DataFrame:
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    rows = {}
    for col in returns:
        r = returns[col].dropna()
        rows[col] = {
            "CAGR": cagr(r, periods),
            "Vol": ann_vol(r, periods),
            "Sharpe": sharpe(r, periods=periods),
            "Sortino": sortino(r, periods=periods),
            "Calmar": calmar(r, periods),
            "MaxDD": max_drawdown(r),
            "MaxDD days": max_drawdown_duration(r),
            "VaR95": var_historical(r),
            "CVaR95": cvar_historical(r),
            "CF-VaR95": var_cornish_fisher(r),
            "Skew": float(r.skew()),
            "Kurt": float(r.kurt()),
            "Hit rate": float((r > 0).mean()),
            "Days": len(r),
        }
    return pd.DataFrame(rows).T
