"""Trading signals in [-1, 1].

All signals at date t use information available at the close of t only.
The backtester then applies them to returns from t to t+1 (one-bar lag),
so there is no look-ahead anywhere in the chain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind


def tsmom(close: pd.Series, lookbacks=(20, 60, 120), vol_n: int = 60) -> pd.Series:
    """Time-series momentum (Moskowitz, Ooi & Pedersen, 2012), volatility-normalised.

    For each lookback L the trailing log return is divided by sigma*sqrt(L), giving a
    t-statistic-like strength. The average is squashed with tanh into [-1, 1].
    """
    lr = np.log(close).diff()
    sigma = lr.rolling(vol_n, min_periods=vol_n // 2).std()
    parts = [np.log(close / close.shift(L)) / (sigma * np.sqrt(L)) for L in lookbacks]
    return np.tanh(pd.concat(parts, axis=1).mean(axis=1, skipna=False))


def trend(df: pd.DataFrame, fast: int = 20, slow: int = 100, adx_n: int = 14) -> pd.Series:
    """EMA-crossover trend, scaled by distance in ATR units and gated by ADX strength."""
    c = df["close"]
    gap = (ind.ema(c, fast) - ind.ema(c, slow)) / ind.atr(df["high"], df["low"], c, adx_n)
    strength = ((ind.adx(df["high"], df["low"], c, adx_n)["ADX"] - 15) / 15).clip(0, 1)
    return (np.tanh(gap / 2) * strength).clip(-1, 1)


def mean_reversion(df: pd.DataFrame, n: int = 20, adx_max: float = 20.0) -> pd.Series:
    """Fade stretched moves, but only when the market is NOT trending (low ADX)."""
    c = df["close"]
    z = ind.zscore(c, n)
    rsi = ind.rsi(c, 14)
    stretched = (z.abs() > 1.5) & ((rsi > 70) | (rsi < 30))
    calm = ind.adx(df["high"], df["low"], c, 14)["ADX"] < adx_max
    sig = (-z / 3).clip(-1, 1)
    return sig.where(stretched & calm, 0.0).where(z.notna())


def composite(df: pd.DataFrame, weights=(0.5, 0.35, 0.15)) -> pd.Series:
    """Blend of momentum, trend and mean-reversion, clipped to [-1, 1]."""
    w_m, w_t, w_r = weights
    s = w_m * tsmom(df["close"]) + w_t * trend(df) + w_r * mean_reversion(df)
    return s.clip(-1, 1)


def panel(universe: dict[str, pd.DataFrame], fn=composite, **kw) -> pd.DataFrame:
    return pd.DataFrame({sym: fn(df, **kw) for sym, df in universe.items()}).sort_index()


def cross_sectional_momentum(prices: pd.DataFrame, lookback: int = 60, skip: int = 2,
                             vol_n: int = 60) -> pd.DataFrame:
    """Risk-adjusted relative strength, skipping the most recent days to avoid
    short-term reversal contaminating the ranking. Returns cross-sectional ranks in [0, 1]."""
    lr = np.log(prices).diff()
    mom = np.log(prices.shift(skip) / prices.shift(skip + lookback))
    score = mom / (lr.rolling(vol_n, min_periods=vol_n // 2).std() * np.sqrt(lookback))
    return score.rank(axis=1, pct=True)


def latest_screen(universe: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per asset describing where it stands today, sorted by composite score."""
    rows = {}
    for sym, df in universe.items():
        c = df["close"].dropna()
        if len(c) < 200:
            continue
        d = df.loc[c.index]
        last = c.index[-1]
        adx_v = ind.adx(d["high"], d["low"], c)["ADX"].iloc[-1]
        sma200 = ind.sma(c, 200).iloc[-1]
        rows[sym] = {
            "Price": c.iloc[-1],
            "1D %": c.pct_change().iloc[-1],
            "30D %": c.iloc[-1] / c.iloc[-31] - 1,
            "vs 200D": c.iloc[-1] / sma200 - 1,
            "RSI": ind.rsi(c).iloc[-1],
            "ADX": adx_v,
            "Vol 30D": ind.realized_vol(c, 30).iloc[-1],
            "Hurst": ind.hurst_exponent(c.iloc[-180:].to_numpy()),
            "Momentum": tsmom(c).iloc[-1],
            "Trend": trend(d).iloc[-1],
            "Composite": composite(d).iloc[-1],
            "As of": last.date(),
        }
    out = pd.DataFrame(rows).T
    num = [c for c in out.columns if c != "As of"]
    out[num] = out[num].astype(float)
    out["Stance"] = pd.cut(out["Composite"], [-1.01, -0.4, -0.1, 0.1, 0.4, 1.01],
                           labels=["Strong bear", "Bear", "Neutral", "Bull", "Strong bull"])
    return out.sort_values("Composite", ascending=False)
