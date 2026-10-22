"""Vectorised technical indicators.

Every function accepts pandas Series (or DataFrames, column-wise) and returns
objects aligned to the input index. No function ever looks forward in time:
the value at row t depends only on rows <= t (enforced by tests/test_no_lookahead.py).

Conventions follow the original definitions:
  * RSI, ATR and ADX use Wilder's smoothing (alpha = 1/n), as in Wilder (1978)
  * EMA uses span-based alpha = 2/(n+1) with no bias adjustment, as charting packages do
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PERIODS_PER_YEAR


# Moving averages
def sma(x, n: int):
    return x.rolling(n, min_periods=n).mean()


def ema(x, n: int):
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(x, n: int):
    """Wilder's smoothing, i.e. an EMA with alpha = 1/n."""
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


# Momentum / oscillators
def rsi(close, n: int = 14):
    delta = close.diff()
    gain = wilder(delta.clip(lower=0.0), n)
    loss = wilder((-delta).clip(lower=0.0), n)
    rs = gain / loss
    out = 100.0 - 100.0 / (1.0 + rs)
    # all-gain windows: loss == 0 -> RSI = 100 (rs = inf already gives 100);
    # flat windows: gain == loss == 0 -> neutral 50
    return out.where(~((gain == 0) & (loss == 0)), 50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def roc(close, n: int = 20):
    return close / close.shift(n) - 1.0


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               n: int = 14, d: int = 3) -> pd.DataFrame:
    hh = high.rolling(n, min_periods=n).max()
    ll = low.rolling(n, min_periods=n).min()
    k = 100.0 * (close - ll) / (hh - ll).replace(0, np.nan)
    return pd.DataFrame({"%K": k, "%D": k.rolling(d, min_periods=d).mean()})


# Volatility / bands
def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)  # population sd, as Bollinger defines it
    upper, lower = mid + k * sd, mid - k * sd
    return pd.DataFrame({
        "mid": mid, "upper": upper, "lower": lower,
        "pct_b": (close - lower) / (upper - lower),
        "bandwidth": (upper - lower) / mid,
    })


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.where(prev.notna(), high - low)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return wilder(true_range(high, low, close), n)


def keltner(high, low, close, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = ema(close, n)
    a = atr(high, low, close, n)
    return pd.DataFrame({"mid": mid, "upper": mid + k * a, "lower": mid - k * a})


def donchian(high: pd.Series, low: pd.Series, n: int = 20) -> pd.DataFrame:
    up = high.rolling(n, min_periods=n).max()
    dn = low.rolling(n, min_periods=n).min()
    return pd.DataFrame({"upper": up, "lower": dn, "mid": (up + dn) / 2})


def realized_vol(close, n: int = 30, periods: int = PERIODS_PER_YEAR):
    return np.log(close).diff().rolling(n, min_periods=n).std() * np.sqrt(periods)


def parkinson_vol(high, low, n: int = 30, periods: int = PERIODS_PER_YEAR):
    """Range-based volatility estimator (Parkinson, 1980): ~5x more efficient than close-to-close."""
    hl = np.log(high / low) ** 2
    return np.sqrt(hl.rolling(n, min_periods=n).mean() / (4 * np.log(2)) * periods)


def ewma_vol(returns, lam: float = 0.94, periods: int = PERIODS_PER_YEAR):
    """RiskMetrics-style EWMA volatility, annualised. lam=0.94 is the standard daily value."""
    return np.sqrt((returns ** 2).ewm(alpha=1 - lam, adjust=False).mean() * periods)


# Trend strength / volume
def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.DataFrame:
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    tr_s = wilder(true_range(high, low, close), n)
    plus_di = 100 * wilder(plus_dm, n) / tr_s
    minus_di = 100 * wilder(minus_dm, n) / tr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame({"+DI": plus_di, "-DI": minus_di, "ADX": wilder(dx, n)})


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def vwap_rolling(close, volume, n: int = 20):
    return (close * volume).rolling(n, min_periods=n).sum() / volume.rolling(n, min_periods=n).sum()


# Statistical features
def zscore(x, n: int = 20):
    m = x.rolling(n, min_periods=n).mean()
    s = x.rolling(n, min_periods=n).std()
    return (x - m) / s


def hurst_exponent(x: np.ndarray, max_lag: int = 50) -> float:
    """Hurst exponent of a price path from the scaling of lagged differences.

    H ~ 0.5 random walk, H > 0.5 persistent (trending), H < 0.5 anti-persistent (mean-reverting).
    Uses std(log p_{t+k} - log p_t) ~ k^H, fitted by OLS in log-log space.
    """
    x = np.log(np.asarray(x, dtype=float))
    x = x[np.isfinite(x)]
    lags = np.arange(2, min(max_lag, len(x) // 2))
    if len(lags) < 5:
        return np.nan
    tau = np.array([np.std(x[lag:] - x[:-lag]) for lag in lags])
    ok = tau > 0
    slope, _ = np.polyfit(np.log(lags[ok]), np.log(tau[ok]), 1)
    return float(slope)


def rolling_hurst(close: pd.Series, window: int = 180, max_lag: int = 40) -> pd.Series:
    return close.rolling(window, min_periods=window).apply(
        lambda w: hurst_exponent(w, max_lag), raw=True)


# Feature table
def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
    feats = {
        "sma_50": sma(c, 50), "sma_200": sma(c, 200),
        "ema_20": ema(c, 20), "ema_50": ema(c, 50),
        "rsi_14": rsi(c, 14), "roc_20": roc(c, 20), "roc_90": roc(c, 90),
        "atr_14": atr(h, l, c, 14), "rv_30": realized_vol(c, 30),
        "pk_vol_30": parkinson_vol(h, l, 30), "obv": obv(c, v),
        "vwap_20": vwap_rolling(c, v, 20), "z_20": zscore(c, 20),
    }
    out = pd.DataFrame(feats, index=df.index)
    out = out.join(macd(c).add_prefix("macd_"))
    out = out.join(bollinger(c).add_prefix("bb_"))
    out = out.join(adx(h, l, c))
    out = out.join(stochastic(h, l, c))
    return out
