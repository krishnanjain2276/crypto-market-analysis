"""Vectorised indicators must match textbook loop implementations exactly."""

import numpy as np
import pandas as pd

from cryptolab import _reference as ref
from cryptolab import indicators as ind


def close_to(a, b, tol=1e-9):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape
    both_nan = np.isnan(a) & np.isnan(b)
    assert np.all(both_nan | np.isclose(a, b, rtol=tol, atol=tol)), \
        f"max diff {np.nanmax(np.abs(a - b))}"


def test_sma_ema(btc):
    c = btc["close"]
    close_to(ind.sma(c, 20), ref.sma(c.tolist(), 20))
    close_to(ind.ema(c, 20), ref.ema(c.tolist(), 20))


def test_rsi_matches_wilder(btc):
    close_to(ind.rsi(btc["close"], 14), ref.rsi(btc["close"].tolist(), 14))


def test_rsi_bounds_and_edge_cases():
    up = pd.Series(np.arange(1.0, 60.0))
    assert ind.rsi(up).dropna().eq(100).all()
    flat = pd.Series(np.full(40, 5.0))
    assert ind.rsi(flat).dropna().eq(50).all()


def test_atr_macd_bollinger_vol(btc):
    h, l, c = (btc[k].tolist() for k in ("high", "low", "close"))
    close_to(ind.atr(btc["high"], btc["low"], btc["close"]), ref.atr(h, l, c))
    m = ind.macd(btc["close"])
    line, sig = ref.macd(c)
    close_to(m["macd"], line)
    close_to(m["signal"], sig)
    bb = ind.bollinger(btc["close"])
    mid, up, lo = ref.bollinger(c)
    close_to(bb["mid"], mid)
    close_to(bb["upper"], up)
    close_to(bb["lower"], lo)
    close_to(ind.realized_vol(btc["close"], 30), ref.realized_vol(c, 30))


def test_true_range_is_at_least_high_low(btc):
    tr = ind.true_range(btc["high"], btc["low"], btc["close"])
    assert (tr >= (btc["high"] - btc["low"]) - 1e-12).all()


def test_adx_range(btc):
    a = ind.adx(btc["high"], btc["low"], btc["close"]).dropna()
    assert ((a >= 0) & (a <= 100)).all().all()


def test_hurst_distinguishes_trend_from_noise():
    rng = np.random.default_rng(0)
    walk = np.exp(np.cumsum(rng.normal(0, 0.02, 3000)))
    trending = np.exp(np.cumsum(np.convolve(rng.normal(0, 0.02, 3100), np.ones(50) / 50, "valid")[:3000]))
    mean_rev = np.exp(np.sin(np.arange(3000) / 3) * 0.05 + rng.normal(0, 0.005, 3000))
    h_walk, h_trend, h_mr = (ind.hurst_exponent(x) for x in (walk, trending, mean_rev))
    assert 0.4 < h_walk < 0.6
    assert h_trend > h_walk > h_mr


def test_compute_all_shape(btc):
    feats = ind.compute_all(btc)
    assert len(feats) == len(btc)
    assert {"rsi_14", "macd_hist", "bb_pct_b", "ADX", "%K"} <= set(feats.columns)
