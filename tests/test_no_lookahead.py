"""Look-ahead bias tests.

If an indicator or signal is causal, scrambling everything AFTER date t must
leave every value up to and including t unchanged. This catches centred
windows, accidental negative shifts, full-sample normalisation and similar bugs.
"""

import numpy as np
import pandas as pd
import pytest

from cryptolab import indicators as ind
from cryptolab import signals as sig
from cryptolab.backtest import run_backtest, vol_target


def scramble_future(df: pd.DataFrame, cut: int, seed: int = 0) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)
    future = out.iloc[cut:].to_numpy()
    out.iloc[cut:] = future * rng.uniform(0.5, 1.5, size=(len(future), 1))
    out["high"] = out[["open", "high", "close"]].max(axis=1)
    out["low"] = out[["open", "low", "close"]].min(axis=1)
    return out


CUT = 600

INDICATOR_FNS = {
    "rsi": lambda d: ind.rsi(d["close"]),
    "macd": lambda d: ind.macd(d["close"]),
    "bollinger": lambda d: ind.bollinger(d["close"]),
    "atr": lambda d: ind.atr(d["high"], d["low"], d["close"]),
    "adx": lambda d: ind.adx(d["high"], d["low"], d["close"]),
    "stoch": lambda d: ind.stochastic(d["high"], d["low"], d["close"]),
    "obv": lambda d: ind.obv(d["close"], d["volume"]),
    "zscore": lambda d: ind.zscore(d["close"]),
    "tsmom": lambda d: sig.tsmom(d["close"]),
    "trend": sig.trend,
    "mean_reversion": sig.mean_reversion,
    "composite": sig.composite,
}


@pytest.mark.parametrize("name", INDICATOR_FNS)
def test_indicator_is_causal(btc, name):
    fn = INDICATOR_FNS[name]
    a = fn(btc).iloc[:CUT]
    b = fn(scramble_future(btc, CUT)).iloc[:CUT]
    pd.testing.assert_frame_equal(pd.DataFrame(a), pd.DataFrame(b))


def test_backtest_return_uses_previous_weights():
    idx = pd.date_range("2024-01-01", periods=4)
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0, 99.0]}, index=idx)
    w = pd.DataFrame({"A": [0.0, 1.0, 1.0, 0.0]}, index=idx)
    res = run_backtest(w, prices, fee_bps=0, slippage_bps=0)
    # weight set at close of day 1 earns day 2's -10%, never day 1's +10%
    np.testing.assert_allclose(res.gross_returns.to_numpy(), [0.0, 0.0, -0.1, 0.0])


def test_costs_charged_on_drifted_weights():
    idx = pd.date_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"A": [100.0, 200.0, 200.0], "B": [100.0, 100.0, 100.0]}, index=idx)
    w = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]}, index=idx)
    res = run_backtest(w, prices, fee_bps=10, slippage_bps=0)
    # day 1: A doubles -> drifted weights 2/3, 1/3 -> rebalance trades 1/6 + 1/6
    assert res.turnover.iloc[1] == pytest.approx(1 / 3)
    assert res.costs.iloc[1] == pytest.approx(1 / 3 * 0.001)


def test_untradable_assets_get_zero_weight():
    idx = pd.date_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [np.nan, np.nan, 1.0]}, index=idx)
    w = pd.DataFrame({"A": [0.5] * 3, "B": [0.5] * 3}, index=idx)
    res = run_backtest(w, prices)
    assert (res.weights["B"].iloc[:2] == 0).all()


def test_vol_target_is_causal(market):
    prices = pd.DataFrame({s: d["close"] for s, d in market.items()})
    w = prices.notna().astype(float).div(prices.notna().sum(axis=1), axis=0)
    a = vol_target(w, prices).iloc[:CUT]
    scrambled = prices.copy()
    scrambled.iloc[CUT:] *= 1.7
    b = vol_target(w, scrambled).iloc[:CUT]
    pd.testing.assert_frame_equal(a, b)
