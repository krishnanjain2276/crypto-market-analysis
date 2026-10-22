"""Backtesting engine.

The timing rule everything else depends on: weights decided with data up to the close of day t
are held over (t, t+1] and earn day t+1's return, never day t's. Getting this wrong is the most
common way a backtest ends up looking better than it should.

Also handled here:
  * weights drift with prices between rebalances, and turnover is measured against the drifted
    weights, so costs aren't understated
  * fees and slippage are charged on traded notional
  * a coin can't be held before it lists (NaN price means weight 0)
  * optional rebalance calendar and no-trade band
  * volatility targeting from past returns only, and a generic walk-forward loop
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import risk
from .config import FEE_BPS, PERIODS_PER_YEAR, SLIPPAGE_BPS


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series  # net daily returns
    gross_returns: pd.Series
    weights: pd.DataFrame  # post-trade weights held into the next day
    turnover: pd.Series  # one-way traded notional / equity, per day
    costs: pd.Series
    target: pd.DataFrame | None = None  # kept so the run can be repeated with other costs
    prices: pd.DataFrame | None = None
    rebalance: str | None = "D"
    band: float = 0.0

    @property
    def equity(self) -> pd.Series:
        return (1 + self.returns).cumprod()

    def with_costs(self, total_bps: float) -> BacktestResult:
        """Same strategy, same trades, different cost assumption (fees + slippage, one-way)."""
        return run_backtest(self.target, self.prices, self.name, fee_bps=total_bps, slippage_bps=0.0,
                            rebalance=self.rebalance, band=self.band)

    def summary(self) -> pd.Series:
        s = risk.performance_summary(self.returns.rename(self.name)).iloc[0]
        s["Turnover/yr"] = float(self.turnover.mean() * PERIODS_PER_YEAR)
        s["Cost drag/yr"] = float(self.costs.mean() * PERIODS_PER_YEAR)
        s["Avg gross exp."] = float(self.weights.abs().sum(axis=1).mean())
        return s


def _rebalance_mask(index: pd.DatetimeIndex, freq: str | None) -> np.ndarray:
    if freq in (None, "D"):
        return np.ones(len(index), dtype=bool)
    periods = index.to_period(freq)
    mask = np.r_[True, periods[1:] != periods[:-1]]
    return mask


def run_backtest(target: pd.DataFrame, prices: pd.DataFrame, name: str = "strategy",
                 fee_bps: float = FEE_BPS, slippage_bps: float = SLIPPAGE_BPS,
                 rebalance: str | None = "D", band: float = 0.0) -> BacktestResult:
    """Simulate holding ``target`` weights (fractions of equity, cash = 1 - sum)."""
    prices = prices.sort_index()
    target = target.reindex(index=prices.index, columns=prices.columns)
    tradable = prices.notna().to_numpy()
    W = np.where(tradable, np.nan_to_num(target.to_numpy(), nan=0.0), 0.0)
    R = np.nan_to_num(prices.pct_change(fill_method=None).to_numpy(), nan=0.0)
    reb = _rebalance_mask(prices.index, rebalance)
    cost_rate = (fee_bps + slippage_bps) / 1e4

    T, N = W.shape
    held = np.zeros((T, N))
    gross = np.zeros(T)
    net = np.zeros(T)
    turn = np.zeros(T)
    cost = np.zeros(T)
    w = np.zeros(N)
    for t in range(T):
        g = w @ R[t]  # earned on yesterday's positions
        gross[t] = g
        drift = w * (1 + R[t]) / (1 + g) if (1 + g) > 0 else np.zeros(N)
        drift = np.where(tradable[t], drift, 0.0)  # nothing held in untradable names
        if reb[t]:
            trade = W[t] - drift
            if band > 0:  # skip tiny trades
                trade = np.where(np.abs(trade) < band, 0.0, trade)
            w = drift + trade
        else:
            w = drift
            trade = np.zeros(N)
        turn[t] = np.abs(trade).sum()
        cost[t] = turn[t] * cost_rate
        net[t] = (1 + g) * (1 - cost[t]) - 1
        held[t] = w

    idx = prices.index
    return BacktestResult(
        name=name,
        returns=pd.Series(net, idx, name=name),
        gross_returns=pd.Series(gross, idx, name=name),
        weights=pd.DataFrame(held, idx, prices.columns),
        turnover=pd.Series(turn, idx),
        costs=pd.Series(cost, idx),
        target=target,
        prices=prices,
        rebalance=rebalance,
        band=band,
    )


def vol_target(weights: pd.DataFrame, prices: pd.DataFrame, target_vol: float = 0.40,
               halflife: int = 20, max_leverage: float = 1.0,
               periods: int = PERIODS_PER_YEAR) -> pd.DataFrame:
    """Scale weights so the strategy's ex-ante volatility is ~``target_vol``.

    The scaler at t uses the strategy's own realised returns up to t only.
    """
    rets = prices.pct_change(fill_method=None).fillna(0.0)
    w = weights.reindex_like(prices).fillna(0.0)
    raw = (w.shift(1) * rets).sum(axis=1)  # unscaled strategy P&L, known at t
    vol = np.sqrt((raw ** 2).ewm(halflife=halflife, min_periods=halflife).mean() * periods)
    scale = (target_vol / vol).clip(upper=max_leverage).fillna(0.0)
    return w.mul(scale, axis=0)


# Walk-forward
@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    info: dict = field(default_factory=dict)


def walk_forward(model: Callable[[pd.Timestamp, pd.Timestamp, pd.Timestamp], tuple[pd.DataFrame, dict]],
                 index: pd.DatetimeIndex, train_days: int = 730, test_days: int = 90,
                 expanding: bool = True) -> tuple[pd.DataFrame, list[Fold]]:
    """Generic walk-forward loop.

    ``model(train_start, train_end, test_end)`` must fit any parameters on
    data in [train_start, train_end] only and return target weights for
    (train_end, test_end] plus a dict of diagnostics. The out-of-sample weight
    blocks are stitched together and can be passed to ``run_backtest``.
    """
    folds, blocks = [], []
    start = index[0]
    i = train_days
    while i < len(index) - 1:
        tr_start = start if expanding else index[i - train_days]
        tr_end = index[i - 1]
        te_end = index[min(i + test_days, len(index)) - 1]
        w, info = model(tr_start, tr_end, te_end)
        w = w.loc[(w.index > tr_end) & (w.index <= te_end)]
        blocks.append(w)
        folds.append(Fold(tr_start, tr_end, index[i], te_end, info))
        i += test_days
    return pd.concat(blocks).sort_index(), folds
