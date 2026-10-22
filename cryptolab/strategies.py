"""Strategy definitions evaluated in the research report.

Every strategy is long-only spot (shorting crypto needs perpetual futures with
funding costs that are out of scope here) and holds cash when it has no view.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import signals as sig
from .backtest import BacktestResult, run_backtest, vol_target, walk_forward
from .config import MARKET_PROXY
from .portfolio import ledoit_wolf, risk_parity
from .regime import GaussianHMM


@dataclass
class Market:
    universe: dict[str, pd.DataFrame]

    def __post_init__(self):
        self.prices = pd.DataFrame({s: d["close"] for s, d in self.universe.items()}).sort_index()
        self.returns = self.prices.pct_change(fill_method=None)
        self.vol = self.returns.rolling(60, min_periods=30).std()


def equal_weight(m: Market) -> BacktestResult:
    live = m.prices.notna()
    w = live.div(live.sum(axis=1), axis=0)
    return run_backtest(w, m.prices, "Equal-weight (monthly)", rebalance="M")


def btc_hold(m: Market) -> BacktestResult:
    w = pd.DataFrame(0.0, m.prices.index, m.prices.columns)
    w[MARKET_PROXY] = 1.0
    return run_backtest(w, m.prices, "BTC buy & hold", rebalance=None)


def composite_trend(m: Market, target_vol: float = 0.35) -> BacktestResult:
    """Per-asset composite signal (momentum + trend + mean reversion), long-only,
    inverse-volatility sized, then volatility-targeted at the portfolio level."""
    s = sig.panel(m.universe, sig.composite).reindex_like(m.prices).clip(lower=0)
    inv = 1.0 / m.vol
    raw = s * inv
    w = raw.div(inv.where(m.prices.notna()).sum(axis=1), axis=0)
    w = vol_target(w, m.prices, target_vol=target_vol)
    return run_backtest(w, m.prices, "Composite trend (vol-targeted)", rebalance="W", band=0.01)


def xs_momentum_weights(m: Market, lookback: int = 60, top_k: int = 4,
                        abs_filter: bool = True) -> pd.DataFrame:
    """Dual momentum: hold the top-k assets by relative strength, but only those
    whose own trend is positive; inverse-volatility weighted."""
    rank = sig.cross_sectional_momentum(m.prices, lookback)
    n_live = rank.notna().sum(axis=1)
    pick = rank.ge((n_live - top_k + 0.5) / n_live, axis=0) & rank.notna()
    if abs_filter:
        own = np.log(m.prices / m.prices.shift(lookback)) > 0
        pick &= own
    inv = (1.0 / m.vol).where(pick)
    return inv.div(inv.sum(axis=1).where(lambda x: x > 0), axis=0).fillna(0.0) \
              .mul(pick.sum(axis=1).clip(upper=top_k) / top_k, axis=0)


def xs_momentum(m: Market, lookback: int = 60, top_k: int = 4) -> BacktestResult:
    w = xs_momentum_weights(m, lookback, top_k)
    return run_backtest(w, m.prices, f"XS momentum L={lookback} k={top_k}", rebalance="W", band=0.01)


def risk_parity_strategy(m: Market, window: int = 90, target_vol: float = 0.35) -> BacktestResult:
    month_ends = m.prices.index.to_series().groupby(m.prices.index.to_period("M")).last()
    w = pd.DataFrame(np.nan, m.prices.index, m.prices.columns)
    for d in month_ends:
        win = m.returns.loc[:d].iloc[-window:]
        live = win.columns[win.notna().all()]
        if len(live) < 3 or len(win) < window:
            continue
        cov, _ = ledoit_wolf(win[live])
        w.loc[d, live] = risk_parity(cov).to_numpy()
    w = w.ffill().fillna(0.0)
    w = vol_target(w, m.prices, target_vol=target_vol)
    return run_backtest(w, m.prices, "Risk parity (Ledoit-Wolf)", rebalance="W", band=0.01)


def regime_features(close: pd.Series) -> pd.DataFrame:
    """Smoothed features for the market regime model: 10-day mean log return and log 20-day vol."""
    lr = np.log(close).diff()
    return pd.DataFrame({
        "ret10": lr.rolling(10).mean(),
        "logvol20": np.log(lr.rolling(20).std()),
    }).dropna()


def regime_overlay(m: Market, base: pd.DataFrame, train_days: int = 730,
                   test_days: int = 90) -> tuple[BacktestResult, pd.Series, list]:
    """Scale a base allocation by the walk-forward HMM's filtered probability of a bull regime.

    The HMM is refitted every ``test_days`` on data up to the fold start only,
    and the forward filter (causal) supplies probabilities inside each test window.
    """
    feats = regime_features(m.prices[MARKET_PROXY].dropna())
    p_bull = pd.Series(np.nan, index=feats.index)

    def model(tr_start, tr_end, te_end):
        train = feats.loc[tr_start:tr_end]
        hmm = GaussianHMM(n_states=2, seed=0).fit(train.to_numpy())
        # run the causal filter from the start of training through the test window
        span = feats.loc[tr_start:te_end]
        prob = pd.Series(hmm.filter_proba(span.to_numpy())[:, -1], index=span.index)
        p_bull.loc[prob.index[prob.index > tr_end]] = prob[prob.index > tr_end]
        exposure = ((prob - 0.3) / 0.4).clip(0, 1)
        w = base.loc[span.index].mul(exposure, axis=0)
        info = {"bull_mean_ret": hmm.means[-1, 0], "bear_mean_ret": hmm.means[0, 0],
                "bull_vol": np.exp(hmm.means[-1, 1]), "bear_vol": np.exp(hmm.means[0, 1]),
                "durations": hmm.expected_durations.round(1).tolist()}
        return w, info

    w, folds = walk_forward(model, feats.index, train_days, test_days)
    w = w.reindex(m.prices.index).fillna(0.0)
    res = run_backtest(w, m.prices, "Regime-gated equal-weight (walk-forward HMM)",
                       rebalance="W", band=0.02)
    return res, p_bull.dropna(), folds


LOOKBACKS = (20, 30, 45, 60, 90, 120)
TOP_KS = (3, 4, 5)


def parameter_sweep(m: Market) -> pd.DataFrame:
    """Full-sample sweep of the XS momentum grid. Used ONLY to measure how many
    trials were run and how dispersed they were (input to the Deflated Sharpe)."""
    rows = []
    for L in LOOKBACKS:
        for k in TOP_KS:
            r = xs_momentum(m, L, k).returns
            r = r[r.index >= m.prices.index[0] + pd.Timedelta(days=L + 60)]
            rows.append({"lookback": L, "top_k": k, "sharpe_pp": r.mean() / r.std(),
                         "sharpe": r.mean() / r.std() * np.sqrt(365)})
    return pd.DataFrame(rows)


def xs_momentum_walk_forward(m: Market, train_days: int = 730,
                             test_days: int = 90) -> tuple[BacktestResult, list]:
    """Pick (lookback, top_k) by in-sample Sharpe on each training window, then trade
    the chosen setting out-of-sample. The stitched OOS track record is the honest one."""
    cache = {(L, k): xs_momentum_weights(m, L, k) for L in LOOKBACKS for k in TOP_KS}
    rets = {key: run_backtest(w, m.prices, rebalance="W", band=0.01).returns for key, w in cache.items()}

    def model(tr_start, tr_end, te_end):
        best, best_sr = None, -np.inf
        for key, r in rets.items():
            ins = r.loc[tr_start + pd.Timedelta(days=key[0] + 60):tr_end]
            sr = ins.mean() / ins.std() if ins.std() > 0 else -np.inf
            if sr > best_sr:
                best, best_sr = key, sr
        return cache[best].loc[:te_end], {"lookback": best[0], "top_k": best[1],
                                          "in_sample_sharpe": best_sr * np.sqrt(365)}

    w, folds = walk_forward(model, m.prices.index, train_days, test_days)
    w = w.reindex(m.prices.index).fillna(0.0)
    return run_backtest(w, m.prices, "XS momentum (walk-forward selected)",
                        rebalance="W", band=0.01), folds
