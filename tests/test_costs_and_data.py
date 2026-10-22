import io
import json
import urllib.error
import urllib.parse

import numpy as np
import pandas as pd
import pytest

from cryptolab import data, risk
from cryptolab import portfolio as pf
from cryptolab.backtest import run_backtest


def test_rerunning_with_costs_changes_net_but_not_gross_returns(market):
    prices = pd.DataFrame({s: d["close"] for s, d in market.items()})
    live = prices.notna()
    weights = live.div(live.sum(axis=1), axis=0)
    base = run_backtest(weights, prices, rebalance="W")
    free = base.with_costs(0)
    expensive = base.with_costs(60)
    pd.testing.assert_series_equal(free.gross_returns, base.gross_returns)
    assert free.equity.iloc[-1] > base.equity.iloc[-1] > expensive.equity.iloc[-1]
    assert free.costs.sum() == 0


def test_buy_and_hold_does_not_care_about_costs_after_day_one():
    idx = pd.date_range("2024-01-01", periods=5)
    prices = pd.DataFrame({"A": [1.0, 1.1, 1.2, 1.1, 1.3]}, index=idx)
    hold = run_backtest(pd.DataFrame({"A": 1.0}, index=idx), prices, rebalance=None)
    assert hold.turnover.sum() == pytest.approx(1.0)  # the initial buy, then nothing
    np.testing.assert_allclose(hold.with_costs(100).returns.iloc[1:], hold.returns.iloc[1:])


def test_calendar_returns_compound_within_each_year():
    idx = pd.to_datetime(["2023-12-30", "2023-12-31", "2024-01-01", "2024-01-02"])
    r = pd.DataFrame({"s": [0.10, 0.10, -0.50, 0.20]}, index=idx)
    years = risk.calendar_returns(r)
    assert years.loc[2023, "s"] == pytest.approx(1.1 * 1.1 - 1)
    assert years.loc[2024, "s"] == pytest.approx(0.5 * 1.2 - 1)


def test_drawdown_duration_counts_the_longest_stretch_underwater():
    r = pd.Series([0.1, -0.1, 0.0, 0.2, -0.05, 0.0, 0.0, 0.0, 0.3])
    assert risk.max_drawdown_duration(r) == 4
    assert risk.max_drawdown_duration(pd.Series([0.01] * 5)) == 0


def test_ledoit_wolf_matches_the_textbook_loop():
    rng = np.random.default_rng(4)
    returns = pd.DataFrame(rng.normal(0, 0.03, (120, 6)))
    _, delta = pf.ledoit_wolf(returns)
    x = returns.to_numpy() - returns.to_numpy().mean(0)
    n, p = x.shape
    s = x.T @ x / n
    target = np.trace(s) / p * np.eye(p)
    d2 = np.sum((s - target) ** 2) / p
    b2 = min(sum(np.sum((np.outer(xi, xi) - s) ** 2) for xi in x) / (n**2 * p), d2)
    assert delta == pytest.approx(b2 / d2)


def test_download_falls_back_to_the_mirror_when_blocked(monkeypatch):
    candle = [1704067200000, "1", "2", "0.5", "1.5", "10", 1704153599999, "15", 3, "0", "0", "0"]
    hosts_tried = []

    def fake_urlopen(url, timeout):
        parts = urllib.parse.urlsplit(url)
        hosts_tried.append(f"{parts.scheme}://{parts.netloc}")
        if "api.binance.com" in url:
            raise urllib.error.HTTPError(url, 451, "Unavailable For Legal Reasons", {}, None)
        return io.BytesIO(json.dumps([candle] if len(hosts_tried) == 2 else []).encode())

    monkeypatch.setattr(data.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    df = data.fetch_binance("BTC", start="2024-01-01", end="2024-01-03")
    assert hosts_tried[:2] == data.BINANCE_HOSTS
    assert df["close"].iloc[0] == 1.5
