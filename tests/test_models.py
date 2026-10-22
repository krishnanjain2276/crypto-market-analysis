"""Regime model, portfolio construction, risk metrics and overfitting statistics."""

import numpy as np
import pandas as pd
import pytest

from cryptolab import portfolio as pf
from cryptolab import risk, stats
from cryptolab.data import validate_ohlcv
from cryptolab.regime import GaussianHMM, select_n_states
from cryptolab.strategies import regime_features


def test_hmm_recovers_planted_regimes():
    rng = np.random.default_rng(1)
    A = np.array([[0.98, 0.02], [0.04, 0.96]])
    states = [0]
    for _ in range(3999):
        states.append(rng.choice(2, p=A[states[-1]]))
    states = np.array(states)
    x = np.where(states == 1, rng.normal(0.01, 0.01, 4000), rng.normal(-0.01, 0.03, 4000))
    hmm = GaussianHMM(2, seed=0).fit(x)
    acc = (hmm.viterbi(x) == states).mean()
    assert acc > 0.9
    np.testing.assert_allclose(hmm.means[:, 0], [-0.01, 0.01], atol=0.004)
    np.testing.assert_allclose(np.diag(hmm.A_), [0.96, 0.98], atol=0.03)


def test_hmm_probabilities_are_valid(market_with_regimes):
    u, _ = market_with_regimes
    f = regime_features(u["BTC"]["close"]).to_numpy()
    hmm = GaussianHMM(2).fit(f)
    for p in (hmm.filter_proba(f), hmm.predict_proba(f)):
        assert np.allclose(p.sum(axis=1), 1)
        assert ((p >= 0) & (p <= 1)).all()


def test_hmm_filter_is_causal(market_with_regimes):
    u, _ = market_with_regimes
    f = regime_features(u["BTC"]["close"]).to_numpy()
    hmm = GaussianHMM(2).fit(f[:1500])
    full = hmm.filter_proba(f)
    part = hmm.filter_proba(f[:1200])
    np.testing.assert_allclose(full[:1200], part)


def test_hmm_tracks_synthetic_market_regime(market_with_regimes):
    u, true = market_with_regimes
    feats = regime_features(u["BTC"]["close"])
    hmm = GaussianHMM(2).fit(feats.to_numpy())
    bear = hmm.predict_proba(feats.to_numpy())[:, 0] > 0.5
    agree = (bear == (true.reindex(feats.index) == 1)).mean()
    assert agree > 0.7


def test_bic_prefers_two_states_for_two_state_data():
    rng = np.random.default_rng(5)
    s = np.repeat([0, 1, 0, 1, 0, 1], 500)
    x = np.where(s == 1, rng.normal(0.02, 0.01, len(s)), rng.normal(-0.02, 0.01, len(s)))
    k, _ = select_n_states(x, (1, 2, 3))
    assert k == 2


@pytest.fixture
def cov(market):
    r = pd.DataFrame({s: d["close"] for s, d in market.items()}).pct_change().dropna()
    return pf.ledoit_wolf(r)[0]


def test_ledoit_wolf_is_well_conditioned(market):
    r = pd.DataFrame({s: d["close"] for s, d in market.items()}).pct_change().dropna().iloc[-60:]
    shrunk, delta = pf.ledoit_wolf(r)
    assert 0 <= delta <= 1
    assert np.linalg.cond(shrunk) <= np.linalg.cond(r.cov())
    assert np.all(np.linalg.eigvalsh(shrunk) > 0)


def test_risk_parity_equalises_contributions(cov):
    w = pf.risk_parity(cov)
    rc = pf.risk_contributions(w, cov)
    assert w.sum() == pytest.approx(1)
    np.testing.assert_allclose(rc, 1 / len(w), atol=1e-6)


def test_min_variance_beats_alternatives_and_respects_cap(cov):
    w = pf.min_variance(cov, max_weight=0.3)
    assert w.sum() == pytest.approx(1)
    assert (w >= -1e-12).all() and (w <= 0.3 + 1e-9).all()

    def var(x):
        return float(np.asarray(x) @ cov.to_numpy() @ np.asarray(x))

    assert var(w) <= var(pf.inverse_vol(cov)) + 1e-12
    assert var(w) <= var(pf.risk_parity(cov)) + 1e-12


def test_drawdown_and_sharpe_known_values():
    r = pd.Series([0.1, -0.5, 0.2])
    assert risk.max_drawdown(r) == pytest.approx(-0.5)
    r2 = pd.Series([0.01, -0.01] * 50)
    assert risk.sharpe(r2) == pytest.approx(0.0, abs=1e-12)


def test_var_ordering_on_fat_tails():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.standard_t(3, 20000) * 0.02)
    assert risk.cvar_historical(r, 0.99) > risk.var_historical(r, 0.99)
    assert risk.var_cornish_fisher(r, 0.99) > risk.var_gaussian(r, 0.99)


def test_validate_ohlcv_repairs_and_reports():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-05"])
    df = pd.DataFrame({"open": [10, 11, 11, 12], "high": [9, 12, 12, 13],
                       "low": [8, 10, 10, 11], "close": [10, 11, 11, 12],
                       "volume": [1, 1, 1, 1]}, index=idx, dtype=float)
    clean, rep = validate_ohlcv(df)
    assert rep.duplicates == 1 and rep.ohlc_violations == 1 and rep.missing_days == 2
    assert (clean["high"] >= clean[["open", "close"]].max(axis=1)).all()
    assert len(clean) == 5


def test_psr_and_dsr_behave():
    rng = np.random.default_rng(0)
    skill = pd.Series(rng.normal(0.002, 0.02, 2000))
    noise = pd.Series(rng.normal(0.0, 0.02, 2000))
    assert stats.probabilistic_sharpe(skill) > 0.95
    assert stats.probabilistic_sharpe(noise) < 0.95
    # more trials -> higher bar -> lower deflated Sharpe
    few = stats.deflated_sharpe(skill, rng.normal(0, 0.02, 3))["DSR"]
    many = stats.deflated_sharpe(skill, rng.normal(0, 0.02, 500))["DSR"]
    assert many < few


def test_expected_max_sharpe_grows_with_trials():
    assert stats.expected_max_sharpe(1000, 0.05) > stats.expected_max_sharpe(10, 0.05) > 0


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.001, 0.02, 1500))
    ci = stats.stationary_bootstrap_sharpe(r, n_boot=500)
    assert ci["low"] < risk.sharpe(r) < ci["high"]
