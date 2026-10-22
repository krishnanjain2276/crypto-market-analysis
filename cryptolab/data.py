"""Getting data in: Binance downloads, a local cache, cleaning, and a fake market for tests.

Real prices come from Binance's public klines endpoint, so no API key is needed. If the main
API is blocked where you are, the downloader falls back to Binance's public market-data mirror.
The synthetic market exists so the whole pipeline (and the test suite) runs offline; anything
built on it is labelled as synthetic in the report.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CACHE_DIR, QUOTE, UNIVERSE

OHLCV = ["open", "high", "low", "close", "volume"]

# The main API is geo-blocked in a few countries; data-api.binance.vision serves the same
# public market data and usually isn't.
BINANCE_HOSTS = ["https://api.binance.com", "https://data-api.binance.vision"]
KLINES_PATH = "/api/v3/klines"


def _to_ms(ts: str | pd.Timestamp) -> int:
    return int(pd.Timestamp(ts, tz="UTC").timestamp() * 1000)


def _get_json(query: str, pair: str, max_retries: int):  # pragma: no cover - network
    """Try each host in turn. Retry with back-off on timeouts and 5xx; move straight to the
    next host on 403/451, which is what a geo-block looks like."""
    errors = []
    for host in BINANCE_HOSTS:
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(f"{host}{KLINES_PATH}?{query}", timeout=20) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                errors.append(f"{host}: HTTP {exc.code}")
                if exc.code < 500:  # 4xx won't fix itself; 403/451 usually means geo-blocked
                    break
            except (urllib.error.URLError, TimeoutError) as exc:
                errors.append(f"{host}: {exc}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Binance request failed for {pair} ({'; '.join(errors[-3:])}). "
                       f"Try --source synthetic to run offline.")


def fetch_binance(symbol: str, start: str, end: str | None = None,
                  interval: str = "1d", pause: float = 0.25,
                  max_retries: int = 4) -> pd.DataFrame:
    """Download closed daily candles for ``{symbol}USDT``, 1000 at a time."""
    pair = f"{symbol}{QUOTE}"
    start_ms = _to_ms(start)
    end_ms = _to_ms(end) if end else int(time.time() * 1000)
    rows: list[list] = []

    while start_ms < end_ms:
        query = urllib.parse.urlencode({
            "symbol": pair, "interval": interval,
            "startTime": start_ms, "endTime": end_ms, "limit": 1000,
        })
        batch = _get_json(query, pair, max_retries)
        if not batch:
            break
        rows.extend(batch)
        start_ms = batch[-1][6] + 1  # 1 ms after the last candle's close time
        time.sleep(pause)  # stay well under the rate limit

    if not rows:
        raise ValueError(f"No data returned for {pair}")

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore"])
    now_ms = int(time.time() * 1000)
    df = df[df["close_time"] < now_ms]  # drop the live candle
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    df.index.name = "date"
    out = df[OHLCV + ["quote_volume", "trades"]].astype(float)
    return out


def load_universe(source: str = "binance", start: str = "2019-01-01",
                  end: str | None = None, symbols: list[str] | None = None,
                  refresh: bool = False, cache_dir: Path = CACHE_DIR,
                  seed: int = 7) -> dict[str, pd.DataFrame]:
    """Load OHLCV for every asset in the universe.

    source="binance"   -> live download, cached as CSV under ``data/cache``
    source="synthetic" -> deterministic simulated market (offline demo / tests)
    """
    symbols = symbols or UNIVERSE
    if source == "synthetic":
        return synthetic_universe(symbols=symbols, start=start, end=end, seed=seed)
    if source != "binance":
        raise ValueError("source must be 'binance' or 'synthetic'")

    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        path = cache_dir / f"{sym}{QUOTE}_1d.csv"
        if path.exists() and not refresh:
            df = pd.read_csv(path, index_col="date", parse_dates=True)
            last = df.index.max()
            if last < pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=1):
                new = fetch_binance(sym, start=str(last + pd.Timedelta(days=1)), end=end)
                df = pd.concat([df, new])
        else:
            df = fetch_binance(sym, start=start, end=end)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_csv(path)
        out[sym] = df.loc[start:end] if end else df.loc[start:]
    return out


# Validation
@dataclass
class QualityReport:
    symbol: str
    rows: int
    first: pd.Timestamp | None
    last: pd.Timestamp | None
    duplicates: int = 0
    missing_days: int = 0
    filled_days: int = 0
    ohlc_violations: int = 0
    non_positive: int = 0
    extreme_moves: int = 0
    notes: list[str] = field(default_factory=list)


def validate_ohlcv(df: pd.DataFrame, symbol: str = "", max_ffill: int = 3,
                   extreme_threshold: float = 0.5) -> tuple[pd.DataFrame, QualityReport]:
    """Clean an OHLCV frame and report every issue found.

    * removes duplicate timestamps and sorts the index
    * enforces high >= max(open, close) and low <= min(open, close)
    * drops non-positive prices
    * reindexes to a continuous daily calendar, forward-filling at most ``max_ffill`` days
    * flags (but keeps) single-day moves larger than ``extreme_threshold`` in log terms
    """
    df = df.copy()
    rep = QualityReport(symbol, len(df), None, None)
    rep.duplicates = int(df.index.duplicated().sum())
    df = df[~df.index.duplicated(keep="last")].sort_index()

    bad = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    rep.non_positive = int(bad.sum())
    df = df[~bad]

    hi = df[["open", "close"]].max(axis=1)
    lo = df[["open", "close"]].min(axis=1)
    viol = (df["high"] < hi) | (df["low"] > lo)
    rep.ohlc_violations = int(viol.sum())
    df["high"] = np.maximum(df["high"], hi)
    df["low"] = np.minimum(df["low"], lo)

    if len(df):
        full = pd.date_range(df.index.min(), df.index.max(), freq="D")
        rep.missing_days = int(len(full) - len(df))
        df = df.reindex(full)
        before = df["close"].isna().sum()
        df[OHLCV] = df[OHLCV].ffill(limit=max_ffill)
        rep.filled_days = int(before - df["close"].isna().sum())
        df.index.name = "date"
        rep.first, rep.last = df.index.min(), df.index.max()

    lr = np.log(df["close"]).diff().abs()
    rep.extreme_moves = int((lr > extreme_threshold).sum())
    if rep.extreme_moves:
        rep.notes.append(f"{rep.extreme_moves} daily moves > {extreme_threshold:.0%} (log) kept, review manually")
    return df, rep


def close_panel(universe: dict[str, pd.DataFrame], field_: str = "close") -> pd.DataFrame:
    return pd.DataFrame({sym: df[field_] for sym, df in universe.items()}).sort_index()


# Synthetic market
def synthetic_universe(symbols: list[str] | None = None, start: str = "2019-01-01",
                       end: str | None = None, n_days: int | None = None, seed: int = 7,
                       return_regimes: bool = False):
    """Simulate a crypto-like market.

    Features that matter for testing a research pipeline honestly:
      * a hidden 2-state Markov regime (bull / bear) with different drift and volatility
      * one common market factor plus asset-specific betas (BTC has beta 1)
      * Student-t (df=4) shocks -> fat tails and volatility clustering via regimes
      * staggered listing dates for a few assets (NaNs before listing)
      * consistent OHLC bars and volume that rises with absolute returns
    """
    symbols = symbols or UNIVERSE
    rng = np.random.default_rng(seed)
    if n_days is None:
        end_ts = pd.Timestamp(end) if end else pd.Timestamp("2026-09-30")
        dates = pd.date_range(start, end_ts, freq="D")
    else:
        dates = pd.date_range(start, periods=n_days, freq="D")
    n = len(dates)

    # Hidden regime path
    p_stay = np.array([0.985, 0.975])  # bull, bear persistence
    mu = np.array([0.0020, -0.0025])  # daily drift by regime
    sig = np.array([0.028, 0.050])  # daily vol by regime
    regime = np.empty(n, dtype=int)
    regime[0] = 0
    for t in range(1, n):
        stay = rng.random() < p_stay[regime[t - 1]]
        regime[t] = regime[t - 1] if stay else 1 - regime[t - 1]

    def t_shock(size):
        dof = 4.0
        return rng.standard_t(dof, size) / np.sqrt(dof / (dof - 2))  # unit variance

    market = mu[regime] + sig[regime] * t_shock(n)

    start_px = {"BTC": 3800, "ETH": 140, "BNB": 6, "SOL": 1.5, "XRP": 0.35, "ADA": 0.04,
                "DOGE": 0.002, "AVAX": 4.0, "DOT": 3.0, "LINK": 0.4, "LTC": 32, "TRX": 0.02,
                "ATOM": 4.5, "XLM": 0.11}
    late_listing = {"SOL": 0.22, "AVAX": 0.24, "DOT": 0.20}  # fraction of sample before listing

    out: dict[str, pd.DataFrame] = {}
    regimes = pd.Series(regime, index=dates, name="regime")
    for sym in symbols:
        beta = 1.0 if sym == "BTC" else rng.uniform(0.9, 1.6)
        idio_vol = (0.004 if sym == "BTC" else rng.uniform(0.012, 0.030)) * np.where(regime == 1, 1.5, 1.0)
        alpha = 0.0 if sym == "BTC" else rng.normal(0, 0.0004)
        r = alpha + beta * market + idio_vol * t_shock(n)
        r = np.clip(r, -0.6, 0.6)
        close = start_px.get(sym, rng.uniform(1, 100)) * np.exp(np.cumsum(r))

        gap = rng.normal(0, 0.002, n)
        open_ = np.r_[close[0] * np.exp(-r[0]), close[:-1]] * np.exp(gap)
        rng_hi = np.abs(rng.normal(0, 0.6, n)) * (sig[regime] + idio_vol)
        rng_lo = np.abs(rng.normal(0, 0.6, n)) * (sig[regime] + idio_vol)
        high = np.maximum(open_, close) * np.exp(rng_hi)
        low = np.minimum(open_, close) * np.exp(-rng_lo)
        base_vol = rng.uniform(1e7, 5e8) / close.mean()
        volume = base_vol * np.exp(rng.normal(0, 0.35, n)) * (1 + 15 * np.abs(r))

        df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                           "volume": volume}, index=dates)
        df.index.name = "date"
        if sym in late_listing:
            df.iloc[: int(late_listing[sym] * n)] = np.nan
            df = df.dropna()
        out[sym] = df

    return (out, regimes) if return_regimes else out
