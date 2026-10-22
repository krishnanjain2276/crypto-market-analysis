"""Measure the speed-up of the vectorised indicator pipeline over naive loops.

Run:  python -m cryptolab benchmark
"""

from __future__ import annotations

import time

import pandas as pd

from . import _reference as ref
from . import indicators as ind


def vectorised_pipeline(df: pd.DataFrame) -> dict:
    h, l, c = df["high"], df["low"], df["close"]
    return {
        "sma_50": ind.sma(c, 50), "sma_200": ind.sma(c, 200), "ema_20": ind.ema(c, 20),
        "rsi_14": ind.rsi(c, 14), "atr_14": ind.atr(h, l, c, 14), "macd": ind.macd(c),
        "bb": ind.bollinger(c), "rv_30": ind.realized_vol(c, 30),
    }


def _best_of(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def run(universe: dict[str, pd.DataFrame], repeats: int = 5) -> dict:
    frames = {s: d.dropna() for s, d in universe.items()}
    lists = {s: [d[col].tolist() for col in ("open", "high", "low", "close", "volume")]
             for s, d in frames.items()}

    def naive():
        for cols in lists.values():
            ref.indicator_pipeline(*cols)

    def vectorised():
        for d in frames.values():
            vectorised_pipeline(d)

    t_naive = _best_of(naive, max(1, repeats // 2))
    t_vec = _best_of(vectorised, repeats)
    rows = int(sum(len(d) for d in frames.values()))
    return {
        "assets": len(frames), "rows": rows,
        "naive_s": t_naive, "vectorised_s": t_vec,
        "speedup_x": t_naive / t_vec,
        "time_saved_pct": 100 * (1 - t_vec / t_naive),
    }


if __name__ == "__main__":  # pragma: no cover
    from .data import synthetic_universe
    print(run(synthetic_universe()))
