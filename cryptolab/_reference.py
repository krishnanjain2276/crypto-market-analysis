"""Slow, loop-based versions of the main indicators.

Nothing in the pipeline uses these. The tests check the vectorised indicators against them,
and benchmark.py times both, so the speed-up in the README is something you can measure.
"""

from __future__ import annotations

import math


def sma(x: list[float], n: int) -> list[float]:
    out = [math.nan] * len(x)
    for i in range(n - 1, len(x)):
        out[i] = sum(x[i - n + 1:i + 1]) / n
    return out


def ema_alpha(x: list[float], alpha: float, n: int) -> list[float]:
    out = [math.nan] * len(x)
    prev = None
    for i, v in enumerate(x):
        prev = v if prev is None else alpha * v + (1 - alpha) * prev
        if i >= n - 1:
            out[i] = prev
    return out


def ema(x, n):
    return ema_alpha(x, 2 / (n + 1), n)


def rsi(close: list[float], n: int = 14) -> list[float]:
    gains, losses = [math.nan], [math.nan]
    for i in range(1, len(close)):
        d = close[i] - close[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    # Wilder smoothing starting from the first valid delta, matching pandas ewm(adjust=False)
    out = [math.nan] * len(close)
    ag = al = None
    for i in range(1, len(close)):
        ag = gains[i] if ag is None else ag + (gains[i] - ag) / n
        al = losses[i] if al is None else al + (losses[i] - al) / n
        if i >= n:
            if ag == 0 and al == 0:
                out[i] = 50.0
            elif al == 0:
                out[i] = 100.0
            else:
                out[i] = 100 - 100 / (1 + ag / al)
    return out


def std_pop(x: list[float], n: int) -> list[float]:
    out = [math.nan] * len(x)
    for i in range(n - 1, len(x)):
        w = x[i - n + 1:i + 1]
        m = sum(w) / n
        out[i] = math.sqrt(sum((v - m) ** 2 for v in w) / n)
    return out


def std_sample(x: list[float], n: int) -> list[float]:
    out = [math.nan] * len(x)
    for i in range(n - 1, len(x)):
        w = x[i - n + 1:i + 1]
        if any(math.isnan(v) for v in w):
            continue
        m = sum(w) / n
        out[i] = math.sqrt(sum((v - m) ** 2 for v in w) / (n - 1))
    return out


def atr(high, low, close, n: int = 14) -> list[float]:
    tr = []
    for i in range(len(close)):
        if i == 0:
            tr.append(high[0] - low[0])
        else:
            tr.append(max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])))
    return ema_alpha(tr, 1 / n, n)


def macd(close, fast=12, slow=26, signal=9):
    f, s = ema(close, fast), ema(close, slow)
    line = [a - b for a, b in zip(f, s)]
    sig = [math.nan] * len(line)
    prev, count = None, 0
    for i, v in enumerate(line):
        if math.isnan(v):
            continue
        prev = v if prev is None else (2 / (signal + 1)) * v + (1 - 2 / (signal + 1)) * prev
        count += 1
        if count >= signal:
            sig[i] = prev
    return line, sig


def bollinger(close, n=20, k=2.0):
    mid, sd = sma(close, n), std_pop(close, n)
    return mid, [m + k * s for m, s in zip(mid, sd)], [m - k * s for m, s in zip(mid, sd)]


def realized_vol(close, n=30, periods=365):
    lr = [math.nan] + [math.log(close[i] / close[i - 1]) for i in range(1, len(close))]
    return [v * math.sqrt(periods) for v in std_sample(lr, n)]


def indicator_pipeline(o, h, l, c, v):
    return {
        "sma_50": sma(c, 50), "sma_200": sma(c, 200), "ema_20": ema(c, 20),
        "rsi_14": rsi(c, 14), "atr_14": atr(h, l, c, 14), "macd": macd(c),
        "bb": bollinger(c), "rv_30": realized_vol(c, 30),
    }
