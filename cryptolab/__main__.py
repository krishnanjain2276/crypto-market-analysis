"""Command-line interface.

    python -m cryptolab report    --source binance --start 2019-01-01
    python -m cryptolab screen    --source binance
    python -m cryptolab benchmark --source synthetic
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from . import benchmark
from .config import REPORT_DIR, UNIVERSE
from .data import load_universe, validate_ohlcv


def _load(args):
    print(f"Loading {len(args.symbols)} assets from {args.source} ...", flush=True)
    return load_universe(args.source, start=args.start, end=args.end, symbols=args.symbols,
                         refresh=getattr(args, "refresh", False))


def cmd_report(args):
    from .report import build_report
    t0 = time.perf_counter()
    u = _load(args)
    label = "Binance spot (USDT pairs), daily candles" if args.source == "binance" \
        else "synthetic regime-switching market (seed 7)"
    out = Path(args.out or REPORT_DIR / f"crypto_report_{args.source}.html")
    res = build_report(u, label, out, synthetic=args.source == "synthetic", run_benchmark=not args.no_bench)
    pd.set_option("display.width", 160)
    cols = ["CAGR", "Vol", "Sharpe", "MaxDD", "Turnover/yr"]
    print(res["summary"][cols].round(3).to_string())
    print(f"\nDeflated Sharpe of best grid config: {res['dsr']['DSR']:.1%}")
    if res["benchmark"]:
        print(f"Indicator engine speed-up vs loops: {res['benchmark']['speedup_x']:.1f}x")
    print(f"\nReport written to {out}  ({time.perf_counter() - t0:.0f}s)")


def cmd_screen(args):
    from .signals import latest_screen
    u = {s: validate_ohlcv(d, s)[0].dropna(subset=["close"]) for s, d in _load(args).items()}
    pd.set_option("display.width", 200)
    screen = latest_screen(u)
    print(screen.round(3).to_string())
    if args.csv:
        screen.to_csv(args.csv)
        print(f"\nSaved to {args.csv}")


def cmd_benchmark(args):
    u = {s: validate_ohlcv(d, s)[0].dropna(subset=["close"]) for s, d in _load(args).items()}
    r = benchmark.run(u, repeats=args.repeats)
    print(f"{r['assets']} assets, {r['rows']:,} rows")
    print(f"naive loops : {r['naive_s'] * 1000:8.1f} ms")
    print(f"vectorised  : {r['vectorised_s'] * 1000:8.1f} ms")
    print(f"speed-up    : {r['speedup_x']:.1f}x  ({r['time_saved_pct']:.0f}% less compute time)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="cryptolab", description="Crypto market analysis toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in [("report", cmd_report), ("screen", cmd_screen), ("benchmark", cmd_benchmark)]:
        sp = sub.add_parser(name)
        sp.add_argument("--source", choices=["binance", "synthetic"], default="binance")
        sp.add_argument("--start", default="2019-01-01")
        sp.add_argument("--end", default=None)
        sp.add_argument("--symbols", nargs="+", default=UNIVERSE)
        sp.add_argument("--refresh", action="store_true", help="re-download instead of using the cache")
        sp.set_defaults(func=fn)
        if name == "report":
            sp.add_argument("--out", default=None)
            sp.add_argument("--no-bench", action="store_true")
        if name == "screen":
            sp.add_argument("--csv", default=None, help="also save the screen to this CSV file")
        if name == "benchmark":
            sp.add_argument("--repeats", type=int, default=5)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
