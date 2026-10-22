"""Project-wide configuration."""

from __future__ import annotations

from pathlib import Path

# The 14-asset research universe (quoted against USDT on Binance).
UNIVERSE: list[str] = [
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE",
    "AVAX", "DOT", "LINK", "LTC", "TRX", "ATOM", "XLM",
]

QUOTE = "USDT"
MARKET_PROXY = "BTC"  # used for beta and the market regime model

# Crypto trades every day, so annualisation uses 365 periods, not 252.
PERIODS_PER_YEAR = 365

# Execution assumptions (basis points, one-way, applied to traded notional).
FEE_BPS = 10.0  # taker fee on a major exchange
SLIPPAGE_BPS = 5.0  # conservative market-impact allowance

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"
REPORT_DIR = ROOT / "reports"
