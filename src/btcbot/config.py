"""Single source of truth for research settings.

Change values here, never inline in scripts, so every result is traceable
to one config.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DATA_FILE = DATA_DIR / "btc_usd_daily.csv"

PRODUCT = "BTC-USD"          # Coinbase spot pair
DATA_START = "2016-01-01"    # 2016 is warmup only; trading starts 2017
STARTING_CASH = 150.0        # matches the planned live Alpaca balance (USD)

# Bars of history loaded before each slice so slow indicators (200-day SMA)
# are fully formed on the slice's first trading day.
WARMUP_DAYS = 250

# Slippage: the gap between the price the backtest assumes (the day's open) and
# the price a real order actually gets. It comes from the bid/ask spread (buyers
# pay slightly above the mid price, sellers get slightly below) and from the price
# moving while the order fills. For a $150 BTC order it is tiny; 0.05% per side
# is a deliberately cautious guess. Applied as a round-trip spread of 0.1%.
SLIPPAGE_SPREAD = 0.001


@dataclass(frozen=True)
class Slice:
    name: str
    start: str
    end: str | None  # None = through the latest candle
    locked: bool = False  # True = refused unless explicitly unlocked for the final run

    def bounds(self, index: pd.DatetimeIndex) -> tuple[pd.Timestamp, pd.Timestamp]:
        start = pd.Timestamp(self.start, tz="UTC")
        end = pd.Timestamp(self.end, tz="UTC") if self.end else index[-1]
        return start, end


# Time-ordered three-way split, like ML train/validation/test.
SLICES = {
    "design": Slice("design", "2017-01-01", "2022-12-31"),
    "validation": Slice("validation", "2023-01-01", "2024-12-31"),
    "test": Slice("test", "2025-01-01", None, locked=True),
}

# Per-side fee scenarios. Bots trade through Coinbase's API, which uses the
# Coinbase Advanced schedule, not the simple app's (~1.84% + spread). Check before relying on them.
FEE_SCENARIOS = {
    "coinbase_taker": 0.009,   # Advanced US entry tier since 2026-09-16
    "coinbase_maker": 0.005,   # limit orders that wait on the book
    "alpaca_taker": 0.0025,    # Alpaca crypto tier 1, confirmed in Alpaca docs (maker is 0.15%)
}
