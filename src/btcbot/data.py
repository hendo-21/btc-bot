"""Download and validate daily BTC-USD candles from Coinbase's public API.

No account or API key needed. Candles are UTC days.
"""

import time

import pandas as pd
import requests

from btcbot.config import DATA_DIR, DATA_FILE, DATA_START, PRODUCT

API = "https://api.exchange.coinbase.com/products/{product}/candles"
GRANULARITY = 86_400          # seconds per daily candle
MAX_CANDLES_PER_REQUEST = 300  # Coinbase limit
HEADERS = {"User-Agent": "btcbot-research"}


def fetch_daily(start: str = DATA_START, product: str = PRODUCT) -> pd.DataFrame:
    """Fetch all complete daily candles from `start` through yesterday (UTC)."""
    # Date range: from `start` up to (not including) today. Today's candle is
    # excluded because it is still forming and its prices would change later.
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp.now(tz="UTC").normalize()  # midnight today, UTC

    # Each request covers 299 days *after* its start date. Both the start and
    # end dates are included, so that is 300 candles: Coinbase's per-request max.
    step = pd.Timedelta(days=MAX_CANDLES_PER_REQUEST - 1)

    # Download in chunks, walking `cursor` forward until it reaches today.
    rows = []
    cursor = start_ts
    while cursor < end_ts:
        # A full chunk, or stop at yesterday if a full chunk would reach today.
        chunk_end = min(cursor + step, end_ts - pd.Timedelta(days=1))

        resp = requests.get(
            API.format(product=product),
            params={
                "granularity": GRANULARITY,
                "start": cursor.isoformat(),
                "end": chunk_end.isoformat(),
            },
            headers=HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        rows.extend(resp.json())  # each row: [time, low, high, open, close, volume]

        # Next chunk starts the day after this one ended, so no day is fetched twice.
        cursor = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.25)  # stay well under the public rate limit

    # Build one table. Coinbase sends unix-second timestamps, newest first.
    df = pd.DataFrame(rows, columns=["time", "Low", "High", "Open", "Close", "Volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    # Clean up: drop_duplicates guards against Coinbase repeating a candle at a
    # chunk boundary; then index by date, sort oldest first, put columns in the
    # usual OHLCV order, and make every column a float so types are consistent.
    df = (
        df.drop_duplicates("time")
        .set_index("time")
        .sort_index()[["Open", "High", "Low", "Close", "Volume"]]
        .astype(float)
    )

    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    """Raise if the candles have gaps, duplicates, or impossible prices.

    Runs after every download and every load, so a hand-edited or damaged
    CSV is caught too.
    """
    # Each day must appear once, in date order.
    if df.index.has_duplicates:
        raise ValueError("duplicate candle timestamps")
    if not df.index.is_monotonic_increasing:
        raise ValueError("candles not sorted by time")

    # No gaps: every calendar day between the first and last candle must be present.
    expected = pd.date_range(df.index[0], df.index[-1], freq="D", tz="UTC")
    missing = expected.difference(df.index)
    if len(missing):
        raise ValueError(f"{len(missing)} missing days, first: {missing[0].date()}")

    # The open and close happen during the day, so they must fall between the
    # day's low and high. This also catches columns in the wrong order
    low_ok = (df["Low"] <= df["Open"]) & (df["Low"] <= df["Close"])
    high_ok = (df["High"] >= df["Open"]) & (df["High"] >= df["Close"])
    if not (low_ok & high_ok).all():
        raise ValueError("found candles where open or close is outside the low-high range")


def save(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DATA_FILE)


def load() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"{DATA_FILE} missing; run `uv run python scripts/fetch_data.py`")
    df = pd.read_csv(DATA_FILE, index_col="time", parse_dates=["time"])
    df.index = pd.to_datetime(df.index, utc=True)
    validate(df)
    return df
