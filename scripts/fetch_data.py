"""Download daily BTC-USD candles from Coinbase into data/btc_usd_daily.csv."""

from btcbot import data
from btcbot.config import DATA_FILE

if __name__ == "__main__":
    df = data.fetch_daily()
    data.save(df)
    print(f"Saved {len(df):,} daily candles "
          f"({df.index[0].date()} to {df.index[-1].date()}) to {DATA_FILE}")
