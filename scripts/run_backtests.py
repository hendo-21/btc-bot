"""Run every strategy at every fee level on one slice and write a results table.

Usage: uv run python scripts/run_backtests.py [design|validation]
The locked test slice is refused here on purpose.
"""

import sys

import pandas as pd

from btcbot import data, strategies
from btcbot.backtest import run
from btcbot.config import FEE_SCENARIOS, RESULTS_DIR, SLICES, STARTING_CASH

COLUMNS = {
    "final_equity": "Final $",
    "total_return_pct": "Return %",
    "cagr_pct": "CAGR %",
    "max_drawdown_pct": "Max DD %",
    "sharpe": "Sharpe",
    "trades": "Trades",
    "trades_per_year": "Trades/yr",
    "exposure_pct": "In market %",
    "fees_paid": "Fees $",
}

KEY = """## Key

- **Final $**: account value at the end of the slice, starting from the cash above.
- **Return %**: total gain or loss over the whole slice.
- **CAGR %** (compound annual growth rate): the steady yearly return that would produce the same
  total return. Lets you compare periods of different lengths.
- **Max DD %** (maximum drawdown): the worst fall from a peak to a later low. The most pain you
  would have sat through. -50% means the account halved at some point.
- **Sharpe**: return per unit of volatility, annualized (average daily return / standard deviation
  of daily returns x sqrt(365)). Above 1 is decent; above 2-3 is suspicious (luck, few trades, or a bug).
- **Trades / Trades/yr**: completed round trips (buy then sell). Few trades = weak evidence.
- **In market %**: share of days holding BTC rather than cash.
- **Fees $**: exchange commissions only. Slippage is included in the returns but not in this column.
"""


def main(slice_name: str) -> None:
    sl = SLICES[slice_name]
    df = data.load()

    # One row per (fee level, strategy) pair.
    rows = []
    for fee_name, fee in FEE_SCENARIOS.items():
        for name, cls in strategies.ALL.items():
            r = run(name, cls, df, sl, fee)
            row = {"fees": fee_name, "strategy": name}
            for field in COLUMNS:
                row[field] = getattr(r, field)  # e.g. r.sharpe when field is "sharpe"
            rows.append(row)
    table = pd.DataFrame(rows).rename(columns=COLUMNS)

    # Markdown report: one table per fee level, in the order listed in config.
    start, end = sl.bounds(df.index)
    report = f"# {slice_name} slice: {start.date()} to {end.date()}, ${STARTING_CASH:.0f} start\n\n"
    for fee_name, group in table.groupby("fees", sort=False):
        report += f"## {fee_name} ({FEE_SCENARIOS[fee_name]:.2%} per side)\n\n"
        report += group.drop(columns="fees").to_markdown(index=False, floatfmt=".2f") + "\n\n"
    print(report)

    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / f"{slice_name}_results.csv", index=False)
    (RESULTS_DIR / f"{slice_name}_results.md").write_text(report + KEY)
    print(f"Wrote {RESULTS_DIR}/{slice_name}_results.csv and .md")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "design")
