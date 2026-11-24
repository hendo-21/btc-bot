"""Run one strategy on one data slice and compute metrics over that slice only."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from backtesting.lib import FractionalBacktest

from btcbot.config import SLIPPAGE_SPREAD, STARTING_CASH, WARMUP_DAYS, Slice


class LockedSliceError(RuntimeError):
    """Raised when code tries to run the locked test slice without unlocking it."""


@dataclass
class Result:
    strategy: str
    slice: str
    fee: float
    start: pd.Timestamp
    end: pd.Timestamp
    final_equity: float
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe: float
    trades: int
    trades_per_year: float
    exposure_pct: float
    fees_paid: float
    equity: pd.Series  # daily equity over the slice


def prepare(data: pd.DataFrame, sl: Slice) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Cut the data to [slice start - warmup, slice end]. Nothing after the slice is visible."""
    start, end = sl.bounds(data.index)
    warmup_start = start - pd.Timedelta(days=WARMUP_DAYS)
    window = data[(data.index >= warmup_start) & (data.index <= end)]
    if window.index[0] > warmup_start:
        raise ValueError(f"not enough history before {start.date()} for warmup")
    return window, start, end


def metrics(equity: pd.Series) -> dict:
    """Standard metrics from a daily equity curve. Crypto trades every day, so annualize by 365."""
    years = (equity.index[-1] - equity.index[0]).days / 365
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (1 + total) ** (1 / years) - 1
    drawdown = equity / equity.cummax() - 1
    daily = equity.pct_change().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else 0.0
    return {
        "total_return_pct": total * 100,
        "cagr_pct": cagr * 100,
        "max_drawdown_pct": drawdown.min() * 100,
        "sharpe": sharpe,
        "years": years,
    }


def run(
    strategy_name: str,
    strategy_cls,
    data: pd.DataFrame,
    sl: Slice,
    fee: float,
    allow_locked: bool = False,
) -> Result:
    """Backtest one strategy on one slice at one fee level (fraction per side, e.g. 0.0025)."""
    if sl.locked and not allow_locked:
        raise LockedSliceError(
            f"'{sl.name}' is the locked test slice. It runs once, on the final pick only."
        )
    window, start, end = prepare(data, sl)

    bt = FractionalBacktest(
        window,
        strategy_cls,
        cash=STARTING_CASH,
        commission=fee,
        spread=SLIPPAGE_SPREAD,
        finalize_trades=True,
    )
    stats = bt.run(trade_start=start)

    equity = stats._equity_curve["Equity"]
    equity = equity[equity.index >= start]
    trades = stats._trades
    m = metrics(equity)

    # Exposure: share of slice days that end holding BTC. A sell fills at the
    # exit day's open, so the exit day itself counts as a day in cash.
    held = pd.Series(False, index=equity.index)
    for t in trades.itertuples():
        held.loc[t.EntryTime : t.ExitTime - pd.Timedelta(days=1)] = True
    exposure = held.mean() * 100

    return Result(
        strategy=strategy_name,
        slice=sl.name,
        fee=fee,
        start=start,
        end=end,
        final_equity=float(equity.iloc[-1]),
        total_return_pct=m["total_return_pct"],
        cagr_pct=m["cagr_pct"],
        max_drawdown_pct=m["max_drawdown_pct"],
        sharpe=m["sharpe"],
        trades=len(trades),
        trades_per_year=len(trades) / m["years"],
        exposure_pct=exposure,
        fees_paid=float(stats.get("Commissions [$]", 0.0)),  # key is missing when fee is 0
        equity=equity,
    )
