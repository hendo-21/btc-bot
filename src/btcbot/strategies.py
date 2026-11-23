"""Long-only strategies. Each one is either fully in BTC or fully in cash.

Rules are decided on a day's close and filled at the next day's open
(backtesting.py's default), so no strategy can act on a price it hasn't seen.

Settings are fixed textbook values, not tuned, so they can't be overfit to the data.
"""

import numpy as np
import pandas as pd
from backtesting import Strategy

POSITION_SIZE = 0.99  # keep 1% in cash, like the headroom a live order needs for fees and price moves


# Indicator functions. Each takes a full price column (every day in the backtest)
# and returns an array with one value per day. Only past and current values are
# used for each day; the first days are NaN until a full window exists.


def sma(values, n: int):
    """Simple moving average: the mean of the last n values, including today."""
    return pd.Series(values).rolling(n).mean().to_numpy()


def rolling_max_prev(values, n: int):
    """Highest value over the previous n bars, excluding today.

    Today is excluded because today's close can never be above today's own high,
    so a breakout must be measured against earlier days.
    """
    return pd.Series(values).rolling(n).max().shift(1).to_numpy()


def rolling_min_prev(values, n: int):
    """Lowest value over the previous n bars, excluding today."""
    return pd.Series(values).rolling(n).min().shift(1).to_numpy()


def rolling_std(values, n: int):
    """Standard deviation of the last n values, including today."""
    return pd.Series(values).rolling(n).std().to_numpy()


def pct_change(values, n: int):
    """Return over the last n days: today's value vs. n days ago, as a fraction (0.1 = +10%)."""
    return pd.Series(values).pct_change(n).to_numpy()


def rsi(values, n: int):
    """Wilder's RSI: 0-100, low means recent closes mostly fell (oversold).

    It compares the average daily gain with the average daily loss. Wilder's
    smoothing is an exponential average with weight 1/n on the newest day,
    which is what ewm(alpha=1/n) computes.
    """
    delta = pd.Series(values).diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return (100 - 100 / (1 + gain / loss)).to_numpy()


def supertrend(high, low, close, n: int, mult: float):
    """Supertrend direction for each day: 1 = uptrend, -1 = downtrend.

    ATR (average true range) is how far the price typically moves in a day.
    Two bands sit `mult` ATRs above and below the day's midpoint. In an uptrend
    the lower band acts as a trailing stop: it can only move up, and a close
    below it flips the trend to down. In a downtrend the upper band does the
    mirror image. Each day's band depends on the previous day's, so this needs
    a loop instead of a one-line pandas calculation.
    """
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)

    # True range: the day's full move, counting a gap from yesterday's close.
    prev_close = close.shift(1)
    day_range = high - low
    gap_up = (high - prev_close).abs()
    gap_down = (low - prev_close).abs()
    true_range = pd.concat([day_range, gap_up, gap_down], axis=1).max(axis=1)  # largest of the three
    atr = true_range.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()  # Wilder smoothing, as in rsi()

    midpoint = (high + low) / 2
    upper = (midpoint + mult * atr).to_numpy(copy=True)  # copy: the loop below edits these
    lower = (midpoint - mult * atr).to_numpy(copy=True)
    close = close.to_numpy()

    direction = np.full(len(close), np.nan)  # stays NaN until ATR has n days of data
    trend = 1  # start by assuming an uptrend, as TradingView's version does
    for i in range(1, len(close)):
        if np.isnan(lower[i - 1]):
            continue  # ATR not ready yet

        # Ratchet: the lower band can only rise and the upper band can only fall,
        # unless yesterday's close broke through the band (then it resets).
        if close[i - 1] > lower[i - 1]:
            lower[i] = max(lower[i], lower[i - 1])
        if close[i - 1] < upper[i - 1]:
            upper[i] = min(upper[i], upper[i - 1])

        # Flip only when today's close crosses yesterday's band; otherwise keep the trend.
        if trend == -1 and close[i] > upper[i - 1]:
            trend = 1
        elif trend == 1 and close[i] < lower[i - 1]:
            trend = -1
        direction[i] = trend
    return direction


# Strategies. Base holds the behavior they all share; each class below it adds
# one trading rule through init() (set up indicators) and next() (decide each day).


class Base(Strategy):
    """Shared behavior for all strategies. Inherits self.data, self.position,
    self.buy() and self.I() from backtesting.py's Strategy.

    Holds off trading until `trade_start` so warmup bars only feed indicators.
    """

    trade_start = None  # pd.Timestamp, set by the backtest runner

    def warming_up(self) -> bool:
        return self.trade_start is not None and self.data.index[-1] < self.trade_start

    def go_long(self):
        """Buy with 99% of equity, unless already holding BTC."""
        if not self.position:
            self.buy(size=POSITION_SIZE)

    def go_flat(self):
        """Sell everything and hold cash, unless already in cash."""
        if self.position:
            self.position.close()


class BuyAndHold(Base):
    """Baseline: buy on the first trading day, never sell."""

    def init(self):
        pass

    def next(self):
        if not self.warming_up():
            self.go_long()


class SmaCross(Base):
    """Golden/death cross: long while the 50-day SMA is above the 200-day SMA."""

    fast = 50
    slow = 200

    def init(self):
        # Runs once: sma() computes the average for every day in a single call.
        self.ma_fast = self.I(sma, self.data.Close, self.fast)
        self.ma_slow = self.I(sma, self.data.Close, self.slow)

    def next(self):
        # Runs once per day. [-1] reads today's precomputed value; no math happens here.
        if self.warming_up():
            return
        if self.ma_fast[-1] > self.ma_slow[-1]:
            self.go_long()
        else:
            self.go_flat()


class AboveSma(Base):
    """Long while the close is above its 50-day SMA."""

    n = 50

    def init(self):
        self.ma = self.I(sma, self.data.Close, self.n)

    def next(self):
        if self.warming_up():
            return
        if self.data.Close[-1] > self.ma[-1]:
            self.go_long()
        else:
            self.go_flat()


class DonchianBreakout(Base):
    """Buy a close above the prior 20-day high; sell a close below the prior 10-day low."""

    entry_n = 20
    exit_n = 10

    def init(self):
        self.upper = self.I(rolling_max_prev, self.data.High, self.entry_n)
        self.lower = self.I(rolling_min_prev, self.data.Low, self.exit_n)

    def next(self):
        if self.warming_up():
            return
        if self.data.Close[-1] > self.upper[-1]:
            self.go_long()
        elif self.data.Close[-1] < self.lower[-1]:
            self.go_flat()


class BollingerReversion(Base):
    """Slow mean reversion: buy 2 std devs below the 20-day mean, sell back at the mean."""

    n = 20
    k = 2.0  # how many standard deviations below the mean counts as "cheap"

    def init(self):
        self.mid = self.I(sma, self.data.Close, self.n)
        self.std = self.I(rolling_std, self.data.Close, self.n)

    def next(self):
        if self.warming_up():
            return
        close = self.data.Close[-1]
        if close < self.mid[-1] - self.k * self.std[-1]:
            self.go_long()
        elif close > self.mid[-1]:
            self.go_flat()


class Rsi2Reversion(Base):
    """Faster mean reversion: in an uptrend (close > 200-day SMA), buy when 2-day
    RSI is deeply oversold and sell once the close recovers above its 5-day SMA.
    Holds for days, not weeks, so it tests whether short trades survive fees."""

    rsi_n = 2
    oversold = 10
    trend_n = 200
    exit_n = 5

    def init(self):
        self.rsi = self.I(rsi, self.data.Close, self.rsi_n)
        self.trend = self.I(sma, self.data.Close, self.trend_n)
        self.exit_ma = self.I(sma, self.data.Close, self.exit_n)

    def next(self):
        if self.warming_up():
            return
        close = self.data.Close[-1]
        if not self.position:
            if close > self.trend[-1] and self.rsi[-1] < self.oversold:
                self.go_long()
        elif close > self.exit_ma[-1]:
            self.go_flat()


class Momentum(Base):
    """Time-series momentum: long while the close is above its level 20 days ago.
    Trades more often than the other trend rules, to test whether faster trading
    survives fees."""

    n = 20

    def init(self):
        self.ret = self.I(pct_change, self.data.Close, self.n)

    def next(self):
        if self.warming_up():
            return
        if self.ret[-1] > 0:
            self.go_long()
        else:
            self.go_flat()


class Supertrend(Base):
    """Long while Supertrend (10-day ATR, 3 ATRs) says uptrend. The exit is a
    trailing stop that sits further from the price when the market is volatile."""

    atr_n = 10
    mult = 3.0

    def init(self):
        self.direction = self.I(
            supertrend, self.data.High, self.data.Low, self.data.Close, self.atr_n, self.mult
        )

    def next(self):
        if self.warming_up():
            return
        if self.direction[-1] > 0:
            self.go_long()
        else:
            self.go_flat()


# Every strategy to backtest. Scripts and tests loop over this, so adding a
# strategy means writing its class and adding one line here.
ALL = {
    "buy_and_hold": BuyAndHold,
    "sma_cross_50_200": SmaCross,
    "above_sma_50": AboveSma,
    "donchian_20_10": DonchianBreakout,
    "bollinger_20_2": BollingerReversion,
    "rsi2_reversion": Rsi2Reversion,
    "momentum_20": Momentum,
    "supertrend_10_3": Supertrend,
}
