"""Trend-filter overlay: hold a position only while the trailing return is positive, otherwise sit in cash.

What the evidence says (Phases 6.7-6.9, docs/PHASE_6_*_PREREGISTRATION.md) -- read before presenting this to anyone:
  * It REDUCED MAXIMUM DRAWDOWN in three separate looks (-67% -> -37.5%; -78% -> -64%; -74% -> -48%), the last two
    pre-registered and on different coins / a disjoint earlier period.
  * It did NOT show a Sharpe / return edge (every Sharpe-difference interval included zero).
  * Crypto only, survivorship-biased coins, one 28-day setting that was fixed in advance and never tuned.
So the honest label is "drawdown reduction / smoother ride", NOT "alpha". Parameters below are the validated ones:
lookback 28 bars, re-evaluated every 7 bars, long/flat. Validation was on DAILY bars; other intervals are unvalidated.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

DEFAULT_LOOKBACK = 28
DEFAULT_REBALANCE = 7
LABEL = "Trend overlay (drawdown reduction, not alpha)"


def trailing_return(closes, lookback: int = DEFAULT_LOOKBACK) -> Optional[float]:
    """Return over the last ``lookback`` bars using the LAST value in ``closes`` as 'now'; None if not enough history."""
    vals = list(closes)
    if lookback < 1 or len(vals) <= lookback or vals[-1 - lookback] in (0, None):
        return None
    return float(vals[-1] / vals[-1 - lookback] - 1.0)


def trend_is_up(closes, lookback: int = DEFAULT_LOOKBACK) -> Optional[bool]:
    """True/False once ``lookback + 1`` closes exist, else None (warm-up -- callers should treat it as 'stay in cash')."""
    r = trailing_return(closes, lookback)
    return None if r is None else bool(r > 0.0)


def is_evaluation_bar(n_bars: int, lookback: int = DEFAULT_LOOKBACK, every: int = DEFAULT_REBALANCE) -> bool:
    """The trend is re-read on the first bar with a full window and then every ``every`` bars (the validated cadence)."""
    return n_bars > lookback and (n_bars - lookback - 1) % every == 0


def portfolio_weights(closes: pd.DataFrame, lookback: int = DEFAULT_LOOKBACK) -> pd.Series:
    """Multi-asset form used in the experiments: weight 1/N on every column whose trailing return is > 0, else 0 (cash).
    ``closes`` holds prices up to and including the decision bar."""
    n = closes.shape[1]
    w = pd.Series(0.0, index=closes.columns)
    if len(closes) <= lookback or n == 0:
        return w
    up = closes.iloc[-1] / closes.iloc[-1 - lookback] - 1.0 > 0.0
    w[up[up].index] = 1.0 / n
    return w


def describe(closes, lookback: int = DEFAULT_LOOKBACK) -> str:
    r = trailing_return(closes, lookback)
    if r is None:
        return f"trend unknown (needs {lookback + 1} bars) -> staying in cash"
    return f"trailing {lookback}-bar return {r * 100:+.1f}% -> trend {'UP' if r > 0 else 'DOWN'}"
