"""How much history a chart should ask for, so minute candles load fast and stay smooth.

Why this exists: with Days = 365 and a 1-minute interval the chart asked for 525,600 candles. Binance returns 1,000 per request
(about 0.9 s each), so the fetch alone took ~8 minutes, and the figure would have been ~40 MB of JSON for the browser. Stocks
were just as bad in a different way: the data providers keep only a short intraday history (Yahoo: 1m 7 days, 5-30m ~60 days,
1h ~2 years), so the request was refused or silently cut to 7 days (even for 1h).

Binance's own chart does not fetch a year of minutes either: it draws the latest few hundred bars at once and fetches older
bars as you scroll left. This module holds the numbers and rules for doing the same here; it is pure (no network, no UI).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

INTERVAL_SECONDS = {"1m": 60, "2m": 120, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "60m": 3600, "1h": 3600,
                    "2h": 7200, "4h": 14400, "6h": 21600, "8h": 28800, "12h": 43200, "1d": 86400, "3d": 259200,
                    "1wk": 604800, "1w": 604800}

INITIAL_BARS = 1500        # first paint: ~0.1 MB of figure, 2 Binance requests
CHUNK_BARS = 1000          # one "scroll left" step: exactly one Binance request
MAX_BARS = 20000           # most bars kept in the figure at once (a 20k-candle figure is ~1.6 MB)
LOAD_OLDER_WHEN_LEFT_EDGE_WITHIN = 0.15   # fetch more when the view comes within 15% of the loaded range's left edge

# Yahoo Finance's intraday history limits in days (its own documented limits, not tunable): 1m 7, other minute bars ~60,
# hourly ~730. Daily and coarser: unlimited for our purposes.
YAHOO_MAX_DAYS = {"1m": 7, "2m": 59, "3m": 59, "5m": 59, "15m": 59, "30m": 59, "90m": 59, "60m": 729, "1h": 729}


def interval_seconds(interval: str) -> int:
    return INTERVAL_SECONDS.get(str(interval or "1d").lower(), 86400)


def span_text(seconds: float) -> str:
    """'25 hours', '5.2 days', '2.1 years' -- for status messages."""
    if seconds < 2 * 3600:
        return f"{max(1, round(seconds / 60))} minutes"
    if seconds < 3 * 86400:
        return f"{seconds / 3600:.0f} hours"
    if seconds < 400 * 86400:
        return f"{seconds / 86400:.1f} days".replace(".0 days", " days")
    return f"{seconds / (365 * 86400):.1f} years"


@dataclass(frozen=True)
class WindowPlan:
    days: int                   # history to request now (never more than the user asked for)
    bars: int                   # roughly how many candles that is
    capped_by: Optional[str]    # None | "initial_window" (crypto: older bars load on scroll) | "provider" (stocks: source limit)
    note: str                   # one sentence for the status line; "" when nothing was cut


def plan_window(interval: str, days: int, *, crypto: bool) -> WindowPlan:
    """The history to fetch for a chart: ``days`` as asked, unless that is too many candles (crypto) or more than the data
    source keeps (stocks). The user is always told when it was cut."""
    days = max(1, int(days))
    step = interval_seconds(interval)
    wanted_bars = math.ceil(days * 86400 / step)
    if crypto:
        if wanted_bars <= INITIAL_BARS:
            return WindowPlan(days, wanted_bars, None, "")
        eff_days = max(1, math.ceil(INITIAL_BARS * step / 86400))
        bars = math.ceil(eff_days * 86400 / step)
        return WindowPlan(eff_days, bars, "initial_window",
                          f"Showing the latest {bars:,} {interval} candles ({span_text(bars * step)}); scroll left to load "
                          f"older ones (up to {days} days) -- like Binance, only what you look at is fetched.")
    limit = YAHOO_MAX_DAYS.get(str(interval).lower())
    if limit is not None and days > limit:
        bars = math.ceil(limit * 86400 / step)
        return WindowPlan(limit, bars, "provider",
                          f"The data source keeps only {limit} days of {interval} candles, so {limit} days are shown "
                          f"instead of {days}. Use a daily interval for longer history.")
    return WindowPlan(days, wanted_bars, None, "")


def should_load_older(view_start_ms: float, loaded_start_ms: float, loaded_end_ms: float) -> bool:
    """True when the visible range's left edge has reached (or passed) the left part of what is loaded, so the next older
    chunk should be fetched before the user runs out of candles."""
    span = max(1.0, loaded_end_ms - loaded_start_ms)
    return view_start_ms <= loaded_start_ms + LOAD_OLDER_WHEN_LEFT_EDGE_WITHIN * span


def older_chunk_range(loaded_start_ms: int, interval: str, bars: int = CHUNK_BARS) -> tuple:
    """(start_ms, end_ms) of the ``bars`` candles immediately before ``loaded_start_ms`` (end exclusive)."""
    step_ms = interval_seconds(interval) * 1000
    return int(loaded_start_ms - bars * step_ms), int(loaded_start_ms)


MAX_BARS_PER_LOAD = 8 * CHUNK_BARS       # one scroll event never fetches more than this (eight parallel Binance pages)


def bars_needed(view_start_ms: float, view_end_ms: float, loaded_start_ms: float, interval: str) -> int:
    """How many older candles to fetch so the visible range is covered plus a margin to its left (30% of the view's width), so
    a fast drag far to the left is filled in one go instead of leaving the chart blank. At least one chunk, at most
    MAX_BARS_PER_LOAD (a still-farther drag simply triggers the next load)."""
    step_ms = interval_seconds(interval) * 1000
    margin = 0.3 * max(0.0, view_end_ms - view_start_ms)
    gap = loaded_start_ms - (view_start_ms - margin)
    return int(min(MAX_BARS_PER_LOAD, max(CHUNK_BARS, math.ceil(gap / step_ms))))


def can_load_more(loaded_bars: int, oldest_allowed_ms: Optional[float], loaded_start_ms: float) -> bool:
    """False once the in-memory cap is reached or the user's Days limit (oldest allowed time) has been reached."""
    if loaded_bars >= MAX_BARS:
        return False
    return oldest_allowed_ms is None or loaded_start_ms > oldest_allowed_ms
