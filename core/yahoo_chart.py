"""Direct Yahoo Finance chart-API fallback for OHLCV history.

Why: the ``yfinance`` library path (used by the data loader and OpenBB) fetches a cookie/crumb first and is the part that
gets HTTP 429 "Too Many Requests" on a busy IP -- at which point EVERY stock, ETF, index, future and FX chart fails, even though
Yahoo's plain chart endpoint still answers. This module talks to that endpoint directly (no crumb) and is used only when the
library path comes back empty.

Returns a frame shaped like the loader's other sources: Open/High/Low/Close/Volume, split/dividend ADJUSTED like yfinance's
default (auto_adjust=True), naive exchange-local index, daily+ bars normalised to midnight.
"""
from __future__ import annotations

import time
from typing import Optional

import pandas as pd
import requests

from core.logger import logger

_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_HEADERS = {"User-Agent": "Mozilla/5.0"}
_INTRADAY = {"1m", "2m", "5m", "15m", "30m", "60m", "90m"}
EMPTY_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _empty() -> pd.DataFrame:
    df = pd.DataFrame(columns=EMPTY_COLUMNS)
    df.index.name = "Datetime"
    return df


def fetch_chart(symbol: str, days: int, interval: str = "1d", timeout: float = 15.0, retries: int = 2,
                session: Optional[requests.Session] = None, now: Optional[float] = None) -> pd.DataFrame:
    """OHLCV for ``symbol`` over the last ``days`` days, or an EMPTY frame if Yahoo has nothing / cannot be reached. Never raises."""
    now = time.time() if now is None else now
    params = {"period1": int(now - days * 86400), "period2": int(now), "interval": interval, "events": "div,splits"}
    getter = (session or requests).get
    payload = None
    for attempt in range(retries + 1):
        try:
            r = getter(_URL.format(symbol=symbol), params=params, headers=_HEADERS, timeout=timeout)
            if r.status_code == 200:
                payload = r.json()
                break
            logger.warning("Yahoo chart %s -> HTTP %s (attempt %d)", symbol, r.status_code, attempt + 1)
        except Exception as exc:  # noqa: BLE001 -- network trouble must degrade to "no data", not crash a chart load
            logger.warning("Yahoo chart %s failed: %s (attempt %d)", symbol, exc, attempt + 1)
        time.sleep(0.5 * (attempt + 1))
    try:
        result = payload["chart"]["result"][0]
        ts = result.get("timestamp") or []
        q = result["indicators"]["quote"][0]
        if not ts:
            return _empty()
        df = pd.DataFrame({"Open": q["open"], "High": q["high"], "Low": q["low"], "Close": q["close"], "Volume": q["volume"]})
        adj = (result["indicators"].get("adjclose") or [{}])[0].get("adjclose")
        if adj and len(adj) == len(df):
            factor = pd.Series(adj, dtype="float64") / df["Close"].astype("float64")
            for col in ("Open", "High", "Low", "Close"):
                df[col] = df[col].astype("float64") * factor
        offset = int(result.get("meta", {}).get("gmtoffset", 0) or 0)
        idx = pd.to_datetime(pd.Series(ts, dtype="int64") + offset, unit="s")
        df.index = pd.DatetimeIndex(idx, name="Datetime")
        if interval not in _INTRADAY:
            df.index = df.index.normalize()
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        df["Volume"] = df["Volume"].fillna(0)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df if len(df) else _empty()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Yahoo chart %s: unusable response (%s)", symbol, exc)
        return _empty()
