"""Order-flow and derivatives features for crypto (Phase 6.5).

Two free Binance public data sets, neither of which the OHLC-only pipeline used:

* **taker-buy volume** (in every spot kline): how much of the volume was buyers lifting
  offers. ``taker_buy_ratio`` = taker_buy_volume / volume, 0.5 = balanced. This is a
  daily-bar version of the signed order flow that Kyle's lambda is built on.
* **perpetual-futures funding rate**: what longs pay shorts (or vice versa) every 8 h.
  Persistently high funding means the market is crowded long / leveraged.

Both are computed only from events inside the bar, so they are known at the bar's close.
Network fetchers cache to disk so repeated experiments don't hammer the API.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests

from core.logger import logger

SPOT_KLINES = "https://api.binance.com/api/v3/klines"
FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "training_ground", "datasets", "cache")
_DAY_MS = 86_400_000


def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name)


def _fresh(path: str, max_age_hours: float) -> bool:
    return os.path.exists(path) and (time.time() - os.path.getmtime(path)) < max_age_hours * 3600


def fetch_klines(symbol: str, days: int = 1500, interval: str = "1d", max_age_hours: float = 12) -> pd.DataFrame:
    """Spot OHLCV **plus taker-buy volume and trade count**, paginated, UTC-naive open-time index."""
    path = _cache_path(f"klines_{symbol}_{interval}_{days}.csv")
    if _fresh(path, max_age_hours):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    rows, start = [], int(time.time() * 1000) - days * _DAY_MS
    while True:
        r = requests.get(SPOT_KLINES, params={"symbol": symbol, "interval": interval, "limit": 1000,
                                              "startTime": start}, timeout=20)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows += batch
        start = batch[-1][0] + 1
        if len(batch) < 1000:
            break
        time.sleep(0.15)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["open_time", "Open", "High", "Low", "Close", "Volume", "close_time",
                                     "quote_vol", "Trades", "TakerBuyBase", "TakerBuyQuote", "ignore"])
    df.index = pd.to_datetime(df["open_time"], unit="ms")
    df.index.name = "datetime"
    df = df[["Open", "High", "Low", "Close", "Volume", "Trades", "TakerBuyBase"]].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.iloc[:-1] if len(df) and df.index[-1] + pd.Timedelta(days=1) > pd.Timestamp.now(tz="UTC").tz_localize(None) else df
    df.to_csv(path)
    return df


def fetch_funding_rates(symbol: str, days: int = 1500, max_age_hours: float = 12) -> pd.Series:
    """Perpetual funding rate per settlement (8-hourly), indexed by settlement time (UTC-naive)."""
    path = _cache_path(f"funding_{symbol}_{days}.csv")
    if _fresh(path, max_age_hours):
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    rows, start = [], int(time.time() * 1000) - days * _DAY_MS
    while True:
        r = requests.get(FUNDING, params={"symbol": symbol, "limit": 1000, "startTime": start}, timeout=20)
        if r.status_code != 200:
            logger.warning("funding %s: HTTP %s %s", symbol, r.status_code, r.text[:80])
            break
        batch = r.json()
        if not batch:
            break
        rows += batch
        start = batch[-1]["fundingTime"] + 1
        if len(batch) < 1000:
            break
        time.sleep(0.15)
    if not rows:
        return pd.Series(dtype=float, name="funding_rate")
    s = pd.Series({pd.to_datetime(x["fundingTime"], unit="ms").round("min"): float(x["fundingRate"]) for x in rows},
                  name="funding_rate").sort_index()
    s.to_frame().to_csv(path)
    return s


def taker_flow_features(klines: pd.DataFrame) -> pd.DataFrame:
    """Taker-flow features from spot klines; every value uses only the current and earlier bars."""
    d = klines.copy()
    d.columns = [str(c) for c in d.columns]
    vol = d["Volume"].replace(0, np.nan)
    ratio = d["TakerBuyBase"] / vol
    f = pd.DataFrame(index=d.index)
    f["tb_ratio"] = ratio
    f["tb_ratio_z20"] = (ratio - ratio.rolling(20).mean()) / ratio.rolling(20).std().replace(0, np.nan)
    f["tb_ratio_ewm5"] = ratio.ewm(span=5, adjust=False).mean()
    f["tb_ratio_chg1"] = ratio.diff()
    if "Trades" in d.columns:
        t = d["Trades"]
        f["trades_z20"] = (t - t.rolling(20).mean()) / t.rolling(20).std().replace(0, np.nan)
    return f.replace([np.inf, -np.inf], np.nan)


def funding_features(funding: pd.Series, bar_index: pd.DatetimeIndex, bar: str = "1D") -> pd.DataFrame:
    """Funding features per bar, from settlements INSIDE the bar ``[t, t + bar)`` only.

    A settlement at exactly ``t + bar`` belongs to the next bar. Bars before the perpetual
    launched have no settlements and come out NaN (they are not treated as zero funding).
    """
    idx = pd.DatetimeIndex(bar_index)
    f = pd.DataFrame(index=idx)
    if funding is None or len(funding) == 0:
        f["funding_bar"] = np.nan
    else:
        width = pd.Timedelta(bar)
        s = funding.sort_index()
        bar_of = idx[np.clip(idx.searchsorted(s.index, side="right") - 1, 0, len(idx) - 1)]
        inside = (s.index >= bar_of) & (s.index < bar_of + width)
        per_bar = s[inside].groupby(bar_of[inside]).sum()
        f["funding_bar"] = per_bar.reindex(idx)
    fb = f["funding_bar"]
    f["funding_mean3"] = fb.rolling(3).mean()
    f["funding_z30"] = (fb - fb.rolling(30).mean()) / fb.rolling(30).std().replace(0, np.nan)
    f["funding_chg1"] = fb.diff()
    return f.replace([np.inf, -np.inf], np.nan)
