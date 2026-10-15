"""Closed Binance candles cached on disk, with exact coverage tracking, so history is fetched once and then instant.

Without this, every chart load, backtest and refresh re-downloaded the whole history from scratch (a year of 1-minute candles
is 526 requests, about 8 minutes). Now: the first request for a range fetches only what is missing, later ones read the file,
and a refresh fetches only the candles that closed since last time. Only CLOSED candles are stored; the still-forming candle is
always fetched fresh and never written, so a cached bar can never be a half-finished one.

Coverage is one contiguous [from, to) range per (symbol, interval). A request that touches the covered range extends it (older
bars on the left, newer on the right); a request wholly outside it replaces the coverage, so a hole can never be mistaken for
data. Binance's own gaps (maintenance) are fine: they are covered ranges with fewer rows.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from typing import Callable, List, Optional, Sequence

import pandas as pd

from core.chart_window import interval_seconds
from core.logger import logger

SPOT_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"}
PAGE = 1000                                  # Binance's maximum candles per request
DEFAULT_PATH = os.path.join("training_ground", "datasets", "klines_cache.sqlite3")
COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS klines(symbol TEXT NOT NULL, interval TEXT NOT NULL, t INTEGER NOT NULL,
    o REAL, h REAL, l REAL, c REAL, v REAL, PRIMARY KEY(symbol, interval, t)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS coverage(symbol TEXT NOT NULL, interval TEXT NOT NULL, from_ms INTEGER NOT NULL, to_ms INTEGER NOT NULL,
    PRIMARY KEY(symbol, interval));
"""

Row = Sequence[float]                          # [open_time_ms, open, high, low, close, volume]
FetchFn = Callable[[str, str, int, int], List[Row]]       # (symbol, interval, start_ms, end_ms_exclusive) -> rows


def _step_ms(interval: str) -> int:
    return interval_seconds(interval) * 1000


def _to_frame(rows: Sequence[Row]) -> pd.DataFrame:
    if not len(rows):
        out = pd.DataFrame(columns=COLUMNS, dtype=float)
    else:
        arr = pd.DataFrame(list(rows), columns=["t", "Open", "High", "Low", "Close", "Volume"])
        arr = arr.drop_duplicates("t").sort_values("t")
        out = arr.set_index(pd.to_datetime(arr["t"].astype("int64"), unit="ms"))[COLUMNS].astype(float)
    out.index.name = "Datetime"
    return out


class KlineCache:
    def __init__(self, path: Optional[str] = None):
        self.path = path or os.getenv("KLINE_CACHE_PATH") or DEFAULT_PATH
        self._lock = threading.Lock()
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        with closing(self._connect()) as con:
            con.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def coverage(self, symbol: str, interval: str) -> Optional[tuple]:
        with closing(self._connect()) as con:
            r = con.execute("SELECT from_ms, to_ms FROM coverage WHERE symbol=? AND interval=?", (symbol, interval)).fetchone()
        return (int(r[0]), int(r[1])) if r else None

    def read(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Cached candles with open time in [start_ms, end_ms)."""
        with closing(self._connect()) as con:
            rows = con.execute("SELECT t,o,h,l,c,v FROM klines WHERE symbol=? AND interval=? AND t>=? AND t<? ORDER BY t",
                               (symbol, interval, int(start_ms), int(end_ms))).fetchall()
        return _to_frame(rows)

    def store(self, symbol: str, interval: str, rows: Sequence[Row], covered_from: int, covered_to: int) -> None:
        """Save closed candles and extend (or, if disjoint, replace) the covered range to include [covered_from, covered_to)."""
        with self._lock, closing(self._connect()) as con, con:
            old = con.execute("SELECT from_ms, to_ms FROM coverage WHERE symbol=? AND interval=?", (symbol, interval)).fetchone()
            if old and covered_from <= old[1] and covered_to >= old[0]:           # overlaps or touches: union
                lo, hi = min(int(old[0]), int(covered_from)), max(int(old[1]), int(covered_to))
            else:                                                                  # nothing before, or disjoint: start over
                if old:
                    con.execute("DELETE FROM klines WHERE symbol=? AND interval=?", (symbol, interval))
                lo, hi = int(covered_from), int(covered_to)
            con.executemany("INSERT OR REPLACE INTO klines(symbol, interval, t, o, h, l, c, v) VALUES(?,?,?,?,?,?,?,?)",
                            [(symbol, interval, int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]))
                             for r in rows])
            con.execute("INSERT OR REPLACE INTO coverage(symbol, interval, from_ms, to_ms) VALUES(?,?,?,?)", (symbol, interval, lo, hi))


def fetch_range_http(symbol: str, interval: str, start_ms: int, end_ms: int, *, workers: int = 4,
                     get: Optional[Callable] = None, sleep: Callable[[float], None] = time.sleep) -> List[Row]:
    """Candles with open time in [start_ms, end_ms) from Binance's public REST API, split into 1,000-bar pages fetched by a few
    threads at once (stateless requests, so thread-safe; 4 workers stays well inside Binance's weight limit).
    Retries timeouts and 5xx; on 429/418 it waits as long as Binance asks and tries again."""
    import requests
    get = get or requests.get
    step = _step_ms(interval)
    pages = [(s, min(s + PAGE * step, end_ms)) for s in range(int(start_ms), int(end_ms), PAGE * step)]

    def one(page) -> List[Row]:
        s, e = page
        last_exc: Optional[Exception] = None
        for attempt in range(4):
            try:
                r = get(SPOT_KLINES, params={"symbol": symbol, "interval": interval, "startTime": s, "endTime": e - 1,
                                             "limit": PAGE}, timeout=15)
                if r.status_code in (418, 429):
                    sleep(min(60.0, float(r.headers.get("Retry-After", 2 ** attempt))))
                    last_exc = RuntimeError(f"Binance rate limit (HTTP {r.status_code})")
                    continue
                r.raise_for_status()
                return [[int(k[0]), k[1], k[2], k[3], k[4], k[5]] for k in r.json()]
            except Exception as exc:  # noqa: BLE001 -- retry, then surface the last error
                last_exc = exc
                sleep(0.4 * (attempt + 1))
        raise RuntimeError(f"Binance klines failed for {symbol} {interval} [{s}, {e}): {last_exc}")

    if not pages:
        return []
    if len(pages) == 1 or workers <= 1:
        chunks = [one(p) for p in pages]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(pages))) as ex:
            chunks = list(ex.map(one, pages))
    return [row for chunk in chunks for row in chunk]


def get_klines(symbol: str, interval: str, start_ms: int, end_ms: int, *, cache: Optional[KlineCache] = None,
               fetch: Optional[FetchFn] = None, now_ms: Optional[int] = None) -> pd.DataFrame:
    """Candles with open time in [start_ms, end_ms), reading the cache and fetching only what it lacks.
    The forming candle (if the range reaches it) is fetched fresh and appended; it is never stored."""
    cache = cache or KlineCache()
    fetch = fetch or fetch_range_http                       # looked up at call time so tests can swap the network out
    step = _step_ms(interval)
    now_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    start_ms, end_ms = int(start_ms), int(end_ms)
    closed_to = (now_ms // step) * step                      # open time of the forming candle = end of the closed ones
    want_to = min(end_ms, closed_to)
    if start_ms < want_to:
        cov = cache.coverage(symbol, interval)
        jobs = []
        if cov is None or want_to < cov[0] - 0 or start_ms > cov[1]:         # nothing cached, or wholly outside what we hold
            jobs.append((start_ms, want_to))
        else:
            if start_ms < cov[0]:
                jobs.append((start_ms, cov[0]))                               # older than the cache
            if want_to > cov[1]:
                jobs.append((cov[1], want_to))                                # newer than the cache
        for s, e in jobs:
            rows = [r for r in fetch(symbol, interval, s, e) if s <= int(r[0]) < e and int(r[0]) + step <= now_ms]
            cache.store(symbol, interval, rows, s, e)
            logger.info("kline cache: fetched %d %s %s candles [%s .. %s)", len(rows), symbol, interval,
                        pd.to_datetime(s, unit="ms"), pd.to_datetime(e, unit="ms"))
    df = cache.read(symbol, interval, start_ms, want_to)
    if end_ms > closed_to:                                    # include the forming candle, live and unstored
        live = [r for r in fetch(symbol, interval, closed_to, closed_to + step) if int(r[0]) == closed_to]
        if live:
            df = pd.concat([df, _to_frame(live)])
            df = df[~df.index.duplicated(keep="last")]
    return df
