"""
core/watchlist.py

A user watchlist shared by both front ends: the symbol list (a small JSON file) and a background quote book.

* Symbols live in ``training_ground/paper/watchlist.json`` (override ``WATCHLIST_PATH``; git-ignored), so the desktop
  app and Dash show the same list.
* Quotes: crypto from Binance's public 24h ticker (last, previous close, change %); everything else from Yahoo's
  chart API (last daily close vs the one before, so it is DELAYED and only as fresh as Yahoo). No API keys.
* ``QuoteBook`` fetches off the UI thread and caches per symbol (crypto 10 s, others 60 s), so a UI timer can call
  ``refresh_async`` + ``snapshot`` freely. Fetch failures are kept as ``error`` text, never raised.

Display only: nothing here places orders.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from core.logger import logger

DEFAULT_SYMBOLS: List[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "SPY", "AAPL"]
MAX_SYMBOLS = 30
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BINANCE_24H = "https://api.binance.com/api/v3/ticker/24hr"


def default_path() -> str:
    return os.environ.get("WATCHLIST_PATH") or os.path.join(_REPO_ROOT, "training_ground", "paper", "watchlist.json")


def _clean(symbol) -> str:
    return str(symbol or "").strip().upper()


def load(path: Optional[str] = None) -> List[str]:
    """The saved list, or the defaults when there is no (readable) file."""
    try:
        with open(path or default_path(), encoding="utf-8") as fh:
            data = json.load(fh)
        symbols = [_clean(s) for s in data.get("symbols", []) if _clean(s)]
        return symbols[:MAX_SYMBOLS]
    except (OSError, ValueError, AttributeError):
        return list(DEFAULT_SYMBOLS)


def save(symbols: List[str], path: Optional[str] = None) -> List[str]:
    out: List[str] = []
    for s in symbols:
        s = _clean(s)
        if s and s not in out:
            out.append(s)
    out = out[:MAX_SYMBOLS]
    path = path or default_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"symbols": out}, fh, indent=1)
    os.replace(tmp, path)
    return out


def add(symbol: str, path: Optional[str] = None) -> List[str]:
    symbols = load(path)
    s = _clean(symbol)
    if s and s not in symbols:
        symbols.append(s)
    return save(symbols, path)


def remove(symbol: str, path: Optional[str] = None) -> List[str]:
    s = _clean(symbol)
    return save([x for x in load(path) if x != s], path)


# ------------------------------------------------------------------------------------------------------- quotes

@dataclass
class Quote:
    symbol: str
    last: Optional[float] = None
    prev_close: Optional[float] = None
    change_pct: Optional[float] = None
    as_of: Optional[float] = None       # time.time() of the fetch
    source: str = ""
    error: Optional[str] = None


def _finite(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _is_crypto(symbol: str) -> bool:
    from core.chart_builder import is_crypto_symbol
    return is_crypto_symbol(symbol)


def fetch_quote(symbol: str, session=None, timeout: float = 8.0) -> Quote:
    """One quote. Never raises: problems come back in ``Quote.error``."""
    import requests
    getter = (session or requests).get
    now = time.time()
    try:
        if _is_crypto(symbol):
            r = getter(_BINANCE_24H, params={"symbol": symbol}, timeout=timeout)
            if r.status_code != 200:
                return Quote(symbol, as_of=now, source="binance", error=f"HTTP {r.status_code}")
            d = r.json()
            return Quote(symbol, last=_finite(d.get("lastPrice")), prev_close=_finite(d.get("prevClosePrice")),
                         change_pct=_finite(d.get("priceChangePercent")), as_of=now, source="binance")
        from core.yahoo_chart import fetch_chart
        df = fetch_chart(symbol, days=10, interval="1d", timeout=timeout, retries=0, session=session)
        if df is None or df.empty or "Close" not in df.columns:
            return Quote(symbol, as_of=now, source="yahoo", error="no data")
        closes = [c for c in (_finite(x) for x in df["Close"].tolist()) if c]
        if not closes:
            return Quote(symbol, as_of=now, source="yahoo", error="no data")
        last = closes[-1]
        prev = closes[-2] if len(closes) > 1 else None
        pct = (last / prev - 1) * 100 if prev else None
        return Quote(symbol, last=last, prev_close=prev, change_pct=pct, as_of=now, source="yahoo (delayed)")
    except Exception as exc:  # noqa: BLE001 -- a watchlist row must never break a UI
        return Quote(symbol, as_of=now, error=str(exc)[:120])


class QuoteBook:
    """Cached, background-refreshed quotes for a list of symbols."""

    def __init__(self, fetch: Callable[[str], Quote] = fetch_quote, ttl_crypto_s: float = 10.0,
                 ttl_other_s: float = 60.0, max_workers: int = 4, clock: Callable[[], float] = time.time):
        self._fetch = fetch
        self._ttl = (ttl_crypto_s, ttl_other_s)
        self._clock = clock
        self._quotes: Dict[str, Quote] = {}
        self._fetched_at: Dict[str, float] = {}
        self._inflight: set = set()
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="watchlist")

    def _due(self, symbol: str, now: float) -> bool:
        ttl = self._ttl[0] if _is_crypto(symbol) else self._ttl[1]
        t = self._fetched_at.get(symbol)
        return symbol not in self._inflight and (t is None or now - t >= ttl)

    def _run(self, symbol: str) -> None:
        try:
            q = self._fetch(symbol)
        except Exception as exc:  # noqa: BLE001
            q = Quote(symbol, as_of=self._clock(), error=str(exc)[:120])
        with self._lock:
            old = self._quotes.get(symbol)
            if q.last is None and old is not None and old.last is not None:   # keep the last good price, note the error
                q = Quote(symbol, old.last, old.prev_close, old.change_pct, old.as_of, old.source, q.error)
            self._quotes[symbol] = q
            self._fetched_at[symbol] = self._clock()
            self._inflight.discard(symbol)

    def refresh_async(self, symbols: List[str]) -> int:
        """Start fetches for symbols whose cache expired. Returns how many were started."""
        now = self._clock()
        started = 0
        with self._lock:
            due = [s for s in symbols if self._due(s, now)]
            self._inflight.update(due)
        for s in due:
            try:
                self._pool.submit(self._run, s)
                started += 1
            except RuntimeError:          # pool shut down (app closing)
                with self._lock:
                    self._inflight.discard(s)
        return started

    def refresh_now(self, symbols: List[str]) -> None:
        """Synchronous refresh of every listed symbol (tests, CLI)."""
        for s in symbols:
            with self._lock:
                self._inflight.add(s)
            self._run(s)

    def snapshot(self, symbols: List[str]) -> List[dict]:
        """Display rows in list order: symbol, last, change_pct, age_s, source, error (None = not fetched yet)."""
        now = self._clock()
        rows = []
        with self._lock:
            for s in symbols:
                q = self._quotes.get(s)
                rows.append({
                    "symbol": s,
                    "last": q.last if q else None,
                    "change_pct": q.change_pct if q else None,
                    "age_s": (now - q.as_of) if (q and q.as_of and q.last is not None) else None,
                    "source": q.source if q else "",
                    "error": q.error if q else None,
                })
        return rows

    def close(self) -> None:
        self._pool.shutdown(wait=False)


def format_price(value: Optional[float]) -> str:
    if value is None:
        return "—"
    if value >= 1000:
        return f"{value:,.2f}"
    if value >= 1:
        return f"{value:,.2f}" if value >= 10 else f"{value:,.4f}"
    return f"{value:.6f}"


def format_change(pct: Optional[float]) -> str:
    if pct is None:
        return "—"
    return ("▲ +" if pct > 0 else "▼ −" if pct < 0 else "  ") + f"{abs(pct):.2f}%"


if __name__ == "__main__":       # python -m core.watchlist  -> print current quotes once
    syms = load()
    book = QuoteBook()
    book.refresh_now(syms)
    for row in book.snapshot(syms):
        print(f"{row['symbol']:<10} {format_price(row['last']):>14} {format_change(row['change_pct']):>10}  "
              f"{row['source']}{'  ERROR ' + row['error'] if row['error'] else ''}")
    book.close()
    logger.debug("watchlist printed")
