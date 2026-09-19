"""Read-only Kalshi market-data client (Phase 9.0).

Deliberately NOT part of ``brokers/``: Kalshi contracts are binary and settle to $0 or $1, which
does not fit the continuous-position ``submit_order`` / ``get_position`` interface. This module can
only read public market data -- there is no code path here that places an order or needs a key.

Verified against the live public API on 2026-09-19 (no authentication needed for market data):
  GET /markets                       list (filter: status, event_ticker, series_ticker; cursor pagination)
  GET /markets/{ticker}              one market
  GET /markets/{ticker}/orderbook    ``orderbook_fp`` with ``yes_dollars`` / ``no_dollars`` = [[price, size], ...]
  GET /events/{event_ticker}         event metadata (``mutually_exclusive``) + its markets
Prices arrive as dollar strings ("0.5600"), sizes as fixed-point strings ("120.00").

Prices here are floats in dollars, 0..1: a YES contract at 0.56 pays $1 if the event happens.
Sizes are contracts. Rate limits were not published in a machine-readable way, so the client
paces itself (``min_interval``) and backs off on HTTP 429 / 5xx.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests

from core.logger import logger

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


class KalshiError(RuntimeError):
    """The API answered with an error, or answered something we cannot interpret."""


def _f(x: Any) -> Optional[float]:
    """Parse a dollar / fixed-point string ('0.5600', '12.00') to float; None if absent or malformed."""
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class Quote:
    price: float
    size: float


@dataclass
class Market:
    ticker: str
    event_ticker: str
    status: str                      # active / closed / finalized ...
    title: str = ""
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    no_bid: Optional[float] = None
    no_ask: Optional[float] = None
    yes_bid_size: float = 0.0
    yes_ask_size: float = 0.0
    last_price: Optional[float] = None
    volume: float = 0.0
    open_interest: float = 0.0
    liquidity: float = 0.0
    result: Optional[bool] = None    # True = resolved YES, False = resolved NO, None = not settled
    close_time: Optional[str] = None
    expiration_time: Optional[str] = None
    market_type: str = "binary"
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def yes_ask_quote(self) -> Optional[Quote]:
        """Best price to BUY yes, only if someone is actually offering (an empty book shows 0.0000)."""
        if self.yes_ask and self.yes_ask > 0 and self.yes_ask_size > 0:
            return Quote(self.yes_ask, self.yes_ask_size)
        return None

    @property
    def no_ask_quote(self) -> Optional[Quote]:
        """Best price to BUY no. The market endpoint gives no size for it; the order book does
        (best NO ask = 1 - best YES bid, sized by that YES bid)."""
        if self.no_ask and 0 < self.no_ask < 1 and self.yes_bid_size > 0:
            return Quote(self.no_ask, self.yes_bid_size)
        return None

    @property
    def mid(self) -> Optional[float]:
        if self.yes_bid is not None and self.yes_ask is not None and self.yes_ask > 0 and self.yes_bid > 0:
            return (self.yes_bid + self.yes_ask) / 2.0
        return self.last_price if self.last_price else None


@dataclass
class OrderBook:
    ticker: str
    yes_bids: List[Quote]            # best (highest) first
    no_bids: List[Quote]             # best (highest) first

    @property
    def best_yes_bid(self) -> Optional[Quote]:
        return self.yes_bids[0] if self.yes_bids else None

    @property
    def best_no_bid(self) -> Optional[Quote]:
        return self.no_bids[0] if self.no_bids else None

    @property
    def implied_yes_ask(self) -> Optional[Quote]:
        """A NO bid at p is a willingness to sell YES at 1 - p."""
        b = self.best_no_bid
        return Quote(round(1.0 - b.price, 4), b.size) if b else None

    @property
    def implied_no_ask(self) -> Optional[Quote]:
        b = self.best_yes_bid
        return Quote(round(1.0 - b.price, 4), b.size) if b else None

    def depth(self, side: str, levels: int = 5) -> float:
        """Contracts resting on the best ``levels`` bid levels of 'yes' or 'no'."""
        book = self.yes_bids if side == "yes" else self.no_bids
        return float(sum(q.size for q in book[:levels]))


@dataclass
class Event:
    event_ticker: str
    title: str
    mutually_exclusive: bool
    category: str
    markets: List[Market]


def parse_market(d: Dict[str, Any]) -> Market:
    result = {"yes": True, "no": False}.get(str(d.get("result", "")).lower())
    return Market(
        ticker=d["ticker"], event_ticker=d.get("event_ticker", ""), status=str(d.get("status", "")),
        title=d.get("title") or d.get("yes_sub_title") or "",
        yes_bid=_f(d.get("yes_bid_dollars")), yes_ask=_f(d.get("yes_ask_dollars")),
        no_bid=_f(d.get("no_bid_dollars")), no_ask=_f(d.get("no_ask_dollars")),
        yes_bid_size=_f(d.get("yes_bid_size_fp")) or 0.0, yes_ask_size=_f(d.get("yes_ask_size_fp")) or 0.0,
        last_price=_f(d.get("last_price_dollars")), volume=_f(d.get("volume_fp")) or 0.0,
        open_interest=_f(d.get("open_interest_fp")) or 0.0, liquidity=_f(d.get("liquidity_dollars")) or 0.0,
        result=result, close_time=d.get("close_time"), expiration_time=d.get("expiration_time"),
        market_type=str(d.get("market_type", "binary")), raw=d)


def _levels(raw: Any) -> List[Quote]:
    quotes = []
    for row in raw or []:
        try:
            p, s = float(row[0]), float(row[1])
        except (TypeError, ValueError, IndexError):
            continue
        if s > 0:
            quotes.append(Quote(p, s))
    return sorted(quotes, key=lambda q: -q.price)


class KalshiClient:
    def __init__(self, session: Optional[requests.Session] = None, base_url: str = BASE_URL, *,
                 min_interval: float = 0.25, timeout: float = 15.0, max_retries: int = 3,
                 sleep=time.sleep, clock=time.monotonic):
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip("/")
        self.min_interval, self.timeout, self.max_retries = min_interval, timeout, max_retries
        self._sleep, self._clock = sleep, clock
        self._last_call = 0.0

    def _get(self, path: str, params: Optional[dict] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        for attempt in range(1, self.max_retries + 1):
            wait = self.min_interval - (self._clock() - self._last_call)
            if wait > 0:
                self._sleep(wait)
            self._last_call = self._clock()
            try:
                resp = self.session.get(url, params={k: v for k, v in (params or {}).items() if v is not None},
                                        timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    raise KalshiError(f"request to {path} failed: {exc}") from exc
                self._sleep(2 ** attempt * 0.5)
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == self.max_retries:
                    raise KalshiError(f"{path}: HTTP {resp.status_code} after {attempt} attempts")
                retry_after = _f(getattr(resp, "headers", {}).get("Retry-After")) or 2 ** attempt * 0.5
                logger.warning("Kalshi %s -> HTTP %s; retrying in %.1fs", path, resp.status_code, retry_after)
                self._sleep(retry_after)
                continue
            if resp.status_code != 200:
                raise KalshiError(f"{path}: HTTP {resp.status_code} {getattr(resp, 'text', '')[:120]}")
            try:
                return resp.json()
            except ValueError as exc:
                raise KalshiError(f"{path}: response was not JSON") from exc
        raise KalshiError(f"{path}: gave up")          # pragma: no cover

    # ---------------------------------------------------------------- markets

    def list_markets(self, *, status: Optional[str] = "open", limit: int = 100, cursor: Optional[str] = None,
                     event_ticker: Optional[str] = None, series_ticker: Optional[str] = None,
                     include_multivariate: bool = False) -> Tuple[List[Market], Optional[str]]:
        """``include_multivariate=False`` (default) sends ``mve_filter=exclude``. Verified 2026-09-19: without it the
        first 200 open markets were ALL empty auto-generated multivariate combos (KXMVE*) with no quotes at all,
        which makes any scan silently blind."""
        data = self._get("/markets", {"status": status, "limit": min(int(limit), 1000), "cursor": cursor,
                                      "event_ticker": event_ticker, "series_ticker": series_ticker,
                                      "mve_filter": None if include_multivariate else "exclude"})
        return [parse_market(m) for m in data.get("markets", [])], (data.get("cursor") or None)

    def iter_markets(self, *, max_items: int = 500, page_size: int = 200, **filters) -> Iterator[Market]:
        """Follow the cursor until ``max_items`` markets or the end."""
        cursor, n = None, 0
        while n < max_items:
            markets, cursor = self.list_markets(limit=min(page_size, max_items - n), cursor=cursor, **filters)
            for m in markets:
                yield m
                n += 1
                if n >= max_items:
                    return
            if not cursor or not markets:
                return

    def get_market(self, ticker: str) -> Market:
        data = self._get(f"/markets/{ticker}")
        if "market" not in data:
            raise KalshiError(f"market {ticker}: unexpected response shape")
        return parse_market(data["market"])

    def get_orderbook(self, ticker: str) -> OrderBook:
        data = self._get(f"/markets/{ticker}/orderbook")
        ob = data.get("orderbook_fp") or data.get("orderbook") or {}
        return OrderBook(ticker=ticker, yes_bids=_levels(ob.get("yes_dollars")), no_bids=_levels(ob.get("no_dollars")))

    def get_event(self, event_ticker: str) -> Event:
        data = self._get(f"/events/{event_ticker}", {"with_nested_markets": "true"})
        ev = data.get("event", {})
        raw_markets = data.get("markets") or ev.get("markets") or []
        return Event(event_ticker=ev.get("event_ticker", event_ticker), title=ev.get("title", ""),
                     mutually_exclusive=bool(ev.get("mutually_exclusive")), category=ev.get("category", ""),
                     markets=[parse_market(m) for m in raw_markets])

    def settled_markets(self, *, event_ticker: Optional[str] = None, series_ticker: Optional[str] = None,
                        limit: int = 100) -> List[Market]:
        """Historical outcomes: settled markets with ``result`` True (YES) / False (NO)."""
        out = []
        for m in self.iter_markets(status="settled", max_items=limit, event_ticker=event_ticker, series_ticker=series_ticker):
            if m.result is not None:
                out.append(m)
        return out
