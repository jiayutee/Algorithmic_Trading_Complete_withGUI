"""Kalshi mispricing / arbitrage *signals* (Phase 9.1). Signals only: this module never places orders.

Two structural, model-free checks -- each is a payoff identity, not a forecast:

1. Within one binary market: YES + NO always pays exactly $1. If the cost of buying both
   (yes_ask + no_ask + fees) is below $1, the difference is locked in. Books are normally
   uncrossed, so this is rare; it is checked because it costs nothing.
2. Across the markets of one *mutually exclusive* event (at most one outcome resolves YES):
   - Buy NO on every outcome: pays n-1 (or n if nothing wins), so it is profitable when
     sum(no_ask) + fees < n-1. Needs only "at most one winner".
   - Buy YES on every outcome: pays exactly $1 *only if the outcomes are exhaustive* (exactly one
     wins). Kalshi's ``mutually_exclusive`` flag does not promise that, so these signals carry
     ``needs_exhaustive=True`` and lower confidence; a human/agent must confirm the outcome list is complete.

Fee model (ASSUMPTION -- Kalshi's published taker schedule at time of writing, verify before trading):
    fee = ceil_to_cent(0.07 * contracts * p * (1 - p))
Every edge reported here is *after* fees and is per 1 contract-set, sized by the thinnest leg's resting
depth at the best price only (deeper levels are ignored on purpose: conservative).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from core.kalshi_data import Event, KalshiClient, KalshiError, Market
from core.logger import logger

FEE_RATE = 0.07
MIN_EDGE = 0.01            # $ per set after fees; below this, execution slippage would eat it
MIN_SIZE = 5.0             # contracts on the thinnest leg


def taker_fee(price: float, contracts: float = 1.0, rate: float = FEE_RATE) -> float:
    """Taker fee in dollars for buying ``contracts`` at ``price`` (rounded UP to the next cent)."""
    if price <= 0 or price >= 1 or contracts <= 0:
        return 0.0
    return math.ceil(rate * contracts * price * (1 - price) * 100 - 1e-9) / 100.0


@dataclass
class Signal:
    kind: str                     # within_market | event_buy_all_no | event_buy_all_yes
    market_id: str                # market ticker, or event ticker for event-level signals
    legs: List[str]               # tickers involved
    edge_per_set: float           # guaranteed $ profit per set after fees
    max_sets: float               # limited by thinnest resting size
    cost_per_set: float           # $ outlay per set including fees
    confidence: float             # 0..1 -- structural certainty, NOT a probability of profit
    needs_exhaustive: bool = False
    notes: str = ""
    detail: dict = field(default_factory=dict)

    @property
    def edge_pct(self) -> float:
        return self.edge_per_set / self.cost_per_set if self.cost_per_set > 0 else 0.0

    @property
    def max_profit(self) -> float:
        return self.edge_per_set * self.max_sets


def scan_market(m: Market, *, min_edge: float = MIN_EDGE, min_size: float = MIN_SIZE) -> Optional[Signal]:
    """Buy YES + buy NO in the same market for less than $1 net of fees."""
    y, n = m.yes_ask_quote, m.no_ask_quote
    if m.status != "active" or y is None or n is None:
        return None
    fees = taker_fee(y.price) + taker_fee(n.price)
    cost = y.price + n.price + fees
    edge = 1.0 - cost
    size = min(y.size, n.size)
    if edge < min_edge or size < min_size:
        return None
    return Signal("within_market", m.ticker, [m.ticker], edge, size, cost, confidence=0.95,
                  notes="YES+NO pays $1 with certainty; check the book is not stale before acting",
                  detail={"yes_ask": y.price, "no_ask": n.price, "fees": fees})


def _all_priced(markets: Iterable[Market], side: str) -> Optional[list]:
    quotes = []
    for m in markets:
        q = m.yes_ask_quote if side == "yes" else m.no_ask_quote
        if m.status != "active" or q is None:
            return None            # one un-priced leg makes the set incomplete -> no valid arbitrage
        quotes.append((m, q))
    return quotes


def scan_event(ev: Event, *, min_edge: float = MIN_EDGE, min_size: float = MIN_SIZE) -> List[Signal]:
    """Cross-market checks for a mutually exclusive event with >= 2 outcomes."""
    out: List[Signal] = []
    if not ev.mutually_exclusive or len(ev.markets) < 2:
        return out
    n = len(ev.markets)

    no_legs = _all_priced(ev.markets, "no")
    if no_legs:
        fees = sum(taker_fee(q.price) for _, q in no_legs)
        cost = sum(q.price for _, q in no_legs) + fees
        edge = (n - 1) - cost                      # guaranteed floor of the payoff
        size = min(q.size for _, q in no_legs)
        if edge >= min_edge and size >= min_size:
            out.append(Signal("event_buy_all_no", ev.event_ticker, [m.ticker for m, _ in no_legs], edge, size, cost,
                              confidence=0.85,
                              notes=f"pays {n-1} if one outcome wins, {n} if none does; relies on 'at most one winner'",
                              detail={"outcomes": n, "fees": fees}))

    yes_legs = _all_priced(ev.markets, "yes")
    if yes_legs:
        fees = sum(taker_fee(q.price) for _, q in yes_legs)
        cost = sum(q.price for _, q in yes_legs) + fees
        edge = 1.0 - cost
        size = min(q.size for _, q in yes_legs)
        if edge >= min_edge and size >= min_size:
            out.append(Signal("event_buy_all_yes", ev.event_ticker, [m.ticker for m, _ in yes_legs], edge, size, cost,
                              confidence=0.5, needs_exhaustive=True,
                              notes="pays $1 only if exactly one outcome wins -- confirm the outcome list is exhaustive",
                              detail={"outcomes": n, "fees": fees}))
    return out


def scan(client: KalshiClient, *, max_markets: int = 300, max_events: int = 60,
         min_edge: float = MIN_EDGE, min_size: float = MIN_SIZE) -> List[Signal]:
    """Scan open markets: every market alone, plus each distinct event once. Read-only.
    Errors on individual events are logged and skipped so one bad response cannot stop a scan."""
    signals: List[Signal] = []
    events_seen: List[str] = []
    for m in client.iter_markets(status="open", max_items=max_markets):
        s = scan_market(m, min_edge=min_edge, min_size=min_size)
        if s:
            signals.append(s)
        if m.event_ticker and m.event_ticker not in events_seen and len(events_seen) < max_events:
            events_seen.append(m.event_ticker)
    for et in events_seen:
        try:
            ev = client.get_event(et)
        except KalshiError as exc:
            logger.warning("Kalshi scan: skipping event %s (%s)", et, exc)
            continue
        signals.extend(scan_event(ev, min_edge=min_edge, min_size=min_size))
    return sorted(signals, key=lambda s: -s.max_profit)
