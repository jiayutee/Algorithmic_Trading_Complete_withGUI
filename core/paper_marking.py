"""Keep EVERY open paper position marked to a real price, not just the symbol on the chart.

The Dash live-price callback and the desktop chart only knew the price of the symbol being displayed, so any other
holding was marked to a stale (or, before the random walk was removed, invented) price and the headline P&L lied.
``HeldPriceMarker`` asks a caller-supplied ``fetch_price(symbol)`` for each held symbol -- throttled per symbol so REST
data sources are not hammered -- and feeds the answer to ``broker.update_price``. Failures are swallowed (a price
feed hiccup must not break the UI) and simply leave that symbol's price to age until the broker treats it as stale.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Iterable, List, Optional

from core.logger import logger


class HeldPriceMarker:
    def __init__(self, fetch_price: Callable[[str], Optional[float]],
                 interval_for: Callable[[str], float] = lambda symbol: 30.0,
                 clock: Callable[[], float] = time.monotonic):
        self._fetch, self._interval_for, self._clock = fetch_price, interval_for, clock
        self._last: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._busy = False

    @staticmethod
    def held_symbols(broker) -> List[str]:
        try:
            with broker._lock:
                return [s for s, p in broker.positions.items() if getattr(p, "qty", 0)]
        except Exception:  # noqa: BLE001 -- brokers without a positions dict simply have nothing to mark
            return []

    def refresh(self, broker, skip: Iterable[str] = ()) -> int:
        """Update prices of held symbols (except ``skip``, which the caller already marks). Returns how many were updated."""
        if broker is None or not hasattr(broker, "update_price"):
            return 0
        skip = set(skip or ())
        updated = 0
        for sym in self.held_symbols(broker):
            if sym in skip:
                continue
            now = self._clock()
            if now - self._last.get(sym, -1e18) < self._interval_for(sym):
                continue
            self._last[sym] = now                  # count the attempt so a failing source is not retried every tick
            try:
                price = self._fetch(sym)
            except Exception as exc:  # noqa: BLE001
                logger.debug("held-price fetch failed for %s: %s", sym, exc)
                continue
            if price:
                broker.update_price(sym, price)
                updated += 1
        return updated

    def refresh_async(self, broker, skip: Iterable[str] = ()) -> None:
        """Run ``refresh`` on a daemon thread (for UI threads that must not block on REST calls). One at a time."""
        with self._lock:
            if self._busy:
                return
            self._busy = True

        def _run():
            try:
                self.refresh(broker, skip)
            finally:
                with self._lock:
                    self._busy = False

        threading.Thread(target=_run, name="held-price-refresh", daemon=True).start()
