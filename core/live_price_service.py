"""core/live_price_service.py — background multi-symbol live-price streaming service.

Usage
-----
    svc = LivePriceService()
    svc.subscribe("BTCUSDT")
    svc.subscribe("ETHUSDT", callback=my_handler)

    price = svc.get_price("BTCUSDT")   # float | None  (non-blocking)

    svc.unsubscribe("ETHUSDT")
    svc.stop()                          # tears down all active streams

Design notes
------------
* One DataLoader instance is created per subscribed symbol so that the
  existing per-symbol WebSocket machinery (reconnect loop, heartbeat liveness
  monitor) inside DataLoader is reused without modification or duplication.
* A thin wrapper callback extracts the mid-price
  ``(best_bid + best_ask) / 2`` from each incoming order-book update,
  mirroring the logic already present in DataLoader.get_latest_price.
* All shared state (price cache, subscription map) is serialised through a
  single threading.Lock, making get_price() safe to call from any thread
  (e.g. a Qt timer callback in the UI).
* WS tunables (connect timeout, heartbeat interval, etc.) are forwarded
  from LivePriceService to each DataLoader before start_realtime_stream is
  called, so tests can override them through the service instance.
* ui/main_window.py is NOT touched — wiring into the UI is deferred to
  Phase 1.2.  The service is intentionally self-contained.
"""

import threading
from typing import Callable, Dict, List, Optional

from core.data_loader import DataLoader
from core.logger import logger

# Sentinel placed in _loaders while a subscribe() call is in flight.
# Prevents a second concurrent subscribe() for the same symbol from
# creating a duplicate DataLoader.
_PENDING = object()


class LivePriceService:
    """Background service that maintains live WebSocket price streams for
    multiple symbols simultaneously.

    Each subscribed symbol gets its own DataLoader instance so that the
    existing per-symbol reconnect / heartbeat machinery is reused without
    any changes to DataLoader's public API.

    Thread safety
    -------------
    All mutations to the price cache and subscription map are serialised
    through ``_lock``.  ``get_price`` acquires the same lock, making it safe
    to call from any thread (e.g. a Qt timer callback in the UI).

    WS tunables
    -----------
    The following attributes mirror the tunables on DataLoader and are
    forwarded to each newly created DataLoader before start_realtime_stream
    is called.  Override them (after __init__, before subscribe) in tests or
    production code to change timing behaviour:

        svc._ws_connect_timeout   = 2      # seconds to wait for initial connect
        svc._ws_reconnect_initial = 0.02   # first backoff delay (seconds)
        svc._ws_reconnect_max     = 0.05   # backoff cap (seconds)
        svc._ws_heartbeat_interval   = 5.0  # liveness check frequency
        svc._ws_heartbeat_staleness  = 60.0 # seconds without message → reconnect
    """

    def __init__(self, **data_loader_kwargs):
        """
        Parameters
        ----------
        **data_loader_kwargs
            Forwarded verbatim to each DataLoader() created for a new symbol.
            Typical keys: ``binance_key``, ``binance_secret``, etc.
        """
        self._data_loader_kwargs = data_loader_kwargs
        self._lock = threading.Lock()

        # symbol -> DataLoader (or _PENDING while subscribe is in flight)
        self._loaders: Dict[str, object] = {}
        # symbol -> latest float price (None if no message received yet)
        self._price_cache: Dict[str, Optional[float]] = {}
        # symbol -> optional user callback (receives raw order-book dict)
        self._callbacks: Dict[str, Optional[Callable]] = {}

        # WS tunables — forwarded to each DataLoader before stream start.
        # These match the attribute names on DataLoader (see DataLoader.__init__).
        self._ws_connect_timeout: float = 10.0
        self._ws_reconnect_initial: float = 1.0
        self._ws_reconnect_max: float = 30.0
        self._ws_heartbeat_interval: float = 5.0
        self._ws_heartbeat_staleness: float = 60.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subscribe(self, symbol: str, callback: Optional[Callable] = None) -> None:
        """Start streaming for *symbol*.

        If *symbol* is already subscribed (or a subscribe is in flight) this
        is a no-op — the existing stream continues unaffected.

        Parameters
        ----------
        symbol : str
            Ticker / trading pair, e.g. ``"BTCUSDT"``.
        callback : callable, optional
            Called on every incoming update with a single argument — the raw
            order-book dict produced by DataLoader (keys: ``symbol``,
            ``bids``, ``asks``, ``timestamp``, ``exchange``, ``type``).
        """
        with self._lock:
            if symbol in self._loaders:
                logger.info("[LivePriceService] %s already subscribed — skipping", symbol)
                return
            # Reserve the slot before releasing the lock so a concurrent
            # subscribe() for the same symbol sees it and returns early.
            self._loaders[symbol] = _PENDING
            self._price_cache[symbol] = None
            self._callbacks[symbol] = callback

        # Build per-symbol wrapper callback outside the lock (DataLoader
        # construction can be slow; holding the lock would block get_price).
        def _on_update(update: dict) -> None:
            bids = update.get("bids", [])
            asks = update.get("asks", [])
            if bids and asks:
                try:
                    mid = (float(bids[0][0]) + float(asks[0][0])) / 2.0
                    with self._lock:
                        # Only update if the symbol is still subscribed
                        if symbol in self._price_cache:
                            self._price_cache[symbol] = mid
                except (IndexError, ValueError, TypeError):
                    pass
            user_cb = self._callbacks.get(symbol)
            if user_cb is not None:
                try:
                    user_cb(update)
                except Exception as exc:
                    logger.error(
                        "[LivePriceService] callback error for %s: %s", symbol, exc
                    )

        loader = DataLoader(**self._data_loader_kwargs)
        # Forward tunables so tests (and production overrides) take effect.
        loader._ws_connect_timeout = self._ws_connect_timeout
        loader._ws_reconnect_initial = self._ws_reconnect_initial
        loader._ws_reconnect_max = self._ws_reconnect_max
        loader._ws_heartbeat_interval = self._ws_heartbeat_interval
        loader._ws_heartbeat_staleness = self._ws_heartbeat_staleness

        try:
            loader.start_realtime_stream(symbol, callback=_on_update)
        except Exception:
            # Clean up the sentinel so the symbol can be retried later.
            with self._lock:
                if self._loaders.get(symbol) is _PENDING:
                    self._loaders.pop(symbol, None)
                    self._price_cache.pop(symbol, None)
                    self._callbacks.pop(symbol, None)
            raise

        with self._lock:
            self._loaders[symbol] = loader

        logger.info("[LivePriceService] subscribed to %s", symbol)

    def unsubscribe(self, symbol: str) -> None:
        """Stop streaming for *symbol* and remove it from the cache.

        Delegates teardown to ``DataLoader.stop_realtime_stream()`` which
        joins both the ws_thread and the heartbeat_thread before returning,
        guaranteeing no orphaned threads remain after this call.
        """
        with self._lock:
            loader = self._loaders.pop(symbol, None)
            self._price_cache.pop(symbol, None)
            self._callbacks.pop(symbol, None)

        if loader is None:
            logger.info(
                "[LivePriceService] unsubscribe called for unknown symbol %s", symbol
            )
            return

        if loader is _PENDING:
            # Subscribe is still in flight — nothing to stop yet.
            logger.info(
                "[LivePriceService] unsubscribe for %s while subscribe was in flight",
                symbol,
            )
            return

        loader.stop_realtime_stream()
        logger.info("[LivePriceService] unsubscribed from %s", symbol)

    def get_price(self, symbol: str) -> Optional[float]:
        """Return the most-recently-cached price for *symbol*, or ``None``.

        This is a synchronous, non-blocking read from the in-memory cache
        that is populated by the background WebSocket thread.  Returns
        ``None`` if *symbol* is not subscribed or if no order-book message
        has arrived yet.
        """
        with self._lock:
            return self._price_cache.get(symbol)

    def subscribed_symbols(self) -> List[str]:
        """Return a snapshot list of currently subscribed symbols."""
        with self._lock:
            return [
                s for s, loader in self._loaders.items() if loader is not _PENDING
            ]

    def stop(self) -> None:
        """Stop all active streams and clear all internal state.

        Calls ``DataLoader.stop_realtime_stream()`` for every subscribed
        symbol.  After this call returns, no DataLoader threads remain active
        and the price cache is empty.
        """
        with self._lock:
            symbols = list(self._loaders.keys())

        for symbol in symbols:
            self.unsubscribe(symbol)

        logger.info("[LivePriceService] all streams stopped")
