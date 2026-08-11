"""
dash_app/callbacks.py

Registers Dash callbacks on *app*.

Phase 1.1 — load_chart:
    Updates main-chart and status-bar when the user clicks "Load Chart".

Phase 1.2 — live price streaming (this file):
    A dcc.Interval fires every 1 500 ms.  On each tick the
    ``update_live_price`` callback:

    * Crypto symbols (contains "USDT"): reads the latest price from the
      module-level ``LivePriceService`` singleton.  The WebSocket
      subscription is started (once, in a daemon thread) when a crypto
      chart is loaded, so get_price() is always a non-blocking cache read.

    * Equity symbols: polls ``DataLoader.get_latest_price()`` but only
      every ``_EQUITY_POLL_EVERY_N_TICKS`` ticks (~10.5 s at 1 500 ms
      intervals) to avoid hammering the REST API.  Between polls the
      previous price is returned from an in-memory cache.

    Rather than rebuilding the full figure on every tick, the callback
    uses ``dash.Patch()`` (Dash 2.9+) to update only the last trace
    (``figure.data[-1]``), which is a dedicated "live tick" scatter trace
    appended by ``add_live_tick_trace()`` at chart-load time.  The
    candlestick body is never touched.

    A "live badge" label above the chart reads:
        "🟢 Live"          for crypto (WebSocket-backed)
        "🟡 Near real-time" for equities (REST-polling-backed)
    updated by the same interval callback.

Later phases will add order-entry, live P&L, backtest triggers, etc.
"""

from __future__ import annotations

import threading
from typing import Optional

import pandas as pd
from dash import Input, Output, Patch, State, no_update
import dash

from core.chart_builder import (
    add_live_tick_trace,
    build_candlestick_figure,
    is_crypto_symbol,
    overlay_signals,
)
from core.logger import logger

# ---------------------------------------------------------------------------
# Module-level singletons (shared across all callback invocations)
# ---------------------------------------------------------------------------

# Lazy-initialised LivePriceService — created on first crypto chart load.
_live_svc: Optional[object] = None


def _get_live_svc():
    """Return the module-level LivePriceService, creating it on first call."""
    global _live_svc
    if _live_svc is None:
        from core.live_price_service import LivePriceService
        _live_svc = LivePriceService()
        logger.info("[Dash] LivePriceService singleton created")
    return _live_svc


# Lazy-initialised DataLoader used exclusively for equity price polling.
# Created on first equity tick so the heavy __init__ (news pipeline, ccxt)
# does not run until it's actually needed.
_equity_loader: Optional[object] = None


def _get_equity_loader():
    """Return the module-level DataLoader for equity REST polling."""
    global _equity_loader
    if _equity_loader is None:
        from core.data_loader import DataLoader
        _equity_loader = DataLoader()
        logger.info("[Dash] Equity DataLoader singleton created")
    return _equity_loader


# ---------------------------------------------------------------------------
# Interval-tick state (not thread-safe by design — Dash uses a single worker
# in default development mode; keep simple per the Phase 1.2 scope note).
# ---------------------------------------------------------------------------

#: How many 1 500 ms ticks to skip between equity REST fetches.
#: 7 ticks ≈ 10.5 s between REST calls — avoids hammering the API.
_EQUITY_POLL_EVERY_N_TICKS: int = 7

_equity_tick_count: int = 0          # counts ticks since last REST fetch
_equity_last_price: Optional[float] = None  # cached REST price


# ---------------------------------------------------------------------------
# Subscription helpers (background threads so callbacks stay non-blocking)
# ---------------------------------------------------------------------------

def _subscribe_async(symbol: str) -> None:
    """Start a LivePriceService WebSocket stream for *symbol* in a daemon thread.

    Errors (e.g. network unreachable) are logged but not propagated — the UI
    still works; the live-badge will remain in the "connecting" state until
    the WebSocket connects and a price arrives.
    """
    def _run():
        try:
            _get_live_svc().subscribe(symbol)
            logger.info("[Dash] subscribed to %s", symbol)
        except Exception as exc:
            logger.error("[Dash] subscribe(%s) failed: %s", symbol, exc)

    t = threading.Thread(target=_run, name=f"sub-{symbol}", daemon=True)
    t.start()


def _unsubscribe_async(symbol: str) -> None:
    """Stop the LivePriceService stream for *symbol* in a daemon thread."""
    def _run():
        try:
            _get_live_svc().unsubscribe(symbol)
            logger.info("[Dash] unsubscribed from %s", symbol)
        except Exception as exc:
            logger.error("[Dash] unsubscribe(%s) failed: %s", symbol, exc)

    t = threading.Thread(target=_run, name=f"unsub-{symbol}", daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Badge text helpers
# ---------------------------------------------------------------------------

def _badge_connecting(symbol: str) -> str:
    """Return the initial badge text right after clicking Load Chart."""
    if is_crypto_symbol(symbol):
        return "🟢 Live — connecting…"
    return "🟡 Near real-time — loading…"


def _badge_with_price(symbol: str, price: float) -> str:
    """Return the badge text once a price is known."""
    price_str = f"{price:,.4f}" if price < 10_000 else f"{price:,.2f}"
    if is_crypto_symbol(symbol):
        return f"🟢 Live — {price_str}"
    return f"🟡 Near real-time — {price_str}"


# ---------------------------------------------------------------------------
# Callback registration
# ---------------------------------------------------------------------------

def register_callbacks(app: dash.Dash) -> None:
    """Attach all callbacks to *app*.

    Called once from ``dash_app/app.py`` after the layout is set.
    """

    # ------------------------------------------------------------------
    # Phase 1.1 + 1.2: load_chart
    # ------------------------------------------------------------------
    @app.callback(
        Output("main-chart", "figure"),
        Output("chart-status", "children"),
        Output("status-bar", "children"),
        Output("price-interval", "disabled"),
        Output("active-symbol-store", "data"),
        Output("live-badge", "children"),
        Input("load-btn", "n_clicks"),
        State("symbol-dropdown", "value"),
        State("interval-dropdown", "value"),
        State("active-symbol-store", "data"),
        prevent_initial_call=True,
    )
    def load_chart(n_clicks: int, symbol: str, interval: str, prev_symbol: Optional[str]):
        """Fetch OHLCV data for *symbol* and re-render the candlestick chart.

        Phase 1.2 additions vs Phase 1.1:
        * Manages LivePriceService subscriptions — unsubscribes the previous
          symbol, subscribes the new one (crypto only, both in daemon threads).
        * Enables the price-interval so update_live_price starts firing.
        * Resets equity tick-count / price cache on every symbol change.
        * Appends the empty live-tick scatter trace (``data[-1]``) so the
          interval callback can Patch it without touching the candlestick.
        """
        global _equity_tick_count, _equity_last_price

        if not n_clicks:
            return no_update, no_update, no_update, no_update, no_update, no_update

        # -- Subscription housekeeping --------------------------------------
        if prev_symbol and is_crypto_symbol(prev_symbol) and prev_symbol != symbol:
            _unsubscribe_async(prev_symbol)

        _equity_tick_count = 0
        _equity_last_price = None

        if is_crypto_symbol(symbol):
            # Subscribe in background — get_price() returns None until
            # the WS connects; the badge shows "connecting…" in the meantime.
            _subscribe_async(symbol)

        # -- Data fetch -----------------------------------------------------
        try:
            from core.data_loader import DataLoader
            loader = DataLoader()
            df = loader.load_data(
                symbol=symbol,
                source="Historical",
                live=False,
                days=365,
                interval=interval,
            )

            if df is None or df.empty:
                fig = build_candlestick_figure(df=None, symbol=symbol)
                add_live_tick_trace(fig)
                status_msg = f"No data returned for {symbol} ({interval})"
                badge = _badge_connecting(symbol)
                return fig, status_msg, f"Warning: {status_msg}", False, symbol, badge

            fig = build_candlestick_figure(df=df, symbol=symbol, show_ma=False)
            add_live_tick_trace(fig)
            n_candles = len(df)
            status_msg = f"Loaded {n_candles:,} candles for {symbol} ({interval})"
            badge = _badge_connecting(symbol)
            return fig, status_msg, status_msg, False, symbol, badge

        except Exception as exc:  # noqa: BLE001
            fig = build_candlestick_figure(df=None, symbol=symbol)
            add_live_tick_trace(fig)
            err_msg = f"Error loading {symbol}: {exc}"
            # Disable interval and clear symbol store on hard error so the
            # interval callback doesn't try to fetch a price for a failed load.
            return fig, err_msg, err_msg, True, None, "⚠ Data unavailable"

    # ------------------------------------------------------------------
    # Phase 1.2: update_live_price (interval-driven)
    # ------------------------------------------------------------------
    @app.callback(
        Output("main-chart", "figure", allow_duplicate=True),
        Output("live-badge", "children", allow_duplicate=True),
        Input("price-interval", "n_intervals"),
        State("active-symbol-store", "data"),
        prevent_initial_call=True,
    )
    def update_live_price(n_intervals: int, symbol: Optional[str]):
        """Update the live-tick trace and badge on each interval tick.

        Performance notes
        -----------------
        * Returns ``dash.Patch()`` rather than a full figure dict so only the
          changed trace arrays are sent to the browser — the candlestick body
          is untouched.
        * Crypto: non-blocking cache read (LivePriceService.get_price()).
        * Equity: throttled — the REST fetch runs at most once per
          ``_EQUITY_POLL_EVERY_N_TICKS`` ticks; between fetches the cached
          price from the last successful call is re-used.
        """
        global _equity_tick_count, _equity_last_price

        if not symbol:
            return no_update, no_update

        price: Optional[float] = None

        if is_crypto_symbol(symbol):
            # Non-blocking read from the WebSocket price cache.
            price = _get_live_svc().get_price(symbol)
            badge_if_none = "🟢 Live — connecting…"
        else:
            # Equity: throttle REST calls to avoid hammering the API.
            _equity_tick_count += 1
            if _equity_tick_count >= _EQUITY_POLL_EVERY_N_TICKS:
                _equity_tick_count = 0
                try:
                    price = _get_equity_loader().get_latest_price(symbol)
                    _equity_last_price = price
                    logger.debug("[Dash] equity REST price for %s = %s", symbol, price)
                except Exception as exc:
                    logger.debug("[Dash] equity REST fetch failed for %s: %s", symbol, exc)
                    price = _equity_last_price  # fall back to last known
            else:
                price = _equity_last_price

            badge_if_none = "🟡 Near real-time — loading…"

        if price is None:
            return no_update, badge_if_none

        # -- Partial figure update via Patch() ------------------------------
        # We only update the last trace (the live-tick scatter appended by
        # add_live_tick_trace) — not the candlestick body.  Patch sends only
        # the delta to the browser; the full figure is never rebuilt.
        now = pd.Timestamp.utcnow()
        p = Patch()
        p["data"][-1]["x"] = [now]
        p["data"][-1]["y"] = [price]

        badge = _badge_with_price(symbol, price)
        return p, badge

    # ------------------------------------------------------------------
    # Placeholder wiring point for future phases
    # ------------------------------------------------------------------
    # Phase 2: order entry callbacks (buy/sell buttons → broker)
    # Phase 3: backtest trigger → update bt-sharpe / bt-winrate / bt-maxdd
    # Phase 4: live P&L polling → account-balance / pnl-value
