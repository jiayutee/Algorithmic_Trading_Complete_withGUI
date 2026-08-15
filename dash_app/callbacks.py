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

import calendar as calendar_mod
import datetime as dt_mod
import threading
from typing import Optional

import pandas as pd
from dash import Input, Output, Patch, State, html, no_update
import dash

from core.chart_builder import (
    THEME,
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


# Lazy-initialised SimulatedBroker — created on first order submission so the
# background price-simulation thread doesn't start until the user actually
# places an order.
_broker: Optional[object] = None


def _get_broker():
    """Return the module-level SimulatedBroker, creating it on first call."""
    global _broker
    if _broker is None:
        from brokers.simulatedbroker import SimulatedBroker
        _broker = SimulatedBroker()
        logger.info("[Dash] SimulatedBroker singleton created")
    return _broker


def _broker_or_none():
    """Return the existing SimulatedBroker singleton without creating one.

    Unlike ``_get_broker()``, this never initialises the broker.  Used by
    display-only callbacks (positions panel, PnL calendar) so the broker's
    background price-simulation thread is not started on page load before
    the user places any order.
    """
    return _broker


# ---------------------------------------------------------------------------
# PnL Calendar cell background colors (no THEME equivalent — matched directly
# from the PyQt5 reference implementation in ui/main_window.py).
# ---------------------------------------------------------------------------

_CAL_GREEN_BG = "#1a4731"  # dark green cell bg for days with PnL >= 0
_CAL_RED_BG   = "#3d1a1a"  # dark red cell bg for days with PnL < 0
_CAL_DIMMED_FG = "#484f58"  # very muted text for out-of-month filler cells


# ---------------------------------------------------------------------------
# Positions and PnL Calendar component builders (pure helpers, no Dash context)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Orders / trade-blotter helper (pure helper, no Dash context)
# ---------------------------------------------------------------------------

def _build_orders_table_data(broker) -> tuple:
    """Build ``(data, status_text)`` for the ``orders-table`` DataTable.

    ``data``        — list of row dicts, one per order in
                      ``broker.order_history``.
    ``status_text`` — summary string matching the PyQt5 label format:
                      "Orders: N total, M filled".

    Column keys and formatting mirror ``_refresh_orders_tab()`` in
    ``ui/main_window.py`` exactly:
    ======= ====== ==== ==== ==== =========== ======
    time    symbol side type qty  fill_price  status
    ======= ====== ==== ==== ==== =========== ======

    Safe to call with ``broker=None`` — returns empty data and
    "Orders: none yet".
    """
    _EMPTY: tuple = ([], "Orders: none yet")

    if broker is None or not hasattr(broker, "order_history"):
        return _EMPTY

    try:
        history = broker.order_history
        if not history:
            return _EMPTY

        data = []
        for order in history:
            ts = dt_mod.datetime.fromtimestamp(order.created_at).strftime("%H:%M:%S")
            side_str   = order.side.value       if hasattr(order.side,       "value") else str(order.side)
            type_str   = order.order_type.value if hasattr(order.order_type, "value") else str(order.order_type)
            status_str = order.status.value     if hasattr(order.status,     "value") else str(order.status)
            fill_price = (
                f"${order.filled_avg_price:,.4f}" if order.filled_avg_price else "—"
            )

            data.append({
                "time":       ts,
                "symbol":     order.symbol,
                "side":       side_str.upper(),
                "type":       type_str.capitalize(),
                "qty":        f"{order.filled_qty:.4f}",
                "fill_price": fill_price,
                "status":     status_str.capitalize(),
            })

        count  = len(history)
        filled = sum(
            1 for o in history
            if (o.status.value if hasattr(o.status, "value") else str(o.status)) == "filled"
        )
        status_text = f"Orders: {count} total, {filled} filled"
        return data, status_text

    except Exception as exc:  # noqa: BLE001
        logger.error("[Dash] orders table build error: %s", exc)
        return [], f"Orders: error — {exc}"

def _build_position_row(symbol: str, pos) -> html.Div:
    """Build one row in the positions panel for a single open position.

    Mirrors the per-symbol block in ``update_positions_display()``
    (ui/main_window.py lines 1538-1545) — same data fields, adapted for Dash.
    """
    pnl = pos.pnl
    sign = "+" if pnl >= 0 else ""
    pnl_color = THEME["green"] if pnl >= 0 else THEME["red"]
    return html.Div(
        style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "padding": "4px 6px",
            "borderRadius": "4px",
            "backgroundColor": THEME["bg_dark"],
            "marginBottom": "4px",
            "fontSize": "11px",
        },
        children=[
            html.Div([
                html.Span(symbol, style={"color": THEME["text_main"], "fontWeight": "600"}),
                html.Span(
                    f"  {pos.qty:+.4f} @ ${pos.avg_price:.2f}",
                    style={"color": THEME["text_muted"]},
                ),
            ]),
            html.Span(f"{sign}${pnl:,.2f}", style={"color": pnl_color, "fontWeight": "600"}),
        ],
    )


def _build_positions_content(broker) -> list:
    """Build the ``children`` list for the ``positions-content`` Div.

    Returns a list of Dash components: one row per open position, or a
    single muted "No active positions" span when the portfolio is flat.
    Safe to call with ``broker=None`` (returns the empty-state message).
    """
    _no_positions = [
        html.Span(
            "No active positions",
            style={"color": THEME["text_muted"], "fontSize": "11px"},
        )
    ]

    if broker is None:
        return _no_positions

    try:
        rows = []
        if hasattr(broker, "positions"):
            for symbol, pos in broker.positions.items():
                if pos.qty != 0:
                    rows.append(_build_position_row(symbol, pos))
        return rows if rows else _no_positions
    except Exception as exc:  # noqa: BLE001
        logger.error("[Dash] positions display error: %s", exc)
        return [
            html.Span(
                f"Error loading positions: {exc}",
                style={"color": THEME["red"], "fontSize": "11px"},
            )
        ]


def _build_pnl_calendar_grid(year: int, month: int, by_day: dict) -> list:
    """Build the 42-cell calendar grid as a single-element list of Dash Divs.

    Returns ``[html.Div(...)]`` — the outer CSS-grid div containing 42 day
    cells.  Each cell is styled to match the PyQt5 reference implementation:
    * in-month + PnL >= 0 : dark-green background, green text
    * in-month + PnL < 0  : dark-red background, red text
    * in-month + no PnL   : card background, muted text
    * out-of-month filler  : page-dark background, very muted text
    * today                : accent-color border (#58a6ff ≈ THEME["accent"])

    Uses ``calendar.Calendar(firstweekday=0).itermonthdates()`` (Monday-first)
    to match the weekday header row built in ``_bottom_tabs_panel()`` in
    layout.py, and the PyQt5 ``_refresh_pnl_calendar()`` method.
    """
    today = dt_mod.date.today()
    cal = calendar_mod.Calendar(firstweekday=0)
    month_dates = list(cal.itermonthdates(year, month))[:42]

    _cell_base: dict = {
        "borderRadius": "3px",
        "minHeight": "44px",
        "padding": "3px",
        "fontSize": "10px",
        "fontWeight": "600",
        "lineHeight": "1.4",
        "overflow": "hidden",
    }

    cells = []
    for cell_date in month_dates:
        in_month = cell_date.month == month
        pnl = by_day.get(cell_date)
        is_today = cell_date == today

        if not in_month:
            cell_style = {
                **_cell_base,
                "backgroundColor": THEME["bg_dark"],
                "border": f"1px solid {THEME['bg_card']}",
                "color": _CAL_DIMMED_FG,
            }
            cell_children: object = str(cell_date.day)
        elif pnl is not None:
            bg = _CAL_GREEN_BG if pnl >= 0 else _CAL_RED_BG
            fg = THEME["green"] if pnl >= 0 else THEME["red"]
            border_color = THEME["accent"] if is_today else THEME["border_dim"]
            sign = "+" if pnl >= 0 else ""
            cell_style = {
                **_cell_base,
                "backgroundColor": bg,
                "border": f"1px solid {border_color}",
                "color": fg,
            }
            cell_children = [str(cell_date.day), html.Br(), f"{sign}${pnl:,.2f}"]
        else:
            border_color = THEME["accent"] if is_today else THEME["border_dim"]
            cell_style = {
                **_cell_base,
                "backgroundColor": THEME["bg_card"],
                "border": f"1px solid {border_color}",
                "color": THEME["text_muted"],
            }
            cell_children = str(cell_date.day)

        cells.append(html.Div(cell_children, style=cell_style))

    return [
        html.Div(
            cells,
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(7, 1fr)",
                "gap": "2px",
            },
        )
    ]


# ---------------------------------------------------------------------------
# Order-status text style (shared between the two order-entry callbacks)
# ---------------------------------------------------------------------------

_ORDER_STATUS_BASE_STYLE: dict = {
    "fontSize": "11px",
    "marginTop": "6px",
    "minHeight": "16px",
    "wordBreak": "break-word",
}

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
# Pure logic helpers (extracted for testability; called by callbacks below)
# ---------------------------------------------------------------------------

def _price_input_style_and_placeholder(order_type: str) -> tuple:
    """Return ``(wrapper_style_dict, price_input_placeholder)`` for the given order type.

    Pure function — no Dash callback context required — extracted so it can be
    tested directly.  Called by ``toggle_price_input`` inside
    ``register_callbacks``.
    """
    if order_type == "limit":
        return {"display": "block"}, "Limit Price"
    if order_type == "stop":
        return {"display": "block"}, "Stop Price"
    # "market" (default) — price input is hidden
    return {"display": "none"}, "Price"


def _validate_and_submit_order(
    broker,
    side: str,
    qty: Optional[float],
    order_type: str,
    price: Optional[float],
    symbol: Optional[str],
) -> tuple:
    """Validate inputs and submit an order to *broker*.

    Returns ``(status_text, style_dict)`` — the same 2-tuple that the Dash
    callback pushes to the ``order-status`` children / style outputs.

    Extracted as a pure function (no Dash callback context) so it can be
    covered by unit tests without spinning up a Dash server.
    """

    def _err(msg: str):
        return msg, {**_ORDER_STATUS_BASE_STYLE, "color": THEME["red"]}

    def _ok(msg: str):
        return msg, {**_ORDER_STATUS_BASE_STYLE, "color": THEME["green"]}

    def _warn(msg: str):
        return msg, {**_ORDER_STATUS_BASE_STYLE, "color": THEME["orange"]}

    def _info(msg: str):
        return msg, {**_ORDER_STATUS_BASE_STYLE, "color": THEME["text_muted"]}

    # Validation ----------------------------------------------------------
    if not symbol:
        return _err("Load a chart first before placing an order.")

    if qty is None or qty <= 0:
        return _err("Qty must be greater than 0.")

    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    if order_type == "limit":
        if not price or price <= 0:
            return _err("Limit Price is required for Limit orders.")
        limit_price = float(price)
    elif order_type == "stop":
        if not price or price <= 0:
            return _err("Stop Price is required for Stop orders.")
        stop_price = float(price)

    # Submit --------------------------------------------------------------
    try:
        order = broker.submit_order(
            symbol=symbol,
            qty=float(qty),
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
        )
        status = order.status.value
        if status == "filled":
            return _ok(
                f"{side.upper()} order filled for {qty} {symbol}"
                f" @ ${order.filled_avg_price:.2f}"
            )
        if status == "pending":
            return _warn(f"{side.upper()} order pending for {qty} {symbol}")
        if status == "rejected":
            return _err(f"Order rejected: {qty} {symbol}")
        return _info(f"Order {status}: {qty} {symbol}")

    except Exception as exc:  # noqa: BLE001
        logger.error("[Dash] submit_order error: %s", exc)
        return _err(f"Order error: {exc}")


def _extract_backtest_metrics(results: dict) -> tuple:
    """Extract display strings from a Backtester results dict.

    Returns a 6-tuple:
        ``(sharpe_str, winrate_str, maxdd_str, alpha_str, beta_str, status_msg)``

    All values are formatted strings suitable for populating the backtest-results
    card spans (``bt-sharpe``, ``bt-winrate``, ``bt-maxdd``, ``bt-alpha``,
    ``bt-beta``) and the ``bt-status`` feedback div.

    Pure function — no Dash callback context required.  Extracted for testability.
    Mirrors the display-field extraction pattern in ``run_backtest()``
    (ui/main_window.py ~line 1435-1458), using the same fallback chain:
    ``summary.get(key, results.get(shorthand, default))``.
    """
    if not results:
        return "N/A", "N/A", "N/A", "N/A", "N/A", "Backtest returned no results."
    if "error" in results:
        return "N/A", "N/A", "N/A", "N/A", "N/A", f"Backtest error: {results['error']}"

    summary   = results.get("summary", {})
    sharpe    = summary.get("Sharpe Ratio",    results.get("sharpe",       0))
    max_dd    = summary.get("Max Drawdown (%)", results.get("max_drawdown", 0))
    # NOTE: unlike the other fields, the "Win Rate" default must NOT be built with an
    # eager f-string (`f"{results.get('win_rate', 0):.2f}%"` as a dict.get() default) —
    # Python evaluates default-argument expressions unconditionally even when the key
    # IS present, so a non-numeric results['win_rate'] (e.g. already a formatted string
    # like "55.00%") would raise ValueError before .get() ever got to ignore the default.
    if "Win Rate" in summary:
        win_rate = summary["Win Rate"]
    else:
        raw_win_rate = results.get("win_rate", 0)
        win_rate = (
            f"{raw_win_rate:.2f}%" if isinstance(raw_win_rate, (int, float)) else raw_win_rate
        )
    alpha     = summary.get("Alpha",           results.get("alpha",        0))
    beta      = summary.get("Beta",            results.get("beta",         0))
    final_val = summary.get("Final Value", 0)
    total_pnl = summary.get("P&L",        0)

    sharpe_str  = f"{sharpe:.2f}"  if isinstance(sharpe,   (int, float)) else "N/A"
    maxdd_str   = f"{max_dd:.2f}%" if isinstance(max_dd,   (int, float)) else "N/A"
    winrate_str = win_rate         if isinstance(win_rate, str)          else f"{win_rate:.2f}%"
    alpha_str   = f"{alpha:.4f}"   if isinstance(alpha,    (int, float)) else "N/A"
    beta_str    = f"{beta:.4f}"    if isinstance(beta,     (int, float)) else "N/A"

    status_msg = (
        f"Backtest complete | Final: ${final_val:,.2f} | "
        f"P&L: ${total_pnl:+,.2f} | Sharpe: {sharpe_str} | "
        f"MaxDD: {maxdd_str} | Win Rate: {winrate_str}"
    )
    return sharpe_str, winrate_str, maxdd_str, alpha_str, beta_str, status_msg


def _build_equity_curve_figure(total_asset_value: list):
    """Build a dark-themed equity-curve line chart from *total_asset_value*.

    Called by ``run_backtest_callback`` in ``register_callbacks``.  Returns an
    empty placeholder figure when *total_asset_value* is falsy (empty list or
    None) so the chart area never shows a broken layout.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    if total_asset_value:
        fig.add_trace(go.Scatter(
            y=total_asset_value,
            mode="lines",
            line=dict(color=THEME["accent"], width=2),
            name="Portfolio Value",
            fill="tozeroy",
            fillcolor="rgba(88, 166, 255, 0.08)",  # THEME["accent"] @ 8% opacity
        ))
    fig.update_layout(
        paper_bgcolor=THEME["bg_dark"],
        plot_bgcolor=THEME["bg_card"],
        font=dict(color=THEME["text_muted"], size=11),
        margin=dict(l=50, r=10, t=10, b=30),
        height=200,
        xaxis=dict(showgrid=False, color=THEME["text_muted"], zeroline=False),
        yaxis=dict(
            showgrid=True,
            gridcolor=THEME["border"],
            color=THEME["text_muted"],
            zeroline=False,
            tickformat="$,.0f",
        ),
        showlegend=False,
    )
    return fig


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
    # Phase 1.3: order-type dropdown → toggle price input visibility
    # ------------------------------------------------------------------
    @app.callback(
        Output("order-price-wrapper", "style"),
        Output("order-price-input", "placeholder"),
        Input("order-type-dropdown", "value"),
        prevent_initial_call=True,
    )
    def toggle_price_input(order_type: str):
        """Show or hide the price input depending on the selected order type.

        Mirrors ``on_order_type_changed()`` in ui/main_window.py — pure UI,
        no broker call.
        """
        return _price_input_style_and_placeholder(order_type)

    # ------------------------------------------------------------------
    # Phase 1.3: buy/sell buttons → SimulatedBroker.submit_order
    # ------------------------------------------------------------------
    @app.callback(
        Output("order-status", "children"),
        Output("order-status", "style"),
        Input("buy-btn", "n_clicks"),
        Input("sell-btn", "n_clicks"),
        State("order-qty-input", "value"),
        State("order-type-dropdown", "value"),
        State("order-price-input", "value"),
        State("active-symbol-store", "data"),
        prevent_initial_call=True,
    )
    def submit_order_callback(
        buy_clicks: Optional[int],
        sell_clicks: Optional[int],
        qty: Optional[float],
        order_type: str,
        price: Optional[float],
        symbol: Optional[str],
    ):
        """Submit a buy or sell order to the SimulatedBroker.

        Mirrors ``place_order()`` in ui/main_window.py, adapted for Dash.
        ``ctx.triggered_id`` identifies which button was clicked — both
        buy-btn and sell-btn feed this single callback output.
        """
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update, no_update

        triggered_id = ctx.triggered_id  # "buy-btn" or "sell-btn"
        side = "buy" if triggered_id == "buy-btn" else "sell"

        return _validate_and_submit_order(
            broker=_get_broker(),
            side=side,
            qty=qty,
            order_type=order_type,
            price=price,
            symbol=symbol,
        )

    # ------------------------------------------------------------------
    # Phase 1.4: PnL Calendar month navigation → update pnl-calendar-store
    # ------------------------------------------------------------------
    @app.callback(
        Output("pnl-calendar-store", "data"),
        Input("pnl-prev-btn", "n_clicks"),
        Input("pnl-next-btn", "n_clicks"),
        Input("pnl-today-btn", "n_clicks"),
        State("pnl-calendar-store", "data"),
        prevent_initial_call=True,
    )
    def update_calendar_store(
        prev_clicks: Optional[int],
        next_clicks: Optional[int],
        today_clicks: Optional[int],
        store_data: dict,
    ) -> dict:
        """Advance or retreat the displayed calendar month in the store.

        Mirrors ``_shift_pnl_calendar_month()`` and
        ``_jump_pnl_calendar_to_today()`` from ui/main_window.py, adapted
        for Dash's stateless callback model via a dcc.Store.
        """
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update

        triggered_id = ctx.triggered_id

        if triggered_id == "pnl-today-btn":
            today = dt_mod.date.today()
            return {"year": today.year, "month": today.month}

        year: int = store_data["year"]
        month: int = store_data["month"]

        if triggered_id == "pnl-prev-btn":
            month -= 1
            if month < 1:
                month = 12
                year -= 1
        elif triggered_id == "pnl-next-btn":
            month += 1
            if month > 12:
                month = 1
                year += 1

        return {"year": year, "month": month}

    # ------------------------------------------------------------------
    # Phase 1.4: pnl-calendar-store + order-status → rebuild calendar display
    # ------------------------------------------------------------------
    @app.callback(
        Output("pnl-calendar-title", "children"),
        Output("pnl-calendar-total", "children"),
        Output("pnl-calendar-total", "style"),
        Output("pnl-calendar-grid", "children"),
        Input("pnl-calendar-store", "data"),
        Input("order-status", "children"),
    )
    def update_pnl_calendar_display(store_data: dict, order_status: object):
        """Rebuild the PnL calendar grid for the year/month held in the store.

        Triggered on page load (both inputs available immediately), on every
        month-navigation click (store changes), and after every order placement
        (order-status changes) — so the calendar always reflects the latest
        broker data without a separate polling interval.

        Mirrors ``_refresh_pnl_calendar()`` in ui/main_window.py.
        """
        year: int = store_data["year"]
        month: int = store_data["month"]

        by_day: dict = {}
        broker = _broker_or_none()
        if broker is not None and hasattr(broker, "get_pnl_by_day"):
            try:
                by_day = broker.get_pnl_by_day()
            except Exception as exc:  # noqa: BLE001
                logger.error("[Dash] get_pnl_by_day error: %s", exc)

        title = f"{calendar_mod.month_name[month]} {year}"
        grid_children = _build_pnl_calendar_grid(year, month, by_day)

        month_total = sum(
            pnl for d, pnl in by_day.items()
            if d.year == year and d.month == month
        )
        sign = "+" if month_total >= 0 else ""
        total_text = f"Month total: {sign}${month_total:,.2f}"
        total_color = THEME["green"] if month_total >= 0 else THEME["red"]
        total_style = {
            "fontSize": "12px",
            "fontWeight": "600",
            "marginLeft": "8px",
            "color": total_color,
        }

        return title, total_text, total_style, grid_children

    # ------------------------------------------------------------------
    # Phase 1.4: order-status → rebuild positions panel
    # ------------------------------------------------------------------
    @app.callback(
        Output("positions-content", "children"),
        Input("order-status", "children"),
    )
    def update_positions(order_status: object):
        """Rebuild the open positions list after an order is placed or on page load.

        Mirrors ``update_positions_display()`` in ui/main_window.py.
        Listening to order-status children means this fires immediately after
        every buy/sell submission — no separate polling interval needed.
        """
        return _build_positions_content(_broker_or_none())

    # ------------------------------------------------------------------
    # Phase 1.5: bt-run-btn → run backtest + update results panel
    # (Supersedes the forward-reference "Phase 3" placeholder that was here.)
    # ------------------------------------------------------------------

    #: Strategy dropdown label → (module, class) for importlib.import_module
    _STRATEGY_CLASS_MAP = {
        "MACD/RSI":      ("strategies.simple_strategies", "MACD_RSI_Strategy"),
        "EMA Crossover": ("strategies.simple_strategies", "EMACrossoverStrategy"),
        "Stochastic":    ("strategies.simple_strategies", "StochasticStrategy"),
    }

    @app.callback(
        Output("bt-sharpe", "children"),
        Output("bt-winrate", "children"),
        Output("bt-maxdd", "children"),
        Output("bt-alpha", "children"),
        Output("bt-beta", "children"),
        Output("bt-status", "children"),
        Output("bt-status", "style"),
        Output("equity-curve-chart", "figure"),
        Output("main-chart", "figure", allow_duplicate=True),
        Input("bt-run-btn", "n_clicks"),
        State("active-symbol-store", "data"),
        State("strategy-dropdown", "value"),
        State("bt-cash-input", "value"),
        prevent_initial_call=True,
    )
    def run_backtest_callback(
        n_clicks: Optional[int],
        symbol: Optional[str],
        strategy_name: Optional[str],
        cash: Optional[float],
    ):
        """Run a backtest and update the Backtest Results panel + Equity Curve tab.

        Phase 1.5 — supersedes the "Phase 3" forward-reference placeholder.

        Flow:
        1. Validate: symbol must be loaded, a strategy must be selected.
        2. Load OHLCV data fresh via DataLoader (same path as load_chart).
        3. Map strategy-dropdown value → backtrader strategy class via
           _STRATEGY_CLASS_MAP (avoids a circular import with ui/).
        4. Run Backtester; extract metrics via _extract_backtest_metrics().
        5. Build equity-curve-chart figure from total_asset_value list.
        6. Rebuild main-chart with signal markers via overlay_signals()
           (allow_duplicate=True since load_chart also writes to main-chart).

        Mirrors ``run_backtest()`` in ui/main_window.py, adapted for Dash's
        stateless callback model.
        """
        _err_style = {**_ORDER_STATUS_BASE_STYLE, "color": THEME["red"]}
        _ok_style  = {**_ORDER_STATUS_BASE_STYLE, "color": THEME["green"]}
        _info_style = {**_ORDER_STATUS_BASE_STYLE, "color": THEME["text_muted"]}

        # 9-output no-op for the defensive early-exit guard
        _noop9 = (no_update,) * 9

        def _err(msg: str):
            return (
                no_update, no_update, no_update, no_update, no_update,
                msg, _err_style,
                no_update, no_update,
            )

        def _info(msg: str):
            return (
                no_update, no_update, no_update, no_update, no_update,
                msg, _info_style,
                no_update, no_update,
            )

        if not n_clicks:
            return _noop9

        # -- Input validation -----------------------------------------------
        if not symbol:
            return _err("Load a chart first before running a backtest.")

        if not strategy_name or strategy_name == "None":
            return _err("Select a strategy in the top bar before running a backtest.")

        if strategy_name not in _STRATEGY_CLASS_MAP:
            return _err(f"Unknown strategy: {strategy_name!r}")

        initial_cash = float(cash) if (cash and cash > 0) else 100_000.0

        # -- Data + backtest ------------------------------------------------
        try:
            from core.data_loader import DataLoader
            loader = DataLoader()
            df = loader.load_data(
                symbol=symbol,
                source="Historical",
                live=False,
                days=365,
            )
            if df is None or df.empty:
                return _err(f"No data available for {symbol}. Load the chart first.")

            mod_name, cls_name = _STRATEGY_CLASS_MAP[strategy_name]
            import importlib
            strategy_cls = getattr(importlib.import_module(mod_name), cls_name)

            from core.backtester import Backtester
            backtester = Backtester()
            backtester.add_data(df.copy())
            backtester.add_strategy(strategy_cls)
            results = backtester.run_backtest(cash=initial_cash)

        except Exception as exc:  # noqa: BLE001
            logger.error("[Dash] run_backtest error: %s", exc)
            return _err(f"Backtest error: {exc}")

        # -- Extract metrics ------------------------------------------------
        sharpe_str, winrate_str, maxdd_str, alpha_str, beta_str, status_msg = (
            _extract_backtest_metrics(results)
        )

        # -- Equity curve figure --------------------------------------------
        equity_fig = _build_equity_curve_figure(results.get("total_asset_value", []))

        # -- Rebuild main chart with signal markers -------------------------
        signals = results.get("signals", [])
        try:
            chart_fig = build_candlestick_figure(df=df, symbol=symbol, show_ma=False)
            add_live_tick_trace(chart_fig)
            if signals:
                overlay_signals(chart_fig, signals)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Dash] signal overlay failed: %s", exc)
            chart_fig = no_update

        logger.info("[Dash] backtest complete: sharpe=%s winrate=%s maxdd=%s", sharpe_str, winrate_str, maxdd_str)

        return (
            sharpe_str,
            winrate_str,
            maxdd_str,
            alpha_str,
            beta_str,
            status_msg,
            _ok_style,
            equity_fig,
            chart_fig,
        )

    # ------------------------------------------------------------------
    # Phase 1.6: order-status → rebuild orders table
    # ------------------------------------------------------------------
    @app.callback(
        Output("orders-table", "data"),
        Output("orders-status", "children"),
        Input("order-status", "children"),
    )
    def update_orders_table(order_status: object):
        """Rebuild the Orders trade-blotter table after every order placement.

        Triggered by changes to ``order-status`` children — the same pattern
        used by ``update_positions`` and ``update_pnl_calendar_display`` so
        the blotter is always up-to-date without a separate polling interval.

        Mirrors ``_refresh_orders_tab()`` in ui/main_window.py: reads
        ``broker.order_history``, formats each row, and emits a summary
        string ("Orders: N total, M filled") to the status label.
        """
        data, status_text = _build_orders_table_data(_broker_or_none())
        return data, status_text

    # ------------------------------------------------------------------
    # Placeholder wiring points for future phases
    # ------------------------------------------------------------------
    # Phase 4: live P&L polling → account-balance / pnl-value
