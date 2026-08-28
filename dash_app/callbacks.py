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
# News & Earnings panel helpers (Phase 1.7)
# ---------------------------------------------------------------------------

def _build_news_content(symbol: Optional[str]) -> list:
    """Build the ``children`` list for the ``news-content`` Div.

    Fetches up to 20 recent news items via ``get_default_news_pipeline()``
    (DuckDuckGo → OpenBB → GDELT).  Returns a list of Dash components:
    one row per news item showing the headline (as a link when a URL is
    available), source name, and publish timestamp.

    Safe to call with ``symbol=None`` — returns a prompt to select a symbol.
    All exceptions are caught and returned as an error message so the
    callback never crashes the app.
    """
    _muted_span = lambda text: [  # noqa: E731
        html.Span(text, style={"color": THEME["text_muted"], "fontSize": "11px"})
    ]

    if not symbol:
        return _muted_span("Select a symbol and click Refresh to load news.")

    try:
        from core.news_pipeline import get_default_news_pipeline
        pipeline = get_default_news_pipeline()
        items = pipeline.fetch_news_items(symbol, limit=20)
    except Exception as exc:  # noqa: BLE001
        logger.error("[Dash] news fetch error for %s: %s", symbol, exc)
        return [
            html.Span(
                f"News fetch error: {exc}",
                style={"color": THEME["red"], "fontSize": "11px"},
            )
        ]

    if not items:
        return _muted_span(f"No news found for {symbol}.")

    rows = []
    for item in items[:20]:
        # Format publish timestamp
        try:
            ts = item.datetime_utc.strftime("%Y-%m-%d %H:%M") if item.datetime_utc else "—"
        except Exception:
            ts = "—"

        # Headline — wrapped in an anchor when a URL is available
        headline_text = item.headline or "(no headline)"
        if item.url:
            headline_el = html.A(
                headline_text,
                href=item.url,
                target="_blank",
                rel="noopener noreferrer",
                style={
                    "color": THEME["accent"],
                    "textDecoration": "none",
                    "fontSize": "11px",
                    "lineHeight": "1.4",
                },
            )
        else:
            headline_el = html.Span(
                headline_text,
                style={"color": THEME["text_main"], "fontSize": "11px", "lineHeight": "1.4"},
            )

        rows.append(
            html.Div(
                style={
                    "borderBottom": f"1px solid {THEME['border_dim']}",
                    "padding": "5px 0",
                },
                children=[
                    headline_el,
                    html.Div(
                        f"{item.source or '—'}  ·  {ts}",
                        style={
                            "color": THEME["text_muted"],
                            "fontSize": "10px",
                            "marginTop": "2px",
                        },
                    ),
                ],
            )
        )

    return rows


def _fmt_eps(val: Optional[float]) -> str:
    """Format an EPS value (float or None) for display."""
    if val is None:
        return "—"
    return f"{val:.4f}"


def _fmt_revenue_millions(val: Optional[float]) -> str:
    """Format a revenue value in dollars to millions for display."""
    if val is None:
        return "—"
    return f"{val / 1_000_000:.1f}"


def _build_earnings_table_data(symbol: Optional[str]) -> tuple:
    """Build ``(data, status_text)`` for the ``earnings-table`` DataTable.

    Fetches earnings via ``DataLoader.get_earnings_calendar(symbol)``.

    Returns:
        data        — list of row dicts for the DataTable (may be empty).
        status_text — short summary string for the ``earnings-status`` Div.

    Handles gracefully:
    * ``symbol=None``     → empty data + prompt message.
    * Crypto symbols      → empty data + "No earnings data — crypto symbol".
    * Empty result list   → empty data + "No upcoming earnings found for {symbol}".
    * Any exception       → empty data + error description.

    Never raises — all exceptions are caught internally so the callback
    that calls this cannot crash the Dash server.
    """
    if not symbol:
        return [], "Select a symbol and click Refresh."

    # Crypto symbols have no earnings — fast path without a DataLoader call
    if "USDT" in symbol.upper():
        return [], "No earnings data — crypto symbol"

    try:
        from core.data_loader import DataLoader
        loader = DataLoader()
        entries = loader.get_earnings_calendar(symbol)
    except Exception as exc:  # noqa: BLE001
        logger.error("[Dash] earnings fetch error for %s: %s", symbol, exc)
        return [], f"Earnings fetch error: {exc}"

    if not entries:
        return [], f"No upcoming earnings found for {symbol}"

    data = [
        {
            "date":             entry.get("date") or "—",
            "eps_estimate":     _fmt_eps(entry.get("eps_estimate")),
            "eps_actual":       _fmt_eps(entry.get("eps_actual")),
            "revenue_estimate": _fmt_revenue_millions(entry.get("revenue_estimate")),
            "revenue_actual":   _fmt_revenue_millions(entry.get("revenue_actual")),
        }
        for entry in entries
    ]
    status_text = f"Earnings: {len(data)} record(s) for {symbol}"
    return data, status_text


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
    # Phase 1.7: news-refresh-btn / active-symbol-store → news + earnings
    # ------------------------------------------------------------------
    @app.callback(
        Output("news-content", "children"),
        Output("earnings-table", "data"),
        Output("earnings-status", "children"),
        Input("news-refresh-btn", "n_clicks"),
        Input("active-symbol-store", "data"),
        State("symbol-dropdown", "value"),
        prevent_initial_call=True,
    )
    def update_news_earnings_panel(
        n_clicks: Optional[int],
        active_symbol: Optional[str],
        symbol_dropdown: Optional[str],
    ):
        """Fetch news and earnings for the currently-selected symbol.

        Triggered by two inputs:
        * Explicit "Refresh" button click (``news-refresh-btn``) — uses the
          current ``symbol-dropdown`` value so news is fetched for whichever
          symbol the user currently sees, even before a chart is loaded.
        * ``active-symbol-store`` change — auto-refreshes whenever the user
          clicks "Load Chart", keeping the panel in sync with the chart.

        Reuses backend singletons where possible:
        * ``get_default_news_pipeline()`` — ``@lru_cache`` singleton, DuckDuckGo
          → OpenBB → GDELT.
        * ``DataLoader().get_earnings_calendar(symbol)`` — yfinance / FMP,
          returns [] for crypto, never raises.

        All exceptions are caught inside the helper functions so this callback
        can never crash the Dash server regardless of network conditions.
        """
        # Prefer the chart-loaded symbol; fall back to whatever the dropdown shows
        symbol = active_symbol or symbol_dropdown

        news_children = _build_news_content(symbol)
        earnings_data, earnings_status = _build_earnings_table_data(symbol)

        return news_children, earnings_data, earnings_status

    # ------------------------------------------------------------------
    # Phase 2: Research Lab — Strategy Lab + Volatility Lab + Signal & Gate
    # ------------------------------------------------------------------

    @app.callback(
        Output("rl-status",               "children"),
        Output("rl-status",               "style"),
        Output("rl-strategy-book-table",  "data"),
        Output("rl-drawdown-chart",       "figure"),
        Output("rl-rolling-sharpe-chart", "figure"),
        Output("rl-pnl-dist-chart",       "figure"),
        Output("rl-monthly-heatmap",      "figure"),
        Output("rl-year-by-year-table",   "data"),
        Output("rl-vol-chart",            "figure"),
        Output("rl-vol-stats",            "children"),
        Output("rl-regime-tape-chart",    "figure"),
        Output("rl-permtest-result",      "children"),
        Output("rl-position-size",        "children"),
        Output("rl-gate-verdict",         "children"),
        Input("rl-run-btn",               "n_clicks"),
        State("active-symbol-store",      "data"),
        State("strategy-dropdown",        "value"),
        State("bt-cash-input",            "value"),
        prevent_initial_call=True,
    )
    def run_research_lab(
        n_clicks: Optional[int],
        symbol: Optional[str],
        strategy_name: Optional[str],
        cash: Optional[float],
    ):
        """Run all Research Lab analytics for the loaded symbol/strategy.

        Orchestrates a four-step pipeline:

        1. **Backtest** — runs the selected (or default MACD/RSI) strategy
           via ``Backtester`` to obtain the full report dict including
           ``profit_per_trade``, ``dates``, ``returns``, ``total_asset_value``.

        2. **Strategy Lab** — calls ``core.research_lab`` functions to build:
           drawdown series, rolling Sharpe, P&L distribution histogram,
           monthly-returns heatmap, year-by-year table, and strategy book.

        3. **Volatility Lab** — calls
           ``core.volatility_lab.compute_volatility_clustering_report`` on
           the symbol's price returns (``Close.pct_change().dropna()``),
           then builds the real-vs-shuffled vol chart, stats block, regime
           tape, permutation-test card, and position-size card.

        4. **Gate** — calls ``core.research_lab.evaluate_gate`` and renders
           the pass/fail verdict with per-check detail rows.

        All 14 outputs are returned from this single callback so that one
        button press populates every Research Lab panel atomically.
        Mirrors the conventions of ``run_backtest_callback`` above: lazy
        imports inside the function, all exceptions caught and surfaced as
        status messages rather than crashes.
        """
        import math
        import plotly.graph_objects as go
        import numpy as np

        _err_style  = {**_ORDER_STATUS_BASE_STYLE, "color": THEME["red"]}
        _ok_style   = {**_ORDER_STATUS_BASE_STYLE, "color": THEME["green"]}
        _info_style = {**_ORDER_STATUS_BASE_STYLE, "color": THEME["text_muted"]}

        _MONTH_LABELS = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]

        # -----------------------------------------------------------------
        # Shared empty-state returns so early exits stay concise
        # -----------------------------------------------------------------
        def _empty_fig(height: int = 180) -> go.Figure:
            fig = go.Figure()
            fig.update_layout(
                paper_bgcolor=THEME["bg_dark"],
                plot_bgcolor=THEME["bg_card"],
                font=dict(color=THEME["text_muted"], size=11),
                margin=dict(l=50, r=10, t=24, b=30),
                height=height,
                xaxis=dict(showgrid=False, color=THEME["text_muted"], zeroline=False),
                yaxis=dict(showgrid=True, gridcolor=THEME["border"],
                           color=THEME["text_muted"], zeroline=False),
                showlegend=False,
            )
            return fig

        _noop14 = (no_update,) * 14

        def _err(msg: str):
            figs = [_empty_fig(h) for h in (180, 180, 160, 180, 200, 120)]
            muted_div = html.Span(
                msg, style={"color": THEME["text_muted"], "fontSize": "11px"}
            )
            return (
                msg, _err_style,
                [],                 # strategy book table
                figs[0],            # drawdown
                figs[1],            # rolling sharpe
                figs[2],            # pnl dist
                figs[3],            # monthly heatmap
                [],                 # year-by-year table
                figs[4],            # vol chart
                muted_div,          # vol stats
                figs[5],            # regime tape
                muted_div,          # permtest
                muted_div,          # position size
                muted_div,          # gate verdict
            )

        if not n_clicks:
            return _noop14

        if not symbol:
            return _err("Load a chart first before running the Research Lab.")

        # Strategy is optional here — fall back to MACD/RSI if unset
        _effective_strategy = (
            strategy_name
            if (strategy_name and strategy_name != "None" and strategy_name in _STRATEGY_CLASS_MAP)
            else "MACD/RSI"
        )
        initial_cash = float(cash) if (cash and cash > 0) else 100_000.0

        # -----------------------------------------------------------------
        # Step 1: Load OHLCV data
        # -----------------------------------------------------------------
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
                return _err(f"No OHLCV data available for {symbol}. Load the chart first.")
        except Exception as exc:  # noqa: BLE001
            logger.error("[Dash-RL] data load error: %s", exc)
            return _err(f"Data load error: {exc}")

        # -----------------------------------------------------------------
        # Step 2: Run backtest
        # -----------------------------------------------------------------
        try:
            import importlib
            mod_name, cls_name = _STRATEGY_CLASS_MAP[_effective_strategy]
            strategy_cls = getattr(importlib.import_module(mod_name), cls_name)

            from core.backtester import Backtester
            backtester = Backtester()
            backtester.add_data(df.copy())
            backtester.add_strategy(strategy_cls)
            report = backtester.run_backtest(cash=initial_cash)
        except Exception as exc:  # noqa: BLE001
            logger.error("[Dash-RL] backtest error: %s", exc)
            return _err(f"Backtest error: {exc}")

        if "error" in report:
            return _err(f"Backtest error: {report['error']}")

        bt_dates   = report.get("dates",            [])
        bt_returns = report.get("returns",           [])
        equity     = report.get("total_asset_value", [])
        ppt        = report.get("profit_per_trade",  [])

        # -----------------------------------------------------------------
        # Step 3a: Research Lab — drawdown series
        # -----------------------------------------------------------------
        from core.research_lab import (
            compute_drawdown_series,
            compute_rolling_sharpe,
            trade_pnl_distribution,
            monthly_returns_table,
            year_by_year_table,
            unit_economics_per_trade,
            build_strategy_book,
            evaluate_gate,
        )

        dd_series = compute_drawdown_series(equity)

        def _build_drawdown_fig() -> go.Figure:
            fig = go.Figure()
            if dd_series:
                x_vals = bt_dates if bt_dates else list(range(len(dd_series)))
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=dd_series,
                    mode="lines",
                    line=dict(color=THEME["red"], width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(248, 81, 73, 0.12)",
                    name="Drawdown %",
                ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor=THEME["bg_dark"],
                plot_bgcolor=THEME["bg_card"],
                font=dict(color=THEME["text_muted"], size=10),
                margin=dict(l=50, r=6, t=6, b=30),
                height=180,
                xaxis=dict(showgrid=False, color=THEME["text_muted"]),
                yaxis=dict(showgrid=True, gridcolor=THEME["border"],
                           color=THEME["text_muted"], ticksuffix="%"),
                showlegend=False,
            )
            return fig

        # -----------------------------------------------------------------
        # Step 3b: Rolling Sharpe
        # -----------------------------------------------------------------
        rs_series = compute_rolling_sharpe(bt_returns, window=63, annualization=252)

        def _build_rolling_sharpe_fig() -> go.Figure:
            fig = go.Figure()
            if rs_series:
                x_vals = bt_dates if bt_dates else list(range(len(rs_series)))
                # Replace nan with None so plotly treats them as gaps
                y_vals = [None if (isinstance(v, float) and math.isnan(v)) else v
                          for v in rs_series]
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines",
                    line=dict(color=THEME["accent"], width=1.5),
                    name="Rolling Sharpe",
                    connectgaps=False,
                ))
                # Zero-reference line
                fig.add_hline(y=0, line_color=THEME["border"], line_dash="dot", line_width=1)
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor=THEME["bg_dark"],
                plot_bgcolor=THEME["bg_card"],
                font=dict(color=THEME["text_muted"], size=10),
                margin=dict(l=50, r=6, t=6, b=30),
                height=180,
                xaxis=dict(showgrid=False, color=THEME["text_muted"]),
                yaxis=dict(showgrid=True, gridcolor=THEME["border"],
                           color=THEME["text_muted"]),
                showlegend=False,
            )
            return fig

        # -----------------------------------------------------------------
        # Step 3c: Trade P&L distribution histogram
        # -----------------------------------------------------------------
        pnl_dist = trade_pnl_distribution(ppt, bins=30)

        def _build_pnl_dist_fig() -> go.Figure:
            fig = go.Figure()
            edges  = pnl_dist.get("bin_edges", [])
            counts = pnl_dist.get("counts",    [])
            if edges and counts:
                centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
                colors  = [THEME["green"] if c > 0 else THEME["red"] for c in centers]
                fig.add_trace(go.Bar(
                    x=centers,
                    y=counts,
                    marker_color=colors,
                    marker_line_width=0,
                    name="Trade P&L",
                ))
                # Add win/loss annotation
                wins   = pnl_dist.get("win_count",  0)
                losses = pnl_dist.get("loss_count", 0)
                fig.add_annotation(
                    text=f"Wins: {wins} | Losses: {losses}",
                    xref="paper", yref="paper",
                    x=0.99, y=0.95,
                    showarrow=False,
                    font=dict(color=THEME["text_muted"], size=10),
                    align="right",
                )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor=THEME["bg_dark"],
                plot_bgcolor=THEME["bg_card"],
                font=dict(color=THEME["text_muted"], size=10),
                margin=dict(l=50, r=6, t=6, b=30),
                height=160,
                xaxis=dict(showgrid=False, color=THEME["text_muted"], title="Net P&L ($)"),
                yaxis=dict(showgrid=True, gridcolor=THEME["border"],
                           color=THEME["text_muted"], title="Count"),
                showlegend=False,
            )
            return fig

        # -----------------------------------------------------------------
        # Step 3d: Monthly returns heatmap
        # -----------------------------------------------------------------
        monthly_tbl = monthly_returns_table(bt_returns, bt_dates)

        def _build_monthly_heatmap_fig() -> go.Figure:
            fig = go.Figure()
            # Collect numeric year keys
            years = sorted(k for k in monthly_tbl if isinstance(k, int))
            if years:
                z_matrix, text_matrix = [], []
                for yr in years:
                    row_z, row_t = [], []
                    for mo in _MONTH_LABELS:
                        val = monthly_tbl[yr].get(mo) if isinstance(monthly_tbl.get(yr), dict) else None
                        row_z.append(val)
                        row_t.append(f"{val:.2f}%" if val is not None else "")
                    z_matrix.append(row_z)
                    text_matrix.append(row_t)

                fig.add_trace(go.Heatmap(
                    z=z_matrix,
                    x=_MONTH_LABELS,
                    y=[str(y) for y in years],
                    text=text_matrix,
                    texttemplate="%{text}",
                    colorscale=[
                        [0.0, THEME["red"]],
                        [0.5, THEME["bg_card"]],
                        [1.0, THEME["green"]],
                    ],
                    zmid=0,
                    showscale=True,
                    colorbar=dict(
                        thickness=10,
                        tickfont=dict(color=THEME["text_muted"], size=9),
                        ticksuffix="%",
                    ),
                    hoverongaps=False,
                ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor=THEME["bg_dark"],
                plot_bgcolor=THEME["bg_card"],
                font=dict(color=THEME["text_muted"], size=10),
                margin=dict(l=50, r=70, t=6, b=30),
                height=max(120, 40 + 30 * len(years)) if years else 180,
                xaxis=dict(showgrid=False, color=THEME["text_muted"]),
                yaxis=dict(showgrid=False, color=THEME["text_muted"]),
            )
            return fig

        # -----------------------------------------------------------------
        # Step 3e: Year-by-year table rows
        # -----------------------------------------------------------------
        yby_rows_raw = year_by_year_table(bt_returns, bt_dates)
        yby_table_data = []
        for row in yby_rows_raw:
            yby_table_data.append({
                "year":            str(row["year"]),
                "return_pct":      f"{row['return_pct']:.2f}" if row["return_pct"] is not None else "—",
                "benchmark_pct":   f"{row['benchmark_pct']:.2f}" if row.get("benchmark_pct") is not None else "—",
                "sharpe":          f"{row['sharpe']:.2f}" if row.get("sharpe") is not None else "—",
                "max_drawdown_pct": f"{row['max_drawdown_pct']:.2f}" if row.get("max_drawdown_pct") is not None else "—",
            })

        # -----------------------------------------------------------------
        # Step 3f: Strategy book (runs all registered strategies — may be slow)
        # -----------------------------------------------------------------
        strategy_book_data = []
        try:
            from core.strategy_manager import StrategyManager
            sm = StrategyManager()
            book = build_strategy_book(sm, df.copy(), cash=initial_cash)
            for entry in book:
                if "error" in entry:
                    strategy_book_data.append({
                        "name": entry["name"],
                        "sharpe": "—", "max_drawdown": "—", "win_rate": "—",
                        "error": entry["error"][:60],
                    })
                else:
                    strategy_book_data.append({
                        "name":         entry["name"],
                        "sharpe":       f"{entry['sharpe']:.2f}",
                        "max_drawdown": f"{entry['max_drawdown']:.2f}",
                        "win_rate":     f"{entry['win_rate']:.1f}",
                        "error":        "",
                    })
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Dash-RL] strategy book error: %s", exc)
            strategy_book_data = [{"name": "—", "sharpe": "—", "max_drawdown": "—",
                                   "win_rate": "—", "error": str(exc)[:80]}]

        # -----------------------------------------------------------------
        # Step 4a: Volatility Lab — price returns from OHLCV Close
        # -----------------------------------------------------------------
        price_returns: list = []
        price_dates: list = []
        try:
            close_col = next((c for c in df.columns if c.lower() == "close"), None)
            if close_col:
                close_s  = df[close_col].dropna()
                ret_s    = close_s.pct_change().dropna()
                price_returns = ret_s.tolist()
                price_dates   = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                                  for d in ret_s.index]
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Dash-RL] price return extraction error: %s", exc)

        from core.volatility_lab import compute_volatility_clustering_report

        vol_report: dict = {}
        if price_returns:
            try:
                vol_report = compute_volatility_clustering_report(
                    price_returns,
                    dates=price_dates,
                    n_permutations=500,
                    seed=42,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[Dash-RL] volatility report error: %s", exc)

        # -----------------------------------------------------------------
        # Step 4b: Real-vs-shuffled rolling vol chart
        # -----------------------------------------------------------------
        def _build_vol_chart() -> go.Figure:
            fig = go.Figure()
            ann_vol = vol_report.get("ann_vol_series", [])
            tape    = vol_report.get("regime_tape", {})
            if ann_vol:
                vdates = vol_report.get("dates") or price_dates
                x_real = vdates if vdates else list(range(len(ann_vol)))
                y_real = [None if v is None else v for v in ann_vol]
                fig.add_trace(go.Scatter(
                    x=x_real,
                    y=y_real,
                    mode="lines",
                    line=dict(color=THEME["accent"], width=1.5),
                    name="Real Vol",
                    connectgaps=False,
                ))

                # Build shuffled vol series from shuffled regime labels for visual comparison
                # Use the shuffled_labels from regime_tape as a proxy indicator overlay
                shuffled_labels = tape.get("shuffled_labels", [])
                if shuffled_labels and len(shuffled_labels) == len(ann_vol):
                    # Map labels to vol-relative multipliers for a visual "shuffled" trace
                    # The shuffled series is the real vol applied to the shuffled label order —
                    # since permutation_test already did the permutation, we approximate the
                    # shuffled baseline by sorting the non-None vol values randomly.
                    non_none = [v for v in ann_vol if v is not None]
                    if non_none:
                        rng = np.random.default_rng(42)
                        shuffled_vols_arr = rng.permutation(non_none)
                        shuffled_y: list = []
                        idx = 0
                        for v in ann_vol:
                            if v is None:
                                shuffled_y.append(None)
                            else:
                                shuffled_y.append(float(shuffled_vols_arr[idx % len(shuffled_vols_arr)]))
                                idx += 1
                        fig.add_trace(go.Scatter(
                            x=x_real,
                            y=shuffled_y,
                            mode="lines",
                            line=dict(color=THEME["text_muted"], width=1, dash="dot"),
                            name="Shuffled Baseline",
                            connectgaps=False,
                        ))

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor=THEME["bg_dark"],
                plot_bgcolor=THEME["bg_card"],
                font=dict(color=THEME["text_muted"], size=10),
                margin=dict(l=50, r=6, t=6, b=30),
                height=200,
                xaxis=dict(showgrid=False, color=THEME["text_muted"]),
                yaxis=dict(showgrid=True, gridcolor=THEME["border"],
                           color=THEME["text_muted"], tickformat=".1%"),
                showlegend=True,
                legend=dict(
                    font=dict(color=THEME["text_muted"], size=9),
                    bgcolor="rgba(0,0,0,0)",
                    x=0.01, y=0.99,
                ),
            )
            return fig

        # -----------------------------------------------------------------
        # Step 4c: Vol stats block (kurtosis, ACF, Ljung-Box, same-sign rate)
        # -----------------------------------------------------------------
        def _build_vol_stats_div():
            if not vol_report:
                return html.Span(
                    "Volatility stats unavailable — insufficient data.",
                    style={"color": THEME["text_muted"], "fontSize": "11px"},
                )

            ek    = vol_report.get("excess_kurtosis")
            acf_d = vol_report.get("acf_abs", {})
            lb    = vol_report.get("ljung_box", {})
            ssr   = vol_report.get("same_sign_rate")

            def _fmt(v, decimals=4):
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    return "N/A"
                return f"{v:.{decimals}f}"

            lb_stat    = _fmt(lb.get("statistic"))
            lb_pval    = _fmt(lb.get("p_value"), 3)
            lb_sig     = ""
            try:
                if lb.get("p_value") is not None and not math.isnan(lb.get("p_value", float("nan"))):
                    lb_sig = " — clustering confirmed" if lb["p_value"] < 0.05 else " — no clustering"
            except Exception:
                pass

            ssr_interp = ""
            if ssr is not None and not (isinstance(ssr, float) and math.isnan(ssr)):
                ssr_interp = "  (momentum)" if ssr > 0.5 else "  (mean-reversion)"

            def _stat_row(label: str, value: str) -> html.Div:
                return html.Div(
                    style={"display": "flex", "gap": "8px", "marginBottom": "3px"},
                    children=[
                        html.Span(label, style={"color": THEME["text_muted"], "minWidth": "180px"}),
                        html.Span(value, style={"color": THEME["text_main"], "fontWeight": "600"}),
                    ],
                )

            return html.Div(
                style={"fontSize": "11px"},
                children=[
                    _stat_row("Excess Kurtosis:",          _fmt(ek)),
                    _stat_row("ACF |ret| lag-1:",          _fmt(acf_d.get(1))),
                    _stat_row("ACF |ret| lag-5:",          _fmt(acf_d.get(5))),
                    _stat_row("ACF |ret| lag-22:",         _fmt(acf_d.get(22))),
                    _stat_row("ACF |ret| lag-66:",         _fmt(acf_d.get(66))),
                    _stat_row("Ljung-Box stat (lag 22):",  lb_stat),
                    _stat_row("Ljung-Box p-value:",        lb_pval + lb_sig),
                    _stat_row("Same-sign rate:",           _fmt(ssr) + ssr_interp),
                ],
            )

        # -----------------------------------------------------------------
        # Step 4d: Regime tape chart (calm / normal / turbulent)
        # -----------------------------------------------------------------
        def _build_regime_tape_fig() -> go.Figure:
            fig = go.Figure()
            tape    = vol_report.get("regime_tape", {})
            labels  = tape.get("labels", [])
            if labels:
                vdates = vol_report.get("dates") or price_dates
                x_vals = vdates if vdates else list(range(len(labels)))
                # Numeric mapping for scatter color
                _label_map = {"calm": 0, "normal": 1, "turbulent": 2}
                _color_map = {"calm": THEME["green"], "normal": THEME["accent"],
                              "turbulent": THEME["red"]}
                for regime, color in _color_map.items():
                    idxs = [i for i, lbl in enumerate(labels) if lbl == regime]
                    if idxs:
                        fig.add_trace(go.Scatter(
                            x=[x_vals[i] for i in idxs],
                            y=[1] * len(idxs),
                            mode="markers",
                            marker=dict(color=color, size=6, symbol="square"),
                            name=regime.capitalize(),
                        ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor=THEME["bg_dark"],
                plot_bgcolor=THEME["bg_card"],
                font=dict(color=THEME["text_muted"], size=10),
                margin=dict(l=10, r=6, t=6, b=30),
                height=120,
                xaxis=dict(showgrid=False, color=THEME["text_muted"]),
                yaxis=dict(visible=False),
                showlegend=True,
                legend=dict(
                    font=dict(color=THEME["text_muted"], size=9),
                    bgcolor="rgba(0,0,0,0)",
                    orientation="h", x=0.0, y=1.1,
                ),
            )
            return fig

        # -----------------------------------------------------------------
        # Step 4e: Permutation test card
        # -----------------------------------------------------------------
        def _build_permtest_div():
            pt = vol_report.get("permutation", {})
            if not pt:
                return html.Span(
                    "Permutation test unavailable.",
                    style={"color": THEME["text_muted"], "fontSize": "11px"},
                )

            def _fmt(v, d=3):
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    return "N/A"
                return f"{v:.{d}f}"

            lift = pt.get("lift_pts")
            lift_str = _fmt(lift, 1) if lift is not None else "N/A"
            pval = pt.get("p_value")
            sig_text = ""
            if pval is not None and not (isinstance(pval, float) and math.isnan(pval)):
                sig_text = "significant" if pval < 0.05 else "not significant"

            return html.Div(
                style={"fontSize": "11px"},
                children=[
                    html.P(
                        "Permutation Test (ACF lag-1 |ret|)",
                        style={"color": THEME["text_muted"], "fontWeight": "600",
                               "marginBottom": "6px", "fontSize": "11px"},
                    ),
                    html.Div([
                        html.Span("Observed ACF: ", style={"color": THEME["text_muted"]}),
                        html.Span(_fmt(pt.get("observed")), style={"color": THEME["text_main"], "fontWeight": "600"}),
                    ], style={"marginBottom": "3px"}),
                    html.Div([
                        html.Span("Shuffled mean: ", style={"color": THEME["text_muted"]}),
                        html.Span(_fmt(pt.get("shuffled_mean")), style={"color": THEME["text_main"]}),
                    ], style={"marginBottom": "3px"}),
                    html.Div([
                        html.Span("Lift: ", style={"color": THEME["text_muted"]}),
                        html.Span(
                            f"+{lift_str} pts",
                            style={"color": THEME["green"], "fontWeight": "600"},
                        ),
                    ], style={"marginBottom": "3px"}),
                    html.Div([
                        html.Span("p-value: ", style={"color": THEME["text_muted"]}),
                        html.Span(
                            f"{_fmt(pval)} ({sig_text})",
                            style={"color": THEME["accent"]},
                        ),
                    ]),
                ],
            )

        # -----------------------------------------------------------------
        # Step 4f: Position size suggestion card
        # -----------------------------------------------------------------
        def _build_position_size_div():
            if not price_returns or not vol_report:
                return html.Span(
                    "Position sizing unavailable.",
                    style={"color": THEME["text_muted"], "fontSize": "11px"},
                )
            try:
                from core.volatility_lab import suggest_position_size
                sizing = suggest_position_size(
                    price_returns,
                    capital=initial_cash,
                    risk_budget_pct=0.02,
                    confidence=0.99,
                )
            except Exception as exc:  # noqa: BLE001
                return html.Span(
                    f"Position sizing error: {exc}",
                    style={"color": THEME["red"], "fontSize": "11px"},
                )

            def _fmt(v, d=4):
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    return "N/A"
                return f"{v:.{d}f}"

            fraction = sizing.get("suggested_fraction", 0)
            notional = sizing.get("suggested_notional", 0)

            return html.Div(
                style={"fontSize": "11px"},
                children=[
                    html.P(
                        "Tail-Risk Position Sizing (99% VaR, 2% budget)",
                        style={"color": THEME["text_muted"], "fontWeight": "600",
                               "marginBottom": "6px", "fontSize": "11px"},
                    ),
                    html.Div([
                        html.Span("VaR 99%: ", style={"color": THEME["text_muted"]}),
                        html.Span(_fmt(sizing.get("var_99"), 4),
                                  style={"color": THEME["red"], "fontWeight": "600"}),
                    ], style={"marginBottom": "3px"}),
                    html.Div([
                        html.Span("CVaR 99%: ", style={"color": THEME["text_muted"]}),
                        html.Span(_fmt(sizing.get("cvar_99"), 4),
                                  style={"color": THEME["red"]}),
                    ], style={"marginBottom": "3px"}),
                    html.Div([
                        html.Span("Suggested fraction: ", style={"color": THEME["text_muted"]}),
                        html.Span(
                            f"{fraction * 100:.1f}%",
                            style={"color": THEME["accent"], "fontWeight": "600"},
                        ),
                    ], style={"marginBottom": "3px"}),
                    html.Div([
                        html.Span("Suggested notional: ", style={"color": THEME["text_muted"]}),
                        html.Span(
                            f"${notional:,.0f}",
                            style={"color": THEME["green"], "fontWeight": "600"},
                        ),
                    ]),
                ],
            )

        # -----------------------------------------------------------------
        # Step 5: Gate verdict
        # -----------------------------------------------------------------
        def _build_gate_verdict_div():
            try:
                gate = evaluate_gate(report)
            except Exception as exc:  # noqa: BLE001
                return html.Span(
                    f"Gate evaluation error: {exc}",
                    style={"color": THEME["red"], "fontSize": "11px"},
                )

            passed  = gate.get("passed", False)
            verdict = gate.get("verdict_text", "")
            checks  = gate.get("checks", [])

            header_color = THEME["green"] if passed else THEME["red"]
            header_text  = "PASSED" if passed else "FAILED"

            check_rows = []
            for chk in checks:
                chk_pass  = chk.get("passed", False)
                chk_color = THEME["green"] if chk_pass else THEME["red"]
                chk_icon  = "✓" if chk_pass else "✗"
                check_rows.append(
                    html.Div(
                        style={
                            "display": "flex",
                            "gap": "8px",
                            "marginBottom": "4px",
                            "fontSize": "11px",
                        },
                        children=[
                            html.Span(chk_icon, style={"color": chk_color, "fontWeight": "bold",
                                                        "minWidth": "14px"}),
                            html.Span(
                                chk["name"],
                                style={"color": THEME["text_main"], "minWidth": "180px"},
                            ),
                            html.Span(
                                chk.get("detail", ""),
                                style={"color": THEME["text_muted"]},
                            ),
                        ],
                    )
                )

            return html.Div(
                children=[
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "10px",
                               "marginBottom": "8px"},
                        children=[
                            html.Span(
                                header_text,
                                style={
                                    "color": header_color,
                                    "fontWeight": "bold",
                                    "fontSize": "14px",
                                    "border": f"1px solid {header_color}",
                                    "borderRadius": "4px",
                                    "padding": "2px 8px",
                                },
                            ),
                            html.Span(
                                f"{_effective_strategy}  ·  {symbol}",
                                style={"color": THEME["text_muted"], "fontSize": "11px"},
                            ),
                        ],
                    ),
                    html.Div(check_rows, style={"marginBottom": "8px"}),
                    html.P(
                        verdict,
                        style={"color": THEME["text_muted"], "fontSize": "11px",
                               "lineHeight": "1.5", "marginBottom": "0"},
                    ),
                ]
            )

        # -----------------------------------------------------------------
        # Assemble and return all 14 outputs
        # -----------------------------------------------------------------
        status_msg = (
            f"Analysis complete — {_effective_strategy} on {symbol}"
            f" | Sharpe: {report.get('sharpe', 0):.2f}"
            f" | Win Rate: {report.get('win_rate', 0):.1f}%"
        )

        return (
            status_msg,
            _ok_style,
            strategy_book_data,
            _build_drawdown_fig(),
            _build_rolling_sharpe_fig(),
            _build_pnl_dist_fig(),
            _build_monthly_heatmap_fig(),
            yby_table_data,
            _build_vol_chart(),
            _build_vol_stats_div(),
            _build_regime_tape_fig(),
            _build_permtest_div(),
            _build_position_size_div(),
            _build_gate_verdict_div(),
        )

    # ------------------------------------------------------------------
    # Placeholder wiring points for future phases
    # ------------------------------------------------------------------
    # Phase 4: live P&L polling → account-balance / pnl-value
