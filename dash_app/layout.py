"""
dash_app/layout.py

Defines the dark-themed base layout for the AlgoTrader Dash web app.

Color palette and overall visual language matches the PyQt5 desktop app
(ui/main_window.py) — same hex values are used so both frontends feel
like the same product.

The chart area uses core.chart_builder.build_candlestick_figure(), the
same shared function used by the PyQt5 app, so figure-building logic
lives in exactly one place.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from core.chart_builder import THEME, build_candlestick_figure

# ---------------------------------------------------------------------------
# Inline CSS that matches the PyQt5 stylesheet hex palette
# ---------------------------------------------------------------------------
_PAGE_STYLE = {
    "backgroundColor": THEME["bg_dark"],
    "color": THEME["text_main"],
    "minHeight": "100vh",
    "fontFamily": "'SF Mono', 'Consolas', 'Menlo', monospace",
    "fontSize": "12px",
}

_TOPBAR_STYLE = {
    "backgroundColor": THEME["bg_card"],
    "borderBottom": f"1px solid {THEME['border']}",
    "padding": "0 12px",
    "height": "44px",
    "display": "flex",
    "alignItems": "center",
    "gap": "12px",
}

_BRAND_STYLE = {
    "color": THEME["accent"],
    "fontWeight": "bold",
    "fontSize": "13px",
    "whiteSpace": "nowrap",
}

_LABEL_MUTED = {
    "color": THEME["text_muted"],
    "fontSize": "11px",
    "marginBottom": "0",
}

_PANEL_STYLE = {
    "backgroundColor": THEME["bg_card"],
    "border": f"1px solid {THEME['border']}",
    "borderRadius": "6px",
    "padding": "10px",
}

_METRIC_VALUE_STYLE = {
    "fontSize": "15px",
    "fontWeight": "bold",
    "color": THEME["green"],
}

_DROPDOWN_STYLE = {
    "backgroundColor": THEME["bg_card"],
    "color": THEME["text_main"],
    "border": f"1px solid {THEME['border']}",
    "borderRadius": "4px",
    "fontSize": "12px",
    "minWidth": "110px",
}

# ---------------------------------------------------------------------------
# Component helpers
# ---------------------------------------------------------------------------

_SYMBOLS    = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AAPL", "TSLA", "SPY", "QQQ"]
_INTERVALS  = ["1d", "1h", "15m", "5m", "1m"]
_STRATEGIES = ["None", "MACD/RSI", "EMA Crossover", "Stochastic"]


def _muted(text: str) -> html.Span:
    return html.Span(text, style=_LABEL_MUTED)


def _topbar() -> html.Div:
    return html.Div(
        style=_TOPBAR_STYLE,
        children=[
            html.Span("◈ AlgoTrader", style=_BRAND_STYLE),
            html.Span("|", style={"color": THEME["border"], "fontSize": "16px"}),

            _muted("Symbol"),
            dcc.Dropdown(
                id="symbol-dropdown",
                options=[{"label": s, "value": s} for s in _SYMBOLS],
                value="AAPL",
                clearable=False,
                style=_DROPDOWN_STYLE,
            ),

            _muted("Interval"),
            dcc.Dropdown(
                id="interval-dropdown",
                options=[{"label": i, "value": i} for i in _INTERVALS],
                value="1d",
                clearable=False,
                style={**_DROPDOWN_STYLE, "minWidth": "70px"},
            ),

            _muted("Strategy"),
            dcc.Dropdown(
                id="strategy-dropdown",
                options=[{"label": s, "value": s} for s in _STRATEGIES],
                value="None",
                clearable=False,
                style={**_DROPDOWN_STYLE, "minWidth": "130px"},
            ),

            html.Div(style={"flex": "1"}),  # spacer

            html.Button(
                "Load Chart",
                id="load-btn",
                n_clicks=0,
                style={
                    "backgroundColor": THEME["bg_dark"],
                    "color": THEME["accent"],
                    "border": f"1px solid {THEME['accent']}",
                    "borderRadius": "4px",
                    "padding": "4px 12px",
                    "cursor": "pointer",
                    "fontSize": "12px",
                },
            ),
        ],
    )


def _chart_panel() -> dbc.Col:
    """Main candlestick chart area with a placeholder empty figure.

    Includes a "live badge" div positioned above the chart that shows
    '🟢 Live' for crypto WebSocket streams or '🟡 Near real-time' for
    equity REST polling.  Both texts are set by the interval callback in
    callbacks.py; the div starts empty so it occupies no visual space
    before a chart is loaded.
    """
    from core.chart_builder import add_live_tick_trace
    placeholder = build_candlestick_figure(df=None, symbol="", height=600)
    add_live_tick_trace(placeholder)  # keep trace index consistent from the start
    return dbc.Col(
        [
            # Live-price badge row — right-aligned, updated by interval callback
            html.Div(
                id="live-badge",
                style={
                    "textAlign": "right",
                    "fontSize": "11px",
                    "paddingRight": "6px",
                    "paddingBottom": "2px",
                    "minHeight": "16px",
                    "fontFamily": "'SF Mono', 'Consolas', 'Menlo', monospace",
                    "color": THEME["text_muted"],
                },
            ),
            dcc.Graph(
                id="main-chart",
                figure=placeholder,
                config={
                    "displayModeBar": True,
                    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                    "displaylogo": False,
                },
                style={"height": "600px"},
            ),
        ],
        width=9,
    )


def _order_entry_panel() -> html.Div:
    """Order entry card: qty, order-type, optional price, buy/sell buttons, status feedback.

    Mirrors the PyQt5 order panel in ui/main_window.py (on_order_type_changed +
    place_order) — same UX flow, adapted for Dash.  Wired to
    SimulatedBroker.submit_order() via callbacks.py.
    """
    _input_style = {
        "backgroundColor": THEME["bg_dark"],
        "color": THEME["text_main"],
        "border": f"1px solid {THEME['border']}",
        "borderRadius": "4px",
        "fontSize": "12px",
        "width": "100%",
        "padding": "4px 6px",
        "outline": "none",
        "boxSizing": "border-box",
    }
    return html.Div(
        style={**_PANEL_STYLE, "marginBottom": "10px"},
        children=[
            html.P("Order Entry", style={**_LABEL_MUTED, "fontWeight": "600", "marginBottom": "6px"}),
            # --- Quantity -------------------------------------------------------
            html.P("Qty", style={**_LABEL_MUTED, "marginBottom": "2px"}),
            dcc.Input(
                id="order-qty-input",
                type="number",
                placeholder="Quantity",
                min=0.000001,
                step="any",
                debounce=False,
                style=_input_style,
            ),
            # --- Order Type ----------------------------------------------------
            html.P("Order Type", style={**_LABEL_MUTED, "marginTop": "6px", "marginBottom": "2px"}),
            dcc.Dropdown(
                id="order-type-dropdown",
                options=[
                    {"label": "Market", "value": "market"},
                    {"label": "Limit",  "value": "limit"},
                    {"label": "Stop",   "value": "stop"},
                ],
                value="market",
                clearable=False,
                style=_DROPDOWN_STYLE,
            ),
            # --- Price input (hidden for Market, shown for Limit/Stop) ---------
            html.Div(
                id="order-price-wrapper",
                style={"display": "none"},  # toggled by toggle_price_input callback
                children=[
                    html.P("Price", style={**_LABEL_MUTED, "marginTop": "6px", "marginBottom": "2px"}),
                    dcc.Input(
                        id="order-price-input",
                        type="number",
                        placeholder="Limit Price",
                        min=0,
                        step="any",
                        debounce=False,
                        style=_input_style,
                    ),
                ],
            ),
            # --- Buy / Sell buttons --------------------------------------------
            html.Div(
                style={"display": "flex", "gap": "4%", "marginTop": "8px"},
                children=[
                    html.Button(
                        "Buy",
                        id="buy-btn",
                        n_clicks=0,
                        style={
                            "backgroundColor": THEME["green"],
                            "color": "#0d1117",
                            "border": "none",
                            "borderRadius": "4px",
                            "padding": "6px 0",
                            "cursor": "pointer",
                            "fontSize": "12px",
                            "fontWeight": "bold",
                            "width": "48%",
                        },
                    ),
                    html.Button(
                        "Sell",
                        id="sell-btn",
                        n_clicks=0,
                        style={
                            "backgroundColor": THEME["red"],
                            "color": "#ffffff",
                            "border": "none",
                            "borderRadius": "4px",
                            "padding": "6px 0",
                            "cursor": "pointer",
                            "fontSize": "12px",
                            "fontWeight": "bold",
                            "width": "48%",
                        },
                    ),
                ],
            ),
            # --- Status / feedback text ----------------------------------------
            html.Div(
                id="order-status",
                style={
                    "color": THEME["text_muted"],
                    "fontSize": "11px",
                    "marginTop": "6px",
                    "minHeight": "16px",
                    "wordBreak": "break-word",
                },
            ),
        ],
    )


def _metrics_panel() -> dbc.Col:
    """Right-side metrics / account panel."""
    return dbc.Col(
        width=3,
        children=[
            # Account card
            html.Div(
                style={**_PANEL_STYLE, "marginBottom": "10px"},
                children=[
                    html.P("Account", style={**_LABEL_MUTED, "fontWeight": "600", "marginBottom": "4px"}),
                    html.P("Simulator", style={"margin": "0", "color": THEME["text_muted"], "fontSize": "11px"}),
                    html.P("$100,000.00", id="account-balance", style=_METRIC_VALUE_STYLE),
                ],
            ),
            # Order Entry card
            _order_entry_panel(),
            # P&L card
            html.Div(
                style={**_PANEL_STYLE, "marginBottom": "10px"},
                children=[
                    html.P("P & L", style={**_LABEL_MUTED, "fontWeight": "600", "marginBottom": "4px"}),
                    html.P("$0.00", id="pnl-value", style={
                        **_METRIC_VALUE_STYLE,
                        "fontSize": "20px",
                        "textAlign": "center",
                    }),
                ],
            ),
            # Backtest results card
            html.Div(
                style={**_PANEL_STYLE, "marginBottom": "10px"},
                children=[
                    html.P("Backtest Results", style={**_LABEL_MUTED, "fontWeight": "600", "marginBottom": "6px"}),
                    html.Div([
                        html.Span("Sharpe: ", style=_LABEL_MUTED),
                        html.Span("—", id="bt-sharpe", style={"color": THEME["accent"]}),
                    ], style={"marginBottom": "4px"}),
                    html.Div([
                        html.Span("Win Rate: ", style=_LABEL_MUTED),
                        html.Span("—", id="bt-winrate", style={"color": THEME["green"]}),
                    ], style={"marginBottom": "4px"}),
                    html.Div([
                        html.Span("Max DD: ", style=_LABEL_MUTED),
                        html.Span("—", id="bt-maxdd", style={"color": THEME["red"]}),
                    ]),
                ],
            ),
            # Chart status info
            html.Div(
                id="chart-status",
                style={**_PANEL_STYLE, "color": THEME["text_muted"], "fontSize": "11px"},
                children="Select a symbol and click Load Chart.",
            ),
        ],
    )


def _status_bar() -> html.Div:
    return html.Div(
        id="status-bar",
        children="Ready",
        style={
            "backgroundColor": THEME["bg_card"],
            "borderTop": f"1px solid {THEME['border']}",
            "color": THEME["text_muted"],
            "fontSize": "11px",
            "padding": "4px 12px",
        },
    )


# ---------------------------------------------------------------------------
# Root layout
# ---------------------------------------------------------------------------

def build_layout() -> html.Div:
    """Assemble and return the complete page layout.

    Called once from ``dash_app/app.py`` to set ``app.layout``.
    """
    return html.Div(
        style=_PAGE_STYLE,
        children=[
            # Hidden store for passing data between callbacks
            dcc.Store(id="ohlcv-store"),
            dcc.Store(id="signals-store", data=[]),

            # Tracks which symbol is currently displayed — used by the
            # interval callback for subscription management.
            dcc.Store(id="active-symbol-store", data=None),

            # Live-price polling interval.  Starts disabled; the load_chart
            # callback enables it once a chart has been successfully loaded.
            # 1 500 ms gives a smooth feel without hammering REST endpoints.
            dcc.Interval(
                id="price-interval",
                interval=1500,   # milliseconds
                n_intervals=0,
                disabled=True,
            ),

            # Top navigation bar
            _topbar(),

            # Main content
            dbc.Container(
                fluid=True,
                style={"padding": "10px 8px"},
                children=[
                    dbc.Row(
                        style={"gap": "0"},
                        children=[
                            _chart_panel(),
                            _metrics_panel(),
                        ],
                    )
                ],
            ),

            # Bottom status bar
            _status_bar(),
        ],
    )
