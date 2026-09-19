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

import datetime

import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table

from core import ui_options
from core.chart_builder import THEME, build_candlestick_figure
from core.strategy_manager import backtrader_strategies

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

# Month-navigation button style for the PnL Calendar panel
_NAV_BTN_STYLE = {
    "backgroundColor": THEME["bg_dark"],
    "color": THEME["text_muted"],
    "border": f"1px solid {THEME['border']}",
    "borderRadius": "4px",
    "padding": "2px 8px",
    "cursor": "pointer",
    "fontSize": "12px",
    "minWidth": "28px",
}

# Tab label styles (header pill, not content area)
_TAB_STYLE = {
    "backgroundColor": THEME["bg_dark"],
    "color": THEME["text_muted"],
    "borderColor": THEME["border"],
    "fontFamily": "'SF Mono', 'Consolas', 'Menlo', monospace",
    "fontSize": "12px",
    "padding": "6px 14px",
}

_TAB_SELECTED_STYLE = {
    **_TAB_STYLE,
    "backgroundColor": THEME["bg_card"],
    "color": THEME["text_main"],
    "borderTop": f"2px solid {THEME['accent']}",
    "borderColor": THEME["border"],
}

# ---------------------------------------------------------------------------
# Component helpers
# ---------------------------------------------------------------------------

_SYMBOLS    = ui_options.SYMBOLS
_INTERVALS  = ui_options.INTERVALS
_STRATEGIES = ["None"] + list(backtrader_strategies())   # same registry the desktop app uses


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
                value=ui_options.DEFAULT_SYMBOL_DASH,
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

            _muted("Days"),
            dcc.Input(
                id="days-input",
                type="number",
                value=ui_options.DEFAULT_DAYS,
                min=1,
                max=10000,
                step=1,
                debounce=True,
                style={**_DROPDOWN_STYLE, "width": "62px", "padding": "4px 6px", "border": f"1px solid {THEME['border']}",
                       "borderRadius": "4px", "backgroundColor": THEME["bg_dark"], "color": THEME["text_main"]},
            ),

            _muted("Strategy"),
            dcc.Dropdown(
                id="strategy-dropdown",
                options=[{"label": s, "value": s} for s in _STRATEGIES],
                value="None",
                clearable=False,
                style={**_DROPDOWN_STYLE, "minWidth": "130px"},
            ),
            html.Div(
                title=ui_options.TREND_OVERLAY_TIP,
                children=dcc.Checklist(
                    id="trend-overlay-check",
                    options=[{"label": " " + ui_options.TREND_OVERLAY_LABEL, "value": "on"}],
                    value=[],
                    inline=True,
                    style={"color": THEME["text_muted"], "fontSize": "12px", "whiteSpace": "nowrap"},
                ),
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
                    html.Div(
                        id="pnl-breakdown",
                        children="Realized $0.00  ·  Unrealized $0.00",
                        style={
                            "textAlign": "center",
                            "fontSize": "10px",
                            "color": THEME["text_muted"],
                            "marginTop": "2px",
                        },
                    ),
                ],
            ),
            # Backtest results card (Phase 1.5)
            html.Div(
                style={**_PANEL_STYLE, "marginBottom": "10px"},
                children=[
                    html.P("Backtest Results", style={**_LABEL_MUTED, "fontWeight": "600", "marginBottom": "6px"}),
                    # -- Initial cash + run button row --------------------------
                    html.Div(
                        style={"display": "flex", "gap": "4px", "marginBottom": "6px", "alignItems": "center"},
                        children=[
                            dcc.Input(
                                id="bt-cash-input",
                                type="number",
                                placeholder="Initial Cash",
                                value=ui_options.DEFAULT_CASH,
                                min=1,
                                step=1000,
                                debounce=False,
                                style={
                                    "backgroundColor": THEME["bg_dark"],
                                    "color": THEME["text_main"],
                                    "border": f"1px solid {THEME['border']}",
                                    "borderRadius": "4px",
                                    "fontSize": "11px",
                                    "flex": "1",
                                    "minWidth": "0",
                                    "padding": "4px 5px",
                                    "outline": "none",
                                    "boxSizing": "border-box",
                                },
                            ),
                            html.Button(
                                "Run Backtest",
                                id="bt-run-btn",
                                n_clicks=0,
                                style={
                                    "backgroundColor": THEME["accent"],
                                    "color": "#0d1117",
                                    "border": "none",
                                    "borderRadius": "4px",
                                    "padding": "4px 8px",
                                    "cursor": "pointer",
                                    "fontSize": "11px",
                                    "fontWeight": "bold",
                                    "whiteSpace": "nowrap",
                                },
                            ),
                    # -- Fees (same fields and defaults as the desktop app's Parameters box) --
                    html.Div(
                        style={"display": "flex", "gap": "4px", "marginBottom": "6px", "alignItems": "center"},
                        children=[
                            _muted("Mkt fee %"),
                            dcc.Input(id="bt-mkt-fee-input", type="number", value=ui_options.DEFAULT_MARKET_FEE_PCT,
                                      min=0, max=10, step=0.01, debounce=True,
                                      style={"backgroundColor": THEME["bg_dark"], "color": THEME["text_main"], "border": f"1px solid {THEME['border']}",
                                       "borderRadius": "4px", "fontSize": "11px", "flex": "1", "minWidth": "0",
                                       "padding": "4px 5px", "outline": "none", "boxSizing": "border-box"}),
                            _muted("Lim fee %"),
                            dcc.Input(id="bt-lim-fee-input", type="number", value=ui_options.DEFAULT_LIMIT_FEE_PCT,
                                      min=0, max=10, step=0.01, debounce=True,
                                      style={"backgroundColor": THEME["bg_dark"], "color": THEME["text_main"], "border": f"1px solid {THEME['border']}",
                                       "borderRadius": "4px", "fontSize": "11px", "flex": "1", "minWidth": "0",
                                       "padding": "4px 5px", "outline": "none", "boxSizing": "border-box"}),
                        ],
                    ),
                        ],
                    ),
                    # -- Metric rows (populated by run_backtest_callback) --------
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
                    ], style={"marginBottom": "4px"}),
                    html.Div([
                        html.Span("Alpha: ", style=_LABEL_MUTED),
                        html.Span("—", id="bt-alpha", style={"color": THEME["text_muted"]}),
                    ], style={"marginBottom": "4px"}),
                    html.Div([
                        html.Span("Beta: ", style=_LABEL_MUTED),
                        html.Span("—", id="bt-beta", style={"color": THEME["text_muted"]}),
                    ], style={"marginBottom": "4px"}),
                    # -- Status / feedback text ---------------------------------
                    html.Div(
                        id="bt-status",
                        style={
                            "color": THEME["text_muted"],
                            "fontSize": "10px",
                            "marginTop": "4px",
                            "minHeight": "14px",
                            "wordBreak": "break-word",
                        },
                    ),
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


def _empty_rl_figure(height: int = 220) -> "go.Figure":
    """Return a dark-themed empty placeholder figure for Research Lab charts.

    Identical visual language to ``_empty_equity_curve_figure`` but
    parameterised on *height* so the various Research Lab panels can choose
    an appropriate vertical size without duplicating boilerplate.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=THEME["bg_dark"],
        plot_bgcolor=THEME["bg_card"],
        font=dict(color=THEME["text_muted"], size=11),
        margin=dict(l=50, r=10, t=24, b=30),
        height=height,
        xaxis=dict(showgrid=False, color=THEME["text_muted"], zeroline=False),
        yaxis=dict(
            showgrid=True,
            gridcolor=THEME["border"],
            color=THEME["text_muted"],
            zeroline=False,
        ),
        showlegend=False,
    )
    return fig


def _rl_datatable(table_id: str, columns: list) -> "dash_table.DataTable":
    """Return a dark-themed DataTable configured for Research Lab sections.

    Reuses the same style tokens as the orders-table in ``_bottom_tabs_panel``
    so all blotter-style tables have a consistent look.
    """
    return dash_table.DataTable(
        id=table_id,
        columns=columns,
        data=[],
        page_action="none",
        sort_action="native",
        sort_mode="single",
        style_table={
            "overflowX": "auto",
            "overflowY": "auto",
            "maxHeight": "200px",
            "backgroundColor": THEME["bg_dark"],
            "border": f"1px solid {THEME['border']}",
            "borderRadius": "4px",
        },
        style_cell={
            "backgroundColor": THEME["bg_dark"],
            "color": THEME["text_main"],
            "border": f"1px solid {THEME['border']}",
            "fontSize": "11px",
            "fontFamily": "'SF Mono', 'Consolas', 'Menlo', monospace",
            "textAlign": "center",
            "padding": "4px 8px",
            "minWidth": "60px",
        },
        style_header={
            "backgroundColor": THEME["bg_card"],
            "color": THEME["text_muted"],
            "fontWeight": "600",
            "fontSize": "10px",
            "border": f"1px solid {THEME['border']}",
            "textTransform": "uppercase",
            "letterSpacing": "0.5px",
            "padding": "4px 8px",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#0f1419"},
        ],
    )


def _research_lab_tab_content() -> html.Div:
    """Build the full content area for the Research Lab bottom tab.

    Three nested sub-tabs mirror the PyQt5 Research Lab panel:

    * **Strategy Lab** — strategy-book ranking table, drawdown series,
      rolling Sharpe, per-trade P&L histogram, monthly-returns heatmap,
      year-by-year performance table.
    * **Volatility Lab** — real-vs-shuffled rolling-vol chart, kurtosis /
      ACF / Ljung-Box stats block, regime tape chart, permutation-test
      result, suggested tail-risk position size.
    * **Signal & Gate** — pass/fail gate verdict with per-check detail rows.

    All chart slots start as empty placeholder figures (``_empty_rl_figure``).
    The "Run Analysis" button above the sub-tabs triggers the
    ``run_research_lab`` callback in callbacks.py, which populates every slot.
    """
    _sub_tab_style = {**_TAB_STYLE, "fontSize": "11px", "padding": "5px 12px"}
    _sub_tab_selected_style = {**_TAB_SELECTED_STYLE, "fontSize": "11px", "padding": "5px 12px"}

    # -----------------------------------------------------------------
    # Sub-tab 1: Strategy Lab
    # -----------------------------------------------------------------
    strategy_lab = dcc.Tab(
        label="Strategy Lab",
        value="rl-strategy-lab-subtab",
        style=_sub_tab_style,
        selected_style=_sub_tab_selected_style,
        children=[
            html.Div(
                style={"padding": "8px 0"},
                children=[
                    # Strategy book ranking table
                    html.P(
                        "Strategy Book (ranked by Sharpe)",
                        style={**_LABEL_MUTED, "fontWeight": "600", "marginBottom": "4px"},
                    ),
                    _rl_datatable(
                        "rl-strategy-book-table",
                        columns=[
                            {"name": "Strategy",     "id": "name"},
                            {"name": "Sharpe",       "id": "sharpe"},
                            {"name": "Max DD (%)",   "id": "max_drawdown"},
                            {"name": "Win Rate (%)", "id": "win_rate"},
                            {"name": "Error",        "id": "error"},
                        ],
                    ),
                    # Drawdown + Rolling Sharpe side by side
                    html.Div(
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "1fr 1fr",
                            "gap": "8px",
                            "marginTop": "8px",
                        },
                        children=[
                            html.Div([
                                html.P("Drawdown Series", style={**_LABEL_MUTED, "marginBottom": "2px"}),
                                dcc.Graph(
                                    id="rl-drawdown-chart",
                                    figure=_empty_rl_figure(height=180),
                                    config={"displayModeBar": False, "displaylogo": False},
                                    style={"height": "180px"},
                                ),
                            ]),
                            html.Div([
                                html.P("Rolling Sharpe (63-bar)", style={**_LABEL_MUTED, "marginBottom": "2px"}),
                                dcc.Graph(
                                    id="rl-rolling-sharpe-chart",
                                    figure=_empty_rl_figure(height=180),
                                    config={"displayModeBar": False, "displaylogo": False},
                                    style={"height": "180px"},
                                ),
                            ]),
                        ],
                    ),
                    # P&L distribution histogram
                    html.P(
                        "Trade P&L Distribution",
                        style={**_LABEL_MUTED, "fontWeight": "600", "marginBottom": "2px", "marginTop": "6px"},
                    ),
                    dcc.Graph(
                        id="rl-pnl-dist-chart",
                        figure=_empty_rl_figure(height=160),
                        config={"displayModeBar": False, "displaylogo": False},
                        style={"height": "160px"},
                    ),
                    # Monthly returns heatmap
                    html.P(
                        "Monthly Returns Heatmap (%)",
                        style={**_LABEL_MUTED, "fontWeight": "600", "marginBottom": "2px", "marginTop": "6px"},
                    ),
                    dcc.Graph(
                        id="rl-monthly-heatmap",
                        figure=_empty_rl_figure(height=180),
                        config={"displayModeBar": False, "displaylogo": False},
                        style={"height": "180px"},
                    ),
                    # Year-by-year table
                    html.P(
                        "Year-by-Year Performance",
                        style={**_LABEL_MUTED, "fontWeight": "600", "marginBottom": "4px", "marginTop": "6px"},
                    ),
                    _rl_datatable(
                        "rl-year-by-year-table",
                        columns=[
                            {"name": "Year",          "id": "year"},
                            {"name": "Return (%)",    "id": "return_pct"},
                            {"name": "Benchmark (%)", "id": "benchmark_pct"},
                            {"name": "Sharpe",        "id": "sharpe"},
                            {"name": "Max DD (%)",    "id": "max_drawdown_pct"},
                        ],
                    ),
                ],
            )
        ],
    )

    # -----------------------------------------------------------------
    # Sub-tab 2: Volatility Lab
    # -----------------------------------------------------------------
    volatility_lab = dcc.Tab(
        label="Volatility Lab",
        value="rl-volatility-lab-subtab",
        style=_sub_tab_style,
        selected_style=_sub_tab_selected_style,
        children=[
            html.Div(
                style={"padding": "8px 0"},
                children=[
                    # Real vs shuffled rolling volatility
                    html.P(
                        "Rolling Annualised Volatility — Real vs Shuffled",
                        style={**_LABEL_MUTED, "fontWeight": "600", "marginBottom": "2px"},
                    ),
                    dcc.Graph(
                        id="rl-vol-chart",
                        figure=_empty_rl_figure(height=200),
                        config={"displayModeBar": False, "displaylogo": False},
                        style={"height": "200px"},
                    ),
                    # Stats text block (kurtosis, ACF, Ljung-Box, same-sign rate)
                    html.Div(
                        id="rl-vol-stats",
                        style={
                            "backgroundColor": THEME["bg_card"],
                            "border": f"1px solid {THEME['border']}",
                            "borderRadius": "4px",
                            "padding": "8px 10px",
                            "marginTop": "6px",
                            "fontSize": "11px",
                            "color": THEME["text_muted"],
                        },
                        children="Run analysis to populate volatility statistics.",
                    ),
                    # Regime tape chart
                    html.P(
                        "Volatility Regime Tape",
                        style={**_LABEL_MUTED, "fontWeight": "600", "marginBottom": "2px", "marginTop": "6px"},
                    ),
                    dcc.Graph(
                        id="rl-regime-tape-chart",
                        figure=_empty_rl_figure(height=120),
                        config={"displayModeBar": False, "displaylogo": False},
                        style={"height": "120px"},
                    ),
                    # Permutation test + position size side by side
                    html.Div(
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "1fr 1fr",
                            "gap": "8px",
                            "marginTop": "6px",
                        },
                        children=[
                            html.Div(
                                id="rl-permtest-result",
                                style={
                                    "backgroundColor": THEME["bg_card"],
                                    "border": f"1px solid {THEME['border']}",
                                    "borderRadius": "4px",
                                    "padding": "8px 10px",
                                    "fontSize": "11px",
                                    "color": THEME["text_muted"],
                                },
                                children="Permutation test result will appear here.",
                            ),
                            html.Div(
                                id="rl-position-size",
                                style={
                                    "backgroundColor": THEME["bg_card"],
                                    "border": f"1px solid {THEME['border']}",
                                    "borderRadius": "4px",
                                    "padding": "8px 10px",
                                    "fontSize": "11px",
                                    "color": THEME["text_muted"],
                                },
                                children="Tail-risk position sizing will appear here.",
                            ),
                        ],
                    ),
                ],
            )
        ],
    )

    # -----------------------------------------------------------------
    # Sub-tab 3: Signal & Gate
    # -----------------------------------------------------------------
    signal_gate = dcc.Tab(
        label="Signal & Gate",
        value="rl-signal-gate-subtab",
        style=_sub_tab_style,
        selected_style=_sub_tab_selected_style,
        children=[
            html.Div(
                style={"padding": "8px 0"},
                children=[
                    html.P(
                        "Quant-Research Gate Verdict",
                        style={**_LABEL_MUTED, "fontWeight": "600", "marginBottom": "6px"},
                    ),
                    html.Div(
                        id="rl-gate-verdict",
                        style={
                            "backgroundColor": THEME["bg_card"],
                            "border": f"1px solid {THEME['border']}",
                            "borderRadius": "6px",
                            "padding": "12px 14px",
                            "fontSize": "12px",
                            "color": THEME["text_muted"],
                            "lineHeight": "1.6",
                        },
                        children="Click 'Run Analysis' to evaluate the strategy against the go/no-go gate.",
                    ),
                ],
            )
        ],
    )

    return html.Div(
        style={**_PANEL_STYLE, "margin": "8px 0"},
        children=[
            # Header row: title + Run Analysis button + status text
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "10px",
                    "marginBottom": "8px",
                },
                children=[
                    html.P(
                        "Research Lab",
                        style={
                            **_LABEL_MUTED,
                            "fontWeight": "600",
                            "marginBottom": "0",
                            "flex": "1",
                        },
                    ),
                    html.Div(
                        id="rl-status",
                        style={
                            "color": THEME["text_muted"],
                            "fontSize": "11px",
                            "flex": "1",
                            "textAlign": "right",
                        },
                        children="Select a symbol, load the chart, then click Run Analysis.",
                    ),
                    html.Button(
                        "Run Analysis",
                        id="rl-run-btn",
                        n_clicks=0,
                        style={
                            "backgroundColor": THEME["bg_dark"],
                            "color": THEME["accent"],
                            "border": f"1px solid {THEME['accent']}",
                            "borderRadius": "4px",
                            "padding": "3px 10px",
                            "cursor": "pointer",
                            "fontSize": "11px",
                            "whiteSpace": "nowrap",
                        },
                    ),
                ],
            ),
            # Nested sub-tabs
            dcc.Tabs(
                id="rl-sub-tabs",
                value="rl-strategy-lab-subtab",
                colors={
                    "border": THEME["border"],
                    "primary": THEME["accent"],
                    "background": THEME["bg_dark"],
                },
                children=[strategy_lab, volatility_lab, signal_gate],
            ),
        ],
    )


def _empty_equity_curve_figure():
    """Return a dark-themed empty line-chart placeholder for the equity curve.

    Returned as the initial ``figure`` for the ``equity-curve-chart`` dcc.Graph.
    The ``run_backtest_callback`` in callbacks.py replaces it with actual data.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
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


def _bottom_tabs_panel() -> html.Div:
    """Positions + PnL Calendar + Equity Curve tab panel, rendered full-width below the main row.

    Three tabs mirror the PyQt5 bottom_tabs area (ui/main_window.py):
    - "Positions"    — open positions list, color-coded by PnL (populated via
                       the update_positions callback in callbacks.py).
    - "PnL Calendar" — month-grid calendar (42-cell, 6-week × 7-day) showing
                       realized PnL per day (populated via
                       update_pnl_calendar_display callback).
    - "Equity Curve" — portfolio value over time, populated by run_backtest_callback
                       after a backtest is run (Phase 1.5).

    Month navigation (◀ / ▶ / Today) drives a dcc.Store which in turn triggers
    the calendar display callback.  Both tabs also refresh after every order
    placement because they listen to order-status children changes.
    """
    return html.Div(
        style={"marginTop": "10px"},
        children=[
            dcc.Tabs(
                id="bottom-tabs",
                value="positions-tab",
                colors={
                    "border": THEME["border"],
                    "primary": THEME["accent"],
                    "background": THEME["bg_dark"],
                },
                children=[
                    # ----------------------------------------------------------
                    # Tab 1: Positions
                    # ----------------------------------------------------------
                    dcc.Tab(
                        label="Positions",
                        value="positions-tab",
                        style=_TAB_STYLE,
                        selected_style=_TAB_SELECTED_STYLE,
                        children=[
                            html.Div(
                                style={**_PANEL_STYLE, "margin": "8px 0"},
                                children=[
                                    html.P(
                                        "Open Positions",
                                        style={**_LABEL_MUTED, "fontWeight": "600", "marginBottom": "6px"},
                                    ),
                                    html.Div(
                                        id="positions-content",
                                        children=html.Span(
                                            "No active positions",
                                            style={"color": THEME["text_muted"], "fontSize": "11px"},
                                        ),
                                    ),
                                ],
                            ),
                        ],
                    ),
                    # ----------------------------------------------------------
                    # Tab 2: PnL Calendar
                    # ----------------------------------------------------------
                    dcc.Tab(
                        label="PnL Calendar",
                        value="pnl-calendar-tab",
                        style=_TAB_STYLE,
                        selected_style=_TAB_SELECTED_STYLE,
                        children=[
                            html.Div(
                                style={**_PANEL_STYLE, "margin": "8px 0"},
                                children=[
                                    # Month navigation header row
                                    html.Div(
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "gap": "8px",
                                            "marginBottom": "8px",
                                        },
                                        children=[
                                            html.Button(
                                                "◀",
                                                id="pnl-prev-btn",
                                                n_clicks=0,
                                                style=_NAV_BTN_STYLE,
                                            ),
                                            html.Div(
                                                id="pnl-calendar-title",
                                                style={
                                                    "flex": "1",
                                                    "textAlign": "center",
                                                    "fontWeight": "600",
                                                    "fontSize": "13px",
                                                    "color": THEME["text_main"],
                                                },
                                            ),
                                            html.Button(
                                                "▶",
                                                id="pnl-next-btn",
                                                n_clicks=0,
                                                style=_NAV_BTN_STYLE,
                                            ),
                                            html.Button(
                                                "Today",
                                                id="pnl-today-btn",
                                                n_clicks=0,
                                                style={**_NAV_BTN_STYLE, "minWidth": "50px"},
                                            ),
                                            html.Div(
                                                id="pnl-calendar-total",
                                                style={
                                                    "fontSize": "12px",
                                                    "fontWeight": "600",
                                                    "marginLeft": "8px",
                                                    "color": THEME["text_muted"],
                                                },
                                            ),
                                        ],
                                    ),
                                    # Weekday header (Mon … Sun, matches PyQt5 Monday-first)
                                    html.Div(
                                        style={
                                            "display": "grid",
                                            "gridTemplateColumns": "repeat(7, 1fr)",
                                            "gap": "2px",
                                            "marginBottom": "2px",
                                        },
                                        children=[
                                            html.Div(
                                                day,
                                                style={
                                                    "textAlign": "center",
                                                    "color": THEME["text_muted"],
                                                    "fontSize": "10px",
                                                    "fontWeight": "600",
                                                    "padding": "2px",
                                                },
                                            )
                                            for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                                        ],
                                    ),
                                    # 42-cell day grid (populated by update_pnl_calendar_display callback)
                                    html.Div(id="pnl-calendar-grid"),
                                ],
                            ),
                        ],
                    ),
                    # ----------------------------------------------------------
                    # Tab 3: Equity Curve (Phase 1.5)
                    # ----------------------------------------------------------
                    dcc.Tab(
                        label="Equity Curve",
                        value="equity-curve-tab",
                        style=_TAB_STYLE,
                        selected_style=_TAB_SELECTED_STYLE,
                        children=[
                            html.Div(
                                style={**_PANEL_STYLE, "margin": "8px 0"},
                                children=[
                                    html.P(
                                        "Portfolio Equity Curve",
                                        style={**_LABEL_MUTED, "fontWeight": "600", "marginBottom": "4px"},
                                    ),
                                    dcc.Graph(
                                        id="equity-curve-chart",
                                        figure=_empty_equity_curve_figure(),
                                        config={
                                            "displayModeBar": False,
                                            "displaylogo": False,
                                        },
                                        style={"height": "200px"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                    # ----------------------------------------------------------
                    # Tab 4: Orders / Trade Blotter (Phase 1.6)
                    # Mirrors the PyQt5 _orders_table (7 cols):
                    #   Time | Symbol | Side | Type | Qty | Fill Price | Status
                    # Populated by the update_orders_table callback in
                    # callbacks.py after every order placement.
                    # ----------------------------------------------------------
                    dcc.Tab(
                        label="Orders",
                        value="orders-tab",
                        style=_TAB_STYLE,
                        selected_style=_TAB_SELECTED_STYLE,
                        children=[
                            html.Div(
                                style={**_PANEL_STYLE, "margin": "8px 0"},
                                children=[
                                    # Header row: title + status summary
                                    html.Div(
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "marginBottom": "6px",
                                        },
                                        children=[
                                            html.P(
                                                "Order History",
                                                style={
                                                    **_LABEL_MUTED,
                                                    "fontWeight": "600",
                                                    "marginBottom": "0",
                                                    "flex": "1",
                                                },
                                            ),
                                            html.Div(
                                                id="orders-status",
                                                style={
                                                    "color": THEME["text_muted"],
                                                    "fontSize": "11px",
                                                },
                                                children="Orders: none yet",
                                            ),
                                        ],
                                    ),
                                    # Trade-blotter DataTable — dark-themed to match
                                    # the PyQt5 QTableWidget style.
                                    dash_table.DataTable(
                                        id="orders-table",
                                        columns=[
                                            {"name": "Time",       "id": "time"},
                                            {"name": "Symbol",     "id": "symbol"},
                                            {"name": "Side",       "id": "side"},
                                            {"name": "Type",       "id": "type"},
                                            {"name": "Qty",        "id": "qty"},
                                            {"name": "Fill Price", "id": "fill_price"},
                                            {"name": "Status",     "id": "status"},
                                            {"name": "Why",        "id": "why"},
                                        ],
                                        data=[],
                                        page_action="none",   # show all rows
                                        sort_action="native", # client-side sort
                                        sort_mode="single",
                                        style_table={
                                            "overflowX": "auto",
                                            "overflowY": "auto",
                                            "maxHeight": "200px",
                                            "backgroundColor": THEME["bg_dark"],
                                            "border": f"1px solid {THEME['border']}",
                                            "borderRadius": "4px",
                                        },
                                        style_cell={
                                            "backgroundColor": THEME["bg_dark"],
                                            "color": THEME["text_main"],
                                            "border": f"1px solid {THEME['border']}",
                                            "fontSize": "11px",
                                            "fontFamily": "'SF Mono', 'Consolas', 'Menlo', monospace",
                                            "textAlign": "center",
                                            "padding": "4px 8px",
                                            "minWidth": "60px",
                                        },
                                        style_header={
                                            "backgroundColor": THEME["bg_card"],
                                            "color": THEME["text_muted"],
                                            "fontWeight": "600",
                                            "fontSize": "10px",
                                            "border": f"1px solid {THEME['border']}",
                                            "textTransform": "uppercase",
                                            "letterSpacing": "0.5px",
                                            "padding": "4px 8px",
                                        },
                                        # "Why" holds the recorded rationale sentence (Phase 11.1):
                                        # left-aligned and wrapping so it stays fully readable.
                                        style_cell_conditional=[
                                            {
                                                "if": {"column_id": "why"},
                                                "textAlign": "left",
                                                "whiteSpace": "normal",
                                                "height": "auto",
                                                "minWidth": "320px",
                                                "maxWidth": "520px",
                                            },
                                        ],
                                        # Color-coding rows by side + status —
                                        # mirrors PyQt5's _SIDE_COLORS / _STATUS_COLORS.
                                        style_data_conditional=[
                                            # Side: BUY → dark-green bg + green text
                                            {
                                                "if": {
                                                    "filter_query": "{side} = 'BUY'",
                                                    "column_id": "side",
                                                },
                                                "backgroundColor": "#1a4731",
                                                "color": "#3fb950",
                                                "fontWeight": "600",
                                            },
                                            # Side: SELL → dark-red bg + red text
                                            {
                                                "if": {
                                                    "filter_query": "{side} = 'SELL'",
                                                    "column_id": "side",
                                                },
                                                "backgroundColor": "#3d1a1a",
                                                "color": "#f85149",
                                                "fontWeight": "600",
                                            },
                                            # Status: Filled → green
                                            {
                                                "if": {
                                                    "filter_query": "{status} = 'Filled'",
                                                    "column_id": "status",
                                                },
                                                "color": "#3fb950",
                                            },
                                            # Status: Pending → orange
                                            {
                                                "if": {
                                                    "filter_query": "{status} = 'Pending'",
                                                    "column_id": "status",
                                                },
                                                "color": "#f0883e",
                                            },
                                            # Status: Rejected → red
                                            {
                                                "if": {
                                                    "filter_query": "{status} = 'Rejected'",
                                                    "column_id": "status",
                                                },
                                                "color": "#f85149",
                                            },
                                            # Status: Canceled → muted
                                            {
                                                "if": {
                                                    "filter_query": "{status} = 'Canceled'",
                                                    "column_id": "status",
                                                },
                                                "color": "#8b949e",
                                            },
                                            # Alternating row shading
                                            {
                                                "if": {"row_index": "odd"},
                                                "backgroundColor": "#0f1419",
                                            },
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                    # ----------------------------------------------------------
                    # Tab 6: Research Lab (Phase 2)
                    # Three nested sub-tabs: Strategy Lab, Volatility Lab,
                    # Signal & Gate.  Populated by run_research_lab callback
                    # in callbacks.py when "Run Analysis" is clicked.
                    # ----------------------------------------------------------
                    dcc.Tab(
                        label="Research Lab",
                        value="research-lab-tab",
                        style=_TAB_STYLE,
                        selected_style=_TAB_SELECTED_STYLE,
                        children=[_research_lab_tab_content()],
                    ),
                    # ----------------------------------------------------------
                    # Tab 5: News & Earnings (Phase 1.7)
                    # Reuses core.news_pipeline (DuckDuckGo → OpenBB → GDELT)
                    # and core.data_loader.DataLoader.get_earnings_calendar().
                    # Populated by the update_news_earnings_panel callback in
                    # callbacks.py when user clicks Refresh or loads a chart.
                    # ----------------------------------------------------------
                    dcc.Tab(
                        label="News & Earnings",
                        value="news-earnings-tab",
                        style=_TAB_STYLE,
                        selected_style=_TAB_SELECTED_STYLE,
                        children=[
                            html.Div(
                                style={**_PANEL_STYLE, "margin": "8px 0"},
                                children=[
                                    # Header row: title + refresh button
                                    html.Div(
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "marginBottom": "8px",
                                        },
                                        children=[
                                            html.P(
                                                "News & Earnings",
                                                style={
                                                    **_LABEL_MUTED,
                                                    "fontWeight": "600",
                                                    "marginBottom": "0",
                                                    "flex": "1",
                                                },
                                            ),
                                            html.Button(
                                                "Refresh",
                                                id="news-refresh-btn",
                                                n_clicks=0,
                                                style={
                                                    "backgroundColor": THEME["bg_dark"],
                                                    "color": THEME["accent"],
                                                    "border": f"1px solid {THEME['accent']}",
                                                    "borderRadius": "4px",
                                                    "padding": "3px 10px",
                                                    "cursor": "pointer",
                                                    "fontSize": "11px",
                                                },
                                            ),
                                        ],
                                    ),
                                    # Two-column body: news (left 55%) + earnings (right 45%)
                                    html.Div(
                                        style={
                                            "display": "grid",
                                            "gridTemplateColumns": "55fr 45fr",
                                            "gap": "12px",
                                            "alignItems": "start",
                                        },
                                        children=[
                                            # --- News section -----------------------------------------
                                            html.Div(
                                                children=[
                                                    html.P(
                                                        "Recent News",
                                                        style={
                                                            **_LABEL_MUTED,
                                                            "fontWeight": "600",
                                                            "marginBottom": "6px",
                                                        },
                                                    ),
                                                    html.Div(
                                                        id="news-content",
                                                        style={
                                                            "maxHeight": "240px",
                                                            "overflowY": "auto",
                                                            "colorScheme": "dark",
                                                        },
                                                        children=html.Span(
                                                            "Select a symbol and click Refresh to load news.",
                                                            style={
                                                                "color": THEME["text_muted"],
                                                                "fontSize": "11px",
                                                            },
                                                        ),
                                                    ),
                                                ],
                                            ),
                                            # --- Earnings section --------------------------------------
                                            html.Div(
                                                children=[
                                                    html.P(
                                                        "Earnings Calendar",
                                                        style={
                                                            **_LABEL_MUTED,
                                                            "fontWeight": "600",
                                                            "marginBottom": "4px",
                                                        },
                                                    ),
                                                    html.Div(
                                                        id="earnings-status",
                                                        style={
                                                            "color": THEME["text_muted"],
                                                            "fontSize": "10px",
                                                            "marginBottom": "4px",
                                                            "minHeight": "14px",
                                                        },
                                                        children="",
                                                    ),
                                                    dash_table.DataTable(
                                                        id="earnings-table",
                                                        columns=[
                                                            {"name": "Date",          "id": "date"},
                                                            {"name": "EPS Est",       "id": "eps_estimate"},
                                                            {"name": "EPS Actual",    "id": "eps_actual"},
                                                            {"name": "Rev Est ($M)",  "id": "revenue_estimate"},
                                                            {"name": "Rev Act ($M)",  "id": "revenue_actual"},
                                                        ],
                                                        data=[],
                                                        page_action="none",
                                                        sort_action="native",
                                                        sort_mode="single",
                                                        style_table={
                                                            "overflowX": "auto",
                                                            "overflowY": "auto",
                                                            "maxHeight": "220px",
                                                            "backgroundColor": THEME["bg_dark"],
                                                            "border": f"1px solid {THEME['border']}",
                                                            "borderRadius": "4px",
                                                        },
                                                        style_cell={
                                                            "backgroundColor": THEME["bg_dark"],
                                                            "color": THEME["text_main"],
                                                            "border": f"1px solid {THEME['border']}",
                                                            "fontSize": "11px",
                                                            "fontFamily": "'SF Mono', 'Consolas', 'Menlo', monospace",
                                                            "textAlign": "center",
                                                            "padding": "4px 6px",
                                                            "minWidth": "50px",
                                                        },
                                                        style_header={
                                                            "backgroundColor": THEME["bg_card"],
                                                            "color": THEME["text_muted"],
                                                            "fontWeight": "600",
                                                            "fontSize": "10px",
                                                            "border": f"1px solid {THEME['border']}",
                                                            "textTransform": "uppercase",
                                                            "letterSpacing": "0.5px",
                                                            "padding": "4px 6px",
                                                        },
                                                        style_data_conditional=[
                                                            {
                                                                "if": {"row_index": "odd"},
                                                                "backgroundColor": "#0f1419",
                                                            },
                                                        ],
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                    _agent_monitor_tab(),
                    _research_loop_tab(),
                    _execution_tab(),
                ],
            ),
        ],
    )


def _agent_table() -> "dash_table.DataTable":
    """Per-agent status table; status text is colour-coded with high-contrast colours."""
    table = _rl_datatable(
        "agent-table",
        [
            {"name": "Agent", "id": "agent"},
            {"name": "Status", "id": "status"},
            {"name": "Runs", "id": "runs"},
            {"name": "Summary", "id": "summary"},
        ],
    )
    table.style_cell_conditional = [
        {"if": {"column_id": "summary"}, "textAlign": "left", "whiteSpace": "normal",
         "height": "auto", "minWidth": "300px"},
        {"if": {"column_id": "agent"}, "textAlign": "left", "fontWeight": "600"},
    ]
    table.style_data_conditional = [
        {"if": {"row_index": "odd"}, "backgroundColor": "#0f1419"},
        {"if": {"filter_query": "{status} = 'ok'", "column_id": "status"}, "color": "#7ee787", "fontWeight": "600"},
        {"if": {"filter_query": "{status} = 'warning'", "column_id": "status"}, "color": "#ffb86b", "fontWeight": "600"},
        {"if": {"filter_query": "{status} = 'error'", "column_id": "status"}, "color": "#ffa198", "fontWeight": "600"},
    ]
    return table


def _agent_monitor_tab() -> "dcc.Tab":
    """Agent Monitor tab -- mirrors the desktop app's Agent Monitor.

    Start/Stop the runtime supervisor, then watch one row per agent
    (portfolio / news / price / stats) plus the LLM's summary line. All
    dynamic content is driven by the single ``agent_monitor`` callback in
    callbacks.py; the interval only exists while this tab is open.
    """
    _btn = {
        "backgroundColor": THEME["bg_dark"],
        "border": f"1px solid {THEME['border']}",
        "borderRadius": "4px",
        "padding": "3px 12px",
        "fontSize": "11px",
        "cursor": "pointer",
        "marginLeft": "8px",
    }
    return dcc.Tab(
        label="Agent Monitor",
        value="agent-monitor-tab",
        style=_TAB_STYLE,
        selected_style=_TAB_SELECTED_STYLE,
        children=[
            html.Div(
                style={**_PANEL_STYLE, "margin": "8px 0"},
                children=[
                    dcc.Interval(id="agent-interval", interval=3000, n_intervals=0),
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "marginBottom": "8px"},
                        children=[
                            html.P("Agent Monitor", style={**_LABEL_MUTED, "fontWeight": "600",
                                                           "marginBottom": "0", "marginRight": "12px"}),
                            html.Span("Agents: stopped", id="agent-status-label",
                                      style={"fontSize": "11px", "color": THEME["text_muted"], "flex": "1"}),
                            html.Button("Start Agents", id="agent-start-btn", n_clicks=0,
                                        style={**_btn, "color": THEME["green"]}),
                            html.Button("Stop Agents", id="agent-stop-btn", n_clicks=0, disabled=True,
                                        style={**_btn, "color": THEME["red"]}),
                        ],
                    ),
                    _agent_table(),
                    html.Div(
                        "LLM summary: —",
                        id="agent-llm-summary",
                        style={"color": THEME["text_muted"], "fontSize": "11px",
                               "fontStyle": "italic", "marginTop": "8px", "lineHeight": "1.5"},
                    ),
                ],
            ),
        ],
    )


def _research_loop_tab() -> "dcc.Tab":
    """Research Loop tab: read-only view of the autonomous research loop and the experiment log.

    Shows each candidate strategy's status (candidate / paper / retired) with its latest evidence
    against buy-and-hold, the forward paper-trading P&L of promoted strategies, and the most recent
    experiment runs. Populated from the same SQLite file the loop writes (core/research_loop.py).
    """
    def _table(table_id, columns):
        t = _rl_datatable(table_id, columns)
        t.style_cell_conditional = [{"if": {"column_id": columns[0]["id"]}, "textAlign": "left", "fontWeight": "600"}]
        t.style_data_conditional = [
            {"if": {"row_index": "odd"}, "backgroundColor": "#0f1419"},
            {"if": {"filter_query": "{status} = 'paper'", "column_id": "status"}, "color": "#7ee787", "fontWeight": "600"},
            {"if": {"filter_query": "{status} = 'retired'", "column_id": "status"}, "color": "#ffa198", "fontWeight": "600"},
            {"if": {"filter_query": "{status} = 'candidate'", "column_id": "status"}, "color": "#c9d1d9"},
            {"if": {"filter_query": "{verdict} = 'PASS'", "column_id": "verdict"}, "color": "#7ee787", "fontWeight": "600"},
        ]
        return t

    label = {**_LABEL_MUTED, "fontWeight": "600", "margin": "10px 0 4px"}
    return dcc.Tab(
        label="Research Loop",
        value="research-loop-tab",
        style=_TAB_STYLE,
        selected_style=_TAB_SELECTED_STYLE,
        children=[
            html.Div(
                style={**_PANEL_STYLE, "margin": "8px 0"},
                children=[
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "marginBottom": "6px"},
                        children=[
                            html.P("Research Loop", style={**_LABEL_MUTED, "fontWeight": "600", "marginBottom": "0", "flex": "1"}),
                            html.Button("Refresh", id="research-refresh-btn", n_clicks=0, style={
                                "backgroundColor": THEME["bg_dark"], "color": THEME["accent"],
                                "border": f"1px solid {THEME['accent']}", "borderRadius": "4px",
                                "padding": "3px 12px", "fontSize": "11px", "cursor": "pointer"}),
                        ],
                    ),
                    html.Div("", id="research-status",
                             style={"color": THEME["text_muted"], "fontSize": "11px", "lineHeight": "1.5", "marginBottom": "6px"}),
                    html.P("Candidate strategies vs buy-and-hold (latest evaluation)", style=label),
                    _table("research-candidates-table", [
                        {"name": "Candidate", "id": "candidate"}, {"name": "Status", "id": "status"},
                        {"name": "Sharpe", "id": "sharpe"}, {"name": "B&H Sharpe", "id": "bh_sharpe"},
                        {"name": "Diff", "id": "diff"}, {"name": "Interval", "id": "ci"},
                        {"name": "Trades", "id": "trades"}, {"name": "Max DD", "id": "max_dd"},
                        {"name": "Test", "id": "verdict"}, {"name": "Evaluated", "id": "evaluated"}]),
                    html.P("Forward paper trading (positions recorded by earlier runs, marked to later prices)", style=label),
                    _table("research-paper-table", [
                        {"name": "Candidate", "id": "candidate"}, {"name": "Days", "id": "days"},
                        {"name": "Return %", "id": "return_pct"}, {"name": "Position changes", "id": "trades"}]),
                    html.P("Recent experiment runs", style=label),
                    _table("research-runs-table", [
                        {"name": "#", "id": "id"}, {"name": "When (UTC)", "id": "when"}, {"name": "Run", "id": "name"},
                        {"name": "Model", "id": "model"}, {"name": "Key result", "id": "result"}, {"name": "Commit", "id": "commit"}]),
                ],
            ),
        ],
    )


def _execution_tab() -> "dcc.Tab":
    """Execution tab: start/stop the PAPER execution service and see exactly what it decided and why.

    Reads the durable journal (core/execution/view.py), so it also shows a service running in the desktop app or the
    command line. Paper only: there is no code path from here to a real broker.
    """
    def _btn(text, btn_id, color):
        return html.Button(text, id=btn_id, n_clicks=0, style={
            "backgroundColor": THEME["bg_dark"], "color": color, "border": f"1px solid {color}", "borderRadius": "4px",
            "padding": "4px 12px", "fontSize": "11px", "cursor": "pointer", "marginRight": "6px"})

    label = {**_LABEL_MUTED, "fontWeight": "600", "margin": "10px 0 4px"}
    return dcc.Tab(
        label="Execution",
        value="execution-tab",
        style=_TAB_STYLE,
        selected_style=_TAB_SELECTED_STYLE,
        children=[
            html.Div(
                style={**_PANEL_STYLE, "margin": "8px 0"},
                children=[
                    dcc.Interval(id="exec-interval", interval=5000, n_intervals=0),
                    html.P("Paper execution (no real orders are possible)", style={**_LABEL_MUTED, "fontWeight": "600", "marginBottom": "6px"}),
                    html.Div(style={"marginBottom": "6px"}, children=[
                        _btn("Start paper trading", "exec-start-btn", THEME["green"]),
                        _btn("Stop", "exec-stop-btn", THEME["text_muted"]),
                        _btn("Halt entries", "exec-halt-btn", THEME["orange"]),
                        _btn("Resume", "exec-resume-btn", THEME["accent"]),
                        _btn("Flatten all", "exec-flatten-btn", THEME["red"]),
                    ]),
                    html.Div("Uses the Symbol, Interval and Strategy chosen in the top bar.",
                             style={"color": THEME["text_muted"], "fontSize": "10px", "marginBottom": "4px"}),
                    html.Div("", id="exec-message", style={"color": THEME["accent"], "fontSize": "11px", "marginBottom": "4px"}),
                    html.Div("", id="exec-headline", style={"color": THEME["text_main"], "fontSize": "12px", "fontWeight": "600", "marginBottom": "4px"}),
                    html.Div("", id="exec-issues", style={"color": THEME["red"], "fontSize": "11px", "whiteSpace": "pre-line", "marginBottom": "6px"}),
                    html.P("Per symbol", style=label),
                    _rl_datatable("exec-symbols-table", [
                        {"name": "Symbol", "id": "symbol"}, {"name": "State", "id": "state"}, {"name": "Signal", "id": "signal"},
                        {"name": "Position", "id": "position"}, {"name": "Bar", "id": "bar"},
                        {"name": "Data age (bars)", "id": "data_age_bars"}, {"name": "Note", "id": "detail"}]),
                    html.P("Every decision, with the reason (newest first)", style=label),
                    _rl_datatable("exec-decisions-table", [
                        {"name": "When", "id": "when"}, {"name": "Symbol", "id": "symbol"}, {"name": "Action", "id": "action"},
                        {"name": "Status", "id": "status"}, {"name": "Detail", "id": "detail"}]),
                ],
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
    _today = datetime.date.today()
    return html.Div(
        style=_PAGE_STYLE,
        children=[
            # Hidden stores for passing data between callbacks
            dcc.Store(id="ohlcv-store"),
            dcc.Store(id="signals-store", data=[]),

            # Holds the last backtest report's serialisable subset so the
            # Research Lab callback can re-run analytics without re-running
            # the full backtest on every sub-tab switch.
            dcc.Store(id="backtest-report-store", data={}),

            # Tracks which symbol is currently displayed — used by the
            # interval callback for subscription management.
            dcc.Store(id="active-symbol-store", data=None),

            # Holds the currently-displayed year/month for the PnL Calendar tab.
            # Initialised to the current calendar month so the calendar shows
            # today's month on first render without any user interaction.
            dcc.Store(
                id="pnl-calendar-store",
                data={"year": _today.year, "month": _today.month},
            ),

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
                    # Primary chart + metrics row (unchanged from Phase 1.3)
                    dbc.Row(
                        style={"gap": "0"},
                        children=[
                            _chart_panel(),
                            _metrics_panel(),
                        ],
                    ),
                    # Full-width bottom tab panel: Positions + PnL Calendar
                    _bottom_tabs_panel(),
                ],
            ),

            # Bottom status bar
            _status_bar(),
        ],
    )
