"""
core/chart_builder.py

UI-framework-agnostic helpers that convert OHLCV price data and trading
signals into ``plotly.graph_objects.Figure`` objects.

Both the PyQt5 desktop app (ui/main_window.py) and the Dash web app
(dash_app/) import from this module so the charting logic is maintained
in exactly one place.

No PyQt5 / Qt dependencies are allowed here — this module must be
importable in any environment that has plotly installed.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Color palette — identical to the hex values in ui/main_window.py stylesheet
# ---------------------------------------------------------------------------
THEME: Dict[str, str] = {
    "bg_dark":    "#0d1117",  # main window / page background
    "bg_card":    "#161b22",  # panel / card background
    "border":     "#30363d",
    "border_dim": "#21262d",
    "text_main":  "#e6edf3",
    "text_muted": "#8b949e",
    "accent":     "#58a6ff",  # blue highlight
    "green":      "#3fb950",  # profit / buy
    "red":        "#f85149",  # loss / sell
    "orange":     "#f0883e",  # warning / pending
}

_BASE_TEMPLATE = "plotly_dark"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_candlestick_figure(
    df: Optional[pd.DataFrame] = None,
    symbol: str = "",
    show_ma: bool = False,
    height: int = 600,
) -> go.Figure:
    """Return a dark-themed candlestick ``go.Figure`` for *df*.

    Parameters
    ----------
    df:
        OHLCV DataFrame with a datetime index and at minimum Open, High,
        Low, Close columns.  Pass ``None`` or an empty DataFrame to get a
        blank placeholder figure.
    symbol:
        Ticker label used in the trace name (e.g. ``"AAPL"``).
    show_ma:
        If ``True`` and *df* contains ``MA20`` / ``MA50`` columns, overlay
        those moving-average lines on the chart.
    height:
        Figure height in pixels.

    Returns
    -------
    go.Figure
        Ready to embed in a ``QWebEngineView`` (PyQt5) or a
        ``dcc.Graph`` component (Dash).
    """
    if df is None or df.empty:
        return _empty_figure(height=height)

    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=symbol or "Price",
            increasing_line_color=THEME["green"],
            decreasing_line_color=THEME["red"],
        )
    ])

    if show_ma:
        for col, color in [("MA20", THEME["accent"]), ("MA50", THEME["orange"])]:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode="lines",
                    line=dict(color=color, width=1.5),
                    name=col,
                ))

    _apply_dark_layout(fig, height=height)
    return fig


def overlay_signals(
    fig: go.Figure,
    signals: List[Dict],
) -> go.Figure:
    """Overlay buy / sell signal markers onto *fig* (mutates and returns it).

    Parameters
    ----------
    fig:
        An existing ``go.Figure`` (e.g. from :func:`build_candlestick_figure`).
    signals:
        List of signal dicts.  Each dict must contain:

        * ``"type"`` — one of ``"buy"``, ``"buy_cover"``, ``"sell"``,
          ``"sell_short"``
        * ``"date"`` — the x-axis value (datetime or string)
        * ``"price"`` — the y-axis value (float)

    Returns
    -------
    go.Figure
        The same figure with signal scatter traces appended.
    """
    buy_signals  = [s for s in signals if s.get("type") in ("buy", "buy_cover")]
    sell_signals = [s for s in signals if s.get("type") in ("sell", "sell_short")]

    if buy_signals:
        fig.add_trace(go.Scatter(
            x=[s["date"] for s in buy_signals],
            y=[s["price"] for s in buy_signals],
            mode="markers",
            marker=dict(symbol="triangle-up", size=15, color=THEME["green"]),
            name="Buy Signal",
        ))

    if sell_signals:
        fig.add_trace(go.Scatter(
            x=[s["date"] for s in sell_signals],
            y=[s["price"] for s in sell_signals],
            mode="markers",
            marker=dict(symbol="triangle-down", size=15, color=THEME["red"]),
            name="Sell Signal",
        ))

    return fig


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def is_crypto_symbol(symbol: str) -> bool:
    """Return True if *symbol* looks like a crypto trading pair.

    Uses the same heuristic as the rest of the codebase: a symbol that
    contains "USDT" (e.g. ``"BTCUSDT"``, ``"ETHUSDT"``) is treated as a
    crypto pair whose live price streams via Binance WebSocket; everything
    else is treated as an equity and uses REST/yfinance polling.

    This mirrors the ``"USDT" in symbol.upper()`` check in
    ``core/data_loader.py``, ``brokers/``, etc. — one source of truth.
    """
    return "USDT" in symbol.upper()


def add_live_tick_trace(fig: go.Figure) -> go.Figure:
    """Append an empty *live-tick* scatter trace to *fig* and return it.

    The Dash interval callback patches this trace (always ``figure.data[-1]``)
    with the latest real-time price without rebuilding the whole figure.
    Call this at the end of any figure-building path (success *and* error
    placeholder) so the trace index is always predictable.

    Parameters
    ----------
    fig:
        An existing ``go.Figure`` (e.g. from :func:`build_candlestick_figure`
        or from :func:`_empty_figure`).

    Returns
    -------
    go.Figure
        The same figure with the live-tick scatter trace appended as the
        final trace.
    """
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode="markers",
        marker=dict(
            symbol="circle-open",
            size=10,
            color="#ffff00",        # yellow — easy to spot on the dark theme
            line=dict(width=2, color="#ffff00"),
        ),
        name="Live",
        showlegend=False,
        hovertemplate="Live: %{y:,.4f}<extra></extra>",
    ))
    return fig


def _apply_dark_layout(fig: go.Figure, height: int = 600) -> None:
    """Apply the AlgoTrader dark theme to *fig* in-place."""
    fig.update_layout(
        template=_BASE_TEMPLATE,
        height=height,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(type="date", rangeslider_visible=False),
        yaxis=dict(title="Price", side="right"),
        hovermode="x unified",
        paper_bgcolor=THEME["bg_dark"],
        plot_bgcolor=THEME["bg_dark"],
        font=dict(color=THEME["text_main"]),
        legend=dict(
            bgcolor=THEME["bg_card"],
            bordercolor=THEME["border"],
            borderwidth=1,
        ),
    )


def _empty_figure(height: int = 600) -> go.Figure:
    """Return a blank dark-themed placeholder figure."""
    fig = go.Figure()
    fig.update_layout(
        template=_BASE_TEMPLATE,
        height=height,
        paper_bgcolor=THEME["bg_dark"],
        plot_bgcolor=THEME["bg_dark"],
        font=dict(color=THEME["text_muted"]),
        annotations=[dict(
            text="No data loaded",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            font=dict(size=18, color=THEME["text_muted"]),
        )],
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig
