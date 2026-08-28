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
    interval: str = "1d",
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
    interval:
        The candle interval used to load *df* (e.g. ``"1d"``, ``"1h"``,
        ``"5m"``). Controls how the x-axis collapses non-trading gaps —
        see :func:`_compute_rangebreaks`. Ignored for crypto symbols
        (see :func:`is_crypto_symbol`), which trade 24/7 and have no gaps
        to close.

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

    # Collapse non-trading gaps (weekends, holidays, and — for intraday
    # intervals — overnight hours) out of the x-axis so candles sit flush
    # against each other instead of leaving a visible stretch of blank grid.
    # Crypto trades 24/7, so there are no such gaps to close there.
    if not is_crypto_symbol(symbol):
        rangebreaks = _compute_rangebreaks(df, interval)
        if rangebreaks:
            fig.update_xaxes(rangebreaks=rangebreaks)

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


_INTRADAY_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}


def _compute_rangebreaks(df: pd.DataFrame, interval: str) -> List[dict]:
    """Build Plotly ``rangebreaks`` that collapse non-trading gaps out of the x-axis.

    Always closes weekends. For intraday intervals also closes overnight
    hours outside the standard 09:30-16:00 US equity session (this codebase
    is yfinance/US-equity-centric elsewhere too — see core/data_loader.py —
    so this is a reasonable default rather than a per-exchange calendar).
    Additionally detects the exact weekdays missing from *df*'s own date
    range (i.e. market holidays) and closes those precisely, so no
    hardcoded holiday calendar or extra dependency is needed.

    Parameters
    ----------
    df:
        The OHLCV DataFrame the figure was built from (datetime index).
    interval:
        The candle interval used to load *df* (e.g. ``"1d"``, ``"5m"``).
        Anything in :data:`_INTRADAY_INTERVALS` also gets an hour-of-day
        rangebreak; everything else (daily and coarser) only gets the
        weekend/holiday breaks.

    Returns
    -------
    list[dict]
        Suitable for ``fig.update_xaxes(rangebreaks=...)``. Empty if *df*
        has no usable datetime index.
    """
    if df is None or df.empty:
        return []

    breaks: List[dict] = [dict(bounds=["sat", "mon"])]  # always hide weekends

    if interval in _INTRADAY_INTERVALS:
        # Standard US equity session (09:30-16:00) — hides the overnight gap.
        breaks.append(dict(bounds=[16, 9.5], pattern="hour"))

    # Exact missing weekdays (holidays), derived from the data itself.
    try:
        idx = pd.DatetimeIndex(df.index)
        present_days = idx.normalize().unique()
        if len(present_days) > 1:
            full_range = pd.date_range(present_days.min(), present_days.max(), freq="D")
            weekday_range = full_range[full_range.weekday < 5]
            missing = weekday_range.difference(present_days)
            if len(missing) > 0:
                breaks.append(dict(values=missing))
    except (TypeError, ValueError):
        pass  # odd/non-datetime index — skip exact holiday closing, keep weekend break

    return breaks


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
