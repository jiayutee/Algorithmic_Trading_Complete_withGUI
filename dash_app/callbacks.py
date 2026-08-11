"""
dash_app/callbacks.py

Registers Dash callbacks on *app*.

Phase 1.1 — scaffold only:
  - load_chart_callback: updates main-chart and status-bar when the user
    clicks "Load Chart".  Calls the shared chart builder (core.chart_builder)
    so no figure logic is duplicated here.

Later phases will add order-entry, live P&L, backtest triggers, etc.
"""

from __future__ import annotations

import pandas as pd
from dash import Input, Output, State, callback, no_update
import dash

from core.chart_builder import build_candlestick_figure, overlay_signals


def register_callbacks(app: dash.Dash) -> None:
    """Attach all callbacks to *app*.

    Called once from ``dash_app/app.py`` after the layout is set.
    """

    @app.callback(
        Output("main-chart", "figure"),
        Output("chart-status", "children"),
        Output("status-bar", "children"),
        Input("load-btn", "n_clicks"),
        State("symbol-dropdown", "value"),
        State("interval-dropdown", "value"),
        prevent_initial_call=True,
    )
    def load_chart(n_clicks: int, symbol: str, interval: str):
        """Fetch OHLCV data for *symbol* and re-render the candlestick chart.

        Falls back to a placeholder figure if data loading fails so the UI
        always shows something meaningful rather than crashing.

        Phase 1.1: uses the DataLoader from the existing backend unchanged.
        """
        if not n_clicks:
            return no_update, no_update, no_update

        try:
            # Import here to avoid requiring all dependencies at module load
            # time (e.g. when running tests that don't need a live broker).
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
                status_msg = f"No data returned for {symbol} ({interval})"
                bar_msg = f"Warning: {status_msg}"
                return fig, status_msg, bar_msg

            fig = build_candlestick_figure(df=df, symbol=symbol, show_ma=False)
            n_candles = len(df)
            status_msg = f"Loaded {n_candles:,} candles for {symbol} ({interval})"
            bar_msg = status_msg
            return fig, status_msg, bar_msg

        except Exception as exc:  # noqa: BLE001
            fig = build_candlestick_figure(df=None, symbol=symbol)
            err_msg = f"Error loading {symbol}: {exc}"
            return fig, err_msg, err_msg

    # ------------------------------------------------------------------
    # Placeholder wiring point for future phases
    # ------------------------------------------------------------------
    # Phase 2: order entry callbacks (buy/sell buttons → broker)
    # Phase 3: backtest trigger → update bt-sharpe / bt-winrate / bt-maxdd
    # Phase 4: live P&L polling → account-balance / pnl-value
