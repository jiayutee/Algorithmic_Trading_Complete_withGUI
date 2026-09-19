"""Regression tests for core.chart_builder.

Verifies that:
- build_candlestick_figure returns a valid go.Figure with correct trace type
- Empty / None input returns a placeholder figure (no crash)
- MA overlays are added only when the columns exist and show_ma=True
- overlay_signals correctly appends buy and sell scatter traces
- THEME dict contains the required color keys
- The shared module has no PyQt5 / Qt dependency

These tests are deliberately lightweight and offline (no network calls).
"""

import pandas as pd
import numpy as np
import pytest
import plotly.graph_objects as go

from core.chart_builder import (
    THEME,
    build_candlestick_figure,
    overlay_signals,
    _empty_figure,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.cumprod(1 + rng.normal(0.001, 0.01, n))
    highs   = closes * (1 + rng.uniform(0.002, 0.02, n))
    lows    = closes * (1 - rng.uniform(0.002, 0.02, n))
    opens   = closes * (1 + rng.normal(0, 0.005, n))
    idx     = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": 1_000.0},
        index=idx,
    )
    return df


# ---------------------------------------------------------------------------
# THEME
# ---------------------------------------------------------------------------

class TestTheme:
    def test_required_keys_present(self):
        required = {"bg_dark", "bg_card", "border", "text_main", "text_muted",
                    "accent", "green", "red", "orange"}
        assert required.issubset(THEME.keys())

    def test_values_are_hex_strings(self):
        for key, val in THEME.items():
            assert isinstance(val, str), f"THEME['{key}'] is not a string"
            assert val.startswith("#"), f"THEME['{key}']={val!r} is not a hex color"
            assert len(val) in (4, 7, 9), f"THEME['{key}']={val!r} has unexpected length"


# ---------------------------------------------------------------------------
# build_candlestick_figure
# ---------------------------------------------------------------------------

class TestBuildCandlestickFigure:
    def test_returns_figure_with_data(self):
        df = _make_ohlcv()
        fig = build_candlestick_figure(df, symbol="AAPL")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1, "Figure should have at least one trace"

    def test_first_trace_is_candlestick(self):
        df = _make_ohlcv()
        fig = build_candlestick_figure(df)
        assert isinstance(fig.data[0], go.Candlestick)

    def test_candlestick_length_matches_df(self):
        df = _make_ohlcv(50)
        fig = build_candlestick_figure(df)
        assert len(fig.data[0].close) == 50

    def test_none_df_returns_placeholder(self):
        fig = build_candlestick_figure(df=None)
        assert isinstance(fig, go.Figure)
        # Placeholder should have no candlestick traces
        candle_traces = [t for t in fig.data if isinstance(t, go.Candlestick)]
        assert len(candle_traces) == 0

    def test_empty_df_returns_placeholder(self):
        empty = pd.DataFrame()
        fig = build_candlestick_figure(df=empty)
        assert isinstance(fig, go.Figure)
        candle_traces = [t for t in fig.data if isinstance(t, go.Candlestick)]
        assert len(candle_traces) == 0

    def test_show_ma_false_no_ma_traces(self):
        df = _make_ohlcv()
        df["MA20"] = df["Close"].rolling(20).mean()
        fig = build_candlestick_figure(df, show_ma=False)
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) == 0

    def test_show_ma_true_adds_ma_traces(self):
        df = _make_ohlcv(60)
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()
        fig = build_candlestick_figure(df, show_ma=True)
        scatter_names = [t.name for t in fig.data if isinstance(t, go.Scatter)]
        assert "MA20" in scatter_names
        assert "MA50" in scatter_names

    def test_show_ma_true_missing_columns_no_crash(self):
        # df has no MA columns — should not crash
        df = _make_ohlcv()
        fig = build_candlestick_figure(df, show_ma=True)
        assert isinstance(fig, go.Figure)

    def test_dark_background_applied(self):
        df = _make_ohlcv()
        fig = build_candlestick_figure(df)
        assert fig.layout.paper_bgcolor == THEME["bg_dark"]
        assert fig.layout.plot_bgcolor == THEME["bg_dark"]

    def test_height_respected(self):
        df = _make_ohlcv()
        fig = build_candlestick_figure(df, height=400)
        assert fig.layout.height == 400

    def test_symbol_used_as_trace_name(self):
        df = _make_ohlcv()
        fig = build_candlestick_figure(df, symbol="TSLA")
        assert fig.data[0].name == "TSLA"


# ---------------------------------------------------------------------------
# overlay_signals
# ---------------------------------------------------------------------------

class TestOverlaySignals:
    def _base_fig(self) -> go.Figure:
        df = _make_ohlcv()
        return build_candlestick_figure(df)

    def test_buy_signal_adds_scatter(self):
        fig = self._base_fig()
        n_before = len(fig.data)
        signals = [{"type": "buy", "date": "2024-01-05", "price": 100.0}]
        overlay_signals(fig, signals)
        assert len(fig.data) == n_before + 1
        assert any(
            isinstance(t, go.Scatter) and "Buy" in (t.name or "")
            for t in fig.data
        )

    def test_sell_signal_adds_scatter(self):
        fig = self._base_fig()
        n_before = len(fig.data)
        signals = [{"type": "sell", "date": "2024-01-10", "price": 105.0}]
        overlay_signals(fig, signals)
        assert len(fig.data) == n_before + 1
        assert any(
            isinstance(t, go.Scatter) and "Sell" in (t.name or "")
            for t in fig.data
        )

    def test_buy_cover_treated_as_buy(self):
        fig = self._base_fig()
        n_before = len(fig.data)
        signals = [{"type": "buy_cover", "date": "2024-01-12", "price": 98.0}]
        overlay_signals(fig, signals)
        assert len(fig.data) == n_before + 1

    def test_sell_short_treated_as_sell(self):
        fig = self._base_fig()
        n_before = len(fig.data)
        signals = [{"type": "sell_short", "date": "2024-01-15", "price": 110.0}]
        overlay_signals(fig, signals)
        assert len(fig.data) == n_before + 1

    def test_both_signals_two_traces(self):
        fig = self._base_fig()
        n_before = len(fig.data)
        signals = [
            {"type": "buy",  "date": "2024-01-05", "price": 100.0},
            {"type": "sell", "date": "2024-01-15", "price": 108.0},
        ]
        overlay_signals(fig, signals)
        assert len(fig.data) == n_before + 2

    def test_empty_signals_no_change(self):
        fig = self._base_fig()
        n_before = len(fig.data)
        overlay_signals(fig, [])
        assert len(fig.data) == n_before

    def test_returns_same_figure_object(self):
        fig = self._base_fig()
        result = overlay_signals(fig, [])
        assert result is fig, "overlay_signals should mutate and return the same figure"

    def test_buy_marker_color_matches_theme_green(self):
        fig = self._base_fig()
        signals = [{"type": "buy", "date": "2024-01-05", "price": 100.0}]
        overlay_signals(fig, signals)
        buy_trace = next(
            t for t in fig.data
            if isinstance(t, go.Scatter) and "Buy" in (t.name or "")
        )
        assert buy_trace.marker.color == THEME["green"]

    def test_sell_marker_color_matches_theme_red(self):
        fig = self._base_fig()
        signals = [{"type": "sell", "date": "2024-01-10", "price": 105.0}]
        overlay_signals(fig, signals)
        sell_trace = next(
            t for t in fig.data
            if isinstance(t, go.Scatter) and "Sell" in (t.name or "")
        )
        assert sell_trace.marker.color == THEME["red"]


# ---------------------------------------------------------------------------
# No Qt dependency guard
# ---------------------------------------------------------------------------

def test_no_pyqt5_import_required():
    """chart_builder must be importable without PyQt5 on the path."""
    import importlib
    import sys

    # Temporarily shadow PyQt5 so it appears uninstalled
    saved = sys.modules.get("PyQt5")
    sys.modules["PyQt5"] = None  # type: ignore[assignment]
    try:
        # Force a fresh import of the module
        if "core.chart_builder" in sys.modules:
            del sys.modules["core.chart_builder"]
        mod = importlib.import_module("core.chart_builder")
        assert hasattr(mod, "build_candlestick_figure")
    finally:
        # Restore
        if saved is None:
            sys.modules.pop("PyQt5", None)
        else:
            sys.modules["PyQt5"] = saved
        # Re-import the real module for subsequent tests
        if "core.chart_builder" in sys.modules:
            del sys.modules["core.chart_builder"]
        importlib.import_module("core.chart_builder")
