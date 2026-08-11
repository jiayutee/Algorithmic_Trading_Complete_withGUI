"""
test_dash_live_price.py

Unit tests for Phase 1.2: live-price chart integration helpers.

Tests are deliberately network-free and Dash-server-free:
  - is_crypto_symbol()  — symbol classification heuristic
  - add_live_tick_trace() — mutates a go.Figure to append the live-tick trace
  - _badge_connecting() / _badge_with_price() — badge text helpers

WebSocket / LivePriceService behaviour is already tested by the existing
LivePriceService test suite; those tests are not duplicated here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from core.chart_builder import (
    add_live_tick_trace,
    build_candlestick_figure,
    is_crypto_symbol,
)


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    closes = 100.0 * np.cumprod(1 + rng.normal(0.001, 0.01, n))
    highs = closes * (1 + rng.uniform(0.002, 0.02, n))
    lows = closes * (1 - rng.uniform(0.002, 0.02, n))
    opens = closes * (1 + rng.normal(0, 0.005, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": 1_000.0},
        index=idx,
    )


# ---------------------------------------------------------------------------
# is_crypto_symbol
# ---------------------------------------------------------------------------

class TestIsCryptoSymbol:
    """Symbol-classification heuristic used throughout the codebase."""

    # --- crypto symbols that must return True ---
    @pytest.mark.parametrize("symbol", [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "ADAUSDT",
        "btcusdt",       # lowercase
        "BtcUsdt",       # mixed case
        "DOGEUSDT",
    ])
    def test_crypto_symbols_return_true(self, symbol: str):
        assert is_crypto_symbol(symbol) is True, (
            f"Expected is_crypto_symbol({symbol!r}) == True"
        )

    # --- equity / index symbols that must return False ---
    @pytest.mark.parametrize("symbol", [
        "AAPL",
        "TSLA",
        "SPY",
        "QQQ",
        "MSFT",
        "GOOG",
        "BTC-USD",   # Yahoo Finance crypto format — no USDT
        "ETH-USD",
        "",          # empty string
    ])
    def test_equity_symbols_return_false(self, symbol: str):
        assert is_crypto_symbol(symbol) is False, (
            f"Expected is_crypto_symbol({symbol!r}) == False"
        )

    def test_returns_bool(self):
        """Return type must be bool, not a truthy object."""
        result = is_crypto_symbol("BTCUSDT")
        assert isinstance(result, bool)

    def test_case_insensitive(self):
        """Upper and lower case should be treated identically."""
        assert is_crypto_symbol("BTCUSDT") == is_crypto_symbol("btcusdt")
        assert is_crypto_symbol("AAPL") == is_crypto_symbol("aapl")


# ---------------------------------------------------------------------------
# add_live_tick_trace
# ---------------------------------------------------------------------------

class TestAddLiveTickTrace:
    """add_live_tick_trace appends a dedicated scatter trace to the figure."""

    def test_appends_one_trace_to_candlestick_figure(self):
        df = _make_ohlcv()
        fig = build_candlestick_figure(df, symbol="AAPL")
        n_before = len(fig.data)
        result = add_live_tick_trace(fig)
        assert len(result.data) == n_before + 1

    def test_appends_one_trace_to_empty_placeholder(self):
        """Works on a placeholder figure (no candlestick body)."""
        fig = build_candlestick_figure(df=None)
        n_before = len(fig.data)
        result = add_live_tick_trace(fig)
        assert len(result.data) == n_before + 1

    def test_last_trace_is_scatter(self):
        df = _make_ohlcv()
        fig = build_candlestick_figure(df)
        add_live_tick_trace(fig)
        assert isinstance(fig.data[-1], go.Scatter)

    def test_last_trace_starts_empty(self):
        """The live-tick trace must start with empty x and y arrays."""
        df = _make_ohlcv()
        fig = build_candlestick_figure(df)
        add_live_tick_trace(fig)
        trace = fig.data[-1]
        # Plotly stores empty lists as () or []
        assert len(trace.x) == 0, "live-tick x should be empty on init"
        assert len(trace.y) == 0, "live-tick y should be empty on init"

    def test_last_trace_is_named_live(self):
        df = _make_ohlcv()
        fig = build_candlestick_figure(df)
        add_live_tick_trace(fig)
        assert fig.data[-1].name == "Live"

    def test_returns_same_figure_object(self):
        """add_live_tick_trace must mutate and return the same figure."""
        df = _make_ohlcv()
        fig = build_candlestick_figure(df)
        returned = add_live_tick_trace(fig)
        assert returned is fig

    def test_showlegend_false(self):
        """The live-tick trace must not appear in the chart legend."""
        df = _make_ohlcv()
        fig = build_candlestick_figure(df)
        add_live_tick_trace(fig)
        assert fig.data[-1].showlegend is False

    def test_marker_mode(self):
        df = _make_ohlcv()
        fig = build_candlestick_figure(df)
        add_live_tick_trace(fig)
        assert "markers" in (fig.data[-1].mode or "")

    def test_idempotent_trace_index(self):
        """After two consecutive add_live_tick_trace calls the last trace is
        always the most recently added one — callers must not double-call."""
        df = _make_ohlcv()
        fig = build_candlestick_figure(df)
        add_live_tick_trace(fig)
        add_live_tick_trace(fig)
        # Both calls add a trace; second is at [-1], first at [-2]
        assert isinstance(fig.data[-1], go.Scatter)
        assert isinstance(fig.data[-2], go.Scatter)
        assert len(fig.data) >= 3  # candlestick + 2 live traces


# ---------------------------------------------------------------------------
# Badge helpers (imported from callbacks module)
# ---------------------------------------------------------------------------

class TestBadgeHelpers:
    """Tests for the badge text helper functions in dash_app/callbacks.py."""

    def setup_method(self):
        from dash_app.callbacks import _badge_connecting, _badge_with_price
        self._badge_connecting = _badge_connecting
        self._badge_with_price = _badge_with_price

    def test_badge_connecting_crypto(self):
        badge = self._badge_connecting("BTCUSDT")
        assert "🟢" in badge
        assert "Live" in badge

    def test_badge_connecting_equity(self):
        badge = self._badge_connecting("AAPL")
        assert "🟡" in badge
        assert "real-time" in badge.lower()

    def test_badge_with_price_crypto(self):
        badge = self._badge_with_price("BTCUSDT", 65_432.10)
        assert "🟢" in badge
        assert "65,432" in badge

    def test_badge_with_price_equity(self):
        badge = self._badge_with_price("AAPL", 213.45)
        assert "🟡" in badge
        assert "213" in badge

    def test_badge_with_price_small_crypto(self):
        """Small prices (< 10_000) should show 4 decimal places."""
        badge = self._badge_with_price("SOLUSDT", 150.1234)
        assert "🟢" in badge
        assert "150.1234" in badge

    def test_badge_with_price_large_crypto(self):
        """Large prices (≥ 10_000) should show 2 decimal places."""
        badge = self._badge_with_price("BTCUSDT", 65_432.10)
        assert "🟢" in badge
        # Should have 2 decimal places, not 4
        assert "65,432.10" in badge


# ---------------------------------------------------------------------------
# Equity throttle counter logic
# ---------------------------------------------------------------------------

class TestEquityThrottle:
    """Smoke-tests the _EQUITY_POLL_EVERY_N_TICKS constant."""

    def test_constant_is_positive_int(self):
        from dash_app.callbacks import _EQUITY_POLL_EVERY_N_TICKS
        assert isinstance(_EQUITY_POLL_EVERY_N_TICKS, int)
        assert _EQUITY_POLL_EVERY_N_TICKS > 0

    def test_effective_poll_interval_is_reasonable(self):
        """At 1 500 ms interval the effective REST cadence must be > 5 s."""
        from dash_app.callbacks import _EQUITY_POLL_EVERY_N_TICKS
        effective_ms = _EQUITY_POLL_EVERY_N_TICKS * 1500
        assert effective_ms >= 5_000, (
            f"Effective equity poll interval {effective_ms}ms is too aggressive"
        )
