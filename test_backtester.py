"""Regression tests for the end-to-end backtest pipeline.

Verifies:
- Backtester runs all 3 strategies on synthetic OHLCV data (252 bars)
- Results dict contains 'sharpe', 'max_drawdown', 'win_rate' keys
- Results dict contains 'summary' with human-readable display keys
- Chart signals list is a list (may be empty for quiet markets)
- At least one of the 3 strategies generates at least 1 trade on trending data
"""

import pytest
import pandas as pd
import numpy as np
from strategies.simple_strategies import (
    MACD_RSI_Strategy,
    EMACrossoverStrategy,
    StochasticStrategy,
)
from core.backtester import Backtester


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 252, seed: int = 42, trend: float = 0.001) -> pd.DataFrame:
    """Generate synthetic OHLCV data with optional trend."""
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.cumprod(1 + rng.normal(trend, 0.01, n))
    highs = closes * (1 + rng.uniform(0.002, 0.02, n))
    lows = closes * (1 - rng.uniform(0.002, 0.02, n))
    opens = closes * (1 + rng.normal(0, 0.005, n))
    volumes = rng.integers(100_000, 1_000_000, n).astype(float)
    idx = pd.date_range(start="2023-01-03", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=idx,
    )


def _run_backtest(strategy_cls, df: pd.DataFrame, cash: float = 100_000) -> dict:
    """Run a single strategy through the Backtester and return the results dict."""
    b = Backtester()
    b.add_data(df.copy())
    b.add_strategy(strategy_cls)
    # Use benchmark_ticker=None-style workaround: pass an unlikely ticker so
    # alpha/beta gracefully returns (0,0) without blocking on network.
    return b.run_backtest(cash=cash, benchmark_ticker="SPY")


# ---------------------------------------------------------------------------
# Tests: result dict structure
# ---------------------------------------------------------------------------

class TestBacktesterResultStructure:
    """Verify that the results dict always has the required top-level keys."""

    @pytest.mark.parametrize("strategy_cls", [
        MACD_RSI_Strategy,
        EMACrossoverStrategy,
        StochasticStrategy,
    ])
    def test_result_has_sharpe_key(self, strategy_cls):
        df = _make_ohlcv(252)
        results = _run_backtest(strategy_cls, df)
        assert "error" not in results, f"Backtest errored: {results.get('error')}"
        assert "sharpe" in results, f"Missing 'sharpe' key in results for {strategy_cls.__name__}"

    @pytest.mark.parametrize("strategy_cls", [
        MACD_RSI_Strategy,
        EMACrossoverStrategy,
        StochasticStrategy,
    ])
    def test_result_has_max_drawdown_key(self, strategy_cls):
        df = _make_ohlcv(252)
        results = _run_backtest(strategy_cls, df)
        assert "error" not in results, f"Backtest errored: {results.get('error')}"
        assert "max_drawdown" in results, f"Missing 'max_drawdown' key in results for {strategy_cls.__name__}"

    @pytest.mark.parametrize("strategy_cls", [
        MACD_RSI_Strategy,
        EMACrossoverStrategy,
        StochasticStrategy,
    ])
    def test_result_has_win_rate_key(self, strategy_cls):
        df = _make_ohlcv(252)
        results = _run_backtest(strategy_cls, df)
        assert "error" not in results, f"Backtest errored: {results.get('error')}"
        assert "win_rate" in results, f"Missing 'win_rate' key in results for {strategy_cls.__name__}"

    @pytest.mark.parametrize("strategy_cls", [
        MACD_RSI_Strategy,
        EMACrossoverStrategy,
        StochasticStrategy,
    ])
    def test_result_has_signals_list(self, strategy_cls):
        df = _make_ohlcv(252)
        results = _run_backtest(strategy_cls, df)
        assert "error" not in results, f"Backtest errored: {results.get('error')}"
        assert isinstance(results.get("signals"), list), "signals must be a list"

    @pytest.mark.parametrize("strategy_cls", [
        MACD_RSI_Strategy,
        EMACrossoverStrategy,
        StochasticStrategy,
    ])
    def test_summary_dict_has_display_keys(self, strategy_cls):
        """The summary sub-dict must have keys for the GUI status bar and statistics window."""
        df = _make_ohlcv(252)
        results = _run_backtest(strategy_cls, df)
        assert "error" not in results, f"Backtest errored: {results.get('error')}"
        summary = results.get("summary", {})
        for key in ("Sharpe Ratio", "Max Drawdown (%)", "Win Rate", "Final Value"):
            assert key in summary, f"Missing '{key}' in summary for {strategy_cls.__name__}"


# ---------------------------------------------------------------------------
# Tests: metric types and ranges
# ---------------------------------------------------------------------------

class TestBacktesterMetricTypes:
    """Verify that metric values have sane types and ranges."""

    @pytest.mark.parametrize("strategy_cls", [
        MACD_RSI_Strategy,
        EMACrossoverStrategy,
        StochasticStrategy,
    ])
    def test_sharpe_is_numeric(self, strategy_cls):
        df = _make_ohlcv(252)
        results = _run_backtest(strategy_cls, df)
        sharpe = results.get("sharpe", None)
        assert isinstance(sharpe, (int, float)), f"sharpe must be numeric, got {type(sharpe)}"

    @pytest.mark.parametrize("strategy_cls", [
        MACD_RSI_Strategy,
        EMACrossoverStrategy,
        StochasticStrategy,
    ])
    def test_max_drawdown_non_negative(self, strategy_cls):
        df = _make_ohlcv(252)
        results = _run_backtest(strategy_cls, df)
        dd = results.get("max_drawdown", -1)
        assert isinstance(dd, (int, float)), f"max_drawdown must be numeric, got {type(dd)}"
        assert dd >= 0, f"max_drawdown must be >= 0, got {dd}"

    @pytest.mark.parametrize("strategy_cls", [
        MACD_RSI_Strategy,
        EMACrossoverStrategy,
        StochasticStrategy,
    ])
    def test_win_rate_in_valid_range(self, strategy_cls):
        df = _make_ohlcv(252)
        results = _run_backtest(strategy_cls, df)
        wr = results.get("win_rate", -1)
        assert isinstance(wr, (int, float)), f"win_rate must be numeric, got {type(wr)}"
        assert 0.0 <= wr <= 100.0, f"win_rate must be between 0 and 100, got {wr}"


# ---------------------------------------------------------------------------
# Tests: trade generation on trending synthetic data
# ---------------------------------------------------------------------------

class TestBacktesterTradeGeneration:
    """Verify at least one strategy produces at least 1 trade on strongly trending data."""

    def test_ema_crossover_generates_trades_on_trending_data(self):
        """EMA crossover should reliably generate trades on strong uptrend data."""
        # Build a downtrend then strong uptrend to trigger EMA crossover
        rng = np.random.default_rng(22)
        down = [100.0]
        for _ in range(40):
            down.append(down[-1] * 0.99)
        up = [down[-1]]
        for _ in range(212):
            up.append(up[-1] * 1.008)
        closes = np.array(down + up[1:], dtype=float)
        n = len(closes)
        highs = closes * (1 + rng.uniform(0.002, 0.015, n))
        lows = closes * (1 - rng.uniform(0.002, 0.015, n))
        opens = closes * (1 + rng.normal(0, 0.003, n))
        volumes = np.full(n, 500_000.0)
        idx = pd.date_range(start="2023-01-03", periods=n, freq="B")
        df = pd.DataFrame(
            {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
            index=idx,
        )
        results = _run_backtest(EMACrossoverStrategy, df)
        assert "error" not in results, f"Backtest errored: {results.get('error')}"
        signals = results.get("signals", [])
        assert len(signals) >= 1, (
            f"Expected at least 1 signal from EMA Crossover on trending data; got {len(signals)}"
        )

    def test_macd_rsi_generates_trades_on_oversold_recovery(self):
        """MACD+RSI should generate at least 1 trade on a sharp drop then recovery."""
        rng = np.random.default_rng(0)
        drop = [100.0]
        for _ in range(39):
            drop.append(drop[-1] * 0.97)
        recovery = [drop[-1]]
        for _ in range(213):
            recovery.append(recovery[-1] * 1.008)
        closes = np.array(drop + recovery[1:], dtype=float)
        n = len(closes)
        highs = closes * (1 + rng.uniform(0.002, 0.015, n))
        lows = closes * (1 - rng.uniform(0.002, 0.015, n))
        opens = closes * (1 + rng.normal(0, 0.003, n))
        volumes = np.full(n, 500_000.0)
        idx = pd.date_range(start="2023-01-03", periods=n, freq="B")
        df = pd.DataFrame(
            {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
            index=idx,
        )
        results = _run_backtest(MACD_RSI_Strategy, df)
        assert "error" not in results, f"Backtest errored: {results.get('error')}"
        signals = results.get("signals", [])
        assert len(signals) >= 1, (
            f"Expected at least 1 signal from MACD+RSI on oversold+recovery data; got {len(signals)}"
        )

    def test_stochastic_generates_trades_on_oversold_bounce(self):
        """Stochastic should generate at least 1 trade on a sharp drop then bounce."""
        rng = np.random.default_rng(44)
        normal = [100.0]
        for _ in range(30):
            normal.append(normal[-1] * (1 + rng.normal(0, 0.005)))
        drop = [normal[-1]]
        for _ in range(20):
            drop.append(drop[-1] * 0.96)
        recovery = [drop[-1]]
        for _ in range(202):
            recovery.append(recovery[-1] * 1.006)
        closes = np.array(normal + drop[1:] + recovery[1:], dtype=float)
        n = len(closes)
        highs = closes * (1 + rng.uniform(0.002, 0.015, n))
        lows = closes * (1 - rng.uniform(0.002, 0.015, n))
        opens = closes * (1 + rng.normal(0, 0.003, n))
        volumes = np.full(n, 500_000.0)
        idx = pd.date_range(start="2023-01-03", periods=n, freq="B")
        df = pd.DataFrame(
            {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
            index=idx,
        )
        results = _run_backtest(StochasticStrategy, df)
        assert "error" not in results, f"Backtest errored: {results.get('error')}"
        signals = results.get("signals", [])
        assert len(signals) >= 1, (
            f"Expected at least 1 signal from Stochastic on oversold bounce data; got {len(signals)}"
        )
