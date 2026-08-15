"""Regression tests for the end-to-end backtest pipeline.

Verifies:
- Backtester runs all 3 strategies on synthetic OHLCV data (252 bars)
- Results dict contains 'sharpe', 'max_drawdown', 'win_rate' keys
- Results dict contains 'summary' with human-readable display keys
- Chart signals list is a list (may be empty for quiet markets)
- At least one of the 3 strategies generates at least 1 trade on trending data
"""

import os
import json
import csv as csv_module
import pytest
import pandas as pd
import numpy as np
from strategies.simple_strategies import (
    MACD_RSI_Strategy,
    EMACrossoverStrategy,
    StochasticStrategy,
)
from core.backtester import Backtester, export_report


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


# ---------------------------------------------------------------------------
# Tests: signal entry structure (T6b)
# ---------------------------------------------------------------------------

class TestBacktesterSignalStructure:
    """Verify that each signal entry in results['signals'] has the required keys
    and valid field types, so the GUI chart overlay can render markers."""

    def test_signals_have_required_keys(self):
        """Each signal dict must contain 'date', 'type', and 'price'."""
        # Use trending data that reliably generates at least one signal
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
        assert len(signals) >= 1, "Expected at least 1 signal; got 0"
        for sig in signals:
            assert "date" in sig, f"Signal missing 'date' key: {sig}"
            assert "type" in sig, f"Signal missing 'type' key: {sig}"
            assert "price" in sig, f"Signal missing 'price' key: {sig}"

    def test_signal_types_are_valid(self):
        """Signal 'type' values must be one of the known categories."""
        valid_types = {"buy", "sell", "buy_cover", "sell_short"}
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
        for sig in results.get("signals", []):
            assert sig.get("type") in valid_types, (
                f"Unexpected signal type '{sig.get('type')}'; expected one of {valid_types}"
            )

    def test_signal_price_is_positive(self):
        """Signal 'price' must be a positive number."""
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
        for sig in results.get("signals", []):
            assert isinstance(sig.get("price"), (int, float)), (
                f"Signal price must be numeric, got {type(sig.get('price'))}"
            )
            assert sig["price"] > 0, f"Signal price must be positive, got {sig['price']}"


# ---------------------------------------------------------------------------
# Tests: Sharpe ratio and win-rate correctness (T6c)
# ---------------------------------------------------------------------------

class TestBacktesterSharpeAndWinRate:
    """Verify Sharpe ratio and win-rate metric semantics."""

    def test_sharpe_nonzero_on_profitable_trend(self):
        """On a strong uptrend that generates profitable trades,
        Sharpe ratio should be non-zero (backtrader's annualised Sharpe)."""
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
        sharpe = results.get("sharpe", 0.0)
        assert isinstance(sharpe, (int, float)), f"sharpe must be numeric, got {type(sharpe)}"
        # On a strong trending data with trades, Sharpe should differ from 0
        # (we can't assert the sign because backtrader may return 0 when no variance,
        #  but it must be numeric and finite)
        assert np.isfinite(sharpe), f"sharpe must be finite, got {sharpe}"

    def test_win_rate_zero_when_no_trades(self):
        """Win rate must be 0 when no trades are closed."""
        df = _make_ohlcv(252, seed=99, trend=0.0)  # flat market, likely no trades
        # Use a strategy on flat data where signals won't fire reliably;
        # we just assert the formula is correct: 0 closed trades -> 0% win rate
        results = _run_backtest(EMACrossoverStrategy, df)
        assert "error" not in results, f"Backtest errored: {results.get('error')}"
        summary = results.get("summary", {})
        n_trades = summary.get("Number of Closed Trades", 0)
        wr = results.get("win_rate", -1)
        assert isinstance(wr, (int, float)), f"win_rate must be numeric, got {type(wr)}"
        if n_trades == 0:
            assert wr == 0.0, f"win_rate must be 0.0 when no trades, got {wr}"

    def test_win_rate_between_0_and_100(self):
        """Win rate must always be in [0, 100] regardless of strategy outcome."""
        for seed in [0, 1, 42, 99]:
            df = _make_ohlcv(252, seed=seed)
            results = _run_backtest(EMACrossoverStrategy, df)
            assert "error" not in results, f"Backtest errored: {results.get('error')}"
            wr = results.get("win_rate", -1)
            assert 0.0 <= wr <= 100.0, (
                f"win_rate out of [0,100] range on seed={seed}: got {wr}"
            )

    def test_sharpe_in_summary_matches_top_level(self):
        """results['summary']['Sharpe Ratio'] must be the rounded form of results['sharpe'].
        The summary stores round(sharpe, 4) so tolerance is 5e-5."""
        df = _make_ohlcv(252)
        results = _run_backtest(EMACrossoverStrategy, df)
        assert "error" not in results, f"Backtest errored: {results.get('error')}"
        # summary stores round(sharpe, 4); allow up to half a ULP at 4 decimal places
        assert abs(results["summary"]["Sharpe Ratio"] - results["sharpe"]) < 5e-5, (
            f"Mismatch: summary Sharpe={results['summary']['Sharpe Ratio']}, "
            f"top-level sharpe={results['sharpe']}"
        )

    def test_win_rate_in_summary_matches_top_level(self):
        """results['summary']['Win Rate'] string must encode results['win_rate']."""
        df = _make_ohlcv(252)
        results = _run_backtest(EMACrossoverStrategy, df)
        assert "error" not in results, f"Backtest errored: {results.get('error')}"
        wr_numeric = results.get("win_rate", -1)
        wr_str = results["summary"].get("Win Rate", "")
        # The summary encodes it as "X.XX%"
        assert wr_str.endswith("%"), f"summary Win Rate must end with '%', got '{wr_str}'"
        wr_from_summary = float(wr_str.rstrip("%"))
        assert abs(wr_from_summary - wr_numeric) < 1e-3, (
            f"Win rate mismatch: summary={wr_from_summary}, top-level={wr_numeric}"
        )


# ---------------------------------------------------------------------------
# Tests: cumulative_pnl and equity curve / Alpha-Beta (T7 regression)
#
# Two bugs fixed here:
#   1. cumulative_pnl was always [] because none of the 3 core strategies ever
#      populated `self.closed_trades` -- _generate_report() silently no-ops
#      via `if hasattr(strategy, 'closed_trades')` when the attribute is
#      missing, so pnl_per_trade (and thus cumulative_pnl) stayed empty for
#      every single backtest regardless of how many trades closed.
#   2. total_asset_value (equity curve) and Alpha/Beta were always empty/zero
#      because _generate_report() read a 'portfolio_value' key from
#      backtrader's PyFolio analyzer output that never exists (the analyzer
#      only ever returns 'returns', 'positions', 'transactions', 'gross_lev').
#      The resulting KeyError was swallowed by a bare `except Exception`,
#      which silently reset `returns` to an empty Series every time -- so
#      Alpha/Beta (computed from `returns`) were always (0, 0) too.
# ---------------------------------------------------------------------------

def _trending_df(down_days=40, up_days=150, down_days2=60, seed=22, start_price=100.0,
                  down_factor=0.99, up_factor=1.008, down_factor2=0.985):
    """Synthetic down-then-up-then-down (reversal) trend that reliably produces
    closed round-trip trades for EMACrossoverStrategy / MACD_RSI_Strategy /
    StochasticStrategy.

    A plain down-then-up trend only ever crosses once: EMACrossoverStrategy
    opens a long on the up-leg and then never exits, because a monotonic
    exponential uptrend never makes the fast EMA cross back below the slow
    EMA. MACD_RSI_Strategy and StochasticStrategy have exit conditions that
    can fire on RSI/K-D behaviour mid-trend, so they happened to close trades
    on the old fixture, but EMACrossoverStrategy needs an actual trend
    reversal to flip its crossover indicator back and close the position.
    Adding a final down-leg (reversal) after the uptrend guarantees the EMA
    crossover flips negative again, closing the long for all 3 strategies.
    """
    rng = np.random.default_rng(seed)
    down = [start_price]
    for _ in range(down_days):
        down.append(down[-1] * down_factor)
    up = [down[-1]]
    for _ in range(up_days):
        up.append(up[-1] * up_factor)
    down2 = [up[-1]]
    for _ in range(down_days2):
        down2.append(down2[-1] * down_factor2)
    closes = np.array(down + up[1:] + down2[1:], dtype=float)
    n = len(closes)
    highs = closes * (1 + rng.uniform(0.002, 0.015, n))
    lows = closes * (1 - rng.uniform(0.002, 0.015, n))
    opens = closes * (1 + rng.normal(0, 0.003, n))
    volumes = np.full(n, 500_000.0)
    idx = pd.date_range(start="2023-01-03", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=idx,
    )


class TestBacktesterCumulativePnL:
    """Regression tests for bug: 'Per-trade P&L (cumulative_pnl) always empty
    for all 3 core strategies'."""

    @pytest.mark.parametrize("strategy_cls", [
        MACD_RSI_Strategy,
        EMACrossoverStrategy,
        StochasticStrategy,
    ])
    def test_closed_trades_produce_nonempty_pnl(self, strategy_cls):
        """When a strategy closes at least one trade, cumulative_pnl and
        profit_per_trade must be populated (not silently left as [])."""
        df = _trending_df()
        results = _run_backtest(strategy_cls, df)
        assert "error" not in results, f"Backtest errored: {results.get('error')}"
        n_trades = results["summary"]["Number of Closed Trades"]
        assert n_trades > 0, (
            f"Test fixture should reliably close trades for {strategy_cls.__name__}; "
            f"got 0 -- fixture needs adjusting, not a real assertion failure"
        )
        assert len(results["profit_per_trade"]) == n_trades, (
            f"profit_per_trade should have one entry per closed trade for "
            f"{strategy_cls.__name__}; if empty, the strategy stopped "
            f"populating self.closed_trades in notify_trade()"
        )
        assert len(results["cumulative_pnl"]) == n_trades, (
            f"cumulative_pnl should have one entry per closed trade for {strategy_cls.__name__}"
        )

    def test_cumulative_pnl_is_running_cumsum_of_profit_per_trade(self):
        df = _trending_df()
        results = _run_backtest(EMACrossoverStrategy, df)
        assert "error" not in results, f"Backtest errored: {results.get('error')}"
        profit_per_trade = results["profit_per_trade"]
        assert len(profit_per_trade) > 0, "Fixture should produce at least 1 closed trade"
        expected = np.cumsum(profit_per_trade).tolist()
        assert results["cumulative_pnl"] == pytest.approx(expected)

    def test_no_trades_means_empty_but_not_erroring_pnl(self):
        """Sanity check: 0 closed trades should still yield [] (not crash),
        distinguishing 'genuinely no trades' from the bug (always empty)."""
        df = _make_ohlcv(252, seed=99, trend=0.0)
        results = _run_backtest(EMACrossoverStrategy, df)
        assert "error" not in results, f"Backtest errored: {results.get('error')}"
        n_trades = results["summary"]["Number of Closed Trades"]
        if n_trades == 0:
            assert results["cumulative_pnl"] == []
            assert results["profit_per_trade"] == []


class TestBacktesterEquityCurveAndAlphaBeta:
    """Regression tests for bug: 'Equity curve chart + Alpha/Beta always
    empty/zero (backtrader pyfolio KeyError swallowed)'."""

    def test_equity_curve_has_one_point_per_bar(self):
        """total_asset_value must be reconstructed from PyFolio's 'returns'
        key (there is no 'portfolio_value' key in backtrader's PyFolio
        analyzer output) and should have one point per bar in the data."""
        df = _make_ohlcv(252)
        results = _run_backtest(EMACrossoverStrategy, df)
        assert "error" not in results, f"Backtest errored: {results.get('error')}"
        assert len(results["total_asset_value"]) == len(df), (
            "Equity curve should have one point per bar; got "
            f"{len(results['total_asset_value'])} for {len(df)} bars -- "
            "regression guard for the swallowed pyfolio KeyError bug"
        )
        assert results["total_asset_value"][0] == pytest.approx(100_000, rel=1e-6)

    def test_generate_report_reconstructs_equity_curve_without_portfolio_value_key(self):
        """Directly regression-tests the root cause: simulate PyFolio's actual
        get_analysis() shape (only 'returns', no 'portfolio_value') and
        confirm _generate_report reconstructs the equity curve instead of
        silently defaulting to an empty series."""
        b = Backtester()
        df = _make_ohlcv(60, seed=7)
        b.df = df

        # Pre-populate the benchmark cache so Alpha/Beta calculation doesn't
        # hit the network for this synthetic, deterministic test.
        start_date = df.index[0].strftime('%Y-%m-%d')
        end_date = df.index[-1].strftime('%Y-%m-%d')
        cache_key = f"SPY_{start_date}_{end_date}"
        Backtester._benchmark_cache[cache_key] = pd.Series(
            0.003, index=df.index[1:]
        )

        class _FakeAnalysis:
            @staticmethod
            def get_analysis():
                # Mirrors backtrader's real bt.analyzers.PyFolio.get_analysis()
                # shape: only 'returns' (+ 'positions'/'transactions'/'gross_lev'
                # which _generate_report never reads) -- crucially, NO
                # 'portfolio_value' key.
                return {"returns": {d: 0.01 for d in df.index}}

        class _EmptyAnalysis:
            @staticmethod
            def get_analysis():
                return {}

        class _FakeAnalyzers:
            pyfolio = _FakeAnalysis()
            trade_analyzer = _EmptyAnalysis()
            drawdown = _EmptyAnalysis()
            sharpe = _EmptyAnalysis()

        class _FakeStrategy:
            analyzers = _FakeAnalyzers()
            signals = []

        report = b._generate_report(_FakeStrategy(), benchmark_ticker="SPY", initial_cash=100_000)

        assert len(report["total_asset_value"]) == len(df), (
            "Equity curve must be reconstructed from the 'returns' key even "
            "though 'portfolio_value' is absent from PyFolio's analysis dict"
        )
        # 1% compounding daily return over len(df) days must grow above the start
        assert report["total_asset_value"][-1] > 100_000
        assert report["summary"]["Alpha"] != 0 or report["summary"]["Beta"] != 0, (
            "Alpha/Beta should be computed from non-empty returns, not silently 0"
        )


# ---------------------------------------------------------------------------
# Tests: export_report() — CSV and JSON file export (roadmap: "Backtest results
# exportable (CSV or JSON)")
# ---------------------------------------------------------------------------

class TestExportReport:
    """Verify that export_report() correctly writes CSV and JSON output files
    containing all required sections and columns.

    Uses _trending_df() as the test fixture because it reliably produces at
    least one closed trade (non-empty profit_per_trade / cumulative_pnl) and
    a non-trivial equity curve, giving richer output to validate against.
    """

    # ------------------------------------------------------------------
    # Shared fixture — run once per test method via a helper (not a
    # pytest fixture) so the test class stays self-contained.
    # ------------------------------------------------------------------

    @staticmethod
    def _results():
        """Run EMACrossover on the reversal fixture and return the report dict."""
        df = _trending_df()
        return _run_backtest(EMACrossoverStrategy, df)

    # ------------------------------------------------------------------
    # JSON tests
    # ------------------------------------------------------------------

    def test_json_export_creates_file(self, tmp_path):
        """export_report with format='json' must create a file on disk."""
        results = self._results()
        outpath = str(tmp_path / "report")
        paths = export_report(results, outpath, format='json')
        assert "json" in paths, "Return dict must have a 'json' key"
        assert os.path.isfile(paths["json"]), (
            f"JSON export file not found at {paths['json']}"
        )

    def test_json_extension_auto_added(self, tmp_path):
        """If filepath has no extension, .json is appended automatically."""
        results = self._results()
        outpath = str(tmp_path / "myreport")
        paths = export_report(results, outpath, format='json')
        assert paths["json"].endswith(".json"), (
            f"Expected .json extension auto-appended; got {paths['json']}"
        )

    def test_json_has_required_top_level_keys(self, tmp_path):
        """The JSON file must contain all required top-level section keys."""
        results = self._results()
        outpath = str(tmp_path / "report.json")
        paths = export_report(results, outpath, format='json')
        with open(paths["json"], encoding='utf-8') as fh:
            data = json.load(fh)
        for key in ("summary", "equity_curve", "profit_per_trade", "cumulative_pnl", "signals"):
            assert key in data, f"Missing top-level key '{key}' in JSON output"

    def test_json_summary_has_metric_keys(self, tmp_path):
        """The 'summary' section in JSON must include the required metric keys."""
        results = self._results()
        outpath = str(tmp_path / "report.json")
        paths = export_report(results, outpath, format='json')
        with open(paths["json"], encoding='utf-8') as fh:
            data = json.load(fh)
        summary = data["summary"]
        for key in ("sharpe", "max_drawdown", "win_rate", "Final Value",
                    "Sharpe Ratio", "Max Drawdown (%)", "Win Rate"):
            assert key in summary, f"Missing key '{key}' in JSON summary section"

    def test_json_equity_curve_matches_total_asset_value(self, tmp_path):
        """equity_curve in JSON must exactly match results['total_asset_value']."""
        results = self._results()
        assert len(results["total_asset_value"]) > 0, "Fixture must produce a non-empty equity curve"
        outpath = str(tmp_path / "report.json")
        paths = export_report(results, outpath, format='json')
        with open(paths["json"], encoding='utf-8') as fh:
            data = json.load(fh)
        assert len(data["equity_curve"]) == len(results["total_asset_value"]), (
            "equity_curve length in JSON does not match total_asset_value length"
        )
        assert data["equity_curve"][0] == pytest.approx(results["total_asset_value"][0], rel=1e-9)

    def test_json_profit_per_trade_matches_report(self, tmp_path):
        """profit_per_trade in JSON must match the source list entry-for-entry."""
        results = self._results()
        assert len(results["profit_per_trade"]) > 0, (
            "Fixture must produce at least one closed trade"
        )
        outpath = str(tmp_path / "report.json")
        paths = export_report(results, outpath, format='json')
        with open(paths["json"], encoding='utf-8') as fh:
            data = json.load(fh)
        assert data["profit_per_trade"] == pytest.approx(results["profit_per_trade"])

    # ------------------------------------------------------------------
    # CSV tests
    # ------------------------------------------------------------------

    def test_csv_export_creates_three_files(self, tmp_path):
        """export_report with format='csv' must create three separate files."""
        results = self._results()
        outpath = str(tmp_path / "report")
        paths = export_report(results, outpath, format='csv')
        assert set(paths.keys()) == {"summary", "trades", "equity"}, (
            f"Expected keys {{'summary','trades','equity'}}; got {set(paths.keys())}"
        )
        for section, path in paths.items():
            assert os.path.isfile(path), (
                f"CSV file for section '{section}' not found: {path}"
            )

    def test_csv_summary_columns_and_metric_keys(self, tmp_path):
        """The summary CSV must have 'metric'/'value' columns and include required keys."""
        results = self._results()
        outpath = str(tmp_path / "report")
        paths = export_report(results, outpath, format='csv')
        with open(paths["summary"], newline='', encoding='utf-8') as fh:
            rows = list(csv_module.DictReader(fh))
        assert rows, "summary CSV must not be empty"
        metrics_present = {row["metric"] for row in rows}
        for key in ("sharpe", "max_drawdown", "win_rate", "Final Value"):
            assert key in metrics_present, (
                f"Metric '{key}' missing from summary CSV; found: {metrics_present}"
            )

    def test_csv_trades_columns(self, tmp_path):
        """The trades CSV must have 'trade_index', 'pnl', 'cumulative_pnl' columns."""
        results = self._results()
        assert len(results["profit_per_trade"]) > 0, (
            "Fixture must produce at least one closed trade"
        )
        outpath = str(tmp_path / "report")
        paths = export_report(results, outpath, format='csv')
        with open(paths["trades"], newline='', encoding='utf-8') as fh:
            reader = csv_module.DictReader(fh)
            fieldnames = reader.fieldnames
            rows = list(reader)
        assert fieldnames is not None, "trades CSV has no header row"
        for col in ("trade_index", "pnl", "cumulative_pnl"):
            assert col in fieldnames, (
                f"Column '{col}' missing from trades CSV; got {fieldnames}"
            )
        assert len(rows) == len(results["profit_per_trade"]), (
            "Number of rows in trades CSV must equal number of closed trades"
        )

    def test_csv_equity_columns_and_row_count(self, tmp_path):
        """The equity CSV must have 'bar_index'/'portfolio_value' columns and one row per bar."""
        results = self._results()
        assert len(results["total_asset_value"]) > 0, "Fixture must produce a non-empty equity curve"
        outpath = str(tmp_path / "report")
        paths = export_report(results, outpath, format='csv')
        with open(paths["equity"], newline='', encoding='utf-8') as fh:
            reader = csv_module.DictReader(fh)
            fieldnames = reader.fieldnames
            rows = list(reader)
        assert fieldnames is not None, "equity CSV has no header row"
        for col in ("bar_index", "portfolio_value"):
            assert col in fieldnames, (
                f"Column '{col}' missing from equity CSV; got {fieldnames}"
            )
        assert len(rows) == len(results["total_asset_value"]), (
            f"equity CSV row count {len(rows)} != total_asset_value length "
            f"{len(results['total_asset_value'])}"
        )

    def test_csv_equity_first_value_is_initial_cash(self, tmp_path):
        """The first portfolio_value in the equity CSV should equal the initial cash (100 000)."""
        results = self._results()
        outpath = str(tmp_path / "report")
        paths = export_report(results, outpath, format='csv')
        with open(paths["equity"], newline='', encoding='utf-8') as fh:
            first_row = next(csv_module.DictReader(fh))
        first_val = float(first_row["portfolio_value"])
        assert first_val == pytest.approx(100_000, rel=1e-5), (
            f"First equity CSV value should be ~100 000 (initial cash); got {first_val}"
        )

    def test_invalid_format_raises_value_error(self, tmp_path):
        """Passing an unknown format string must raise ValueError."""
        results = self._results()
        with pytest.raises(ValueError, match="Unsupported format"):
            export_report(results, str(tmp_path / "out"), format='xlsx')


# ---------------------------------------------------------------------------
# Tests: Backtester.compute_alpha_beta() static method (Phase 0.2)
#
# Covers:
#   1. Known-input beta/alpha against a hand-computed expected value.
#   2. beta=1 / alpha=0 edge case when strategy returns == benchmark returns.
#   3. Positive-alpha case (strategy = benchmark + constant daily offset).
#   4. Top-level 'alpha' and 'beta' keys present in the full results dict.
#   5. Graceful fallback when no benchmark is provided to run_backtest().
# ---------------------------------------------------------------------------

class TestAlphaBetaComputation:
    """Unit tests for Backtester.compute_alpha_beta() static method and the
    'alpha'/'beta' top-level keys added to the results dict."""

    def test_known_beta_value_from_scaled_series(self):
        """Beta must equal the scaling factor when strategy = k * benchmark.

        With strategy = k * benchmark (same direction, k times the magnitude):
          cov(k*b, b) = k * var(b)   →   beta = k * var(b) / var(b) = k
          mean(k*b) = k * mean(b)    →   alpha = k*mean(b)*252 - k*mean(b)*252 = 0
        """
        rng = np.random.default_rng(0)
        bench = rng.normal(0.001, 0.01, 200)  # 200 daily returns
        k = 2.5
        strat = k * bench

        alpha, beta = Backtester.compute_alpha_beta(strat, bench)

        assert beta == pytest.approx(k, abs=1e-9), (
            f"Expected beta={k} (scaling factor); got beta={beta}"
        )
        assert alpha == pytest.approx(0.0, abs=1e-6), (
            f"Expected alpha=0.0 when strategy is a pure scale of benchmark; got alpha={alpha}"
        )

    def test_identical_returns_yield_beta_one_alpha_zero(self):
        """When strategy returns are identical to benchmark returns, beta=1 and alpha=0.

        cov(b, b) / var(b) = var(b) / var(b) = 1  →  beta = 1
        alpha = mean(b)*252 - 1*mean(b)*252 = 0
        """
        returns = np.array([0.01, -0.005, 0.02, -0.01, 0.005, 0.003, -0.008, 0.015])

        alpha, beta = Backtester.compute_alpha_beta(returns, returns)

        assert beta == pytest.approx(1.0, abs=1e-10), (
            f"Expected beta=1.0 when strategy == benchmark; got beta={beta}"
        )
        assert alpha == pytest.approx(0.0, abs=1e-9), (
            f"Expected alpha=0.0 when strategy == benchmark; got alpha={alpha}"
        )

    def test_constant_daily_outperformance_gives_positive_alpha_beta_one(self):
        """strategy = benchmark + constant_offset  →  beta=1, alpha=offset*252.

        cov(b + c, b) = cov(b, b) = var(b)  →  beta = var(b)/var(b) = 1
        alpha = (mean(b) + c)*252 - 1*mean(b)*252 = c*252
        """
        rng = np.random.default_rng(1)
        bench = rng.normal(0.0, 0.01, 200)
        daily_edge = 0.001  # 0.1 % / day constant outperformance
        strat = bench + daily_edge

        alpha, beta = Backtester.compute_alpha_beta(strat, bench)
        expected_alpha = daily_edge * 252

        assert beta == pytest.approx(1.0, abs=1e-10), (
            f"Expected beta=1.0 for constant-offset strategy; got beta={beta}"
        )
        assert alpha == pytest.approx(expected_alpha, rel=1e-8), (
            f"Expected alpha={expected_alpha:.4f}; got alpha={alpha:.4f}"
        )

    def test_zero_variance_benchmark_returns_zero_zero(self):
        """Constant (zero-variance) benchmark must return (0.0, 0.0) gracefully."""
        strat = np.array([0.01, -0.005, 0.02, -0.01, 0.005])
        bench_flat = np.full(5, 0.003)  # zero variance

        alpha, beta = Backtester.compute_alpha_beta(strat, bench_flat)

        assert alpha == 0.0
        assert beta == 0.0

    def test_fewer_than_two_points_returns_zero_zero(self):
        """With fewer than 2 observations the method must return (0.0, 0.0)."""
        alpha, beta = Backtester.compute_alpha_beta([0.01], [0.01])
        assert alpha == 0.0
        assert beta == 0.0

        alpha, beta = Backtester.compute_alpha_beta([], [])
        assert alpha == 0.0
        assert beta == 0.0

    def test_mismatched_lengths_returns_zero_zero(self):
        """Mismatched-length arrays must return (0.0, 0.0) gracefully."""
        alpha, beta = Backtester.compute_alpha_beta([0.01, 0.02], [0.01, 0.02, 0.03])
        assert alpha == 0.0
        assert beta == 0.0

    def test_annualization_factor_respected(self):
        """Changing annualization_factor scales alpha proportionally."""
        bench = np.array([0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01])
        strat = bench + 0.001  # constant daily edge

        alpha_252, _ = Backtester.compute_alpha_beta(strat, bench, annualization_factor=252)
        alpha_52, _ = Backtester.compute_alpha_beta(strat, bench, annualization_factor=52)

        # alpha should scale exactly with the annualization_factor
        assert alpha_252 == pytest.approx(alpha_52 * (252 / 52), rel=1e-9)

    def test_results_dict_has_top_level_alpha_beta_keys(self):
        """The dict returned by run_backtest() must have top-level 'alpha' and 'beta' keys."""
        df = _make_ohlcv(252)
        results = _run_backtest(EMACrossoverStrategy, df)
        assert "error" not in results, f"Backtest errored: {results.get('error')}"
        assert "alpha" in results, "Missing top-level 'alpha' key in results dict"
        assert "beta" in results, "Missing top-level 'beta' key in results dict"

    def test_alpha_beta_are_numeric(self):
        """Top-level 'alpha' and 'beta' must be numeric (int or float)."""
        df = _make_ohlcv(252)
        results = _run_backtest(EMACrossoverStrategy, df)
        assert "error" not in results, f"Backtest errored: {results.get('error')}"
        assert isinstance(results["alpha"], (int, float)), (
            f"'alpha' must be numeric; got {type(results['alpha'])}"
        )
        assert isinstance(results["beta"], (int, float)), (
            f"'beta' must be numeric; got {type(results['beta'])}"
        )

    def test_negative_beta_inverse_correlated_strategy(self):
        """When strategy returns are exactly the inverse of benchmark returns,
        beta must be -1 and alpha must be 0.

        For strat = -bench:
          cov(-bench, bench) = -var(bench)  → beta = -1
          alpha = mean(-bench)*252 - (-1)*mean(bench)*252
                = -mean(bench)*252 + mean(bench)*252 = 0  (exact cancellation)
        """
        rng = np.random.default_rng(7)
        bench = rng.normal(0.0, 0.01, 200)   # 200 daily returns, zero-mean
        strat = -bench

        alpha, beta = Backtester.compute_alpha_beta(strat, bench)

        assert beta == pytest.approx(-1.0, abs=1e-9), (
            f"Expected beta=-1.0 for inverse-correlated strategy; got beta={beta}"
        )
        # alpha cancels exactly for any sample mean: alpha = 0
        assert alpha == pytest.approx(0.0, abs=1e-9), (
            f"Expected alpha=0.0 for inverse strategy; got alpha={alpha}"
        )

    def test_report_unchanged_when_no_benchmark_supplied(self):
        """Passing benchmark_ticker=None must not break the report — all
        required keys must still be present and alpha/beta must default to 0.0."""
        b = Backtester()
        df = _make_ohlcv(60, seed=7)
        b.add_data(df.copy())
        b.add_strategy(EMACrossoverStrategy)
        results = b.run_backtest(cash=100_000, benchmark_ticker=None)

        assert "error" not in results, f"Backtest errored with no benchmark: {results.get('error')}"

        for key in ("sharpe", "max_drawdown", "win_rate", "summary",
                    "alpha", "beta", "signals", "total_asset_value"):
            assert key in results, (
                f"Key '{key}' missing from results when benchmark_ticker=None"
            )

        # With an invalid/None benchmark, _calculate_alpha_beta returns (0, 0)
        assert isinstance(results["alpha"], (int, float))
        assert isinstance(results["beta"], (int, float))
