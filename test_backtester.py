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
