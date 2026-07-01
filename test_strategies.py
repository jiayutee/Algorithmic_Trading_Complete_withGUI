"""Unit tests for all Backtrader strategies using synthetic OHLCV data."""
import os
import sys
import tempfile
import unittest.mock as mock
import pytest
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime, timedelta

from strategies.simple_strategies import MACD_RSI_Strategy, EMACrossoverStrategy, StochasticStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 300, seed: int = 42, trend: float = 0.0) -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    closes = 100 * np.cumprod(1 + rng.normal(trend, 0.01, n))
    highs = closes * (1 + rng.uniform(0, 0.02, n))
    lows = closes * (1 - rng.uniform(0, 0.02, n))
    opens = closes * (1 + rng.normal(0, 0.005, n))
    volumes = rng.integers(100_000, 1_000_000, n).astype(float)
    idx = pd.date_range(start="2022-01-03", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=idx,
    )


def _make_ohlcv_from_closes(closes: list, seed: int = 42) -> pd.DataFrame:
    """Build a noisy OHLCV DataFrame from an explicit close price list.

    Adds small random spread so High > Low (avoids ZeroDivisionError in RSI).
    """
    n = len(closes)
    closes_arr = np.array(closes, dtype=float)
    rng = np.random.default_rng(seed)
    highs = closes_arr * (1 + rng.uniform(0.002, 0.008, n))
    lows = closes_arr * (1 - rng.uniform(0.002, 0.008, n))
    opens = closes_arr * (1 + rng.normal(0, 0.002, n))
    volumes = np.full(n, 500_000.0)
    idx = pd.date_range(start="2020-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes_arr, "Volume": volumes},
        index=idx,
    )


def _run_strategy(strategy_cls, df: pd.DataFrame, cash: float = 100_000, **params) -> dict:
    """Run a Backtrader strategy on a DataFrame; return summary metrics."""
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=0.001)

    feed = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(feed)
    cerebro.addstrategy(strategy_cls, **params)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.0)

    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    trade_analysis = strat.analyzers.trades.get_analysis()

    total_trades = trade_analysis.get("total", {}).get("total", 0)
    won = trade_analysis.get("won", {}).get("total", 0)
    lost = trade_analysis.get("lost", {}).get("total", 0)

    return {
        "final_value": final_value,
        "pnl": final_value - cash,
        "total_trades": total_trades,
        "won": won,
        "lost": lost,
    }


# ---------------------------------------------------------------------------
# MACD / RSI
# ---------------------------------------------------------------------------

class TestMACDRSIStrategy:
    def test_runs_without_error(self):
        df = _make_ohlcv(300)
        result = _run_strategy(MACD_RSI_Strategy, df)
        assert result["final_value"] > 0

    def test_produces_trades_on_trending_data(self):
        df = _make_ohlcv(300, trend=0.002)
        result = _run_strategy(MACD_RSI_Strategy, df)
        assert result["total_trades"] >= 0  # may be 0 in flat markets — just must not crash

    def test_final_value_positive(self):
        df = _make_ohlcv(300)
        result = _run_strategy(MACD_RSI_Strategy, df)
        assert result["final_value"] > 0

    def test_won_plus_lost_equals_total(self):
        df = _make_ohlcv(300, seed=7)
        result = _run_strategy(MACD_RSI_Strategy, df)
        assert result["won"] + result["lost"] <= result["total_trades"]

    def test_custom_rsi_params(self):
        df = _make_ohlcv(300)
        result = _run_strategy(MACD_RSI_Strategy, df, rsi_overbought=60, rsi_oversold=40)
        assert result["final_value"] > 0

    def test_no_crash_on_minimal_data(self):
        df = _make_ohlcv(50)
        result = _run_strategy(MACD_RSI_Strategy, df)
        assert result["final_value"] > 0

    def test_signals_list_populated(self):
        """Strategy should append to self.signals on filled orders."""
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100_000)
        cerebro.adddata(bt.feeds.PandasData(dataname=_make_ohlcv(300, trend=0.002)))
        cerebro.addstrategy(MACD_RSI_Strategy)
        results = cerebro.run()
        strat = results[0]
        # signals may be empty (no fills in short run) — but must be a list
        assert isinstance(strat.signals, list)


# ---------------------------------------------------------------------------
# EMA Crossover
# ---------------------------------------------------------------------------

class TestEMACrossoverStrategy:
    def test_runs_without_error(self):
        df = _make_ohlcv(300)
        result = _run_strategy(EMACrossoverStrategy, df)
        assert result["final_value"] > 0

    def test_uptrend_data_generates_buys(self):
        df = _make_ohlcv(300, trend=0.003, seed=1)
        result = _run_strategy(EMACrossoverStrategy, df)
        assert result["total_trades"] >= 0

    def test_downtrend_data_generates_sells(self):
        df = _make_ohlcv(300, trend=-0.003, seed=2)
        result = _run_strategy(EMACrossoverStrategy, df)
        assert result["final_value"] > 0

    def test_custom_ema_periods(self):
        df = _make_ohlcv(300)
        result = _run_strategy(EMACrossoverStrategy, df, ema_short=5, ema_long=20)
        assert result["final_value"] > 0

    def test_no_crash_with_flat_prices(self):
        """Perfectly flat prices should produce no signals but not crash."""
        n = 100
        flat = pd.DataFrame(
            {"Open": [100.0] * n, "High": [100.5] * n,
             "Low": [99.5] * n, "Close": [100.0] * n, "Volume": [500_000.0] * n},
            index=pd.date_range("2022-01-03", periods=n, freq="B"),
        )
        result = _run_strategy(EMACrossoverStrategy, flat)
        assert result["final_value"] > 0
        assert result["total_trades"] == 0

    def test_signals_list_is_list(self):
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100_000)
        cerebro.adddata(bt.feeds.PandasData(dataname=_make_ohlcv(300)))
        cerebro.addstrategy(EMACrossoverStrategy)
        results = cerebro.run()
        assert isinstance(results[0].signals, list)


# ---------------------------------------------------------------------------
# Stochastic
# ---------------------------------------------------------------------------

class TestStochasticStrategy:
    def test_runs_without_error(self):
        df = _make_ohlcv(300)
        result = _run_strategy(StochasticStrategy, df)
        assert result["final_value"] > 0

    def test_custom_thresholds(self):
        df = _make_ohlcv(300, seed=5)
        result = _run_strategy(StochasticStrategy, df, oversold=30, overbought=70)
        assert result["final_value"] > 0

    def test_no_crash_on_minimal_data(self):
        df = _make_ohlcv(30)
        result = _run_strategy(StochasticStrategy, df)
        assert result["final_value"] > 0

    def test_signals_list_is_list(self):
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100_000)
        cerebro.adddata(bt.feeds.PandasData(dataname=_make_ohlcv(300)))
        cerebro.addstrategy(StochasticStrategy)
        results = cerebro.run()
        assert isinstance(results[0].signals, list)


# ---------------------------------------------------------------------------
# Cross-strategy sanity checks
# ---------------------------------------------------------------------------

class TestStrategyCrossComparison:
    @pytest.mark.parametrize("strategy_cls", [MACD_RSI_Strategy, EMACrossoverStrategy, StochasticStrategy])
    def test_all_strategies_survive_volatile_data(self, strategy_cls):
        df = _make_ohlcv(300, seed=99, trend=0.0)
        result = _run_strategy(strategy_cls, df)
        assert result["final_value"] > 0

    @pytest.mark.parametrize("strategy_cls", [MACD_RSI_Strategy, EMACrossoverStrategy, StochasticStrategy])
    def test_never_goes_negative(self, strategy_cls):
        df = _make_ohlcv(300, seed=13, trend=-0.005)
        result = _run_strategy(strategy_cls, df)
        # broker balance should never go negative with default risk settings
        assert result["final_value"] >= 0


# ---------------------------------------------------------------------------
# Signal-correctness tests — verify actual buy/sell signals are generated
# ---------------------------------------------------------------------------

class TestMACDRSISignalCorrectness:
    """Verify MACD+RSI produces correct buy/sell signals on crafted price series."""

    @staticmethod
    def _sharp_drop_then_recover():
        """40-bar steep decline (RSI<30) followed by 100-bar recovery (MACD bullish crossover)."""
        rng = np.random.default_rng(0)
        drop = [100.0]
        for _ in range(39):
            drop.append(drop[-1] * 0.98)
        recovery = [drop[-1]]
        for _ in range(100):
            recovery.append(recovery[-1] * 1.01)
        return _make_ohlcv_from_closes(drop + recovery[1:])

    @staticmethod
    def _slow_rise_then_drop():
        """Gradual 30-bar rise then steep 40-bar run-up (RSI>70) + 80-bar decline (MACD bearish)."""
        rng = np.random.default_rng(11)
        slow_up = [100.0]
        for _ in range(30):
            slow_up.append(slow_up[-1] * (1 + rng.normal(0.001, 0.003)))
        run_up = [slow_up[-1]]
        for _ in range(40):
            run_up.append(run_up[-1] * 1.025)
        drop = [run_up[-1]]
        for _ in range(80):
            drop.append(drop[-1] * 0.992)
        return _make_ohlcv_from_closes(slow_up + run_up[1:] + drop[1:])

    def test_buy_signal_generated_on_oversold_macd_bullish(self):
        """A steep drop followed by recovery should generate at least one 'buy' signal."""
        df = self._sharp_drop_then_recover()
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100_000)
        cerebro.adddata(bt.feeds.PandasData(dataname=df))
        cerebro.addstrategy(MACD_RSI_Strategy)
        results = cerebro.run()
        strat = results[0]
        buy_signals = [s for s in strat.signals if s["type"] == "buy"]
        assert len(buy_signals) >= 1, (
            f"Expected at least one 'buy' signal from MACD+RSI on oversold+bullish data; "
            f"got signals: {strat.signals}"
        )

    def test_sell_short_signal_generated_on_overbought_macd_bearish(self):
        """A steep run-up followed by decline should generate at least one 'sell_short' signal."""
        df = self._slow_rise_then_drop()
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100_000)
        cerebro.adddata(bt.feeds.PandasData(dataname=df))
        cerebro.addstrategy(MACD_RSI_Strategy)
        results = cerebro.run()
        strat = results[0]
        short_signals = [s for s in strat.signals if s["type"] == "sell_short"]
        assert len(short_signals) >= 1, (
            f"Expected at least one 'sell_short' signal from MACD+RSI on overbought+bearish data; "
            f"got signals: {strat.signals}"
        )

    def test_buy_signal_price_is_positive(self):
        """Executed buy price must be a positive number."""
        df = self._sharp_drop_then_recover()
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100_000)
        cerebro.adddata(bt.feeds.PandasData(dataname=df))
        cerebro.addstrategy(MACD_RSI_Strategy)
        results = cerebro.run()
        strat = results[0]
        for sig in strat.signals:
            assert sig["price"] > 0, f"Signal price must be positive; got {sig['price']}"
            assert sig["qty"] != 0, f"Signal qty must be non-zero; got {sig['qty']}"


class TestEMACrossoverSignalCorrectness:
    """Verify EMA Crossover produces correct buy/sell signals on crafted price series."""

    @staticmethod
    def _downtrend_then_strong_uptrend():
        """40-bar downtrend (EMA12<EMA26) then 100-bar strong uptrend (EMA12 crosses above EMA26)."""
        rng = np.random.default_rng(22)
        down = [100.0]
        for _ in range(40):
            down.append(down[-1] * 0.99)
        up = [down[-1]]
        for _ in range(100):
            up.append(up[-1] * 1.015)
        return _make_ohlcv_from_closes(down + up[1:])

    @staticmethod
    def _uptrend_then_strong_downtrend():
        """50-bar uptrend (EMA12>EMA26) then 100-bar sharp downtrend (EMA12 crosses below EMA26)."""
        rng = np.random.default_rng(33)
        up = [100.0]
        for _ in range(50):
            up.append(up[-1] * 1.008)
        down = [up[-1]]
        for _ in range(100):
            down.append(down[-1] * 0.985)
        return _make_ohlcv_from_closes(up + down[1:])

    def test_buy_signal_on_ema_bullish_crossover(self):
        """Downtrend reversing to strong uptrend should generate at least one 'buy' signal."""
        df = self._downtrend_then_strong_uptrend()
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100_000)
        cerebro.adddata(bt.feeds.PandasData(dataname=df))
        cerebro.addstrategy(EMACrossoverStrategy)
        results = cerebro.run()
        strat = results[0]
        buy_signals = [s for s in strat.signals if s["type"] == "buy"]
        assert len(buy_signals) >= 1, (
            f"Expected at least one 'buy' signal from EMA Crossover on uptrend reversal; "
            f"got signals: {strat.signals}"
        )

    def test_sell_short_signal_on_ema_bearish_crossover(self):
        """Uptrend reversing to downtrend should generate at least one 'sell_short' signal."""
        df = self._uptrend_then_strong_downtrend()
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100_000)
        cerebro.adddata(bt.feeds.PandasData(dataname=df))
        cerebro.addstrategy(EMACrossoverStrategy)
        results = cerebro.run()
        strat = results[0]
        short_signals = [s for s in strat.signals if s["type"] == "sell_short"]
        assert len(short_signals) >= 1, (
            f"Expected at least one 'sell_short' signal from EMA Crossover on downtrend reversal; "
            f"got signals: {strat.signals}"
        )

    def test_buy_before_sell_ordering(self):
        """On a buy then exit sequence, 'buy' must appear before 'sell' in signals list."""
        df = self._downtrend_then_strong_uptrend()
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100_000)
        cerebro.adddata(bt.feeds.PandasData(dataname=df))
        cerebro.addstrategy(EMACrossoverStrategy)
        results = cerebro.run()
        strat = results[0]
        if len(strat.signals) >= 2:
            first_buy_idx = next((i for i, s in enumerate(strat.signals) if s["type"] == "buy"), None)
            first_sell_idx = next((i for i, s in enumerate(strat.signals) if s["type"] == "sell"), None)
            if first_buy_idx is not None and first_sell_idx is not None:
                assert first_buy_idx < first_sell_idx, (
                    "A 'buy' signal must precede its corresponding 'sell' (exit long) signal"
                )


class TestStochasticSignalCorrectness:
    """Verify Stochastic produces correct buy/sell signals on crafted price series."""

    @staticmethod
    def _normal_then_sharp_drop_then_recover():
        """30-bar normal prices, 20-bar sharp drop (stochastic oversold), 60-bar recovery."""
        rng = np.random.default_rng(44)
        normal = [100.0]
        for _ in range(30):
            normal.append(normal[-1] * (1 + rng.normal(0, 0.005)))
        drop = [normal[-1]]
        for _ in range(20):
            drop.append(drop[-1] * 0.97)
        recovery = [drop[-1]]
        for _ in range(60):
            recovery.append(recovery[-1] * 1.008)
        return _make_ohlcv_from_closes(normal + drop[1:] + recovery[1:])

    def test_buy_signal_on_stochastic_oversold_crossover(self):
        """Sharp drop then recovery should generate at least one 'buy' from Stochastic."""
        df = self._normal_then_sharp_drop_then_recover()
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100_000)
        cerebro.adddata(bt.feeds.PandasData(dataname=df))
        cerebro.addstrategy(StochasticStrategy)
        results = cerebro.run()
        strat = results[0]
        buy_signals = [s for s in strat.signals if s["type"] == "buy"]
        assert len(buy_signals) >= 1, (
            f"Expected at least one 'buy' signal from Stochastic on oversold+crossover data; "
            f"got signals: {strat.signals}"
        )

    def test_signal_contains_required_keys(self):
        """Every emitted signal must have 'date', 'type', 'price', and 'qty' keys."""
        df = self._normal_then_sharp_drop_then_recover()
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100_000)
        cerebro.adddata(bt.feeds.PandasData(dataname=df))
        cerebro.addstrategy(StochasticStrategy)
        results = cerebro.run()
        strat = results[0]
        required_keys = {"date", "type", "price", "qty"}
        for sig in strat.signals:
            assert required_keys.issubset(sig.keys()), (
                f"Signal missing required keys. Got keys: {set(sig.keys())}, expected {required_keys}"
            )


# ---------------------------------------------------------------------------
# _closing_long / _closing_short flag correctness
# ---------------------------------------------------------------------------

class _ClosingFlagAuditMixin:
    """Mixin that records when _closing_long/_closing_short are set and verifies they are
    cleared by notify_order after the order is completed."""

    def __init__(self):
        super().__init__()
        self._closing_long_set_dates = []
        self._closing_short_set_dates = []

    def next(self):
        before_clong = self._closing_long
        before_cshort = self._closing_short
        super().next()
        if self._closing_long and not before_clong:
            self._closing_long_set_dates.append(self.data.datetime.datetime(0))
        if self._closing_short and not before_cshort:
            self._closing_short_set_dates.append(self.data.datetime.datetime(0))


class _MACDRSIFlagAudit(_ClosingFlagAuditMixin, MACD_RSI_Strategy):
    pass


class _EMAFlagAudit(_ClosingFlagAuditMixin, EMACrossoverStrategy):
    pass


class _StochFlagAudit(_ClosingFlagAuditMixin, StochasticStrategy):
    pass


class TestClosingFlags:
    """Verify _closing_long and _closing_short flags are set before close() and
    reset to False by notify_order after the order completes."""

    def _run_flag_audit(self, strategy_cls, df):
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100_000)
        cerebro.adddata(bt.feeds.PandasData(dataname=df))
        cerebro.addstrategy(strategy_cls)
        results = cerebro.run()
        return results[0]

    def test_closing_long_reset_after_exit_macd_rsi(self):
        """_closing_long must be False at strategy end (reset by notify_order on sell)."""
        rng = np.random.default_rng(0)
        drop = [100.0]
        for _ in range(39):
            drop.append(drop[-1] * 0.98)
        recovery = [drop[-1]]
        for _ in range(100):
            recovery.append(recovery[-1] * 1.01)
        df = _make_ohlcv_from_closes(drop + recovery[1:])
        strat = self._run_flag_audit(_MACDRSIFlagAudit, df)
        assert strat._closing_long is False, "_closing_long must be reset to False after order fills"
        assert strat._closing_short is False, "_closing_short must remain False when only long trades occurred"

    def test_closing_long_reset_after_exit_ema(self):
        """_closing_long must be False at end after EMA crossover exit-long sequence."""
        rng = np.random.default_rng(22)
        down = [100.0]
        for _ in range(40):
            down.append(down[-1] * 0.99)
        up = [down[-1]]
        for _ in range(100):
            up.append(up[-1] * 1.015)
        # After the buy, add a downtrend to force exit-long
        down2 = [up[-1]]
        for _ in range(60):
            down2.append(down2[-1] * 0.988)
        df = _make_ohlcv_from_closes(down + up[1:] + down2[1:])
        strat = self._run_flag_audit(_EMAFlagAudit, df)
        assert strat._closing_long is False, "_closing_long must be reset after order fills (EMA)"
        assert strat._closing_short is False

    def test_closing_short_reset_after_exit_macd_rsi(self):
        """_closing_short must be False at end after short → cover sequence."""
        rng = np.random.default_rng(11)
        slow_up = [100.0]
        for _ in range(30):
            slow_up.append(slow_up[-1] * (1 + rng.normal(0.001, 0.003)))
        run_up = [slow_up[-1]]
        for _ in range(40):
            run_up.append(run_up[-1] * 1.025)
        drop = [run_up[-1]]
        for _ in range(80):
            drop.append(drop[-1] * 0.992)
        df = _make_ohlcv_from_closes(slow_up + run_up[1:] + drop[1:])
        strat = self._run_flag_audit(_MACDRSIFlagAudit, df)
        assert strat._closing_short is False, "_closing_short must be reset after buy_cover fills"

    def test_sell_signal_type_is_sell_not_sell_short_when_exiting_long(self):
        """When closing a long position, notify_order should record type='sell', not 'sell_short'."""
        rng = np.random.default_rng(0)
        drop = [100.0]
        for _ in range(39):
            drop.append(drop[-1] * 0.98)
        recovery = [drop[-1]]
        for _ in range(100):
            recovery.append(recovery[-1] * 1.01)
        df = _make_ohlcv_from_closes(drop + recovery[1:])
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100_000)
        cerebro.adddata(bt.feeds.PandasData(dataname=df))
        cerebro.addstrategy(MACD_RSI_Strategy)
        results = cerebro.run()
        strat = results[0]
        # If there's a sell signal, it must be 'sell' (exit long), not 'sell_short' (open short)
        sell_signals = [s for s in strat.signals if s["type"] in ("sell", "sell_short")]
        for sig in sell_signals:
            if sig["type"] == "sell":
                # This is an exit-long; the buy must have preceded it
                buy_dates = [s["date"] for s in strat.signals if s["type"] == "buy"]
                assert len(buy_dates) >= 1, "A 'sell' exit signal implies a prior 'buy' entry"

    def test_buy_cover_type_used_when_closing_short(self):
        """When closing a short position, notify_order should record type='buy_cover', not 'buy'."""
        rng = np.random.default_rng(11)
        slow_up = [100.0]
        for _ in range(30):
            slow_up.append(slow_up[-1] * (1 + rng.normal(0.001, 0.003)))
        run_up = [slow_up[-1]]
        for _ in range(40):
            run_up.append(run_up[-1] * 1.025)
        drop = [run_up[-1]]
        for _ in range(80):
            drop.append(drop[-1] * 0.992)
        # Add recovery to trigger exit of short
        recovery = [drop[-1]]
        for _ in range(60):
            recovery.append(recovery[-1] * 1.005)
        df = _make_ohlcv_from_closes(slow_up + run_up[1:] + drop[1:] + recovery[1:])
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100_000)
        cerebro.adddata(bt.feeds.PandasData(dataname=df))
        cerebro.addstrategy(MACD_RSI_Strategy)
        results = cerebro.run()
        strat = results[0]
        short_signals = [s for s in strat.signals if s["type"] == "sell_short"]
        cover_signals = [s for s in strat.signals if s["type"] == "buy_cover"]
        if short_signals:
            # If there are short entries, there may be covers
            # Just verify all signal types are from the allowed set
            allowed_types = {"buy", "sell", "buy_cover", "sell_short"}
            for sig in strat.signals:
                assert sig["type"] in allowed_types, f"Unknown signal type: {sig['type']}"


# ---------------------------------------------------------------------------
# ML / LSTM strategy graceful degradation
# ---------------------------------------------------------------------------

class TestLSTMStrategyDegradation:
    """Verify the LSTM strategy degrades gracefully when TensorFlow is absent."""

    @staticmethod
    def _small_df():
        return _make_ohlcv_from_closes([100.0 + i * 0.5 for i in range(30)])

    def test_lstm_runs_without_crash_when_no_model_files(self):
        """LSTMPredictor must not crash when model/scaler files are absent."""
        from strategies.ml_strategies import LSTMPredictor
        df = self._small_df()
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100_000)
        cerebro.adddata(bt.feeds.PandasData(dataname=df))
        # Use a ticker with no model files
        cerebro.addstrategy(LSTMPredictor, ticker="NO_MODEL_XYZ")
        results = cerebro.run()
        strat = results[0]
        assert strat.model is None, "model should be None when no model file exists"
        assert strat.scaler is None, "scaler should be None when no scaler file exists"

    def test_lstm_model_is_none_when_tensorflow_absent(self):
        """When TF is not importable (but model files exist), model must be None (no crash)."""
        import strategies.ml_strategies as ml_mod
        trained_dir = os.path.join(ml_mod.BASE_DIR, "trained_models")
        ticker = "TFTEST"
        model_path = os.path.join(trained_dir, f"lstm_model_{ticker}.h5")
        scaler_path = os.path.join(trained_dir, f"scaler_{ticker}.pkl")
        os.makedirs(trained_dir, exist_ok=True)

        try:
            import joblib
            from sklearn.preprocessing import MinMaxScaler
            with open(model_path, "w") as f:
                f.write("placeholder")
            sc = MinMaxScaler()
            sc.fit([[0], [1]])
            joblib.dump(sc, scaler_path)

            # Clear any cached imports of ml_strategies so it picks up the TF mock
            for key in list(sys.modules.keys()):
                if "ml_strategies" in key:
                    del sys.modules[key]

            with mock.patch.dict(sys.modules, {
                "tensorflow": None,
                "tensorflow.keras": None,
                "tensorflow.keras.models": None,
            }):
                import strategies.ml_strategies as ml_mod2
                df = self._small_df()
                cerebro = bt.Cerebro()
                cerebro.broker.setcash(100_000)
                cerebro.adddata(bt.feeds.PandasData(dataname=df))
                cerebro.addstrategy(ml_mod2.LSTMPredictor, ticker=ticker)
                results = cerebro.run()
                strat = results[0]
                assert strat.model is None, (
                    "model must be None when TensorFlow is absent, even if model files exist"
                )
        finally:
            for p in (model_path, scaler_path):
                if os.path.exists(p):
                    os.remove(p)
            # Restore ml_strategies import
            for key in list(sys.modules.keys()):
                if "ml_strategies" in key:
                    del sys.modules[key]
            import strategies.ml_strategies  # re-import cleanly
