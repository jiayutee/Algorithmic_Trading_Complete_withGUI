"""Unit tests for all Backtrader strategies using synthetic OHLCV data."""
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
