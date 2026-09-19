"""Trend-filter overlay (core + backtrader strategy + wrapper)."""
import numpy as np
import pandas as pd
import pytest

from core import trend_overlay as t
from core.backtester import Backtester
from strategies.simple_strategies import EMACrossoverStrategy
from strategies.trend_filter_strategy import TrendFilterStrategy, with_trend_overlay


# ------------------------------------------------------------------ pure functions

def test_trailing_return_and_warmup():
    c = list(range(100, 130))                       # 30 closes, +1 per bar
    assert t.trailing_return(c, 28) == pytest.approx(129 / 101 - 1)
    assert t.trailing_return(c[:28], 28) is None    # needs lookback + 1 closes
    assert t.trend_is_up(c[:28], 28) is None
    assert t.trend_is_up(c, 28) is True
    assert t.trend_is_up(list(range(130, 100, -1)), 28) is False
    assert t.trend_is_up([100.0] * 40, 28) is False  # flat is NOT up (strictly > 0)


def test_evaluation_cadence_matches_the_validated_rule():
    bars = [n for n in range(1, 80) if t.is_evaluation_bar(n, 28, 7)]
    assert bars == [29, 36, 43, 50, 57, 64, 71, 78]


def test_portfolio_weights_match_the_experiment_implementation():
    from training_ground.experiments_6_7 import target_weights
    r = np.random.default_rng(1)
    rets = pd.DataFrame(r.normal(0.001, 0.03, (28, 8)), columns=list("ABCDEFGH"))
    closes = (1 + rets).cumprod()
    closes.loc[-1] = 1.0                                       # a starting price row so 28 returns == 29 closes
    closes = closes.sort_index()
    expected = target_weights(rets, "tsmom")
    pd.testing.assert_series_equal(t.portfolio_weights(closes, 28), expected, check_names=False)


def test_describe_is_plain_language():
    assert "staying in cash" in t.describe([1, 2, 3])
    assert "UP" in t.describe(list(range(100, 130)))


# ------------------------------------------------------------------ backtrader integration

def crash_frame(n=300, seed=0):
    """Up 120 bars, crash 60 bars (-35%), recover 120 bars."""
    r = np.random.default_rng(seed)
    drift = np.r_[np.full(120, 0.004), np.full(60, -0.007), np.full(120, 0.004)]
    close = 100 * np.cumprod(1 + drift + r.normal(0, 0.004, n))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.002, "Low": np.minimum(open_, close) * 0.998,
                         "Close": close, "Volume": 1e6}, index=pd.date_range("2023-01-01", periods=n))


def falling_frame(n=300, seed=1):
    """Falls steadily the whole time: the trend is never up."""
    r = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + np.full(n, -0.004) + r.normal(0, 0.003, n))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.002, "Low": np.minimum(open_, close) * 0.998,
                         "Close": close, "Volume": 1e6}, index=pd.date_range("2023-01-01", periods=n))


def run(strategy, df=None, **params):
    b = Backtester()
    b.add_data(df if df is not None else crash_frame())
    b.add_strategy(strategy, **params)
    report = b.run_backtest(cash=100_000, benchmark_ticker=None, market_fee=0.0, limit_fee=0.0)
    return report, b.cerebro.runstrats[0][0]


def max_dd(equity):
    e = np.asarray(equity, float)
    return float((e / np.maximum.accumulate(e) - 1).min())


def test_standalone_strategy_sits_out_the_crash():
    df = crash_frame()
    report, strat = run(TrendFilterStrategy)
    kinds = [s["type"] for s in strat.signals]
    assert kinds[0] == "buy" and "sell" in kinds                 # entered the uptrend, left before the end of the crash
    sell_date = next(s["date"] for s in strat.signals if s["type"] == "sell")
    assert df.index[120] <= pd.Timestamp(sell_date) <= df.index[190]        # exited during/just after the turn, not much later
    assert all(s.get("rationale") for s in strat.signals)


def test_standalone_never_goes_short_and_stays_flat_in_a_pure_downtrend():
    _, strat = run(TrendFilterStrategy, falling_frame())
    assert not strat.signals                                      # trend never up -> never traded


def test_wrapper_keeps_the_wrapped_class_and_blocks_entries_when_trend_is_down():
    Wrapped = with_trend_overlay(EMACrossoverStrategy)
    assert issubclass(Wrapped, EMACrossoverStrategy) and Wrapped is not EMACrossoverStrategy
    assert with_trend_overlay(Wrapped) is Wrapped                 # idempotent
    _, strat = run(Wrapped, falling_frame())
    assert not strat.signals


def test_wrapper_reduces_drawdown_versus_the_bare_strategy_on_a_crash():
    df = crash_frame()
    bare_report, bare = run(EMACrossoverStrategy, df, risk_per_trade=0.9)
    over_report, over = run(with_trend_overlay(EMACrossoverStrategy), df, risk_per_trade=0.9)
    assert bare.signals and over.signals                          # both actually traded (no vacuous pass)
    assert bare_report["max_drawdown"] > 0
    assert over_report["max_drawdown"] < bare_report["max_drawdown"]


def test_overlay_goes_to_cash_when_trend_flips_down_even_from_a_profitable_short():
    """Documented cost: the validated rule is 'cash when the trend reads down'. It applies to shorts too, so a
    strategy that profits from shorting a downtrend has that profit cut short (here: covered days after entry)."""
    df = crash_frame()
    bare_report, bare = run(EMACrossoverStrategy, df, risk_per_trade=0.9)
    over_report, over = run(with_trend_overlay(EMACrossoverStrategy), df, risk_per_trade=0.9)
    held = lambda sigs: (pd.Timestamp(sigs[1]["date"]) - pd.Timestamp(sigs[0]["date"])).days
    assert [s["type"] for s in over.signals] == ["sell_short", "buy_cover"]
    assert held(over.signals) < held(bare.signals) / 4            # covered as soon as the trend read flipped
    assert "trend overlay" in over.signals[1]["rationale"]["summary"]
    assert over_report["total_asset_value"][-1] < bare_report["total_asset_value"][-1]


def test_standalone_drawdown_is_shallower_than_buy_and_hold():
    df = crash_frame()
    report, _ = run(TrendFilterStrategy, df)
    buy_hold_dd = -max_dd(df["Close"].to_numpy()) * 100
    assert report["max_drawdown"] < buy_hold_dd * 0.6             # far shallower, not merely equal


def test_overlay_flatten_uses_correct_signal_type_and_rationale():
    df = crash_frame()
    _, over = run(with_trend_overlay(EMACrossoverStrategy), df, risk_per_trade=0.9)
    for s in over.signals:
        assert s["type"] in ("buy", "sell", "sell_short", "buy_cover")
        assert s.get("rationale")


# ------------------------------------------------------------------ app wiring

def test_strategy_manager_registers_trend_filter_and_wraps_on_request():
    from core.strategy_manager import StrategyManager
    sm = StrategyManager()
    assert "Trend Filter (28d)" in sm.get_available_strategies()
    plain = sm.get_strategy("EMA Crossover")
    assert plain.strategy_obj is EMACrossoverStrategy and plain.name == "EMA Crossover"
    wrapped = sm.get_strategy("EMA Crossover", trend_overlay=True)
    assert wrapped.is_backtrader and issubclass(wrapped.strategy_obj, EMACrossoverStrategy)
    assert wrapped.strategy_obj is not EMACrossoverStrategy and "trend overlay" in wrapped.name


def test_strategy_manager_overlay_runs_end_to_end_through_run_backtest():
    from core.strategy_manager import StrategyManager
    sm = StrategyManager()
    w = sm.get_strategy("EMA Crossover", trend_overlay=True)
    res = sm.run_backtest(w, crash_frame(), cash=100_000, market_fee=0.0, limit_fee=0.0)
    assert res and "max_drawdown" in res


def test_dash_backtest_callback_receives_the_overlay_checkbox():
    import dash_app.app as dash_module
    states = [s["id"] for cb in dash_module.app.callback_map.values() for s in cb.get("state", [])]
    assert "trend-overlay-check" in states
    ids = str(dash_module.app.layout)
    assert "trend-overlay-check" in ids and "Trend Filter (28d)" in ids
