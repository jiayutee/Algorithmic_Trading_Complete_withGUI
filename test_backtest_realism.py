"""Phase 11.4: opt-in execution latency and partial fills; defaults must not change anything."""
import backtrader as bt
import numpy as np
import pandas as pd
import pytest

from core.backtester import Backtester, LatencyBroker


class BuyOnce(bt.Strategy):
    """Buys ``size`` on bar ``bar`` and records every execution: (bar index, cumulative size, price)."""
    params = (("bar", 3), ("size", 250.0), ("cancel_next_bar", False))

    def __init__(self):
        self.fills, self.order, self.statuses = [], None, []

    def next(self):
        if len(self) == self.p.bar:
            self.order = self.buy(size=self.p.size)
        elif self.p.cancel_next_bar and self.order is not None and len(self) == self.p.bar + 1:
            self.cancel(self.order)

    def notify_order(self, order):
        self.statuses.append(order.getstatusname())
        if order.status in (order.Completed, order.Partial):
            self.fills.append((len(self), float(order.executed.size), float(order.executed.price)))


def _frame(n=40, volume=100.0):
    close = 100 + np.arange(n, dtype=float)                        # +1 per bar: every bar of delay costs a point
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"Open": open_, "High": close + 0.5, "Low": open_ - 0.5, "Close": close, "Volume": volume},
                        index=pd.date_range("2024-01-01", periods=n))


def _run(df=None, strategy=BuyOnce, params=None, **kwargs):
    b = Backtester()
    b.add_data(df if df is not None else _frame())
    b.add_strategy(strategy, **(params or {}))
    report = b.run_backtest(cash=1_000_000, benchmark_ticker=None, market_fee=0.0, limit_fee=0.0, **kwargs)
    return report, b.cerebro.runstrats[0][0]


# --------------------------------------------------------------------- defaults unchanged

def test_default_fills_at_the_next_bar_open_in_full_and_reports_the_execution_model():
    report, strat = _run()
    assert strat.fills == [(4, 250.0, 102.0)]          # decided on bar 3 -> filled on bar 4 at its open (= bar 3 close)
    assert report["execution_model"] == {"latency_bars": 0, "max_volume_pct": None}


def test_explicit_disabled_settings_are_identical_to_not_passing_them():
    a, sa = _run()
    b, sb = _run(latency_bars=0, max_volume_pct=None)
    assert sa.fills == sb.fills
    assert a["total_asset_value"] == b["total_asset_value"]


def test_the_default_broker_is_still_the_plain_backtrader_broker():
    b = Backtester()
    b.add_data(_frame()); b.add_strategy(BuyOnce)
    b.run_backtest(benchmark_ticker=None)
    assert not isinstance(b.cerebro.broker, LatencyBroker)


# ------------------------------------------------------------------------------ latency

@pytest.mark.parametrize("latency", [1, 2, 4])
def test_latency_delays_the_fill_by_exactly_that_many_bars_at_that_bars_open(latency):
    _, base = _run()
    _, slow = _run(latency_bars=latency)
    base_bar, base_px = base.fills[0][0], base.fills[0][2]
    bar, _, px = slow.fills[0]
    assert bar == base_bar + latency
    assert px == pytest.approx(base_px + latency)                     # price rose 1/bar while the order waited


def test_latency_makes_a_trend_following_entry_worse():
    fast, _ = _run()
    slow, _ = _run(latency_bars=3)
    assert slow["total_asset_value"][-1] < fast["total_asset_value"][-1]


def test_an_order_cancelled_while_held_never_fills():
    _, strat = _run(latency_bars=3, params={"cancel_next_bar": True})
    assert strat.fills == [] and "Canceled" in strat.statuses


def test_latency_broker_with_zero_latency_behaves_like_the_default_broker():
    df = _frame()
    plain, sp = _run(df)
    b = Backtester()
    b.add_data(df); b.add_strategy(BuyOnce)
    b.cerebro.broker = LatencyBroker(latency_bars=0)
    rep = b.run_backtest(cash=1_000_000, benchmark_ticker=None, market_fee=0.0, limit_fee=0.0)
    assert b.cerebro.runstrats[0][0].fills == sp.fills and rep["total_asset_value"] == plain["total_asset_value"]


# ------------------------------------------------------------------------ partial fills

def test_order_larger_than_liquidity_fills_over_several_bars_never_exceeding_the_cap():
    _, strat = _run(max_volume_pct=50)                                # 50% of 100 = 50 per bar, order is 250
    cumulative = [f[1] for f in strat.fills]
    increments = np.diff([0.0] + cumulative)
    assert len(strat.fills) == 5 and cumulative[-1] == pytest.approx(250.0)
    assert (increments <= 50.0 + 1e-9).all() and (increments > 0).all()
    assert [f[0] for f in strat.fills] == list(range(strat.fills[0][0], strat.fills[0][0] + 5))   # consecutive bars
    assert "Partial" in strat.statuses and strat.statuses[-1] == "Completed"


def test_an_order_within_liquidity_is_unaffected_by_the_cap():
    _, capped = _run(max_volume_pct=50, params={"size": 40.0})
    _, plain = _run(params={"size": 40.0})
    assert capped.fills == plain.fills


def test_partial_fills_cost_more_in_a_trend_than_one_instant_fill():
    instant, _ = _run()
    drip, _ = _run(max_volume_pct=25)
    assert drip["total_asset_value"][-1] < instant["total_asset_value"][-1]


def test_partial_fill_and_latency_combine():
    _, strat = _run(latency_bars=2, max_volume_pct=50)
    _, base = _run(max_volume_pct=50)
    assert strat.fills[0][0] == base.fills[0][0] + 2 and strat.fills[-1][1] == pytest.approx(250.0)


@pytest.mark.parametrize("bad", [0, -5, 101])
def test_invalid_volume_cap_is_rejected(bad):
    b = Backtester()
    b.add_data(_frame()); b.add_strategy(BuyOnce)
    with pytest.raises(ValueError):
        b.run_backtest(benchmark_ticker=None, max_volume_pct=bad)
