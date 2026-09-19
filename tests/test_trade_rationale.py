"""Phase 11.1: every trade carries a structured 'why'."""
import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from brokers.simulatedbroker import SimulatedBroker
from core.trade_rationale import (
    build_rationale, format_rationale, format_rationale_detail,
    manual_rationale, submit_with_rationale,
)


# ---- schema / formatting -------------------------------------------------

def test_build_rationale_is_json_safe_and_has_ml_slots():
    import json
    r = build_rationale(source="strategy", action="open_long", summary="x",
                        features={"rsi": np.float64(24.123456789), "bad": float("nan")})
    json.dumps(r)  # numpy scalars / NaN must not leak through
    assert r["features"]["rsi"] == 24.123457
    assert r["features"]["bad"] is None
    # Phase 6 hooks exist now so ML strategies can fill them without a format change.
    assert "confidence" in r and "feature_importance" in r


def test_detail_shows_values_rules_and_model_output():
    r = build_rationale(source="strategy", action="open_long", summary="Opened LONG",
                        strategy="GBM", signal="p_up", features={"rsi": 20},
                        thresholds={"min_p": 0.6}, confidence=0.72,
                        feature_importance={"rsi": 0.5, "sentiment": -0.2})
    text = format_rationale_detail(r)
    assert "rsi=20" in text and "min_p=0.6" in text
    assert "72%" in text and "Top drivers" in text
    assert format_rationale(None) == "—"


# ---- broker ---------------------------------------------------------------

def test_every_order_gets_a_rationale_even_if_caller_gives_none():
    b = SimulatedBroker(market_fee=0.0, limit_fee=0.0)
    o = b.submit_order("X", 1, "buy", "market", execution_price=100.0)
    assert o.rationale["source"] == "unspecified"
    b.close()


def test_supplied_rationale_is_stored_with_order_history():
    b = SimulatedBroker(market_fee=0.0, limit_fee=0.0)
    r = manual_rationale("buy", "X", "market", price=100.0)
    o = submit_with_rationale(b, r, symbol="X", qty=1, side="buy", order_type="market",
                              execution_price=100.0)
    assert o.rationale is r
    assert b.order_history[-1].rationale["source"] == "manual"
    assert "Manual BUY X" in b.order_history[-1].rationale["summary"]
    b.close()


def test_non_rationale_aware_broker_is_called_without_the_kwarg():
    """Live connectors have fixed signatures; a stray kwarg would raise TypeError."""
    class LiveLike:
        def submit_order(self, symbol, qty, side, order_type="market"):
            return SimpleNamespace(symbol=symbol, qty=qty)
    out = submit_with_rationale(LiveLike(), manual_rationale("buy", "BTC"),
                                symbol="BTC", qty=1, side="buy", order_type="market")
    assert out.symbol == "BTC"


# ---- strategies: the real "why" ------------------------------------------

def _trending_df(n=400, seed=3):
    rng = np.random.default_rng(seed)
    price = 100 + np.cumsum(rng.normal(0, 1.5, n)) + 8 * np.sin(np.arange(n) / 12)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"open": price, "high": price + 1, "low": price - 1,
                         "close": price, "volume": 1000.0}, index=idx)


# Loosened MACD/RSI thresholds so the synthetic series produces entries AND exits.
@pytest.mark.parametrize("name,params", [
    ("MACD_RSI_Strategy", {"rsi_oversold": 45, "rsi_overbought": 55}),
    ("EMACrossoverStrategy", {}),
    ("StochasticStrategy", {}),
])
def test_every_strategy_signal_carries_a_rationale(name, params):
    import backtrader as bt
    import strategies.simple_strategies as ss
    df = _trending_df()
    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.addstrategy(getattr(ss, name), **params)
    cerebro.broker.setcash(100000)
    strat = cerebro.run()[0]
    assert strat.signals, f"{name} produced no trades on the fixture; pick a livelier series"
    for sig in strat.signals:
        r = sig.get("rationale")
        assert r and r["source"] == "strategy", sig
        assert r["summary"] and r["features"], sig
        assert r["strategy"], sig
    # opens and closes are explained differently
    actions = {s["rationale"]["action"] for s in strat.signals}
    assert any(a.startswith("open") for a in actions)
    assert any(a.startswith("close") for a in actions)


# ---- surfaces --------------------------------------------------------------

def test_chart_marker_hover_shows_the_rationale():
    import plotly.graph_objects as go
    from core.chart_builder import overlay_signals
    r = build_rationale(source="strategy", action="open_long", summary="Opened LONG: RSI 24 < 30",
                        strategy="MACD_RSI", features={"rsi": 24})
    fig = overlay_signals(go.Figure(), [
        {"type": "buy", "date": datetime.datetime(2026, 1, 1), "price": 100.0, "rationale": r},
        {"type": "sell", "date": datetime.datetime(2026, 1, 5), "price": 110.0},  # legacy: no rationale
    ])
    buy_hover, sell_hover = fig.data[0].text[0], fig.data[1].text[0]
    assert "Opened LONG: RSI 24 &lt; 30" in buy_hover or "Opened LONG: RSI 24 < 30" in buy_hover
    assert "rsi=24" in buy_hover
    assert "No rationale recorded" in sell_hover   # old signals still render


def test_dash_orders_table_includes_why():
    from dash_app.callbacks import _build_orders_table_data
    b = SimulatedBroker(market_fee=0.0, limit_fee=0.0)
    submit_with_rationale(b, manual_rationale("buy", "X", "market", price=100.0),
                          symbol="X", qty=1, side="buy", order_type="market", execution_price=100.0)
    data, _ = _build_orders_table_data(b)
    assert "Manual BUY X" in data[0]["why"]
    b.close()


def test_dash_manual_order_records_manual_rationale():
    from dash_app.callbacks import _validate_and_submit_order
    b = SimulatedBroker(market_fee=0.0, limit_fee=0.0)
    b.market_data["X"] = 100.0
    _validate_and_submit_order(b, "buy", 1, "market", None, "X")
    r = b.order_history[-1].rationale
    assert r["source"] == "manual" and "Dash Order Entry panel" in r["summary"]
    b.close()


# ---- real prices reach the paper broker (was: every UI trade filled at a fake $100) ----

def test_dash_order_fills_at_the_real_price_not_the_fake_default():
    from dash_app.callbacks import _validate_and_submit_order
    b = SimulatedBroker(market_fee=0.0, limit_fee=0.0)
    text, _ = _validate_and_submit_order(b, "buy", 0.5, "market", None, "BTCUSDT", market_price=64_250.0)
    o = b.order_history[-1]
    assert o.filled_avg_price == 64_250.0, text
    assert b.market_data["BTCUSDT"] == 64_250.0            # so unrealized P&L marks to market
    assert "$64,250.00" in o.rationale["summary"]          # rationale records the real decision price
    b.close()


def test_dash_order_without_a_price_still_works():
    from dash_app.callbacks import _validate_and_submit_order
    b = SimulatedBroker(market_fee=0.0, limit_fee=0.0)
    _validate_and_submit_order(b, "buy", 1, "market", None, "X")   # no feed available
    assert b.order_history[-1].status.value == "filled"
    b.close()


def test_sync_broker_price_is_safe_with_no_broker_or_price():
    from dash_app.callbacks import _sync_broker_price
    _sync_broker_price(None, "X", 1.0)
    b = SimulatedBroker()
    _sync_broker_price(b, "X", None)
    assert "X" not in b.market_data or b.market_data["X"] == 100.0
    b.close()
