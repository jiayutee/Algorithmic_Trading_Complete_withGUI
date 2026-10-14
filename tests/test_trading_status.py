"""Tests for core/trading_status.py: account mode, kill switch file, price freshness and order preview."""

from types import SimpleNamespace

import pytest

from brokers.execution_guard import ExecutionGuard
from core import trading_status as ts


def _guard(tmp_path, **env):
    env = {"KILL_SWITCH_FILE": str(tmp_path / ".kill_switch"), **env}
    return ExecutionGuard(env=env)


class _Live:
    paper_mode = False


class _Paper:
    paper_mode = True


# ------------------------------------------------------------------------------------------------ account mode

def test_simulator_is_paper(tmp_path):
    mode = ts.account_mode("Simulator", None, guard=_guard(tmp_path))
    assert mode.mode == ts.MODE_PAPER
    assert mode.label == "PAPER · Simulator"
    assert not mode.real_money


def test_paper_connector_is_paper(tmp_path):
    assert ts.account_mode("Binance_testnet", _Paper(), guard=_guard(tmp_path)).mode == ts.MODE_PAPER


def test_unconnected_live_broker_is_unavailable(tmp_path):
    mode = ts.account_mode("Alpaca", None, guard=_guard(tmp_path))
    assert mode.mode == ts.MODE_UNAVAILABLE
    assert "NOT CONNECTED" in mode.label


def test_live_connector_blocked_by_default(tmp_path):
    mode = ts.account_mode("Binance", _Live(), guard=_guard(tmp_path))
    assert mode.mode == ts.MODE_BLOCKED
    assert "LIVE_TRADING_ENABLED" in mode.detail


def test_live_connector_dry_run_and_live(tmp_path):
    assert ts.account_mode("Binance", _Live(), guard=_guard(tmp_path, LIVE_TRADING_ENABLED="true")).mode == ts.MODE_DRY_RUN
    live = ts.account_mode("Binance", _Live(), guard=_guard(tmp_path, LIVE_TRADING_ENABLED="true", LIVE_DRY_RUN="false"))
    assert live.mode == ts.MODE_LIVE and live.real_money


def test_kill_switch_file_round_trip_blocks_live(tmp_path):
    guard = _guard(tmp_path, LIVE_TRADING_ENABLED="true", LIVE_DRY_RUN="false")
    assert not ts.kill_switch_engaged(guard)
    path = ts.engage_kill_switch("test", guard=guard)
    assert path == str(tmp_path / ".kill_switch")
    assert ts.kill_switch_engaged(guard)
    assert ts.account_mode("Binance", _Live(), guard=guard).mode == ts.MODE_BLOCKED
    assert ts.release_kill_switch(guard=guard) is True
    assert ts.release_kill_switch(guard=guard) is False
    assert ts.account_mode("Binance", _Live(), guard=guard).mode == ts.MODE_LIVE


# --------------------------------------------------------------------------------------------- price freshness

@pytest.mark.parametrize("age,state", [(None, "unknown"), (0, "fresh"), (60, "fresh"), (61, "aging"),
                                       (300, "aging"), (301, "stale"), (float("nan"), "unknown")])
def test_price_freshness_states(age, state):
    assert ts.price_freshness(age).state == state


def test_format_age_units():
    assert ts.format_age(5) == "5s"
    assert ts.format_age(125) == "2m"
    assert ts.format_age(7300) == "2h"
    assert ts.format_age(200000) == "2d"
    assert ts.price_freshness(400).label == "6m ago · stale"


# ----------------------------------------------------------------------------------------------- order preview

def test_market_buy_preview_math():
    p = ts.preview_order("buy", 0.15, "BTCUSDT", "market", last_price=62418.5, fee_rate=0.001, cash=50000,
                         price_age_s=2)
    assert p.ok and not p.warnings
    assert p.notional == pytest.approx(9362.775)
    assert p.fee == pytest.approx(9.362775)
    assert p.cash_after == pytest.approx(50000 - 9362.775 - 9.362775)
    assert p.title == "BUY 0.15 BTCUSDT · Market"
    assert any(line.startswith("Est. notional: $9,362.7") for line in p.lines())


def test_limit_uses_order_price_and_requires_it():
    p = ts.preview_order("buy", 2, "AAPL", "limit", last_price=230, order_price=225, fee_rate=0.0005, cash=1000)
    assert p.price == 225 and p.notional == 450
    missing = ts.preview_order("buy", 2, "AAPL", "limit", last_price=230)
    assert not missing.ok and "limit price" in missing.blocking[0]


def test_blocking_problems():
    assert not ts.preview_order("buy", 0, "AAPL", last_price=1).ok
    assert not ts.preview_order("buy", "abc", "AAPL", last_price=1).ok
    assert not ts.preview_order("buy", 1, "", last_price=1).ok
    no_price = ts.preview_order("buy", 1, "AAPL")
    assert not no_price.ok and no_price.notional is None


def test_blocked_mode_blocks_and_live_mode_warns(tmp_path):
    blocked = ts.account_mode("Binance", _Live(), guard=_guard(tmp_path))
    assert not ts.preview_order("buy", 1, "BTCUSDT", last_price=10, mode=blocked).ok
    live = ts.account_mode("Binance", _Live(), guard=_guard(tmp_path, LIVE_TRADING_ENABLED="1", LIVE_DRY_RUN="0"))
    p = ts.preview_order("buy", 1, "BTCUSDT", last_price=10, mode=live)
    assert p.ok and any("real money" in w for w in p.warnings)


def test_warnings_stale_cash_and_oversell():
    stale = ts.preview_order("buy", 1, "AAPL", last_price=10, price_age_s=900)
    assert stale.ok and any("old" in w for w in stale.warnings)
    broke = ts.preview_order("buy", 10, "AAPL", last_price=10, cash=50, fee_rate=0)
    assert any("cash" in w for w in broke.warnings)
    capped = ts.preview_order("sell", 5, "AAPL", last_price=10, position_qty=2)
    assert any("caps it" in w for w in capped.warnings)
    short = ts.preview_order("sell", 5, "AAPL", last_price=10, position_qty=0)
    assert any("short" in w for w in short.warnings)


def test_broker_fee_rate():
    sim = SimpleNamespace(market_fee=0.001, limit_fee=0.0005)
    assert ts.broker_fee_rate(sim, "market") == 0.001
    assert ts.broker_fee_rate(sim, "limit") == 0.0005
    assert ts.broker_fee_rate(object(), "market") is None


def test_money_format():
    assert ts.money(None) == "n/a"
    assert ts.money(-12.5) == "−$12.50"
    assert ts.money(1234.567) == "$1,234.57"


# ------------------------------------------------------------------------------------------- Dash wiring (slice 1)

def _text(component):
    """All text in a Dash component tree."""
    from dash.development.base_component import Component
    if isinstance(component, str):
        return component
    if isinstance(component, (list, tuple)):
        return " ".join(_text(c) for c in component)
    if isinstance(component, Component):
        return _text(getattr(component, "children", None) or "")
    return ""


def _ids(component, out=None):
    from dash.development.base_component import Component
    out = [] if out is None else out
    if isinstance(component, (list, tuple)):
        for c in component:
            _ids(c, out)
    elif isinstance(component, Component):
        if getattr(component, "id", None) is not None:
            out.append(component.id)
        _ids(getattr(component, "children", None), out)
    return out


def test_dash_layout_has_safety_and_review_ids():
    pytest.importorskip("dash")
    from dash_app.layout import build_layout
    ids = _ids(build_layout())
    for needed in ("mode-pill", "kill-switch-btn", "kill-release-confirm", "safety-interval", "kill-switch-store",
                   "price-age", "order-review-toggle", "order-review", "order-review-body", "order-draft-store",
                   "order-confirm-btn", "order-cancel-btn"):
        assert needed in ids, needed


def test_dash_positions_table_has_close_buttons():
    pytest.importorskip("dash")
    from brokers.simulatedbroker import SimulatedBroker
    from dash_app.callbacks import _build_positions_content
    broker = SimulatedBroker()
    broker.update_price("AAPL", 100.0)
    broker.submit_order("AAPL", 3, "buy", "market")
    content = _build_positions_content(broker)
    assert {"type": "close-position-btn", "symbol": "AAPL"} in _ids(content)
    assert "AAPL" in _text(content) and "Long" in _text(content)


def test_dash_order_preview_and_review_body():
    pytest.importorskip("dash")
    from brokers.simulatedbroker import SimulatedBroker
    from dash_app.callbacks import _order_preview, _order_review_body
    broker = SimulatedBroker(initial_balance=1000)
    p = _order_preview(broker, "buy", 2, "market", None, "AAPL", 100.0, 1.0)
    assert p.ok and p.notional == 200 and p.cash == 1000
    assert p.mode.mode == ts.MODE_PAPER
    body = _text(_order_review_body(p))
    assert "BUY 2 AAPL" in body and "PAPER · Simulator" in body
    bad = _order_review_body(_order_preview(broker, "buy", None, "market", None, "AAPL", None, None))
    assert "Quantity must be greater than 0" in _text(bad)


def test_dash_price_for_uses_position_mark_for_other_symbols():
    pytest.importorskip("dash")
    from brokers.simulatedbroker import SimulatedBroker
    from dash_app.callbacks import _price_for
    broker = SimulatedBroker()
    broker.update_price("MSFT", 410.0)
    broker.submit_order("MSFT", 1, "buy", "market")
    price, age = _price_for("MSFT", "AAPL", broker)
    assert price == 410.0 and age is not None and age < 5


def test_dash_safety_header_view_reflects_kill_switch(tmp_path, monkeypatch):
    pytest.importorskip("dash")
    from brokers import execution_guard
    from dash_app import callbacks as cb
    monkeypatch.setenv("KILL_SWITCH_FILE", str(tmp_path / ".kill_switch"))
    execution_guard.set_guard(None)
    try:
        pill, detail, pill_style, kill_text, _tip, _style, age_text, _age_style = cb._safety_header_view(None, None)
        assert pill == "PAPER · Simulator" and kill_text == "Kill switch"
        assert age_text.startswith("No price yet")
        ts.engage_kill_switch("test")
        assert cb._safety_header_view(None, None)[3] == "Kill switch ON · Release"
        ts.release_kill_switch()
    finally:
        execution_guard.set_guard(None)


def test_live_price_service_reports_price_age():
    from core.live_price_service import LivePriceService
    svc = LivePriceService.__new__(LivePriceService)
    import threading
    svc._lock = threading.Lock()
    svc._price_cache = {"BTCUSDT": 1.0}
    svc._price_time = {}
    assert svc.get_price_age("BTCUSDT") is None
    import time
    svc._price_time["BTCUSDT"] = time.monotonic() - 3
    assert 2.5 < svc.get_price_age("BTCUSDT") < 10
