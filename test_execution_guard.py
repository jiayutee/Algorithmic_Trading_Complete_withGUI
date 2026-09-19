"""Phase 11.2: live-order safety must hold at the CONNECTOR level, not just in the UI."""
import ast
import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from brokers.execution_guard import (
    DEFAULTS, DryRunResult, ExecutionGuard, OrderBlockedError, get_guard, guarded_live_order, set_guard,
)

ENABLED_LIVE = {"LIVE_TRADING_ENABLED": "true", "LIVE_DRY_RUN": "false"}


class Clock:
    def __init__(self):
        self.t = 1000.0
    def __call__(self):
        return self.t


def make_guard(env=None, price=50_000.0, tmp_path=None, clock=None):
    env = dict(env or {})
    if tmp_path is not None:
        env.setdefault("KILL_SWITCH_FILE", str(tmp_path / ".kill_switch"))
    return ExecutionGuard(env=env, clock=clock or Clock(), price_provider=lambda s: price)


@pytest.fixture(autouse=True)
def _restore_guard():
    yield
    set_guard(None)


# ---------------------------------------------------------------- the kill switch

def test_default_configuration_blocks_everything(tmp_path):
    g = make_guard({}, tmp_path=tmp_path)
    d = g.evaluate("KuCoin", "BTC/USDT", 0.001, "buy")
    assert d.action == "block" and "LIVE_TRADING_ENABLED" in d.reasons[0]
    assert g.describe().startswith("LIVE TRADING BLOCKED")


@pytest.mark.parametrize("value", ["", "false", "0", "no", "maybe", "  "])
def test_only_an_explicit_truthy_value_enables_trading(value, tmp_path):
    assert make_guard({"LIVE_TRADING_ENABLED": value}, tmp_path=tmp_path).evaluate("X", "S", 1, "buy").action == "block"


def test_kill_switch_file_blocks_even_when_everything_else_is_enabled_and_needs_no_restart(tmp_path):
    g = make_guard(ENABLED_LIVE, price=100.0, tmp_path=tmp_path)
    assert g.evaluate("X", "S", 0.1, "buy").action == "allow"
    (tmp_path / ".kill_switch").write_text("stop")
    d = g.evaluate("X", "S", 0.1, "buy")
    assert d.action == "block" and "kill-switch file" in d.reasons[0]
    (tmp_path / ".kill_switch").unlink()
    assert g.evaluate("X", "S", 0.1, "buy").action == "allow"


def test_engaging_the_switch_in_code_blocks_until_released(tmp_path):
    g = make_guard(ENABLED_LIVE, price=100.0, tmp_path=tmp_path)
    g.engage_kill_switch("drawdown limit hit")
    assert g.evaluate("X", "S", 0.1, "buy").action == "block"
    g.release_kill_switch()
    assert g.evaluate("X", "S", 0.1, "buy").action == "allow"


# ---------------------------------------------------------------------- dry run

def test_dry_run_is_the_default_once_trading_is_enabled(tmp_path):
    g = make_guard({"LIVE_TRADING_ENABLED": "true"}, price=100.0, tmp_path=tmp_path)
    assert g.dry_run() is True and g.evaluate("X", "S", 0.1, "buy").action == "dry_run"
    assert "DRY-RUN" in g.describe()
    assert "REAL ORDERS" in make_guard(ENABLED_LIVE, tmp_path=tmp_path).describe()


def test_dry_run_reports_which_checks_would_have_rejected_the_order(tmp_path):
    g = make_guard({"LIVE_TRADING_ENABLED": "true"}, price=50_000.0, tmp_path=tmp_path)
    d = g.evaluate("X", "BTC", 1.0, "buy")                       # $50k vs the $100 default cap
    assert d.action == "dry_run" and any("MAX_ORDER_NOTIONAL_USD" in r for r in d.reasons)


# ------------------------------------------------------------- pre-trade limits

def test_order_value_limit(tmp_path):
    g = make_guard({**ENABLED_LIVE, "MAX_ORDER_NOTIONAL_USD": "50"}, price=100.0, tmp_path=tmp_path)
    assert g.evaluate("X", "S", 0.4, "buy").action == "allow"           # $40
    d = g.evaluate("X", "S", 0.6, "buy")                                # $60
    assert d.action == "block" and "MAX_ORDER_NOTIONAL_USD" in d.reasons[0]


def test_session_notional_cap_accumulates_per_broker(tmp_path):
    g = make_guard({**ENABLED_LIVE, "MAX_ORDER_NOTIONAL_USD": "100", "MAX_SESSION_NOTIONAL_USD": "150"},
                   price=100.0, tmp_path=tmp_path)
    assert g.evaluate("A", "S", 0.9, "buy").action == "allow"
    g.note_submitted("A", 90.0)
    d = g.evaluate("A", "S", 0.9, "buy")                                # 90 + 90 > 150
    assert d.action == "block" and "MAX_SESSION_NOTIONAL_USD" in d.reasons[0]
    assert g.evaluate("B", "S", 0.9, "buy").action == "allow"           # other broker has its own budget


def test_rate_limit_uses_a_sliding_minute(tmp_path):
    clock = Clock()
    g = make_guard({**ENABLED_LIVE, "MAX_ORDERS_PER_MINUTE": "2"}, price=1.0, tmp_path=tmp_path, clock=clock)
    for _ in range(2):
        assert g.evaluate("A", "S", 1, "buy").action == "allow"
        g.note_submitted("A", 1.0)
    d = g.evaluate("A", "S", 1, "buy")
    assert d.action == "block" and "rate limit" in d.reasons[0]
    clock.t += 61
    assert g.evaluate("A", "S", 1, "buy").action == "allow"


def test_an_order_whose_value_cannot_be_determined_is_refused(tmp_path):
    g = ExecutionGuard(env={**ENABLED_LIVE, "KILL_SWITCH_FILE": str(tmp_path / "k")}, price_provider=lambda s: None)
    d = g.evaluate("A", "S", 1, "buy")
    assert d.action == "block" and "cannot determine" in d.reasons[0]


@pytest.mark.parametrize("qty", [0, -1, float("nan"), float("inf"), "abc", None])
def test_invalid_quantities_are_refused(qty, tmp_path):
    assert make_guard(ENABLED_LIVE, tmp_path=tmp_path).evaluate("A", "S", qty, "buy").action == "block"


def test_invalid_side_is_refused(tmp_path):
    d = make_guard(ENABLED_LIVE, price=1.0, tmp_path=tmp_path).evaluate("A", "S", 1, "hodl")
    assert d.action == "block" and "side" in d.reasons[0]


def test_malformed_limits_fall_back_to_the_conservative_defaults(tmp_path):
    g = make_guard({**ENABLED_LIVE, "MAX_ORDER_NOTIONAL_USD": "lots", "MAX_ORDERS_PER_MINUTE": "-3"},
                   price=1000.0, tmp_path=tmp_path)
    d = g.evaluate("A", "S", 1.0, "buy")                                 # $1000 > default $100
    assert d.action == "block" and f"${DEFAULTS['MAX_ORDER_NOTIONAL_USD']:,.2f}" in d.reasons[0]


def test_every_decision_lands_in_the_audit_log(tmp_path):
    g = make_guard({}, tmp_path=tmp_path)
    g.evaluate("KuCoin", "BTC", 1, "buy")
    entry = g.audit[-1]
    assert entry["action"] == "block" and entry["broker"] == "KuCoin" and entry["reasons"]


# ------------------------------------------- CONNECTOR LEVEL: the exchange is never reached

def _live_connector(cls_path, cls_name):
    mod = importlib.import_module(cls_path)
    cls = getattr(mod, cls_name)
    obj = cls.__new__(cls)                       # skip the constructor: no network, no keys
    obj.paper_mode = False
    obj.client = MagicMock(name="exchange_client")
    return obj


CONNECTORS = [("brokers.kucoin_connector", "KuCoinConnector"),
              ("brokers.mexc_connector", "MexcConnector"),
              ("brokers.binance_connector", "BinanceConnector")]


@pytest.mark.parametrize("path,name", CONNECTORS)
def test_default_config_blocks_at_the_connector_and_never_calls_the_exchange(path, name):
    set_guard(ExecutionGuard(env={}, price_provider=lambda s: 1.0))
    c = _live_connector(path, name)
    with pytest.raises(OrderBlockedError):
        c.submit_order("BTC/USDT", 0.001, "buy")
    assert c.client.mock_calls == [], "the kill switch must stop the order BEFORE any exchange call"


@pytest.mark.parametrize("path,name", CONNECTORS)
def test_dry_run_returns_a_placeholder_and_never_calls_the_exchange(path, name):
    set_guard(ExecutionGuard(env={"LIVE_TRADING_ENABLED": "true"}, price_provider=lambda s: 50.0))
    c = _live_connector(path, name)
    out = c.submit_order("BTC/USDT", 0.1, "buy")
    assert isinstance(out, DryRunResult) and out.status.value == "dry_run"
    assert c.client.mock_calls == []


@pytest.mark.parametrize("path,name", CONNECTORS)
def test_over_limit_order_is_blocked_even_when_live_and_not_dry_run(path, name):
    set_guard(ExecutionGuard(env=ENABLED_LIVE, price_provider=lambda s: 60_000.0))
    c = _live_connector(path, name)
    with pytest.raises(OrderBlockedError, match="MAX_ORDER_NOTIONAL_USD"):
        c.submit_order("BTC/USDT", 1.0, "buy")
    assert c.client.mock_calls == []


@pytest.mark.parametrize("path,name", CONNECTORS)
def test_a_fully_enabled_in_limit_order_does_reach_the_exchange_once(path, name):
    set_guard(ExecutionGuard(env=ENABLED_LIVE, price_provider=lambda s: 50.0))
    c = _live_connector(path, name)
    c.submit_order("BTC/USDT", 0.5, "buy")
    submitted = [m for m in c.client.mock_calls if "order" in m[0]]
    assert len(submitted) == 1


@pytest.mark.parametrize("path,name", CONNECTORS)
def test_paper_mode_connectors_bypass_the_guard(path, name):
    set_guard(ExecutionGuard(env={}, price_provider=lambda s: 1.0))
    c = _live_connector(path, name)
    c.paper_mode = True
    with pytest.raises(RuntimeError, match="paper_mode"):          # the connector's own paper-mode message
        c.submit_order("BTC/USDT", 0.001, "buy")


# ------------------------------------------------ structural: no unguarded live path

def _submit_order_defs():
    for path in sorted(Path(__file__).parent.joinpath("brokers").glob("*.py")):
        tree = ast.parse(path.read_text())
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "submit_order"]:
                decos = [d.func.id for d in fn.decorator_list
                         if isinstance(d, ast.Call) and isinstance(d.func, ast.Name)]
                yield path.name, cls.name, decos


def test_every_live_connectors_submit_order_is_guarded():
    found = list(_submit_order_defs())
    assert len(found) >= 5
    unguarded = [(f, c) for f, c, decos in found if c != "SimulatedBroker" and "guarded_live_order" not in decos]
    assert unguarded == [], f"live connector(s) can reach an exchange without the guard: {unguarded}"


# ---------------------------------------------------------------------- Alpaca

@pytest.fixture
def alpaca(monkeypatch):
    """brokers.alpaca_connector imported against stub alpaca-py modules (alpaca-py may not be installed)."""
    class OrderSide:
        BUY, SELL = "BUY", "SELL"
    class MarketOrderRequest:
        def __init__(self, **kw):
            self.__dict__.update(kw)
    stubs = {
        "alpaca": types.ModuleType("alpaca"),
        "alpaca.trading": types.ModuleType("alpaca.trading"),
        "alpaca.trading.client": types.SimpleNamespace(TradingClient=MagicMock(name="TradingClient")),
        "alpaca.trading.requests": types.SimpleNamespace(MarketOrderRequest=MarketOrderRequest),
        "alpaca.trading.enums": types.SimpleNamespace(OrderSide=OrderSide),
    }
    for k, v in stubs.items():
        monkeypatch.setitem(sys.modules, k, v)
    sys.modules.pop("brokers.alpaca_connector", None)
    mod = importlib.import_module("brokers.alpaca_connector")
    yield mod
    sys.modules.pop("brokers.alpaca_connector", None)


def test_alpaca_buy_is_a_buy_not_a_sell(alpaca):
    set_guard(ExecutionGuard(env=ENABLED_LIVE, price_provider=lambda s: 10.0))
    c = alpaca.AlpacaConnector("k", "s", paper=False)              # live account -> guarded
    for side, expected in (("buy", "BUY"), ("long", "BUY"), ("sell", "SELL"), ("short", "SELL")):
        c.client.submit_order.reset_mock()
        c.submit_order("AAPL", 1, side)
        assert c.client.submit_order.call_args[0][0].side == expected, side
    with pytest.raises(OrderBlockedError, match="side"):            # live account: the guard rejects first
        c.submit_order("AAPL", 1, "hodl")
    paper = alpaca.AlpacaConnector("k", "s", paper=True)
    with pytest.raises(ValueError):                                 # paper account: the connector rejects
        paper.submit_order("AAPL", 1, "hodl")


def test_alpaca_live_account_is_blocked_by_default_but_paper_account_is_not(alpaca):
    set_guard(ExecutionGuard(env={}, price_provider=lambda s: 10.0))
    live = alpaca.AlpacaConnector("k", "s", paper=False)
    with pytest.raises(OrderBlockedError):
        live.submit_order("AAPL", 1, "buy")
    live.client.submit_order.assert_not_called()
    paper = alpaca.AlpacaConnector("k", "s", paper=True)          # no real money at risk
    paper.submit_order("AAPL", 1, "buy")
    paper.client.submit_order.assert_called_once()


# ------------------------------------------------------------------ decorator basics

def test_decorator_preserves_the_function_and_marks_it():
    class C:
        paper_mode = False
        @guarded_live_order("Fake")
        def submit_order(self, symbol, qty, side, order_type="market"):
            """doc"""
            return "sent"
    assert C.submit_order.__name__ == "submit_order" and C.submit_order.__doc__ == "doc"
    assert C.submit_order.__guarded_live_order__ is True
    set_guard(ExecutionGuard(env=ENABLED_LIVE, price_provider=lambda s: 1.0))
    assert C().submit_order("S", 1, "buy") == "sent"


def test_get_guard_returns_a_singleton():
    set_guard(None)
    assert get_guard() is get_guard()
