"""IBKRConnector wiring into BrokerManager, against a MOCKED ib_insync (no TWS/Gateway needed)."""
import importlib
import sys
import types
from types import SimpleNamespace as NS
from unittest import mock

import pytest


@pytest.fixture
def ib_env(monkeypatch):
    """Install a fake ib_insync module, reload the connector + manager so they import it."""
    fake_ib = mock.MagicMock(name="IB()")
    fake_ib.isConnected.return_value = False
    mod = types.ModuleType("ib_insync")
    mod.IB = mock.MagicMock(return_value=fake_ib)
    mod.MarketOrder = lambda action, qty: NS(action=action, totalQuantity=qty)
    mod.Contract = lambda **kw: NS(**kw)
    monkeypatch.setitem(sys.modules, "ib_insync", mod)
    import brokers.ib_connector as ibc
    import core.broker_manager as bm
    importlib.reload(ibc)
    importlib.reload(bm)
    yield NS(ib=fake_ib, mod=mod, bm=bm, ibc=ibc)
    monkeypatch.delitem(sys.modules, "ib_insync", raising=False)
    sys.modules.pop("brokers.ib_connector", None)                # drop the copy bound to the fake ib_insync
    # restore the "ib_insync not installed" state so later tests see the real environment
    importlib.reload(importlib.import_module("core.broker_manager"))


def make_position(sym, qty, cost, sec="STK"):
    return NS(position=qty, avgCost=cost, contract=NS(symbol=sym, secType=sec, currency="USD", exchange="SMART"))


def test_not_registered_unless_enabled(ib_env, monkeypatch):
    monkeypatch.delenv("IBKR_ENABLED", raising=False)
    bm = ib_env.bm.BrokerManager()
    assert bm.brokers["IBKR"] is None and "IBKR" not in bm.get_available_brokers()
    ib_env.mod.IB.assert_not_called()                            # never even opened a socket


def test_enabled_by_argument_connects_with_defaults_and_registers(ib_env, monkeypatch):
    for k in ("IBKR_HOST", "IBKR_PORT", "IBKR_CLIENT_ID"):
        monkeypatch.delenv(k, raising=False)
    bm = ib_env.bm.BrokerManager(ibkr_enabled=True)
    assert "IBKR" in bm.get_available_brokers()
    ib_env.ib.connect.assert_called_once_with("127.0.0.1", 7497, 1)


def test_enabled_by_env_with_custom_settings(ib_env, monkeypatch):
    monkeypatch.setenv("IBKR_ENABLED", "1"); monkeypatch.setenv("IBKR_PORT", "4002")
    monkeypatch.setenv("IBKR_HOST", "10.0.0.5"); monkeypatch.setenv("IBKR_CLIENT_ID", "7")
    ib_env.bm.BrokerManager()
    ib_env.ib.connect.assert_called_once_with("10.0.0.5", 4002, 7)


def test_connection_failure_leaves_ibkr_unconfigured_not_crashing(ib_env):
    ib_env.ib.connect.side_effect = ConnectionRefusedError("TWS not running")
    bm = ib_env.bm.BrokerManager(ibkr_enabled=True)
    assert bm.brokers["IBKR"] is None
    with pytest.raises(ValueError):
        bm.get_broker("IBKR")


def test_portfolio_shape_matches_other_brokers(ib_env):
    ib_env.ib.accountSummary.return_value = [NS(tag="AvailableFunds", value="12345.67"), NS(tag="NetLiquidation", value="20000.5"),
                                             NS(tag="BuyingPower", value="50000")]
    ib_env.ib.positions.return_value = [make_position("AAPL", 10, 150.0), make_position("AAPL", 1, 3.2, "OPT")]
    entry = ib_env.bm.BrokerManager(ibkr_enabled=True).get_portfolio()["IBKR"]
    assert entry["cash"] == 12345.67 and isinstance(entry["cash"], float)     # IB reports strings -> floats
    assert entry["portfolio_value"] == 20000.5
    assert entry["positions"]["AAPL"]["qty"] == 10.0
    assert entry["positions"]["AAPL:OPT"]["sec_type"] == "OPT"                # second AAPL contract not overwritten


def test_portfolio_error_isolated(ib_env):
    ib_env.ib.accountSummary.side_effect = RuntimeError("socket closed")
    entry = ib_env.bm.BrokerManager(ibkr_enabled=True).get_portfolio()
    assert "error" in entry["IBKR"] and "Simulator" in entry                  # one broken broker never hides the others


def test_missing_ib_insync_is_handled(monkeypatch):
    import core.broker_manager as bm
    monkeypatch.setattr(bm, "_IBKR_AVAILABLE", False)
    assert bm.BrokerManager(ibkr_enabled=True).brokers["IBKR"] is None


def test_ibkr_orders_still_blocked_by_guard(ib_env, monkeypatch):
    """Wiring must not open a live-order path: with the default env the kill switch blocks before ib_insync is touched."""
    from brokers.execution_guard import OrderBlockedError, set_guard
    monkeypatch.delenv("LIVE_TRADING_ENABLED", raising=False)
    set_guard(None)                                              # fresh guard reading the (default-off) environment
    try:
        broker = ib_env.bm.BrokerManager(ibkr_enabled=True).get_broker("IBKR")
        with pytest.raises(OrderBlockedError):
            broker.submit_order("AAPL", 1, "buy")
        ib_env.ib.placeOrder.assert_not_called()
    finally:
        set_guard(None)
