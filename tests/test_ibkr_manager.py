"""IBKRConnector wiring into BrokerManager, against a MOCKED ib_insync (no TWS/Gateway needed)."""
from types import SimpleNamespace as NS

import pytest


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


# ---------------------------------------------------------------- connector-level behaviour (guard bypassed via paper_mode)

@pytest.fixture
def conn(ib_env):
    c = ib_env.ibc.IBKRConnector()
    c.paper_mode = True                       # exercise the order logic itself; guard behaviour is tested above
    return c


def _trade(ib_env, status="Filled"):
    return NS(order=NS(orderId=42), orderStatus=NS(status=status, filled=5, remaining=0, avgFillPrice=101.5))


@pytest.mark.parametrize("side,qty,expected", [("buy", 5, "BUY"), ("long", 5, "BUY"), ("sell", 5, "SELL"), ("short", 5, "SELL")])
def test_submit_order_side_mapping(ib_env, conn, side, qty, expected):
    ib_env.ib.placeOrder.return_value = _trade(ib_env)
    r = conn.submit_order("AAPL", qty, side)
    order = ib_env.ib.placeOrder.call_args[0][1]
    assert order.action == expected and order.totalQuantity == 5
    assert r["side"] == expected and r["order_id"] == 42 and r["avg_fill_price"] == 101.5


def test_submit_order_builds_contract_with_given_venue(ib_env, conn):
    ib_env.ib.placeOrder.return_value = _trade(ib_env)
    conn.submit_order("EUR", 1, "buy", sec_type="CASH", currency="USD", exchange="IDEALPRO")
    contract = ib_env.ib.placeOrder.call_args[0][0]
    assert (contract.symbol, contract.secType, contract.currency, contract.exchange) == ("EUR", "CASH", "USD", "IDEALPRO")


def test_get_position_matches_on_symbol_type_and_currency(ib_env, conn):
    ib_env.ib.positions.return_value = [make_position("AAPL", 3, 10.0, "OPT"), make_position("AAPL", 7, 150.0)]
    assert conn.get_position("AAPL")["position"] == 7            # the STK one, not the option
    assert conn.get_position("AAPL", sec_type="OPT")["position"] == 3
    assert conn.get_position("MSFT") is None


def test_account_info_missing_tags_are_none(ib_env, conn):
    ib_env.ib.accountSummary.return_value = [NS(tag="NetLiquidation", value="1")]
    info = conn.get_account_info()
    assert info["net_liquidation"] == "1" and info["buying_power"] is None


def test_connect_is_idempotent_and_context_manager_disconnects(ib_env):
    ib_env.ib.isConnected.return_value = True
    with ib_env.ibc.IBKRConnector() as c:
        assert c is not None
    ib_env.ib.connect.assert_not_called()                        # already connected -> no second connect
    ib_env.ib.disconnect.assert_called_once()


def test_no_live_connection_is_ever_attempted_by_the_suite():
    """4.5 audit: the only ib.connect() call site is IBKRConnector.connect; tests always mock ib_insync."""
    import ast, pathlib
    calls = []
    for p in pathlib.Path(".").rglob("*.py"):
        if any(part in (".git", ".claude", "node_modules") for part in p.parts):
            continue
        try:
            tree = ast.parse(p.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "connect" \
                    and isinstance(n.func.value, ast.Attribute) and n.func.value.attr == "ib":
                calls.append(str(p))
    assert calls == ["brokers/ib_connector.py"]


# ---------------------------------------------------------------------------
# Gap 3: live-order-guard price gap (Phase 4.5 IBKR mock-harness gap-check)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("symbol", [
    "AAPL260101C00200000",  # option-style ticker
    "ESH25",                # futures-style ticker
    "EUR",                  # FX-style ticker (base currency)
])
def test_ibkr_unresolvable_price_refused_before_placeorder(ib_env, monkeypatch, symbol):
    """When live trading is enabled but the guard cannot determine a price for the
    symbol, the order is BLOCKED and ib.placeOrder is never called.

    This pins the CURRENT fail-closed behaviour for IBKRConnector.submit_order:
    the method has no ``price`` parameter, so the guard always queries its price
    provider (DataLoader.get_latest_price) to compute the notional.  When that
    lookup returns None (exotic symbol, network error, …) the guard refuses the
    order rather than sending it with an unverifiable value.

    Production code (execution_guard.py, ib_connector.py) is unchanged; this
    test only observes the existing behaviour.
    """
    from brokers.execution_guard import ExecutionGuard, OrderBlockedError, set_guard

    # Inject a guard with live trading fully enabled but no price resolution.
    test_guard = ExecutionGuard(
        env={"LIVE_TRADING_ENABLED": "true", "LIVE_DRY_RUN": "false",
             "KILL_SWITCH_FILE": "/tmp/_ibkr_gapcheck_no_ks"},
        price_provider=lambda s: None,   # simulates a symbol whose price cannot be fetched
    )
    set_guard(test_guard)
    try:
        conn = ib_env.ibc.IBKRConnector()
        conn.paper_mode = False   # exercise the live-order guard path

        with pytest.raises(OrderBlockedError, match="cannot determine"):
            conn.submit_order(symbol, 1, "buy")

        # The order must have been stopped BEFORE ib.placeOrder was ever reached.
        ib_env.ib.placeOrder.assert_not_called()
    finally:
        set_guard(None)
