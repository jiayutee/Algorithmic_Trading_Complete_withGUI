"""Phases 4.3 / 4.4: single-leg option orders, futures contracts, margin fields -- against a MOCKED ib_insync only.

The point of most tests here is the live-order guard: an option contract controls 100 shares and a future its contract size, so
dollar caps must use qty x price x multiplier, and the guard must never value a derivative with the UNDERLYING's stock price."""
from datetime import date
from types import SimpleNamespace as NS

import pytest

from brokers.execution_guard import DryRunResult, ExecutionGuard, OrderBlockedError, set_guard
from core.instrument import AssetClass, Instrument, OptionType

LIVE = {"LIVE_TRADING_ENABLED": "true", "LIVE_DRY_RUN": "false", "KILL_SWITCH_FILE": "/tmp/_ibkr_optfut_no_ks",
        "MAX_ORDER_NOTIONAL_USD": "100", "MAX_SESSION_NOTIONAL_USD": "1000", "MAX_ORDERS_PER_MINUTE": "50"}


def call(strike=200.0, expiry=date(2026, 12, 18), kind=OptionType.CALL, **kw):
    return Instrument(symbol="AAPL261218C00200000", asset_class=AssetClass.OPTION, underlying="AAPL", strike=strike,
                      expiry=expiry, option_type=kind, exchange="SMART", **kw)


def es(size=50.0, exchange="CME", **kw):
    return Instrument(symbol="ESZ6", asset_class=AssetClass.FUTURE, underlying="ES", expiry=date(2026, 12, 18),
                      contract_size=size, exchange=exchange, **kw)


def trade(status="Submitted"):
    return NS(order=NS(orderId=7), orderStatus=NS(status=status, filled=0, remaining=1, avgFillPrice=0.0))


@pytest.fixture
def live_guard():
    def make(**env):
        g = ExecutionGuard(env={**LIVE, **env}, price_provider=lambda s: pytest.fail("the underlying's stock price must not be used"))
        set_guard(g)
        return g
    yield make
    set_guard(None)


@pytest.fixture
def conn(ib_env):
    c = ib_env.ibc.IBKRConnector()
    c.paper_mode = False
    ib_env.ib.placeOrder.return_value = trade()
    return c


# ------------------------------------------------------------------ guard arithmetic
def test_option_notional_includes_the_100_share_multiplier(conn, ib_env, live_guard):
    live_guard()                                                    # cap $100 per order
    with pytest.raises(OrderBlockedError, match=r"\$500"):          # 1 contract x $5.00 premium x 100 = $500
        conn.submit_instrument_order(call(), 1, "buy", price=5.0)
    ib_env.ib.placeOrder.assert_not_called()


def test_option_within_the_cap_is_sent_and_counted_at_multiplied_value(conn, ib_env, live_guard):
    g = live_guard()
    conn.submit_instrument_order(call(), 1, "buy", price=0.5)       # $50
    ib_env.ib.placeOrder.assert_called_once()
    assert g._session_notional["IBKR"] == pytest.approx(50.0)


def test_future_notional_uses_contract_size(conn, ib_env, live_guard):
    live_guard()
    with pytest.raises(OrderBlockedError, match=r"250,000"):        # 1 x 5000 x 50
        conn.submit_instrument_order(es(), 1, "buy", price=5000.0)
    ib_env.ib.placeOrder.assert_not_called()


def test_derivative_without_price_is_refused_not_priced_as_the_underlying(conn, ib_env, live_guard):
    live_guard(MAX_ORDER_NOTIONAL_USD="1000000")
    with pytest.raises(ValueError, match="price"):
        conn.submit_instrument_order(call(), 1, "buy", price=None)
    ib_env.ib.placeOrder.assert_not_called()


def test_legacy_submit_order_for_a_derivative_sec_type_is_refused(conn, ib_env, live_guard):
    live_guard()                                                    # provider would fail the test if consulted
    for sec in ("OPT", "FUT"):
        with pytest.raises(OrderBlockedError, match="LIMIT orders"):        # legacy path: market order AND no price to value it with
            conn.submit_order("AAPL", 1, "buy", sec_type=sec)
    ib_env.ib.placeOrder.assert_not_called()


def test_stock_order_valuation_is_unchanged(ib_env):
    g = ExecutionGuard(env=LIVE, price_provider=lambda s: 10.0)
    assert g.evaluate("IBKR", "AAPL", 2, "buy").notional == pytest.approx(20.0)


def test_bad_multiplier_is_refused():
    g = ExecutionGuard(env=LIVE, price_provider=lambda s: 1.0)
    d = g.evaluate("IBKR", "X", 1, "buy", price=1.0, multiplier=float("nan"))
    assert d.action == "block" and any("multiplier" in r for r in d.reasons)


# ------------------------------------------------------------------ the guard still blocks by default
@pytest.mark.parametrize("inst", [call(), es()])
def test_default_environment_blocks_before_ib_is_touched(ib_env, monkeypatch, inst):
    monkeypatch.delenv("LIVE_TRADING_ENABLED", raising=False)
    set_guard(None)
    try:
        c = ib_env.ibc.IBKRConnector()
        with pytest.raises(OrderBlockedError):
            c.submit_instrument_order(inst, 1, "buy", price=1.0)
        ib_env.ib.placeOrder.assert_not_called()
    finally:
        set_guard(None)


def test_dry_run_is_the_default_when_live_is_enabled(ib_env):
    set_guard(ExecutionGuard(env={"LIVE_TRADING_ENABLED": "true", "KILL_SWITCH_FILE": "/tmp/_ibkr_optfut_no_ks"}))
    try:
        c = ib_env.ibc.IBKRConnector()
        r = c.submit_instrument_order(call(), 1, "buy", price=0.5)
        assert isinstance(r, DryRunResult)
        ib_env.ib.placeOrder.assert_not_called()
    finally:
        set_guard(None)


# ------------------------------------------------------------------ contract construction (guard bypassed via paper_mode)
@pytest.fixture
def pconn(ib_env):
    c = ib_env.ibc.IBKRConnector()
    c.paper_mode = True
    ib_env.ib.placeOrder.return_value = trade("Filled")
    return c


@pytest.mark.parametrize("kind,right,side,action", [(OptionType.CALL, "C", "buy", "BUY"), (OptionType.CALL, "C", "sell", "SELL"),
                                                    (OptionType.PUT, "P", "buy", "BUY"), (OptionType.PUT, "P", "sell", "SELL")])
def test_single_leg_option_orders_build_the_right_contract(ib_env, pconn, kind, right, side, action):
    if side == "sell":                                               # selling is only allowed to close a long position you hold
        ib_env.ib.positions.return_value = [held("OPT", "AAPL", 2, "20261218", 185.5, right, "100")]
    r = pconn.submit_instrument_order(call(strike=185.5, kind=kind), 2, side, price=3.1)
    contract, order = ib_env.ib.placeOrder.call_args[0]
    assert (contract.secType, contract.symbol, contract.strike, contract.right) == ("OPT", "AAPL", 185.5, right)
    assert contract.lastTradeDateOrContractMonth == "20261218" and contract.multiplier == "100"
    assert order.action == action and order.totalQuantity == 2
    assert r["side"] == action and r["order_id"] == 7 and r["sec_type"] == "OPT"          # same shape as submit_order + contract info
    assert {"order_id", "symbol", "quantity", "side", "status", "filled", "remaining", "avg_fill_price"} <= set(r)


def test_orders_are_limit_orders_and_carry_their_price(ib_env, pconn):
    pconn.submit_instrument_order(call(), 1, "buy", price=2.25)                      # default is LMT
    order = ib_env.ib.placeOrder.call_args[0][1]
    assert order.orderType == "LMT" and order.lmtPrice == 2.25


def test_future_contract_uses_contract_month_multiplier_and_exchange(ib_env, pconn):
    pconn.submit_instrument_order(es(), 1, "sell", price=5000.0)
    contract, order = ib_env.ib.placeOrder.call_args[0]
    assert (contract.secType, contract.symbol, contract.exchange) == ("FUT", "ES", "CME")
    assert contract.lastTradeDateOrContractMonth == "202612" and contract.multiplier == "50"
    assert order.action == "SELL"


@pytest.mark.parametrize("inst,why", [
    (Instrument(symbol="AAPL", asset_class=AssetClass.EQUITY), "OPTION and FUTURE only"),
    (call(strike=None), "strike"),
    (call(strike=0.0), "strike"),
    (Instrument(symbol="X", asset_class=AssetClass.OPTION, underlying="AAPL", strike=1.0, option_type=OptionType.PUT), "expiry"),
    (es(exchange=""), "exchange"),
    (es(size=None), "contract_size"),
])
def test_incomplete_or_wrong_instruments_are_rejected_before_anything_is_sent(ib_env, pconn, inst, why):
    with pytest.raises(ValueError, match=why):
        pconn.submit_instrument_order(inst, 1, "buy", price=1.0)
    ib_env.ib.placeOrder.assert_not_called()


def test_bad_order_type_and_missing_price_are_rejected(ib_env, pconn):
    for bad in ("STP", "MKT", "market"):
        with pytest.raises(ValueError, match="LIMIT"):
            pconn.submit_instrument_order(call(), 1, "buy", price=1.0, order_type=bad)
    with pytest.raises(ValueError):
        pconn.submit_instrument_order(call(), 1, "buy", price=0)
    ib_env.ib.placeOrder.assert_not_called()


# ------------------------------------------------------------------ positions round-trip and margin
def held(sec, sym, qty, expiry, strike=0.0, right="", mult="", cur="USD", exch="SMART"):
    return NS(position=qty, avgCost=1.0, contract=NS(symbol=sym, secType=sec, currency=cur, exchange=exch,
              lastTradeDateOrContractMonth=expiry, strike=strike, right=right, multiplier=mult))


def test_option_position_round_trips_and_other_strikes_do_not_match(ib_env, pconn):
    ib_env.ib.positions.return_value = [held("OPT", "AAPL", 3, "20261218", 190.0, "C", "100"),
                                        held("OPT", "AAPL", 2, "20261218", 200.0, "C", "100"),
                                        held("OPT", "AAPL", 1, "20261218", 200.0, "P", "100")]
    p = pconn.get_position("AAPL", instrument=call(strike=200.0))
    assert p["position"] == 2 and p["contract"]["strike"] == 200.0 and p["contract"]["right"] == "C"
    assert p["contract"]["lastTradeDateOrContractMonth"] == "20261218" and p["contract"]["multiplier"] == "100"
    assert pconn.get_position("AAPL", instrument=call(strike=205.0)) is None


def test_future_position_matches_on_contract_month_prefix(ib_env, pconn):
    ib_env.ib.positions.return_value = [held("FUT", "ES", -2, "20261218", mult="50", exch="CME"),
                                        held("FUT", "ES", 5, "20270319", mult="50", exch="CME")]
    p = pconn.get_position("ES", instrument=es())
    assert p["position"] == -2 and p["contract"]["multiplier"] == "50"
    assert pconn.get_position("ES", sec_type="FUT")["position"] == -2        # legacy lookup by symbol still works


def test_account_info_surfaces_margin_fields_and_keeps_the_old_keys(ib_env, pconn):
    ib_env.ib.accountSummary.return_value = [NS(tag=t, value=v) for t, v in
        [("NetLiquidation", "100000"), ("BuyingPower", "400000"), ("GrossPositionValue", "250000"), ("AvailableFunds", "80000"),
         ("InitMarginReq", "20000"), ("MaintMarginReq", "18000"), ("ExcessLiquidity", "82000"), ("Cushion", "0.82")]]
    info = pconn.get_account_info()
    assert info["initial_margin_req"] == "20000" and info["maintenance_margin_req"] == "18000"
    assert info["excess_liquidity"] == "82000" and info["cushion"] == "0.82"
    assert info["net_liquidation"] == "100000" and info["leverage"] == "250000"      # unchanged keys
    ib_env.ib.accountSummary.return_value = [NS(tag="NetLiquidation", value="1")]
    assert pconn.get_account_info()["initial_margin_req"] is None                    # not reported -> None, not an error


# ------------------------------------------------------------------ hardening from the PR review
def test_guard_refuses_a_market_order_on_a_derivative_even_with_a_tiny_stated_price():
    """A market order for 100 contracts stated at $0.01 used to value at $100 and slip under a $100 cap (real value ~ $50,000)."""
    g = ExecutionGuard(env=LIVE, price_provider=lambda s: pytest.fail("underlying price must not be used"))
    d = g.evaluate("IBKR", "AAPL", 100, "buy", "MKT", price=0.01, multiplier=100.0, require_price=True)
    assert d.action == "block" and any("LIMIT orders" in r for r in d.reasons)
    ok = g.evaluate("IBKR", "AAPL", 1, "buy", "LMT", price=0.5, multiplier=100.0, require_price=True)
    assert ok.action == "allow" and ok.notional == pytest.approx(50.0)


def test_stocks_may_still_use_market_orders():
    g = ExecutionGuard(env=LIVE, price_provider=lambda s: 10.0)
    assert g.evaluate("IBKR", "AAPL", 2, "buy", "MKT").action == "allow"


def test_the_exploit_end_to_end_a_market_order_is_refused_before_ib_is_touched(conn, ib_env, live_guard):
    live_guard()
    with pytest.raises(ValueError, match="LIMIT"):
        conn.submit_instrument_order(call(), 100, "buy", price=0.01, order_type="MKT")
    ib_env.ib.placeOrder.assert_not_called()


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
def test_selling_an_option_to_open_a_short_is_refused_in_every_mode(ib_env, kind, live_guard):
    ib_env.ib.positions.return_value = []
    for paper, dry in ((True, False), (False, False), (False, True)):
        live_guard(LIVE_DRY_RUN="true" if dry else "false")
        c = ib_env.ibc.IBKRConnector()
        c.paper_mode = paper
        with pytest.raises(ValueError, match="open a short option"):
            c.submit_instrument_order(call(kind=kind), 1, "sell", price=0.05)         # $5 'value' of unlimited risk
    ib_env.ib.placeOrder.assert_not_called()


def test_selling_to_close_a_long_option_is_allowed_but_not_more_than_you_hold(ib_env, pconn):
    ib_env.ib.positions.return_value = [held("OPT", "AAPL", 2, "20261218", 200.0, "C", "100")]
    pconn.submit_instrument_order(call(), 2, "sell", price=1.0)
    ib_env.ib.placeOrder.assert_called_once()
    ib_env.ib.placeOrder.reset_mock()
    with pytest.raises(ValueError, match="you hold 2"):
        pconn.submit_instrument_order(call(), 3, "sell", price=1.0)
    ib_env.ib.placeOrder.assert_not_called()


def test_a_long_position_in_a_different_strike_does_not_license_the_sale(ib_env, pconn):
    ib_env.ib.positions.return_value = [held("OPT", "AAPL", 5, "20261218", 150.0, "C", "100")]      # holds the 150 call, sells the 200
    with pytest.raises(ValueError, match="open a short option"):
        pconn.submit_instrument_order(call(strike=200.0), 1, "sell", price=1.0)


def test_futures_sells_are_unaffected_by_the_option_rule(ib_env, pconn):
    ib_env.ib.positions.return_value = []
    pconn.submit_instrument_order(es(), 1, "sell", price=5000.0)                      # notional is the real exposure for a future
    ib_env.ib.placeOrder.assert_called_once()


def test_the_private_placement_method_also_enforces_the_short_rule(ib_env, pconn):
    ib_env.ib.positions.return_value = []
    with pytest.raises(ValueError, match="open a short option"):
        pconn._place_instrument_order(symbol="AAPL261218C00200000", qty=1, side="sell", instrument=call(), order_type="LMT",
                                      price=1.0, _contract=pconn.contract_for(call()))
    ib_env.ib.placeOrder.assert_not_called()
