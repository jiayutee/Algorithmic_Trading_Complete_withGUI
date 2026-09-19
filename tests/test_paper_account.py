"""Truthful, durable paper account: real prices only, rejects instead of inventing, re-checks pending orders, survives restarts."""
import time

import pytest

from brokers.simulatedbroker import OrderStatus, SimulatedBroker


def make(**kw):
    b = SimulatedBroker(**kw)
    return b


# ------------------------------------------------------------------ no invented prices

def test_prices_do_not_move_on_their_own_by_default():
    b = make()
    b.update_price("AAPL", 150.0)
    time.sleep(1.3)                                   # the old thread would have moved it within a second
    assert b.market_data["AAPL"] == 150.0
    assert not b._data_thread.is_alive()
    b.close()


def test_random_walk_is_opt_in():
    b = make(simulate_prices=True)
    b.update_price("AAPL", 150.0)
    time.sleep(1.4)
    assert b._data_thread.is_alive() and b.market_data["AAPL"] != 150.0
    b.close()


def test_strict_mode_rejects_a_market_order_with_no_price_instead_of_filling_at_100():
    b = make(strict_prices=True)
    o = b.submit_order("XYZ", 1, "buy")
    assert o.status == OrderStatus.REJECTED and "no current price" in o.reject_reason
    assert b.balance == b.initial_balance and not b.positions
    # default (non-strict) keeps the old behaviour so existing callers are unaffected
    assert make().submit_order("XYZ", 1, "buy").status == OrderStatus.FILLED


def test_strict_mode_fills_at_the_fed_price_and_rejects_a_stale_one():
    b = make(strict_prices=True, max_price_age_s=60)
    b.update_price("AAPL", 150.0)
    assert b.submit_order("AAPL", 2, "buy").filled_avg_price == 150.0
    b._price_time["AAPL"] -= 120                      # the quote is now two minutes old
    o = b.submit_order("AAPL", 1, "buy")
    assert o.status == OrderStatus.REJECTED and "stale" in o.reject_reason
    b.update_price("AAPL", 151.0)                     # a fresh quote makes it tradable again
    assert b.submit_order("AAPL", 1, "buy").status == OrderStatus.FILLED


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -5.0, 0.0, "x", None])
def test_update_price_ignores_garbage(bad):
    b = make()
    b.update_price("AAPL", 100.0)
    b.update_price("AAPL", bad)
    assert b.market_data["AAPL"] == 100.0


def test_positions_are_marked_to_fed_prices():
    b = make(strict_prices=True)
    b.update_price("AAPL", 100.0)
    b.submit_order("AAPL", 10, "buy")
    b.update_price("AAPL", 110.0)
    assert b.get_unrealized_pnl() == pytest.approx(100.0)
    assert b.get_position("AAPL").last_price == 110.0


# ------------------------------------------------------------------ pending orders are re-checked

def test_limit_buy_waits_then_fills_when_the_price_drops_to_it():
    b = make(strict_prices=True)
    b.update_price("AAPL", 100.0)
    o = b.submit_order("AAPL", 5, "buy", order_type="limit", limit_price=95.0)
    assert o.status == OrderStatus.PENDING
    b.update_price("AAPL", 97.0)
    assert o.status == OrderStatus.PENDING            # not yet
    b.update_price("AAPL", 94.0)
    assert o.status == OrderStatus.FILLED and o.filled_avg_price == 94.0   # fills at the better price
    assert b.get_position("AAPL").qty == 5


def test_stop_sell_triggers_when_price_falls_through_it_and_limit_without_price_stays_pending():
    b = make(strict_prices=True)
    b.update_price("AAPL", 100.0)
    b.submit_order("AAPL", 5, "buy")
    stop = b.submit_order("AAPL", 5, "sell", order_type="stop", stop_price=90.0)
    assert stop.status == OrderStatus.PENDING
    b.update_price("AAPL", 89.0)
    assert stop.status == OrderStatus.FILLED and not b.positions
    waiting = b.submit_order("NEW", 1, "buy", order_type="limit", limit_price=10.0)
    assert waiting.status == OrderStatus.PENDING      # no price yet: waits instead of being rejected or filled at $100
    b.update_price("NEW", 9.0)
    assert waiting.status == OrderStatus.FILLED


def test_cancelled_pending_order_is_not_filled_later():
    b = make(strict_prices=True)
    b.update_price("AAPL", 100.0)
    o = b.submit_order("AAPL", 5, "buy", order_type="limit", limit_price=90.0)
    assert b.cancel_order(o.id)
    b.update_price("AAPL", 80.0)
    assert o.status == OrderStatus.CANCELED and not b.positions


# ------------------------------------------------------------------ durability

def test_account_survives_a_restart(tmp_path):
    path = str(tmp_path / "acct.sqlite3")
    b = make(persist_path=path, strict_prices=True)
    b.update_price("AAPL", 100.0)
    b.submit_order("AAPL", 10, "buy", rationale={"summary": "test why"})
    b.update_price("AAPL", 120.0)
    b.submit_order("AAPL", 4, "sell")
    b.submit_order("AAPL", 1, "buy", order_type="limit", limit_price=50.0)        # stays pending
    info = b.get_account_info()
    b.close()

    r = make(persist_path=path, strict_prices=True)                                 # "restart"
    info2 = r.get_account_info()
    assert info2["balance"] == pytest.approx(info["balance"])
    assert info2["realized_pnl"] == pytest.approx(info["realized_pnl"]) and info2["realized_pnl"] > 0
    assert r.get_position("AAPL").qty == pytest.approx(6)
    orders = r.get_orders()
    assert [o.status.value for o in orders] == ["filled", "filled", "pending"]
    assert orders[0].rationale["summary"] == "test why" and orders[0].side.value == "buy"
    r.update_price("AAPL", 49.0)                                                    # the restored pending order still works
    assert r.get_orders()[2].status == OrderStatus.FILLED
    r.close()


def test_saved_prices_are_not_restored_only_the_account(tmp_path):
    path = str(tmp_path / "acct.sqlite3")
    b = make(persist_path=path, strict_prices=True)
    b.update_price("AAPL", 100.0)
    b.submit_order("AAPL", 1, "buy")
    b.close()
    r = make(persist_path=path, strict_prices=True)
    assert "AAPL" not in r.market_data                # a stale saved mark would be exactly the fake price we removed
    assert r.submit_order("AAPL", 1, "buy").status == OrderStatus.REJECTED
    r.close()


def test_rejections_and_reasons_are_persisted(tmp_path):
    path = str(tmp_path / "acct.sqlite3")
    b = make(persist_path=path, strict_prices=True)
    b.submit_order("NOPRICE", 1, "buy")
    b.close()
    r = make(persist_path=path)
    o = r.get_orders()[0]
    assert o.status == OrderStatus.REJECTED and "no current price" in o.reject_reason
    r.close()


def test_reset_wipes_the_account_durably(tmp_path):
    path = str(tmp_path / "acct.sqlite3")
    b = make(persist_path=path, strict_prices=True)
    b.update_price("AAPL", 100.0)
    b.submit_order("AAPL", 3, "buy")
    b.reset(50_000)
    assert b.get_account_info()["balance"] == 50_000 and not b.get_orders()
    b.close()
    r = make(persist_path=path)
    assert r.get_account_info()["balance"] == 50_000 and not r.positions and not r.get_orders()
    r.close()


def test_two_processes_share_one_account(tmp_path):
    """Two broker instances on one file stand in for the desktop app and the Dash view."""
    path = str(tmp_path / "acct.sqlite3")
    a = make(persist_path=path, strict_prices=True)
    b = make(persist_path=path, strict_prices=True)
    a.update_price("AAPL", 100.0)
    b.update_price("AAPL", 100.0)
    a.submit_order("AAPL", 10, "buy")
    assert b.get_position("AAPL").qty == 10                       # b sees a's trade without reopening anything
    b.submit_order("AAPL", 4, "sell")
    assert a.get_position("AAPL").qty == pytest.approx(6)         # and a sees b's
    assert a.get_account_info()["balance"] == pytest.approx(b.get_account_info()["balance"])
    a.submit_order("AAPL", 1, "buy"); b.submit_order("AAPL", 1, "buy"); a.submit_order("AAPL", 1, "buy")
    assert a.get_position("AAPL").qty == pytest.approx(9) == pytest.approx(b.get_position("AAPL").qty)
    assert len(a.get_orders()) == len(b.get_orders()) == 5
    a.close(); b.close()


def test_in_memory_mode_is_unchanged_when_no_path_is_given():
    b = make()
    assert b._store is None
    b.market_data["X"] = 100.0
    assert b.submit_order("X", 1, "buy").status == OrderStatus.FILLED
