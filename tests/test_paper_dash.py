"""Dash side of the truthful paper account."""
import pytest

import dash_app.callbacks as cb
from brokers.simulatedbroker import OrderStatus, SimulatedBroker


@pytest.fixture(autouse=True)
def clean_module_state(monkeypatch):
    monkeypatch.setattr(cb, "_broker", None)
    monkeypatch.setattr(cb, "_held_marker", None)
    monkeypatch.delenv("PAPER_ACCOUNT_PATH", raising=False)


def test_without_a_configured_account_nothing_is_created_on_page_load():
    assert cb._broker_or_none() is None                                # display callbacks must not create a broker


def test_with_a_configured_account_a_restart_restores_positions_on_page_load(tmp_path, monkeypatch):
    path = str(tmp_path / "acct.sqlite3")
    monkeypatch.setenv("PAPER_ACCOUNT_PATH", path)
    b = cb._get_broker()
    assert b.strict_prices and b.max_price_age_s == 300.0             # app mode: real prices only
    b.update_price("AAPL", 100.0)
    b.submit_order("AAPL", 3, "buy")
    b.close()
    monkeypatch.setattr(cb, "_broker", None)                           # "restart" the Dash process
    restored = cb._broker_or_none()                                    # a display callback alone must load it
    assert restored is not None and restored.get_position("AAPL").qty == 3


def test_sync_broker_price_goes_through_update_price_so_pending_orders_fire():
    b = SimulatedBroker(strict_prices=True)
    b.update_price("AAPL", 100.0)
    o = b.submit_order("AAPL", 2, "buy", order_type="limit", limit_price=95.0)
    cb._sync_broker_price(b, "AAPL", 94.0)
    assert o.status == OrderStatus.FILLED and b.price_age("AAPL") is not None


def test_other_holdings_are_marked_while_the_charted_symbol_is_left_to_its_own_feed(monkeypatch):
    b = SimulatedBroker(strict_prices=True)
    for s in ("AAPL", "BTCUSDT"):
        b.update_price(s, 100.0)
        b.submit_order(s, 1, "buy")

    class Svc:
        subscribed = []
        def subscribed_symbols(self): return list(self.subscribed)
        def subscribe(self, s): self.subscribed.append(s)
        def get_price(self, s): return {"BTCUSDT": 30500.0}.get(s)

    class Loader:
        def get_latest_price(self, s): return {"AAPL": 111.0}.get(s)

    svc = Svc()
    monkeypatch.setattr(cb, "_get_live_svc", lambda: svc)
    monkeypatch.setattr(cb, "_get_equity_loader", lambda: Loader())
    cb._mark_other_holdings(b, "TSLA")                                   # chart shows something else entirely
    assert b.market_data["AAPL"] == 111.0 and b.market_data["BTCUSDT"] == 30500.0
    assert "BTCUSDT" in svc.subscribed                                   # crypto holding stays streamed


def test_marking_never_raises_when_the_feed_is_broken(monkeypatch):
    b = SimulatedBroker(strict_prices=True)
    b.update_price("AAPL", 100.0); b.submit_order("AAPL", 1, "buy")
    monkeypatch.setattr(cb, "_get_equity_loader", lambda: (_ for _ in ()).throw(RuntimeError("down")))
    cb._mark_other_holdings(b, None)
    cb._mark_other_holdings(None, None)
    assert b.market_data["AAPL"] == 100.0


def test_positions_panel_refreshes_on_the_price_tick_too():
    import dash_app.app as m
    for key, c in m.app.callback_map.items():
        if key.startswith("positions-content"):
            assert {"order-status", "price-interval"} <= {i["id"] for i in c["inputs"]}
            return
    pytest.fail("positions-content callback not found")


# ------------------------------------------------------------------ held positions are marked before any chart is loaded

def _live_price_callback():
    import dash_app.app as m
    for key, c in m.app.callback_map.items():
        if key.startswith("..main-chart.figure") and "live-badge" in key and "price-interval" in str(c["inputs"]):
            return c["callback"].__wrapped__ if hasattr(c["callback"], "__wrapped__") else c["callback"]
    pytest.fail("update_live_price callback not found")


def test_no_chart_loaded_still_marks_open_positions_to_real_prices(monkeypatch):
    import dash_app.callbacks as cb
    from brokers.simulatedbroker import SimulatedBroker
    broker = SimulatedBroker(strict_prices=True)
    broker.update_price("BTCUSDT", 100.0)
    broker.submit_order("BTCUSDT", 1, "buy")
    broker.market_data.pop("BTCUSDT")                      # a fresh process: positions loaded from the file, no price known
    seen = []
    monkeypatch.setattr(cb, "_broker_or_none", lambda: broker)
    monkeypatch.setattr(cb, "_mark_other_holdings", lambda b, active: seen.append((b, active)))
    fn = cb.register_callbacks and _live_price_callback()
    out = fn(1, None)                                       # no active symbol: nothing loaded yet
    assert seen == [(broker, None)]                         # it asked for the held positions to be marked
    assert all(v is cb.no_update for v in out)              # and touched nothing in the UI
    broker.close()


def test_marking_with_no_symbol_never_raises_without_a_broker(monkeypatch):
    import dash_app.callbacks as cb
    monkeypatch.setattr(cb, "_broker_or_none", lambda: None)
    out = _live_price_callback()(1, None)
    assert all(v is cb.no_update for v in out)
