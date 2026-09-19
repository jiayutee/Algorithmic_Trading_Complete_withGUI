import time

from brokers.simulatedbroker import SimulatedBroker
from core.paper_marking import HeldPriceMarker


def broker_with(*symbols):
    b = SimulatedBroker(strict_prices=True)
    for s in symbols:
        b.update_price(s, 100.0)
        b.submit_order(s, 1, "buy")
    return b


def test_marks_every_held_symbol_but_skips_the_charted_one():
    b = broker_with("AAPL", "TSLA", "BTCUSDT")
    prices = {"AAPL": 110.0, "TSLA": 90.0, "BTCUSDT": 105.0}
    m = HeldPriceMarker(prices.get, interval_for=lambda s: 0)
    assert m.refresh(b, skip=["BTCUSDT"]) == 2
    assert b.market_data["AAPL"] == 110.0 and b.market_data["TSLA"] == 90.0 and b.market_data["BTCUSDT"] == 100.0


def test_per_symbol_throttle_and_failures_do_not_raise():
    b = broker_with("AAPL", "TSLA")
    calls = []
    def fetch(s):
        calls.append(s)
        if s == "TSLA":
            raise RuntimeError("feed down")
        return 111.0
    t = [0.0]
    m = HeldPriceMarker(fetch, interval_for=lambda s: 30, clock=lambda: t[0])
    assert m.refresh(b) == 1 and sorted(calls) == ["AAPL", "TSLA"]
    assert m.refresh(b) == 0 and len(calls) == 2                  # inside the 30s window: nothing is re-fetched (even the failed one)
    t[0] = 31.0
    assert m.refresh(b) == 1 and len(calls) == 4


def test_unpriceable_symbols_and_non_paper_brokers_are_left_alone():
    b = broker_with("AAPL")
    m = HeldPriceMarker(lambda s: None, interval_for=lambda s: 0)
    assert m.refresh(b) == 0 and b.market_data["AAPL"] == 100.0
    assert m.refresh(None) == 0 and m.refresh(object()) == 0


def test_closed_positions_are_not_marked():
    b = broker_with("AAPL")
    b.submit_order("AAPL", 1, "sell")
    seen = []
    HeldPriceMarker(lambda s: seen.append(s) or 1.0, interval_for=lambda s: 0).refresh(b)
    assert seen == []


def test_async_refresh_runs_once_at_a_time():
    b = broker_with("AAPL")
    gate = []
    def slow(s):
        gate.append(1); time.sleep(0.3); return 120.0
    m = HeldPriceMarker(slow, interval_for=lambda s: 0)
    m.refresh_async(b); m.refresh_async(b)                       # second call is dropped while the first is running
    time.sleep(0.6)
    assert len(gate) == 1 and b.market_data["AAPL"] == 120.0
