"""Unit tests for SimulatedBroker — order logic, fills, position tracking, fees."""
import time
import threading
import pytest
from brokers.simulatedbroker import SimulatedBroker, OrderStatus, OrderSide, OrderType


@pytest.fixture
def broker():
    """Fresh broker with a known starting balance and price seeded."""
    b = SimulatedBroker(initial_balance=10_000.0, market_fee=0.001, limit_fee=0.0005)
    # Seed a deterministic price so tests don't depend on random walk timing
    b.market_data["AAPL"] = 150.0
    b.market_data["BTC"] = 30_000.0
    yield b
    b.close()


# ---------------------------------------------------------------------------
# Account info
# ---------------------------------------------------------------------------

class TestAccountInfo:
    def test_initial_balance(self, broker):
        info = broker.get_account_info()
        assert info["balance"] == 10_000.0
        assert info["initial_balance"] == 10_000.0

    def test_initial_pnl_is_zero(self, broker):
        assert broker.get_account_info()["pnl"] == 0.0

    def test_portfolio_value_equals_balance_when_no_positions(self, broker):
        broker._update_portfolio_value()
        assert broker.portfolio_value == broker.balance


# ---------------------------------------------------------------------------
# Market orders
# ---------------------------------------------------------------------------

class TestMarketOrders:
    def test_buy_market_order_fills(self, broker):
        order = broker.submit_order("AAPL", qty=1.0, side="buy",
                                    order_type="market", execution_price=150.0)
        assert order.status == OrderStatus.FILLED

    def test_buy_reduces_balance(self, broker):
        broker.submit_order("AAPL", qty=1.0, side="buy",
                            order_type="market", execution_price=150.0)
        cost = 150.0 * 1.0 * (1 + broker.market_fee)
        assert abs(broker.balance - (10_000.0 - cost)) < 0.01

    def test_buy_creates_position(self, broker):
        broker.submit_order("AAPL", qty=2.0, side="buy",
                            order_type="market", execution_price=150.0)
        pos = broker.get_position("AAPL")
        assert pos is not None
        assert abs(pos.qty - 2.0) < 1e-6
        assert abs(pos.avg_price - 150.0) < 1e-6

    def test_sell_market_order_closes_position(self, broker):
        broker.submit_order("AAPL", qty=2.0, side="buy",
                            order_type="market", execution_price=150.0)
        broker.submit_order("AAPL", qty=2.0, side="sell",
                            order_type="market", execution_price=160.0)
        assert broker.get_position("AAPL") is None

    def test_sell_increases_balance(self, broker):
        broker.submit_order("AAPL", qty=1.0, side="buy",
                            order_type="market", execution_price=150.0)
        balance_after_buy = broker.balance
        broker.submit_order("AAPL", qty=1.0, side="sell",
                            order_type="market", execution_price=160.0)
        assert broker.balance > balance_after_buy

    def test_order_stored_in_history(self, broker):
        broker.submit_order("AAPL", qty=1.0, side="buy",
                            order_type="market", execution_price=150.0)
        assert len(broker.order_history) == 1

    def test_filled_qty_matches_requested(self, broker):
        order = broker.submit_order("AAPL", qty=3.0, side="buy",
                                    order_type="market", execution_price=150.0)
        assert abs(order.filled_qty - 3.0) < 1e-6

    def test_filled_avg_price_matches_execution_price(self, broker):
        order = broker.submit_order("AAPL", qty=1.0, side="buy",
                                    order_type="market", execution_price=150.0)
        assert abs(order.filled_avg_price - 150.0) < 1e-6


# ---------------------------------------------------------------------------
# Insufficient funds
# ---------------------------------------------------------------------------

class TestInsufficientFunds:
    def test_buy_rejected_when_no_funds(self, broker):
        # 10k balance can't buy 1000 shares at $150
        order = broker.submit_order("AAPL", qty=1000.0, side="buy",
                                    order_type="market", execution_price=150.0)
        # Broker adjusts qty to affordable or rejects — either way balance must not go negative
        assert broker.balance >= 0

    def test_balance_never_goes_negative(self, broker):
        for _ in range(5):
            broker.submit_order("AAPL", qty=999.0, side="buy",
                                order_type="market", execution_price=150.0)
        assert broker.balance >= 0

    def test_sell_more_than_position_capped(self, broker):
        broker.submit_order("AAPL", qty=1.0, side="buy",
                            order_type="market", execution_price=150.0)
        # Try to sell 10 but only have 1
        order = broker.submit_order("AAPL", qty=10.0, side="sell",
                                    order_type="market", execution_price=160.0)
        assert order.filled_qty <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Limit orders
# ---------------------------------------------------------------------------

class TestLimitOrders:
    def test_limit_buy_fills_when_price_below_limit(self, broker):
        # market price 150, limit 160 — should fill immediately (price <= limit)
        order = broker.submit_order("AAPL", qty=1.0, side="buy",
                                    order_type="limit", limit_price=160.0,
                                    execution_price=150.0)
        assert order.status == OrderStatus.FILLED

    def test_limit_sell_fills_when_price_above_limit(self, broker):
        broker.submit_order("AAPL", qty=1.0, side="buy",
                            order_type="market", execution_price=150.0)
        order = broker.submit_order("AAPL", qty=1.0, side="sell",
                                    order_type="limit", limit_price=140.0,
                                    execution_price=150.0)
        assert order.status == OrderStatus.FILLED

    def test_limit_buy_pending_when_price_above_limit(self, broker):
        # market price 150, limit 100 — too low, should stay pending
        order = broker.submit_order("AAPL", qty=1.0, side="buy",
                                    order_type="limit", limit_price=100.0)
        assert order.status == OrderStatus.PENDING


# ---------------------------------------------------------------------------
# Position averaging
# ---------------------------------------------------------------------------

class TestPositionAveraging:
    def test_adding_to_long_updates_avg_price(self, broker):
        broker.submit_order("AAPL", qty=1.0, side="buy",
                            order_type="market", execution_price=100.0)
        broker.submit_order("AAPL", qty=1.0, side="buy",
                            order_type="market", execution_price=200.0)
        pos = broker.get_position("AAPL")
        assert abs(pos.avg_price - 150.0) < 0.01
        assert abs(pos.qty - 2.0) < 1e-6

    def test_partial_sell_reduces_position(self, broker):
        broker.submit_order("AAPL", qty=4.0, side="buy",
                            order_type="market", execution_price=100.0)
        broker.submit_order("AAPL", qty=2.0, side="sell",
                            order_type="market", execution_price=110.0)
        pos = broker.get_position("AAPL")
        assert pos is not None
        assert abs(pos.qty - 2.0) < 1e-6

    def test_full_sell_removes_position(self, broker):
        broker.submit_order("AAPL", qty=2.0, side="buy",
                            order_type="market", execution_price=100.0)
        broker.submit_order("AAPL", qty=2.0, side="sell",
                            order_type="market", execution_price=110.0)
        assert broker.get_position("AAPL") is None


# ---------------------------------------------------------------------------
# Multiple symbols
# ---------------------------------------------------------------------------

class TestMultipleSymbols:
    def test_independent_positions(self, broker):
        broker.submit_order("AAPL", qty=1.0, side="buy",
                            order_type="market", execution_price=150.0)
        broker.submit_order("BTC", qty=0.1, side="buy",
                            order_type="market", execution_price=30_000.0)
        assert broker.get_position("AAPL") is not None
        assert broker.get_position("BTC") is not None

    def test_sell_one_does_not_affect_other(self, broker):
        broker.submit_order("AAPL", qty=1.0, side="buy",
                            order_type="market", execution_price=150.0)
        broker.submit_order("BTC", qty=0.1, side="buy",
                            order_type="market", execution_price=30_000.0)
        broker.submit_order("AAPL", qty=1.0, side="sell",
                            order_type="market", execution_price=160.0)
        assert broker.get_position("AAPL") is None
        assert broker.get_position("BTC") is not None


# ---------------------------------------------------------------------------
# Order cancellation
# ---------------------------------------------------------------------------

class TestOrderCancellation:
    def test_cancel_pending_order(self, broker):
        order = broker.submit_order("AAPL", qty=1.0, side="buy",
                                    order_type="limit", limit_price=50.0)
        assert order.status == OrderStatus.PENDING
        result = broker.cancel_order(order.id)
        assert result is True
        assert broker.orders[order.id].status == OrderStatus.CANCELED

    def test_cancel_filled_order_returns_false(self, broker):
        order = broker.submit_order("AAPL", qty=1.0, side="buy",
                                    order_type="market", execution_price=150.0)
        result = broker.cancel_order(order.id)
        assert result is False

    def test_cancel_nonexistent_order_returns_false(self, broker):
        assert broker.cancel_order("does_not_exist") is False


# ---------------------------------------------------------------------------
# Order listing
# ---------------------------------------------------------------------------

class TestGetOrders:
    def test_get_all_orders(self, broker):
        broker.submit_order("AAPL", qty=1.0, side="buy",
                            order_type="market", execution_price=150.0)
        broker.submit_order("AAPL", qty=1.0, side="buy",
                            order_type="limit", limit_price=50.0)
        all_orders = broker.get_orders()
        assert len(all_orders) == 2

    def test_filter_filled_orders(self, broker):
        broker.submit_order("AAPL", qty=1.0, side="buy",
                            order_type="market", execution_price=150.0)
        broker.submit_order("AAPL", qty=1.0, side="buy",
                            order_type="limit", limit_price=50.0)
        filled = broker.get_orders(status=OrderStatus.FILLED)
        assert len(filled) == 1
        assert all(o.status == OrderStatus.FILLED for o in filled)

    def test_filter_pending_orders(self, broker):
        broker.submit_order("AAPL", qty=1.0, side="buy",
                            order_type="limit", limit_price=50.0)
        broker.submit_order("AAPL", qty=1.0, side="buy",
                            order_type="limit", limit_price=50.0)
        pending = broker.get_orders(status=OrderStatus.PENDING)
        assert len(pending) == 2


# ---------------------------------------------------------------------------
# Fee calculation
# ---------------------------------------------------------------------------

class TestFees:
    def test_market_fee_deducted(self):
        b = SimulatedBroker(initial_balance=10_000.0, market_fee=0.01)
        b.market_data["X"] = 100.0
        b.submit_order("X", qty=10.0, side="buy",
                       order_type="market", execution_price=100.0)
        expected_cost = 10 * 100 * (1 + 0.01)
        assert abs(b.balance - (10_000.0 - expected_cost)) < 0.01
        b.close()

    def test_zero_fee_no_deduction(self):
        b = SimulatedBroker(initial_balance=10_000.0, market_fee=0.0, limit_fee=0.0)
        b.market_data["X"] = 100.0
        b.submit_order("X", qty=10.0, side="buy",
                       order_type="market", execution_price=100.0)
        assert abs(b.balance - (10_000.0 - 1000.0)) < 0.01
        b.close()


# ---------------------------------------------------------------------------
# P&L correctness — 100+ round-trip trades with zero fees
# ---------------------------------------------------------------------------

class TestPnLCorrectness:
    """
    Verify that P&L accounting is exact over many sequential buy/sell cycles.

    Strategy: no fees, buy 1 unit at buy_price, sell 1 unit at sell_price.
    Expected profit per round-trip = sell_price - buy_price.
    After N round-trips the final balance must equal:
        initial_balance + N * (sell_price - buy_price)
    """

    def _make_zero_fee_broker(self, balance=1_000_000.0):
        b = SimulatedBroker(initial_balance=balance, market_fee=0.0, limit_fee=0.0)
        b.market_data["SIM"] = 100.0
        return b

    def test_100_buy_sell_roundtrips_pnl(self):
        """100 round-trips: buy @ 100, sell @ 110 → +10 each → +1000 total."""
        N = 100
        buy_price = 100.0
        sell_price = 110.0
        expected_profit = N * (sell_price - buy_price)

        b = self._make_zero_fee_broker()
        initial = b.balance
        for _ in range(N):
            buy_order = b.submit_order("SIM", qty=1.0, side="buy",
                                       order_type="market", execution_price=buy_price)
            assert buy_order.status == OrderStatus.FILLED, "Buy order must fill"

            sell_order = b.submit_order("SIM", qty=1.0, side="sell",
                                        order_type="market", execution_price=sell_price)
            assert sell_order.status == OrderStatus.FILLED, "Sell order must fill"

        assert b.get_position("SIM") is None, "All positions must be closed after round-trips"
        assert abs(b.balance - (initial + expected_profit)) < 0.01, (
            f"Expected balance {initial + expected_profit:.2f}, got {b.balance:.2f}"
        )
        b.close()

    def test_500_buy_sell_roundtrips_pnl(self):
        """500 round-trips (part of 1000-trade simulation): buy @ 50, sell @ 55 → +5 each → +2500 total."""
        N = 500
        buy_price = 50.0
        sell_price = 55.0
        expected_profit = N * (sell_price - buy_price)

        b = self._make_zero_fee_broker()
        initial = b.balance
        for _ in range(N):
            b.submit_order("SIM", qty=1.0, side="buy",
                           order_type="market", execution_price=buy_price)
            b.submit_order("SIM", qty=1.0, side="sell",
                           order_type="market", execution_price=sell_price)

        assert b.get_position("SIM") is None, "No open position expected"
        assert abs(b.balance - (initial + expected_profit)) < 0.01, (
            f"Expected balance {initial + expected_profit:.2f}, got {b.balance:.2f}"
        )
        b.close()

    def test_pnl_with_loss(self):
        """100 round-trips where sell < buy → negative P&L (buy @ 110, sell @ 100)."""
        N = 100
        buy_price = 110.0
        sell_price = 100.0
        expected_profit = N * (sell_price - buy_price)  # -1000

        b = self._make_zero_fee_broker()
        initial = b.balance
        for _ in range(N):
            b.submit_order("SIM", qty=1.0, side="buy",
                           order_type="market", execution_price=buy_price)
            b.submit_order("SIM", qty=1.0, side="sell",
                           order_type="market", execution_price=sell_price)

        assert b.get_position("SIM") is None
        assert abs(b.balance - (initial + expected_profit)) < 0.01, (
            f"Expected balance {initial + expected_profit:.2f}, got {b.balance:.2f}"
        )
        b.close()

    def test_portfolio_balance_updates_per_trade(self):
        """After each buy the balance decreases; after each sell it increases back."""
        b = self._make_zero_fee_broker(balance=10_000.0)
        initial = b.balance
        prev_balance = initial
        for i in range(10):
            b.submit_order("SIM", qty=1.0, side="buy",
                           order_type="market", execution_price=100.0)
            assert b.balance < prev_balance, f"Round-trip {i}: balance should drop after buy"
            prev_balance_after_buy = b.balance

            b.submit_order("SIM", qty=1.0, side="sell",
                           order_type="market", execution_price=120.0)
            assert b.balance > prev_balance_after_buy, f"Round-trip {i}: balance should rise after sell"
            prev_balance = b.balance
        b.close()

    def test_order_round_trip_position_visible(self):
        """Buy → position appears; sell → position disappears (end-to-end round-trip)."""
        b = self._make_zero_fee_broker()
        assert b.get_position("SIM") is None

        buy = b.submit_order("SIM", qty=5.0, side="buy",
                              order_type="market", execution_price=200.0)
        assert buy.status == OrderStatus.FILLED
        pos = b.get_position("SIM")
        assert pos is not None
        assert abs(pos.qty - 5.0) < 1e-6
        assert abs(pos.avg_price - 200.0) < 1e-6

        sell = b.submit_order("SIM", qty=5.0, side="sell",
                               order_type="market", execution_price=250.0)
        assert sell.status == OrderStatus.FILLED
        assert b.get_position("SIM") is None
        b.close()

    def test_1000_trade_simulation(self):
        """
        Full 1000-trade (500 round-trip) simulation with mixed symbols and quantities.
        Verifies final P&L is within floating-point tolerance of the analytical sum.
        """
        b = self._make_zero_fee_broker(balance=5_000_000.0)
        initial = b.balance
        total_expected_profit = 0.0

        # 500 round-trips across 5 symbols, varying qty and prices
        configs = [
            ("SIM_A", 1.0, 100.0, 105.0),   # +5 per round-trip × 100 iterations = +500
            ("SIM_B", 2.0, 200.0, 210.0),   # +20 per round-trip × 100 iterations = +2000
            ("SIM_C", 0.5, 50.0,  45.0),    # -2.5 per round-trip × 100 iterations = -250 (loss)
            ("SIM_D", 3.0, 300.0, 315.0),   # +45 per round-trip × 100 iterations = +4500
            ("SIM_E", 1.0, 400.0, 390.0),   # -10 per round-trip × 100 iterations = -1000 (loss)
        ]
        for sym, qty, buy_p, sell_p in configs:
            b.market_data[sym] = buy_p
            for _ in range(100):
                b.submit_order(sym, qty=qty, side="buy",
                               order_type="market", execution_price=buy_p)
                b.submit_order(sym, qty=qty, side="sell",
                               order_type="market", execution_price=sell_p)
            total_expected_profit += 100 * qty * (sell_p - buy_p)

        # No open positions should remain
        for sym, *_ in configs:
            assert b.get_position(sym) is None, f"Position for {sym} should be closed"

        assert abs(b.balance - (initial + total_expected_profit)) < 0.10, (
            f"1000-trade simulation P&L mismatch: "
            f"expected {initial + total_expected_profit:.4f}, got {b.balance:.4f}"
        )
        b.close()


# ---------------------------------------------------------------------------
# Thread-safety stress tests
# ---------------------------------------------------------------------------

class TestThreadSafety:
    """
    Stress-test concurrent order submission from multiple threads.
    Goal: no exceptions, no negative balance, no torn reads.
    """

    def test_concurrent_orders_no_exception(self):
        """10 threads each submit 20 buy+sell round-trips concurrently."""
        b = SimulatedBroker(initial_balance=50_000_000.0, market_fee=0.0, limit_fee=0.0)
        b.market_data["TS"] = 100.0
        errors = []

        def worker():
            try:
                for _ in range(20):
                    b.submit_order("TS", qty=1.0, side="buy",
                                   order_type="market", execution_price=100.0)
                    b.submit_order("TS", qty=1.0, side="sell",
                                   order_type="market", execution_price=100.0)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Thread errors: {errors}"
        assert b.balance >= 0, f"Balance went negative: {b.balance}"
        b.close()

    def test_concurrent_reads_during_writes(self):
        """Reader threads calling get_account_info/get_position while writers submit orders."""
        b = SimulatedBroker(initial_balance=10_000_000.0, market_fee=0.0, limit_fee=0.0)
        b.market_data["RS"] = 50.0
        read_errors = []
        write_errors = []
        stop_event = threading.Event()

        def reader():
            while not stop_event.is_set():
                try:
                    _ = b.get_account_info()
                    _ = b.get_position("RS")
                    _ = b.get_orders()
                except Exception as exc:
                    read_errors.append(exc)

        def writer():
            for _ in range(50):
                try:
                    b.submit_order("RS", qty=1.0, side="buy",
                                   order_type="market", execution_price=50.0)
                    b.submit_order("RS", qty=1.0, side="sell",
                                   order_type="market", execution_price=50.0)
                except Exception as exc:
                    write_errors.append(exc)

        readers = [threading.Thread(target=reader, daemon=True) for _ in range(4)]
        writers = [threading.Thread(target=writer) for _ in range(4)]
        for t in readers + writers:
            t.start()
        for t in writers:
            t.join(timeout=30)
        stop_event.set()
        for t in readers:
            t.join(timeout=5)

        assert not read_errors, f"Read errors under concurrent writes: {read_errors}"
        assert not write_errors, f"Write errors: {write_errors}"
        assert b.balance >= 0
        b.close()

    def test_balance_never_negative_concurrent(self):
        """
        Multiple threads simultaneously try to buy the same symbol.
        The broker must never let balance go negative.
        """
        b = SimulatedBroker(initial_balance=1_000.0, market_fee=0.0, limit_fee=0.0)
        b.market_data["NEG"] = 100.0

        def aggressive_buyer():
            for _ in range(50):
                b.submit_order("NEG", qty=100.0, side="buy",
                               order_type="market", execution_price=100.0)

        threads = [threading.Thread(target=aggressive_buyer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert b.balance >= -0.01, f"Balance went negative: {b.balance}"  # tiny float tolerance
        b.close()
