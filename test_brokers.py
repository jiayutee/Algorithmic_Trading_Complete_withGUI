"""Unit tests for SimulatedBroker — order logic, fills, position tracking, fees."""
import time
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
