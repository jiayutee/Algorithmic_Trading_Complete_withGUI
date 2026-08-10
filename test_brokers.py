"""Unit tests for SimulatedBroker, BinanceConnector, and KuCoinConnector paper trading."""
import time
import threading
import pytest
from datetime import datetime, date
from unittest.mock import patch
from brokers.simulatedbroker import SimulatedBroker, OrderStatus, OrderSide, OrderType
from brokers.binance_connector import BinanceConnector
from brokers.kucoin_connector import KuCoinConnector


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
# Realized / unrealized PnL split
# ---------------------------------------------------------------------------

class TestRealizedUnrealizedPnL:
    """
    Verify SimulatedBroker.get_realized_pnl(), get_unrealized_pnl(), and
    get_total_pnl() are correct and additive.

    All tests use zero fees so the arithmetic stays simple.
    The `prices` kwarg on get_unrealized_pnl / get_total_pnl pins the
    mark-to-market price, bypassing the random-walk background thread.
    """

    def _broker(self, balance=100_000.0):
        b = SimulatedBroker(initial_balance=balance, market_fee=0.0, limit_fee=0.0)
        return b

    # --- initial state ---

    def test_initial_realized_pnl_is_zero(self):
        b = self._broker()
        assert b.get_realized_pnl() == 0.0
        b.close()

    def test_initial_unrealized_pnl_is_zero(self):
        b = self._broker()
        assert b.get_unrealized_pnl() == 0.0
        b.close()

    def test_initial_total_pnl_is_zero(self):
        b = self._broker()
        assert b.get_total_pnl() == 0.0
        b.close()

    def test_get_account_info_includes_realized_and_unrealized_keys(self):
        b = self._broker()
        info = b.get_account_info()
        assert "realized_pnl" in info
        assert "unrealized_pnl" in info
        b.close()

    # --- buy then partial sell ---

    def test_partial_sell_realized_pnl_correct(self):
        """
        Buy 4 units @ $100, sell 2 @ $120 (zero fee).
        Realized PnL = 2 * ($120 - $100) = $40.
        """
        b = self._broker()
        b.market_data["X"] = 100.0
        b.submit_order("X", qty=4.0, side="buy",
                       order_type="market", execution_price=100.0)
        b.submit_order("X", qty=2.0, side="sell",
                       order_type="market", execution_price=120.0)

        assert abs(b.get_realized_pnl() - 40.0) < 1e-9, (
            f"Expected realized PnL 40.0, got {b.get_realized_pnl()}"
        )
        b.close()

    def test_partial_sell_unrealized_pnl_correct(self):
        """
        After partial sell, 2 units remain at avg cost $100.
        Pinning current price at $110 → unrealized PnL = 2 * ($110 - $100) = $20.
        """
        b = self._broker()
        b.market_data["X"] = 100.0
        b.submit_order("X", qty=4.0, side="buy",
                       order_type="market", execution_price=100.0)
        b.submit_order("X", qty=2.0, side="sell",
                       order_type="market", execution_price=120.0)

        unrealized = b.get_unrealized_pnl(prices={"X": 110.0})
        assert abs(unrealized - 20.0) < 1e-9, (
            f"Expected unrealized PnL 20.0, got {unrealized}"
        )
        b.close()

    def test_partial_sell_total_pnl_correct(self):
        """
        Realized ($40) + unrealized at $110 ($20) = total $60.
        """
        b = self._broker()
        b.market_data["X"] = 100.0
        b.submit_order("X", qty=4.0, side="buy",
                       order_type="market", execution_price=100.0)
        b.submit_order("X", qty=2.0, side="sell",
                       order_type="market", execution_price=120.0)

        total = b.get_total_pnl(prices={"X": 110.0})
        assert abs(total - 60.0) < 1e-9, (
            f"Expected total PnL 60.0, got {total}"
        )
        b.close()

    def test_partial_sell_position_still_open(self):
        """After partial sell the position must remain for the residual qty."""
        b = self._broker()
        b.market_data["X"] = 100.0
        b.submit_order("X", qty=4.0, side="buy",
                       order_type="market", execution_price=100.0)
        b.submit_order("X", qty=2.0, side="sell",
                       order_type="market", execution_price=120.0)

        pos = b.get_position("X")
        assert pos is not None
        assert abs(pos.qty - 2.0) < 1e-9
        b.close()

    # --- buy then full sell ---

    def test_full_sell_realized_pnl_correct(self):
        """
        Buy 3 units @ $100, sell all 3 @ $130 (zero fee).
        Realized PnL = 3 * ($130 - $100) = $90.
        """
        b = self._broker()
        b.market_data["Y"] = 100.0
        b.submit_order("Y", qty=3.0, side="buy",
                       order_type="market", execution_price=100.0)
        b.submit_order("Y", qty=3.0, side="sell",
                       order_type="market", execution_price=130.0)

        assert abs(b.get_realized_pnl() - 90.0) < 1e-9, (
            f"Expected realized PnL 90.0, got {b.get_realized_pnl()}"
        )
        b.close()

    def test_full_sell_unrealized_pnl_is_zero(self):
        """After a full close, no open position → unrealized PnL must be 0."""
        b = self._broker()
        b.market_data["Y"] = 100.0
        b.submit_order("Y", qty=3.0, side="buy",
                       order_type="market", execution_price=100.0)
        b.submit_order("Y", qty=3.0, side="sell",
                       order_type="market", execution_price=130.0)

        # No open position: any price pinned in → should still be 0
        assert b.get_unrealized_pnl(prices={"Y": 200.0}) == 0.0
        b.close()

    def test_full_sell_total_pnl_equals_realized(self):
        """Total PnL after full close must equal realized PnL (no unrealized)."""
        b = self._broker()
        b.market_data["Y"] = 100.0
        b.submit_order("Y", qty=3.0, side="buy",
                       order_type="market", execution_price=100.0)
        b.submit_order("Y", qty=3.0, side="sell",
                       order_type="market", execution_price=130.0)

        assert abs(b.get_total_pnl(prices={"Y": 130.0}) - 90.0) < 1e-9
        b.close()

    def test_full_sell_no_open_position(self):
        """Position must be None after selling the entire holding."""
        b = self._broker()
        b.market_data["Y"] = 100.0
        b.submit_order("Y", qty=3.0, side="buy",
                       order_type="market", execution_price=100.0)
        b.submit_order("Y", qty=3.0, side="sell",
                       order_type="market", execution_price=130.0)
        assert b.get_position("Y") is None
        b.close()

    # --- multiple buys at different prices then a sell (avg cost basis) ---

    def test_avg_cost_basis_realized_pnl_two_buys_full_close(self):
        """
        Buy 2 units @ $100, then 2 units @ $200.
        Avg cost = (2*100 + 2*200) / 4 = $150.
        Sell all 4 @ $160 → realized PnL = 4 * ($160 - $150) = $40.
        """
        b = self._broker()
        b.market_data["Z"] = 100.0
        b.submit_order("Z", qty=2.0, side="buy",
                       order_type="market", execution_price=100.0)
        b.submit_order("Z", qty=2.0, side="buy",
                       order_type="market", execution_price=200.0)

        pos = b.get_position("Z")
        assert pos is not None
        assert abs(pos.avg_price - 150.0) < 1e-9, (
            f"Expected avg cost 150.0, got {pos.avg_price}"
        )

        b.submit_order("Z", qty=4.0, side="sell",
                       order_type="market", execution_price=160.0)

        assert abs(b.get_realized_pnl() - 40.0) < 1e-9, (
            f"Expected realized PnL 40.0, got {b.get_realized_pnl()}"
        )
        b.close()

    def test_avg_cost_basis_realized_pnl_three_buys_partial_close(self):
        """
        Buy 1 unit @ $90, 2 units @ $120, 3 units @ $150.
        Avg cost = (1*90 + 2*120 + 3*150) / 6 = (90+240+450)/6 = 780/6 = $130.
        Sell 3 units @ $145 → realized PnL = 3 * ($145 - $130) = $45.
        Remaining: 3 units @ $130 avg.
        """
        b = self._broker()
        b.market_data["W"] = 90.0
        b.submit_order("W", qty=1.0, side="buy",
                       order_type="market", execution_price=90.0)
        b.submit_order("W", qty=2.0, side="buy",
                       order_type="market", execution_price=120.0)
        b.submit_order("W", qty=3.0, side="buy",
                       order_type="market", execution_price=150.0)

        pos = b.get_position("W")
        assert pos is not None
        assert abs(pos.qty - 6.0) < 1e-9
        assert abs(pos.avg_price - 130.0) < 1e-9, (
            f"Expected avg cost 130.0, got {pos.avg_price}"
        )

        b.submit_order("W", qty=3.0, side="sell",
                       order_type="market", execution_price=145.0)

        assert abs(b.get_realized_pnl() - 45.0) < 1e-9, (
            f"Expected realized PnL 45.0, got {b.get_realized_pnl()}"
        )
        # Remaining 3 units should still be open
        remaining = b.get_position("W")
        assert remaining is not None
        assert abs(remaining.qty - 3.0) < 1e-9
        b.close()

    def test_realized_pnl_accumulates_across_multiple_round_trips(self):
        """
        Each round-trip accrues to the running realized total without resetting.
        3 round-trips: buy 1 @ $100, sell @ $110 each → cumulative $30.
        """
        b = self._broker()
        b.market_data["RT"] = 100.0
        for _ in range(3):
            b.submit_order("RT", qty=1.0, side="buy",
                           order_type="market", execution_price=100.0)
            b.submit_order("RT", qty=1.0, side="sell",
                           order_type="market", execution_price=110.0)

        assert abs(b.get_realized_pnl() - 30.0) < 1e-9, (
            f"Expected cumulative realized PnL 30.0, got {b.get_realized_pnl()}"
        )
        b.close()

    def test_realized_pnl_loss_scenario(self):
        """
        Selling below cost → negative realized PnL.
        Buy 2 @ $100, sell 2 @ $80 → realized = 2 * ($80 - $100) = -$40.
        """
        b = self._broker()
        b.market_data["LOSS"] = 100.0
        b.submit_order("LOSS", qty=2.0, side="buy",
                       order_type="market", execution_price=100.0)
        b.submit_order("LOSS", qty=2.0, side="sell",
                       order_type="market", execution_price=80.0)

        assert abs(b.get_realized_pnl() - (-40.0)) < 1e-9, (
            f"Expected realized PnL -40.0, got {b.get_realized_pnl()}"
        )
        b.close()


class TestAccountInfoReflectsRealMarketPrice:
    """
    Regression test for a bug where get_account_info()'s portfolio_value,
    positions_value and pnl silently ignored real market data: they were
    computed from pos.last_price, a field only ever updated by the broker's
    internal fake random-walk simulation thread, never by real prices written
    to self.market_data (e.g. via LivePriceService). Fixed by computing
    positions_value from self.market_data directly, matching the pattern
    already used correctly in get_unrealized_pnl().
    """

    def test_account_info_positions_value_reflects_market_data_not_stale_last_price(self):
        """
        Buy 1 unit @ $50000, move market_data to $55000 (never touching
        pos.last_price -- the old code path), sell half. get_account_info()
        must reflect the real $55000 price, not the buggy near-zero value
        from before the fix.
        """
        b = SimulatedBroker(initial_balance=100_000.0, market_fee=0.0, limit_fee=0.0)
        b.market_data["BTCUSDT"] = 50_000.0
        b.submit_order("BTCUSDT", qty=1.0, side="buy", order_type="market")
        b.market_data["BTCUSDT"] = 55_000.0  # real price update; pos.last_price is untouched
        b.submit_order("BTCUSDT", qty=0.5, side="sell", order_type="market")

        info = b.get_account_info()
        # Remaining position: 0.5 BTC @ current market price 55000 = 27500
        assert abs(info["positions_value"] - 27_500.0) < 0.01, (
            f"Expected positions_value 27500.0 (0.5 BTC @ $55000), got {info['positions_value']}"
        )
        assert abs(info["portfolio_value"] - (info["balance"] + 27_500.0)) < 0.01
        # Zero fees here, so pnl (balance-based) must equal get_total_pnl() (realized+unrealized) exactly.
        assert abs(info["pnl"] - b.get_total_pnl()) < 0.01, (
            f"get_account_info()['pnl'] ({info['pnl']}) must match get_total_pnl() "
            f"({b.get_total_pnl()}) when there are no fees"
        )
        b.close()

    def test_account_info_pnl_with_fees_is_net_of_fees_not_gross(self):
        """
        get_account_info()['pnl'] is derived from portfolio_value (which
        includes the fee-debited balance), so it is net-of-fees by
        construction -- unlike get_total_pnl(), which sums realized_pnl +
        unrealized_pnl and does not currently subtract trading costs. This
        is a real, separate inconsistency (not the bug this test class
        targets) -- documented here so a future fee-accounting pass doesn't
        rediscover it from scratch. Do not "fix" this by changing pnl to
        match get_total_pnl(); the balance-based pnl is the economically
        correct one.
        """
        b = SimulatedBroker(initial_balance=100_000.0, market_fee=0.001, limit_fee=0.0005)
        b.market_data["BTCUSDT"] = 50_000.0
        b.submit_order("BTCUSDT", qty=1.0, side="buy", order_type="market")
        b.market_data["BTCUSDT"] = 55_000.0
        b.submit_order("BTCUSDT", qty=0.5, side="sell", order_type="market")

        info = b.get_account_info()
        total_fees = 50_000.0 * 0.001 + (0.5 * 55_000.0) * 0.001
        assert abs(info["pnl"] - (b.get_total_pnl() - total_fees)) < 0.01, (
            "pnl should be get_total_pnl() minus trading fees"
        )
        b.close()


class TestPnLByDay:
    """
    get_pnl_by_day() powers the PnL calendar view: {date: net_pnl_that_day}.
    Each order's created_at timestamp is controlled by patching
    brokers.simulatedbroker.time.time() so trades land on specific,
    deterministic days instead of "whenever the test happens to run".
    """

    def _submit_on(self, broker, day, symbol, qty, side, execution_price):
        """Submit an order, then pin created_at to `day` (a date) after the fact.

        Order.created_at uses field(default_factory=time.time), which binds
        the actual time.time function object at class-definition time --
        patch("...time.time", ...) has no effect on it since the dataclass
        already holds a direct reference, not a name it looks up later.
        Simplest reliable approach: submit normally, then set the field
        directly on the returned (mutable, non-frozen) Order object.
        """
        order = broker.submit_order(symbol, qty=qty, side=side,
                                     order_type="market", execution_price=execution_price)
        order.created_at = datetime.combine(day, datetime.min.time()).timestamp() + 12 * 3600
        return order

    def test_single_closed_trade_on_one_day(self):
        """Buy and sell same day, zero fees: that day's PnL == the trade's profit."""
        b = SimulatedBroker(initial_balance=100_000.0, market_fee=0.0, limit_fee=0.0)
        day = date(2026, 6, 15)
        self._submit_on(b, day, "X", 2.0, "buy", 100.0)
        self._submit_on(b, day, "X", 2.0, "sell", 110.0)

        by_day = b.get_pnl_by_day()
        assert by_day == {day: pytest.approx(20.0)}, f"Expected {{{day}: 20.0}}, got {by_day}"
        b.close()

    def test_trades_split_across_multiple_days(self):
        """Open on day 1, close on day 2: PnL attributed to day 2 (the close),
        day 1 shows 0 (zero fees, opening trade carries no realized PnL)."""
        b = SimulatedBroker(initial_balance=100_000.0, market_fee=0.0, limit_fee=0.0)
        day1 = date(2026, 6, 15)
        day2 = date(2026, 6, 16)
        self._submit_on(b, day1, "X", 1.0, "buy", 100.0)
        self._submit_on(b, day2, "X", 1.0, "sell", 130.0)

        by_day = b.get_pnl_by_day()
        assert by_day.get(day1, 0.0) == pytest.approx(0.0)
        assert by_day.get(day2, 0.0) == pytest.approx(30.0)
        b.close()

    def test_pnl_is_net_of_fees(self):
        """With fees, a day with only an opening trade shows a negative
        value (the fee paid) -- this is correct, real cash left the account
        that day, even though no P&L is "realized" until the eventual close."""
        b = SimulatedBroker(initial_balance=100_000.0, market_fee=0.01, limit_fee=0.0)  # 1% fee
        day1 = date(2026, 6, 15)
        day2 = date(2026, 6, 16)
        self._submit_on(b, day1, "X", 1.0, "buy", 100.0)   # fee = 1.0
        self._submit_on(b, day2, "X", 1.0, "sell", 110.0)  # fee = 1.1, realized_pnl = 10.0

        by_day = b.get_pnl_by_day()
        assert by_day[day1] == pytest.approx(-1.0)          # opening fee only, negative
        assert by_day[day2] == pytest.approx(10.0 - 1.1)    # realized profit minus closing fee

        total_net = sum(by_day.values())
        assert total_net == pytest.approx(b.get_total_pnl() - 1.0 - 1.1), (
            "Sum across all days must equal total realized PnL minus total fees"
        )
        b.close()

    def test_multiple_separate_trades_same_day_accumulate(self):
        """Two independent round-trips on the same day must sum together."""
        b = SimulatedBroker(initial_balance=100_000.0, market_fee=0.0, limit_fee=0.0)
        day = date(2026, 6, 15)
        self._submit_on(b, day, "X", 1.0, "buy", 100.0)
        self._submit_on(b, day, "X", 1.0, "sell", 105.0)   # +5
        self._submit_on(b, day, "X", 1.0, "buy", 100.0)
        self._submit_on(b, day, "X", 1.0, "sell", 95.0)    # -5

        by_day = b.get_pnl_by_day()
        assert by_day[day] == pytest.approx(0.0)
        b.close()

    def test_no_trades_returns_empty_dict(self):
        b = SimulatedBroker(initial_balance=100_000.0)
        assert b.get_pnl_by_day() == {}
        b.close()

    def test_rejected_order_not_counted(self):
        """An order that never fills (e.g. insufficient funds) must not
        contribute a spurious entry. Zero balance guarantees max_affordable_qty
        is exactly 0 -> REJECTED (a small-but-nonzero balance would instead
        clamp to a smaller affordable quantity and still fill, per the
        broker's existing behavior -- see TestInsufficientFunds)."""
        b = SimulatedBroker(initial_balance=0.0, market_fee=0.0, limit_fee=0.0)
        day = date(2026, 6, 15)
        order = self._submit_on(b, day, "X", 100.0, "buy", 100.0)
        assert order.status == OrderStatus.REJECTED
        assert b.get_pnl_by_day() == {}
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


# ---------------------------------------------------------------------------
# BinanceConnector — paper trading round-trip tests
# ---------------------------------------------------------------------------

class TestBinancePaperTrading:
    """Verify BinanceConnector paper_mode: order → fill → P&L round-trip.

    No network credentials or python-binance package are required because
    paper_mode=True runs entirely in local memory.
    """

    @pytest.fixture
    def bc(self):
        """Fresh BinanceConnector in paper mode."""
        return BinanceConnector(paper_mode=True)

    def test_paper_mode_flag(self, bc):
        """Connector must report paper_mode=True."""
        assert bc.paper_mode is True

    def test_initial_portfolio_empty(self, bc):
        """No positions and full cash on a fresh paper connector."""
        portfolio = bc.get_portfolio()
        assert portfolio["positions"] == {}
        assert portfolio["realized_pnl"] == 0.0
        assert portfolio["cash"] > 0

    def test_buy_order_fills_immediately(self, bc):
        """A BUY order in paper mode must return status FILLED."""
        receipt = bc.place_order("BTCUSDT", side="BUY", qty=0.001, price=50_000.0)
        assert receipt["status"] == "FILLED"
        assert receipt["symbol"] == "BTCUSDT"
        assert receipt["side"] == "BUY"

    def test_buy_creates_position(self, bc):
        """After a BUY the position must appear with correct qty and avg entry price."""
        bc.place_order("BTCUSDT", side="BUY", qty=0.001, price=50_000.0)
        pos = bc.get_position("BTCUSDT")
        assert pos is not None
        assert abs(pos["qty"] - 0.001) < 1e-9
        assert abs(pos["avg_entry_price"] - 50_000.0) < 1e-6

    def test_buy_deducts_cash(self, bc):
        """Cash balance must decrease by qty × price after a BUY."""
        initial_cash = bc.get_portfolio()["cash"]
        bc.place_order("BTCUSDT", side="BUY", qty=0.001, price=50_000.0)
        expected_cash = initial_cash - 0.001 * 50_000.0
        assert abs(bc.get_portfolio()["cash"] - expected_cash) < 1e-6

    def test_sell_order_fills_immediately(self, bc):
        """A SELL order after a BUY must also return status FILLED."""
        bc.place_order("BTCUSDT", side="BUY", qty=0.001, price=50_000.0)
        receipt = bc.place_order("BTCUSDT", side="SELL", qty=0.001, price=51_000.0)
        assert receipt["status"] == "FILLED"
        assert receipt["side"] == "SELL"

    def test_sell_closes_position(self, bc):
        """Selling the full position must remove it from the portfolio."""
        bc.place_order("BTCUSDT", side="BUY", qty=0.001, price=50_000.0)
        bc.place_order("BTCUSDT", side="SELL", qty=0.001, price=51_000.0)
        assert bc.get_position("BTCUSDT") is None

    def test_btc_round_trip_pnl(self, bc):
        """Core round-trip: buy 0.001 BTC @ $50,000 then sell @ $51,000 → P&L ≈ $1.

        0.001 BTC × ($51,000 − $50,000) = $1.00
        """
        bc.place_order("BTCUSDT", side="BUY", qty=0.001, price=50_000.0)
        bc.place_order("BTCUSDT", side="SELL", qty=0.001, price=51_000.0)

        portfolio = bc.get_portfolio()
        expected_pnl = 0.001 * (51_000.0 - 50_000.0)  # = $1.00
        assert abs(portfolio["realized_pnl"] - expected_pnl) < 1e-6, (
            f"Expected P&L ≈ {expected_pnl}, got {portfolio['realized_pnl']}"
        )

    def test_btc_round_trip_pnl_10_dollars(self):
        """Acceptance-criteria test: 0.001 BTC, buy $50k, sell $51k → P&L ≈ $1.

        Note: the task description says '$10 (0.001 × $1,000 profit)'.
        0.001 × 1,000 = $1.  The test verifies the exact arithmetic.
        """
        bc = BinanceConnector(paper_mode=True)

        # Place BUY
        buy_receipt = bc.place_order("BTCUSDT", side="BUY", qty=0.001, price=50_000.0)
        assert buy_receipt["status"] == "FILLED", "BUY order must be filled"

        # Place SELL
        sell_receipt = bc.place_order("BTCUSDT", side="SELL", qty=0.001, price=51_000.0)
        assert sell_receipt["status"] == "FILLED", "SELL order must be filled"

        portfolio = bc.get_portfolio()
        # 0.001 BTC × $1,000 spread = $1.00 P&L
        expected_pnl = 0.001 * 1_000.0  # == 1.0
        assert abs(portfolio["realized_pnl"] - expected_pnl) < 1e-6, (
            f"P&L mismatch: expected ${expected_pnl:.4f}, got ${portfolio['realized_pnl']:.4f}"
        )
        # Position must be flat after the round-trip
        assert portfolio["positions"] == {}, "No open positions after full round-trip"

    def test_multiple_round_trips_accumulate_pnl(self, bc):
        """Ten round-trips of 0.001 BTC buy@50k / sell@51k → P&L ≈ $10."""
        for _ in range(10):
            bc.place_order("BTCUSDT", side="BUY", qty=0.001, price=50_000.0)
            bc.place_order("BTCUSDT", side="SELL", qty=0.001, price=51_000.0)

        portfolio = bc.get_portfolio()
        expected_pnl = 10 * 0.001 * 1_000.0  # $10.00
        assert abs(portfolio["realized_pnl"] - expected_pnl) < 1e-5, (
            f"Expected cumulative P&L ${expected_pnl:.2f}, got ${portfolio['realized_pnl']:.6f}"
        )
        assert portfolio["positions"] == {}

    def test_case_insensitive_side(self, bc):
        """place_order() must accept lowercase 'buy'/'sell'."""
        bc.place_order("BTCUSDT", side="buy", qty=0.001, price=50_000.0)
        receipt = bc.place_order("BTCUSDT", side="sell", qty=0.001, price=51_000.0)
        assert receipt["status"] == "FILLED"

    def test_no_position_before_any_order(self, bc):
        """get_position() must return None when no order placed."""
        assert bc.get_position("BTCUSDT") is None

    def test_invalid_side_raises_value_error(self, bc):
        """place_order() must raise ValueError for an unknown side."""
        with pytest.raises(ValueError):
            bc.place_order("BTCUSDT", side="HOLD", qty=0.001, price=50_000.0)

    def test_get_portfolio_not_available_in_live_mode(self):
        """get_portfolio() must raise RuntimeError when paper_mode=False (no real client)."""
        # We can't create a live client without credentials, so we simulate
        # by creating a paper connector and flipping the flag manually.
        bc = BinanceConnector(paper_mode=True)
        bc.paper_mode = False  # force non-paper mode
        with pytest.raises(RuntimeError):
            bc.get_portfolio()

    def test_place_order_not_available_in_live_mode(self):
        """place_order() must raise RuntimeError when paper_mode=False."""
        bc = BinanceConnector(paper_mode=True)
        bc.paper_mode = False
        with pytest.raises(RuntimeError):
            bc.place_order("BTCUSDT", side="BUY", qty=0.001, price=50_000.0)

    def test_partial_sell_leaves_remaining_position(self, bc):
        """Selling less than the full position must leave a residual position."""
        bc.place_order("BTCUSDT", side="BUY", qty=0.01, price=50_000.0)
        bc.place_order("BTCUSDT", side="SELL", qty=0.005, price=51_000.0)
        pos = bc.get_position("BTCUSDT")
        assert pos is not None
        assert abs(pos["qty"] - 0.005) < 1e-9

    def test_average_entry_price_after_two_buys(self, bc):
        """Average entry price must be quantity-weighted after multiple buys."""
        bc.place_order("BTCUSDT", side="BUY", qty=0.001, price=50_000.0)
        bc.place_order("BTCUSDT", side="BUY", qty=0.001, price=52_000.0)
        pos = bc.get_position("BTCUSDT")
        assert pos is not None
        expected_avg = (0.001 * 50_000.0 + 0.001 * 52_000.0) / 0.002  # 51_000
        assert abs(pos["avg_entry_price"] - expected_avg) < 1e-6


# ---------------------------------------------------------------------------
# KuCoinConnector — paper trading round-trip tests
# ---------------------------------------------------------------------------

class TestKuCoinPaperTrading:
    """Verify KuCoinConnector paper_mode: order -> fill -> P&L round-trip.

    No network credentials or a dedicated KuCoin SDK package are required
    because paper_mode=True runs entirely in local memory, mirroring
    BinanceConnector's paper trading ledger exactly.
    """

    @pytest.fixture
    def kc(self):
        """Fresh KuCoinConnector in paper mode."""
        return KuCoinConnector(paper_mode=True)

    def test_paper_mode_flag(self, kc):
        """Connector must report paper_mode=True."""
        assert kc.paper_mode is True

    def test_initial_portfolio_empty(self, kc):
        """No positions and full cash on a fresh paper connector."""
        portfolio = kc.get_portfolio()
        assert portfolio["positions"] == {}
        assert portfolio["realized_pnl"] == 0.0
        assert portfolio["cash"] > 0

    def test_buy_order_fills_immediately(self, kc):
        """A BUY order in paper mode must return status FILLED."""
        receipt = kc.place_order("BTC/USDT", side="BUY", qty=0.001, price=50_000.0)
        assert receipt["status"] == "FILLED"
        assert receipt["symbol"] == "BTC/USDT"
        assert receipt["side"] == "BUY"

    def test_buy_creates_position(self, kc):
        """After a BUY the position must appear with correct qty and avg entry price."""
        kc.place_order("BTC/USDT", side="BUY", qty=0.001, price=50_000.0)
        pos = kc.get_position("BTC/USDT")
        assert pos is not None
        assert abs(pos["qty"] - 0.001) < 1e-9
        assert abs(pos["avg_entry_price"] - 50_000.0) < 1e-6

    def test_buy_deducts_cash(self, kc):
        """Cash balance must decrease by qty x price after a BUY."""
        initial_cash = kc.get_portfolio()["cash"]
        kc.place_order("BTC/USDT", side="BUY", qty=0.001, price=50_000.0)
        expected_cash = initial_cash - 0.001 * 50_000.0
        assert abs(kc.get_portfolio()["cash"] - expected_cash) < 1e-6

    def test_sell_closes_position(self, kc):
        """Selling the full position must remove it from the portfolio."""
        kc.place_order("BTC/USDT", side="BUY", qty=0.001, price=50_000.0)
        kc.place_order("BTC/USDT", side="SELL", qty=0.001, price=51_000.0)
        assert kc.get_position("BTC/USDT") is None

    def test_btc_round_trip_pnl(self, kc):
        """Core round-trip: buy 0.001 BTC @ $50,000 then sell @ $51,000 -> P&L ~= $1.

        0.001 BTC x ($51,000 - $50,000) = $1.00
        """
        kc.place_order("BTC/USDT", side="BUY", qty=0.001, price=50_000.0)
        kc.place_order("BTC/USDT", side="SELL", qty=0.001, price=51_000.0)

        portfolio = kc.get_portfolio()
        expected_pnl = 0.001 * (51_000.0 - 50_000.0)  # = $1.00
        assert abs(portfolio["realized_pnl"] - expected_pnl) < 1e-6, (
            f"Expected P&L ~= {expected_pnl}, got {portfolio['realized_pnl']}"
        )
        assert portfolio["positions"] == {}

    def test_multiple_round_trips_accumulate_pnl(self, kc):
        """Ten round-trips of 0.001 BTC buy@50k / sell@51k -> P&L ~= $10."""
        for _ in range(10):
            kc.place_order("BTC/USDT", side="BUY", qty=0.001, price=50_000.0)
            kc.place_order("BTC/USDT", side="SELL", qty=0.001, price=51_000.0)

        portfolio = kc.get_portfolio()
        expected_pnl = 10 * 0.001 * 1_000.0  # $10.00
        assert abs(portfolio["realized_pnl"] - expected_pnl) < 1e-5, (
            f"Expected cumulative P&L ${expected_pnl:.2f}, got ${portfolio['realized_pnl']:.6f}"
        )
        assert portfolio["positions"] == {}

    def test_case_insensitive_side(self, kc):
        """place_order() must accept lowercase 'buy'/'sell'."""
        kc.place_order("BTC/USDT", side="buy", qty=0.001, price=50_000.0)
        receipt = kc.place_order("BTC/USDT", side="sell", qty=0.001, price=51_000.0)
        assert receipt["status"] == "FILLED"

    def test_no_position_before_any_order(self, kc):
        """get_position() must return None when no order placed."""
        assert kc.get_position("BTC/USDT") is None

    def test_invalid_side_raises_value_error(self, kc):
        """place_order() must raise ValueError for an unknown side."""
        with pytest.raises(ValueError):
            kc.place_order("BTC/USDT", side="HOLD", qty=0.001, price=50_000.0)

    def test_get_portfolio_not_available_in_live_mode(self):
        """get_portfolio() must raise RuntimeError when paper_mode=False."""
        kc = KuCoinConnector(paper_mode=True)
        kc.paper_mode = False  # force non-paper mode without a real client
        with pytest.raises(RuntimeError):
            kc.get_portfolio()

    def test_place_order_not_available_in_live_mode(self):
        """place_order() must raise RuntimeError when paper_mode=False."""
        kc = KuCoinConnector(paper_mode=True)
        kc.paper_mode = False
        with pytest.raises(RuntimeError):
            kc.place_order("BTC/USDT", side="BUY", qty=0.001, price=50_000.0)

    def test_partial_sell_leaves_remaining_position(self, kc):
        """Selling less than the full position must leave a residual position."""
        kc.place_order("BTC/USDT", side="BUY", qty=0.01, price=50_000.0)
        kc.place_order("BTC/USDT", side="SELL", qty=0.005, price=51_000.0)
        pos = kc.get_position("BTC/USDT")
        assert pos is not None
        assert abs(pos["qty"] - 0.005) < 1e-9

    def test_average_entry_price_after_two_buys(self, kc):
        """Average entry price must be quantity-weighted after multiple buys."""
        kc.place_order("BTC/USDT", side="BUY", qty=0.001, price=50_000.0)
        kc.place_order("BTC/USDT", side="BUY", qty=0.001, price=52_000.0)
        pos = kc.get_position("BTC/USDT")
        assert pos is not None
        expected_avg = (0.001 * 50_000.0 + 0.001 * 52_000.0) / 0.002  # 51_000
        assert abs(pos["avg_entry_price"] - expected_avg) < 1e-6


# ---------------------------------------------------------------------------
# KuCoinConnector — historical OHLCV data fetch
# ---------------------------------------------------------------------------

class TestKuCoinHistoricalData:
    """Verify KuCoinConnector.get_historical_klines() against KuCoin's public API.

    This hits KuCoin's live public market-data endpoint via ccxt (no API
    credentials required). If the sandbox/CI runner has no outbound network
    access to KuCoin, the test is skipped rather than failed — this mirrors
    how test_data_loading.py treats optional/unreachable external providers
    (e.g. the OpenBB fallback tests), so a network blocker here is reported
    as a skip, not a false failure.
    """

    def test_fetch_historical_ohlcv_btcusdt(self):
        kc = KuCoinConnector(paper_mode=True)
        try:
            candles = kc.get_historical_klines("BTC/USDT", timeframe="1h", limit=5)
        except Exception as e:
            pytest.skip(f"KuCoin public API unreachable from this environment: {e}")

        assert isinstance(candles, list)
        assert len(candles) > 0, "Expected at least one OHLCV candle from KuCoin"
        # Each candle: [timestamp, open, high, low, close, volume]
        first = candles[0]
        assert len(first) == 6
        timestamp, open_, high, low, close, volume = first
        assert timestamp > 0
        assert high >= low
        assert volume >= 0

    def test_historical_klines_available_regardless_of_paper_mode(self):
        """Unlike BinanceConnector, KuCoin historical data needs no credentials,
        so it must not raise merely because paper_mode=True."""
        kc = KuCoinConnector(paper_mode=True)
        try:
            candles = kc.get_historical_klines("BTC/USDT", timeframe="1d", limit=3)
        except Exception as e:
            pytest.skip(f"KuCoin public API unreachable from this environment: {e}")
        assert isinstance(candles, list)
