import time
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import random
import threading
import numpy as np
from collections import defaultdict
from core.logger import get_logger

logger = get_logger(__name__)


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELED = "canceled"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    id: str
    symbol: str
    qty: float
    side: OrderSide
    order_type: OrderType
    price: Optional[float] = None
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    execution_price: Optional[float] = None 
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0
    filled_avg_price: float = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class Position:
    symbol: str
    qty: float
    avg_price: float
    leverage: float = 1.0
    pnl: float = 0
    last_price: float = 0


class SimulatedBroker:
    """
    Simulated broker for paper trading with market data generation.
    Supports market/limit/stop orders, leverage, and PnL tracking.
    """

    def __init__(self, initial_balance: float = 100000.0, market_fee: float = 0.001, limit_fee: float = 0.0005):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.market_fee = market_fee
        self.limit_fee = limit_fee
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.portfolio_value = initial_balance
        self.realized_pnl: float = 0.0
        self.market_data: Dict[str, float] = defaultdict(lambda: 100.0)
        self._running = True
        self._lock = threading.Lock()

        # Start market data simulation thread
        self._data_thread = threading.Thread(target=self._simulate_market_data)
        self._data_thread.daemon = True
        self._data_thread.start()

    def _simulate_market_data(self):
        """Background thread to simulate changing market prices"""
        while self._running:
            with self._lock:
                for symbol in list(self.market_data.keys()):
                    # Random walk with slight upward bias
                    change = random.uniform(-0.5, 1.0)
                    self.market_data[symbol] = max(0.01, self.market_data[symbol] * (1 + change / 100))

                    # Update position PnL if we have positions
                    if symbol in self.positions:
                        pos = self.positions[symbol]
                        pos.last_price = self.market_data[symbol]
                        pos.pnl = pos.qty * (pos.last_price - pos.avg_price)

                # Update portfolio value inside the lock so reads see a consistent state
                self._update_portfolio_value()
            time.sleep(1)  # Update prices every second

    def _update_portfolio_value(self):
        """Calculate current portfolio value.

        Uses self.market_data (current price) rather than pos.last_price --
        that field defaults to 0 and is only ever updated by
        _simulate_market_data(), the broker's internal fake random-walk
        generator. It is never refreshed by real market data (e.g. from
        LivePriceService or any direct write to self.market_data), so
        portfolio_value/pnl/positions_value in get_account_info() were
        silently wrong for any real (non-simulated) position. Mirrors the
        already-correct pattern in _get_unrealized_pnl_locked().
        """
        total_positions_value = sum(
            pos.qty * self.market_data.get(symbol, pos.avg_price)
            for symbol, pos in self.positions.items()
        )
        self.portfolio_value = self.balance + total_positions_value

    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        return f"simorder_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"

    def _get_market_price(self, symbol: str) -> float:
        """Get current simulated market price"""
        return self.market_data.get(symbol, 100.0)

    def submit_order(
            self,
            symbol: str,
            qty: float,
            side: Union[str, OrderSide],
            order_type: Union[str, OrderType] = OrderType.MARKET,
            limit_price: Optional[float] = None,
            stop_price: Optional[float] = None,
            leverage: float = 1.0,
            execution_price: Optional[float] = None
    ) -> Order:
        """
        Submit an order to the simulated broker

        Args:
            symbol: Trading symbol (e.g. 'AAPL')
            qty: Number of shares/contracts
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', or 'stop'
            limit_price: Required for limit orders
            stop_price: Required for stop orders
            leverage: Leverage multiplier (1.0 = no leverage)

        Returns:
            Order object with status
        """
        with self._lock:
            # Convert string inputs to enums
            if isinstance(side, str):
                side = OrderSide(side.lower())
            if isinstance(order_type, str):
                order_type = OrderType(order_type.lower())

            # Create order
            order_id = self._generate_order_id()
            # Use execution_price if provided, otherwise get from market data
            if execution_price is not None:
                current_price = execution_price
                logger.debug("Using execution price: $%.2f", execution_price)
            else:
                current_price = self._get_market_price(symbol)

            order = Order(
                id=order_id,
                symbol=symbol,
                qty=qty,
                side=side,
                order_type=order_type,
                price=current_price,
                limit_price=limit_price,
                stop_price=stop_price,
                execution_price=execution_price
            )

            # Process order based on type
            if order_type == OrderType.MARKET:
                self._process_market_order(order)
            elif order_type == OrderType.LIMIT:
                self._process_limit_order(order)
            elif order_type == OrderType.STOP:
                self._process_stop_order(order)

            # Store order
            self.orders[order_id] = order
            self.order_history.append(order)

            return order

    def _process_market_order(self, order: Order):
        """Execute market order immediately"""
        fill_price = self._get_market_price(order.symbol)
        self._fill_order(order, fill_price)

    def _process_limit_order(self, order: Order):
        """Process limit order (may not fill immediately)"""
        current_price = self._get_market_price(order.symbol)

        if order.side == OrderSide.BUY and order.limit_price >= current_price:
            self._fill_order(order, min(order.limit_price, current_price))
        elif order.side == OrderSide.SELL and order.limit_price <= current_price:
            self._fill_order(order, max(order.limit_price, current_price))

    def _process_stop_order(self, order: Order):
        """Process stop order (may not fill immediately)"""
        current_price = self._get_market_price(order.symbol)

        if order.side == OrderSide.BUY and order.stop_price <= current_price:
            self._fill_order(order, current_price)
        elif order.side == OrderSide.SELL and order.stop_price >= current_price:
            self._fill_order(order, current_price)

    def _fill_order(self, order: Order, fill_price: float):
        """Execute an order fill"""
        logger.debug("Filling order — balance before: $%.2f, positions: %s", self.balance, list(self.positions.keys()))
        if order.execution_price is not None:
            actual_fill_price = order.execution_price
        else:
            actual_fill_price = fill_price

        # CALCULATE FEE
        fee_rate = self.limit_fee if order.order_type == OrderType.LIMIT else self.market_fee
        order_value = order.qty * actual_fill_price
        fee_amount = order_value * fee_rate

        # CALCULATE MAX AFFORDABLE QUANTITY (including fee)
        # balance >= qty * price + qty * price * fee_rate
        # balance >= qty * price * (1 + fee_rate)
        # qty <= balance / (price * (1 + fee_rate))
        max_affordable_qty = self.balance / (actual_fill_price * (1 + fee_rate)) if actual_fill_price > 0 else 0
        
        # For buys: use minimum of requested quantity and affordable quantity
        if order.side == OrderSide.BUY:
            executable_qty = min(order.qty, max_affordable_qty)
            if executable_qty <= 0:
                logger.warning("Insufficient funds to buy %s (incl. fees)", order.symbol)
                order.status = OrderStatus.REJECTED
                return
            if executable_qty < order.qty:
                logger.debug("Adjusting buy qty %.6f -> %.6f (max affordable)", order.qty, executable_qty)
                order.qty = executable_qty
        else:
            executable_qty = order.qty
            if order.symbol in self.positions:
                current_position = self.positions[order.symbol].qty
                executable_qty = min(order.qty, abs(current_position))
                if executable_qty < order.qty:
                    logger.debug("Adjusting sell qty %.6f -> %.6f (position size)", order.qty, executable_qty)
                    order.qty = executable_qty

        # Calculate required capital and actual fee
        required_capital = executable_qty * actual_fill_price
        actual_fee = required_capital * fee_rate
        
        if order.side == OrderSide.BUY:
            total_cost = required_capital + actual_fee
        else:
            # For sells, fee is deducted from proceeds
            total_cost = -(required_capital - actual_fee)

        logger.debug("Order: %s %.6f %s @ $%.2f | fee $%.2f | total $%.2f",
                     order.side.value, executable_qty, order.symbol,
                     actual_fill_price, actual_fee, abs(total_cost))

        # Check if we have enough buying power (for buys)
        if order.side == OrderSide.BUY and total_cost > self.balance:
            logger.warning("Insufficient funds: need $%.2f, have $%.2f", total_cost, self.balance)
            order.status = OrderStatus.REJECTED
            return

        # Update position
        if order.symbol in self.positions:
            position = self.positions[order.symbol]
            logger.debug("Existing position: %.6f @ $%.2f", position.qty, position.avg_price)
            
            if (position.qty > 0 and order.side == OrderSide.BUY) or \
                (position.qty < 0 and order.side == OrderSide.SELL):
                total_qty = position.qty + (executable_qty if order.side == OrderSide.BUY else -executable_qty)
                old_value = position.avg_price * abs(position.qty)
                new_value = actual_fill_price * executable_qty
                position.avg_price = (old_value + new_value) / abs(total_qty)
                position.qty = total_qty
                logger.debug("Added to position -> %.6f @ $%.2f", position.qty, position.avg_price)
            else:
                old_qty = position.qty
                # Accrue realized PnL for the portion being closed
                if order.side == OrderSide.SELL and old_qty > 0:
                    # Closing/reducing a long position
                    self.realized_pnl += executable_qty * (actual_fill_price - position.avg_price)
                elif order.side == OrderSide.BUY and old_qty < 0:
                    # Covering a short position
                    self.realized_pnl += executable_qty * (position.avg_price - actual_fill_price)
                position.qty += (executable_qty if order.side == OrderSide.BUY else -executable_qty)
                logger.debug("Changed position: %.6f -> %.6f", old_qty, position.qty)
                if abs(position.qty) < 0.000001:
                    del self.positions[order.symbol]
                    logger.debug("Position closed for %s", order.symbol)
        else:
            position_qty = executable_qty if order.side == OrderSide.BUY else -executable_qty
            self.positions[order.symbol] = Position(
                symbol=order.symbol,
                qty=position_qty,
                avg_price=actual_fill_price,
                leverage=1.0,
            )
            logger.debug("New position: %.6f @ $%.2f", position_qty, actual_fill_price)

        old_balance = self.balance
        self.balance -= total_cost
        logger.debug("Balance: $%.2f -> $%.2f", old_balance, self.balance)

        order.status = OrderStatus.FILLED
        order.filled_qty = executable_qty
        order.filled_avg_price = actual_fill_price
        order.updated_at = time.time()

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order"""
        with self._lock:
            if order_id in self.orders and self.orders[order_id].status == OrderStatus.PENDING:
                self.orders[order_id].status = OrderStatus.CANCELED
                self.orders[order_id].updated_at = time.time()
                return True
            return False

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for a symbol (thread-safe)"""
        with self._lock:
            return self.positions.get(symbol)

    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get orders filtered by status (thread-safe snapshot)"""
        with self._lock:
            if status is None:
                return list(self.orders.values())
            return [o for o in self.orders.values() if o.status == status]

    def get_account_info(self) -> dict:
        """Get current account information (thread-safe snapshot)"""
        with self._lock:
            self._update_portfolio_value()
            unrealized = self._get_unrealized_pnl_locked()
            return {
                "balance": self.balance,
                "portfolio_value": self.portfolio_value,
                "cash": self.balance,
                "buying_power": self.balance * 2,  # Simple 2x leverage
                "positions_value": self.portfolio_value - self.balance,
                "initial_balance": self.initial_balance,
                "pnl": self.portfolio_value - self.initial_balance,
                "realized_pnl": self.realized_pnl,
                "unrealized_pnl": unrealized,
            }

    # ------------------------------------------------------------------
    # PnL split: realized vs. unrealized
    # ------------------------------------------------------------------

    def _get_unrealized_pnl_locked(self, prices: Optional[Dict[str, float]] = None) -> float:
        """Mark-to-market PnL for open positions.  Caller must hold self._lock."""
        total = 0.0
        for symbol, pos in self.positions.items():
            if prices and symbol in prices:
                current_price = prices[symbol]
            else:
                current_price = self.market_data.get(symbol, pos.avg_price)
            total += pos.qty * (current_price - pos.avg_price)
        return total

    def get_realized_pnl(self) -> float:
        """Return total PnL locked in from closed / partially-closed positions."""
        with self._lock:
            return self.realized_pnl

    def get_unrealized_pnl(self, prices: Optional[Dict[str, float]] = None) -> float:
        """Return mark-to-market PnL for all currently open positions.

        Args:
            prices: Optional mapping of symbol -> current price to use instead
                    of the broker's internal random-walk market data.  Pass
                    deterministic values in tests to avoid flakiness.
        """
        with self._lock:
            return self._get_unrealized_pnl_locked(prices=prices)

    def get_total_pnl(self, prices: Optional[Dict[str, float]] = None) -> float:
        """Return realized + unrealized PnL in a single consistent snapshot."""
        with self._lock:
            return self.realized_pnl + self._get_unrealized_pnl_locked(prices=prices)

    def close(self):
        """Clean up the broker"""
        self._running = False
        if self._data_thread.is_alive():
            self._data_thread.join()