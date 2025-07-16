import time
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import random
import threading
import numpy as np
from collections import defaultdict


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

    def __init__(self, initial_balance: float = 100000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.portfolio_value = initial_balance
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

            # Update portfolio value
            self._update_portfolio_value()
            time.sleep(1)  # Update prices every second

    def _update_portfolio_value(self):
        """Calculate current portfolio value"""
        total_positions_value = sum(
            pos.qty * pos.last_price
            for pos in self.positions.values()
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
            leverage: float = 1.0
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
            current_price = self._get_market_price(symbol)

            order = Order(
                id=order_id,
                symbol=symbol,
                qty=qty,
                side=side,
                order_type=order_type,
                price=current_price,
                limit_price=limit_price,
                stop_price=stop_price
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
        # Calculate required capital (negative for sells)
        required_capital = order.qty * fill_price
        if order.side == OrderSide.SELL:
            required_capital *= -1

        # Check if we have enough buying power
        if required_capital > self.balance:
            order.status = OrderStatus.REJECTED
            return

        # Update position
        if order.symbol in self.positions:
            position = self.positions[order.symbol]
            if (position.qty > 0 and order.side == OrderSide.BUY) or \
                (position.qty < 0 and order.side == OrderSide.SELL):
                # Adding to position
                total_qty = position.qty + (order.qty if order.side == OrderSide.BUY else -order.qty)
                position.avg_price = (position.avg_price * abs(position.qty) + (fill_price * order.qty)) / abs(total_qty)
                position.qty = total_qty
            else:
                # Reducing or reversing position
                position.qty += (order.qty if order.side == OrderSide.BUY else -order.qty)
        else:
            #   New position
            self.positions[order.symbol] = Position(
                symbol=order.symbol,
                qty=order.qty if order.side == OrderSide.BUY else -order.qty,
                avg_price=fill_price,
                leverage=1.0
            )

        # Update balance
        self.balance -= required_capital

        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_qty = order.qty
        order.filled_avg_price = fill_price
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
        """Get current position for a symbol"""
        return self.positions.get(symbol)

    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get orders filtered by status"""
        if status is None:
            return list(self.orders.values())
        return [o for o in self.orders.values() if o.status == status]

    def get_account_info(self) -> dict:
        """Get current account information"""
        return {
            "balance": self.balance,
            "portfolio_value": self.portfolio_value,
            "cash": self.balance,
            "buying_power": self.balance * 2,  # Simple 2x leverage
            "positions_value": self.portfolio_value - self.balance,
            "initial_balance": self.initial_balance,
            "pnl": self.portfolio_value - self.initial_balance
        }

    def close(self):
        """Clean up the broker"""
        self._running = False
        if self._data_thread.is_alive():
            self._data_thread.join()