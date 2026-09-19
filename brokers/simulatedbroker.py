import time
from datetime import datetime, date as date_cls
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import math
import random
import threading
from contextlib import contextmanager
import numpy as np
from collections import defaultdict
from core.logger import get_logger
from core.trade_rationale import unspecified_rationale
from brokers.paper_store import PaperStore

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
    realized_pnl: float = 0.0  # this order's own contribution to closing/reducing a position; 0 for opens
    fee: float = 0.0  # commission paid on this fill
    rationale: Dict = field(default_factory=unspecified_rationale)  # structured "why" (core/trade_rationale.py)
    reject_reason: str = ""  # why a REJECTED order was rejected (empty otherwise)


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
    Paper-trading broker: market/limit/stop orders, leverage flag, realized/unrealized P&L, fees.

    Prices come from the caller (``update_price`` / ``execution_price``); by default nothing is invented.
      * ``simulate_prices=True`` restores the old random-walk generator (upward-biased -- for demos/tests only).
      * ``strict_prices=True``: a market order for a symbol with no known (or, with ``max_price_age_s``, a stale) price is
        REJECTED with a reason instead of silently filling at a made-up $100.
      * ``persist_path``: keep the account in a SQLite file (see brokers/paper_store.py) so it survives restarts and can be
        shared by the desktop app and the Dash view.
    Pending limit/stop orders are re-checked whenever ``update_price`` delivers a new price.
    """

    def __init__(self, initial_balance: float = 100000.0, market_fee: float = 0.001, limit_fee: float = 0.0005,
                 *, simulate_prices: bool = False, strict_prices: bool = False,
                 max_price_age_s: Optional[float] = None, persist_path: Optional[str] = None):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.market_fee = market_fee
        self.limit_fee = limit_fee
        self.strict_prices = strict_prices
        self.max_price_age_s = max_price_age_s
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.portfolio_value = initial_balance
        self.realized_pnl: float = 0.0
        self.market_data: Dict[str, float] = defaultdict(lambda: 100.0)
        self._price_time: Dict[str, float] = {}      # when update_price last set each symbol
        self._dirty_orders: set = set()              # order ids created/changed during the current operation
        self._running = True
        self._lock = threading.RLock()
        self._store: Optional[PaperStore] = PaperStore(persist_path) if persist_path else None
        if self._store is not None:
            with self._lock:
                if self._store.has_state():
                    self._load_from_store_locked()
                else:
                    with self._store.transaction():
                        self._persist_locked()

        # The random-walk generator is opt-in: it invents prices, so paper P&L built on it is not a market result.
        self._data_thread = threading.Thread(target=self._simulate_market_data, daemon=True)
        if simulate_prices:
            self._data_thread.start()

    # ------------------------------------------------------------------
    # Durable account (no-ops when persist_path is None)
    # ------------------------------------------------------------------

    @staticmethod
    def _order_to_dict(o: "Order") -> dict:
        d = dict(o.__dict__)
        d["side"], d["order_type"], d["status"] = o.side.value, o.order_type.value, o.status.value
        return d

    @staticmethod
    def _order_from_dict(d: dict) -> "Order":
        d = dict(d)
        d["side"], d["order_type"], d["status"] = OrderSide(d["side"]), OrderType(d["order_type"]), OrderStatus(d["status"])
        known = {f for f in Order.__dataclass_fields__}
        return Order(**{k: v for k, v in d.items() if k in known})

    def _persist_locked(self) -> None:
        """Write account state + the orders touched by this operation. Caller holds the lock and a store transaction."""
        if self._store is None:
            return
        meta = {"balance": self.balance, "initial_balance": self.initial_balance, "realized_pnl": self.realized_pnl,
                "market_fee": self.market_fee, "limit_fee": self.limit_fee}
        positions = [{"symbol": p.symbol, "qty": p.qty, "avg_price": p.avg_price, "leverage": p.leverage}
                     for p in self.positions.values()]
        touched = [self._order_to_dict(self.orders[i]) for i in self._dirty_orders if i in self.orders]
        self._store.save(meta, positions, touched)
        self._dirty_orders.clear()

    def _load_from_store_locked(self) -> None:
        state = self._store.load()
        m = state["meta"]
        self.balance = float(m.get("balance", self.balance))
        self.initial_balance = float(m.get("initial_balance", self.initial_balance))
        self.realized_pnl = float(m.get("realized_pnl", 0.0))
        self.positions = {p["symbol"]: Position(symbol=p["symbol"], qty=p["qty"], avg_price=p["avg_price"], leverage=p["leverage"],
                                                last_price=self.market_data.get(p["symbol"], 0) if p["symbol"] in self.market_data else 0)
                          for p in state["positions"]}
        self.order_history = [self._order_from_dict(o) for o in state["orders"]]
        self.orders = {o.id: o for o in self.order_history}
        self._update_portfolio_value()

    def _refresh_locked(self) -> None:
        """Pick up commits made by another process (e.g. the Dash view while the desktop app is open)."""
        if self._store is not None and self._store.changed_externally():
            self._load_from_store_locked()

    @contextmanager
    def _operation(self):
        """One atomic read-modify-write on the account. With a store: BEGIN IMMEDIATE, reload, run, save, commit."""
        with self._lock:
            if self._store is None:
                yield
                return
            with self._store.transaction():
                self._load_from_store_locked()
                yield
                self._persist_locked()

    def reset(self, initial_balance: Optional[float] = None) -> None:
        """Wipe the account back to a fresh balance (positions, orders, realized P&L). Prices are kept."""
        with self._lock:
            if initial_balance is not None:
                self.initial_balance = float(initial_balance)
            self.balance = self.initial_balance
            self.positions, self.orders, self.order_history = {}, {}, []
            self.realized_pnl = 0.0
            self._dirty_orders.clear()
            self._update_portfolio_value()
            if self._store is not None:
                with self._store.transaction():
                    self._store.wipe()
                    self._persist_locked()

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------

    def update_price(self, symbol: str, price: float) -> None:
        """Feed a REAL market price. Marks positions to it and re-checks pending limit/stop orders for that symbol."""
        try:
            price = float(price)
        except (TypeError, ValueError):
            return
        if not symbol or not math.isfinite(price) or price <= 0:
            return
        with self._lock:
            self.market_data[symbol] = price
            self._price_time[symbol] = time.time()
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos.last_price = price
                pos.pnl = pos.qty * (price - pos.avg_price)
            if any(o.status == OrderStatus.PENDING and o.symbol == symbol for o in self.orders.values()):
                with self._operation():                       # reloads state, so pick the pending orders AFTER entering
                    for o in [o for o in self.orders.values() if o.status == OrderStatus.PENDING and o.symbol == symbol]:
                        before = o.status
                        if o.order_type == OrderType.LIMIT:
                            self._process_limit_order(o)
                        elif o.order_type == OrderType.STOP:
                            self._process_stop_order(o)
                        if o.status != before:
                            self._dirty_orders.add(o.id)
            self._update_portfolio_value()

    def price_age(self, symbol: str) -> Optional[float]:
        """Seconds since ``update_price`` last set this symbol; None if it was never fed (or set directly)."""
        t = self._price_time.get(symbol)
        return None if t is None else time.time() - t

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

    def _get_market_price(self, symbol: str) -> Optional[float]:
        """Current price for ``symbol``.

        Non-strict (default, for backward compatibility): unknown symbols fall back to 100.0.
        Strict: None when the price is unknown, or older than ``max_price_age_s`` -- callers must not invent one.
        """
        if symbol in self.market_data:
            if self.strict_prices and self.max_price_age_s is not None:
                age = self.price_age(symbol)
                if age is not None and age > self.max_price_age_s:
                    return None
            return self.market_data[symbol]
        return None if self.strict_prices else 100.0

    def submit_order(
            self,
            symbol: str,
            qty: float,
            side: Union[str, OrderSide],
            order_type: Union[str, OrderType] = OrderType.MARKET,
            limit_price: Optional[float] = None,
            stop_price: Optional[float] = None,
            leverage: float = 1.0,
            execution_price: Optional[float] = None,
            rationale: Optional[Dict] = None
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
            rationale: Structured "why" for this order (see core/trade_rationale.py);
                       defaults to an explicit "unspecified" record so no order is
                       ever stored without one.

        Returns:
            Order object with status
        """
        with self._operation():
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
                current_price = self._get_market_price(symbol)      # None only in strict mode (unknown / stale price)

            order = Order(
                id=order_id,
                symbol=symbol,
                qty=qty,
                side=side,
                order_type=order_type,
                price=current_price,
                limit_price=limit_price,
                stop_price=stop_price,
                execution_price=execution_price,
                rationale=rationale if rationale is not None else unspecified_rationale(),
            )

            # Process order based on type
            if order_type == OrderType.MARKET and current_price is None:
                order.status = OrderStatus.REJECTED
                order.reject_reason = (f"no current price for {symbol}" if symbol not in self.market_data
                                       else f"price for {symbol} is stale (> {self.max_price_age_s:.0f}s old)")
                logger.warning("Rejected market order for %s: %s", symbol, order.reject_reason)
            elif order_type == OrderType.MARKET:
                self._process_market_order(order)
            elif order_type == OrderType.LIMIT:
                self._process_limit_order(order)     # stays PENDING until a price update crosses the limit
            elif order_type == OrderType.STOP:
                self._process_stop_order(order)

            # Store order
            self.orders[order_id] = order
            self.order_history.append(order)
            self._dirty_orders.add(order_id)
            return order

    def _process_market_order(self, order: Order):
        """Execute market order immediately"""
        fill_price = self._get_market_price(order.symbol)
        if fill_price is None:
            order.status = OrderStatus.REJECTED
            order.reject_reason = f"no current price for {order.symbol}"
            return
        self._fill_order(order, fill_price)

    def _process_limit_order(self, order: Order):
        """Process limit order (may not fill immediately)"""
        current_price = self._get_market_price(order.symbol)
        if current_price is None:
            return                                   # cannot evaluate without a price: stay pending

        if order.side == OrderSide.BUY and order.limit_price >= current_price:
            self._fill_order(order, min(order.limit_price, current_price))
        elif order.side == OrderSide.SELL and order.limit_price <= current_price:
            self._fill_order(order, max(order.limit_price, current_price))

    def _process_stop_order(self, order: Order):
        """Process stop order (may not fill immediately)"""
        current_price = self._get_market_price(order.symbol)
        if current_price is None:
            return

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
                order.reject_reason = "insufficient funds (including fees)"
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
        order.fee = actual_fee

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
            order.reject_reason = f"insufficient funds: need ${total_cost:,.2f}, have ${self.balance:,.2f}"
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
                    order.realized_pnl = executable_qty * (actual_fill_price - position.avg_price)
                    self.realized_pnl += order.realized_pnl
                elif order.side == OrderSide.BUY and old_qty < 0:
                    # Covering a short position
                    order.realized_pnl = executable_qty * (position.avg_price - actual_fill_price)
                    self.realized_pnl += order.realized_pnl
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
        with self._operation():
            if order_id in self.orders and self.orders[order_id].status == OrderStatus.PENDING:
                self.orders[order_id].status = OrderStatus.CANCELED
                self.orders[order_id].updated_at = time.time()
                self._dirty_orders.add(order_id)
                return True
            return False

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for a symbol (thread-safe)"""
        with self._lock:
            self._refresh_locked()
            return self.positions.get(symbol)

    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get orders filtered by status (thread-safe snapshot)"""
        with self._lock:
            self._refresh_locked()
            if status is None:
                return list(self.orders.values())
            return [o for o in self.orders.values() if o.status == status]

    def get_account_info(self) -> dict:
        """Get current account information (thread-safe snapshot)"""
        with self._lock:
            self._refresh_locked()
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
            self._refresh_locked()
            return self.realized_pnl

    def get_unrealized_pnl(self, prices: Optional[Dict[str, float]] = None) -> float:
        """Return mark-to-market PnL for all currently open positions.

        Args:
            prices: Optional mapping of symbol -> current price to use instead
                    of the broker's internal random-walk market data.  Pass
                    deterministic values in tests to avoid flakiness.
        """
        with self._lock:
            self._refresh_locked()
            return self._get_unrealized_pnl_locked(prices=prices)

    def get_total_pnl(self, prices: Optional[Dict[str, float]] = None) -> float:
        """Return realized + unrealized PnL in a single consistent snapshot."""
        with self._lock:
            self._refresh_locked()
            return self.realized_pnl + self._get_unrealized_pnl_locked(prices=prices)

    def get_pnl_by_day(self) -> Dict[date_cls, float]:
        """Return realized PnL grouped by calendar day, net of fees.

        Powers the PnL calendar view: {date: net_pnl_that_day}. Each closing
        order's own realized_pnl (set at fill time in _fill_order) is summed
        per day, minus ALL fees paid that day -- both closing-order fees and
        opening-order fees, since an opening trade's commission is real cash
        that left the balance that day even though its P&L impact isn't
        "realized" until the position eventually closes (possibly on a
        different day). This matches what a trader would actually see in
        their day-over-day account balance change, not just booked trade P&L.

        Days with only opening trades (no closes) still appear, with a
        negative value equal to that day's opening fees -- this is correct,
        not a bug: opening a position is a real cash outflow on that day.
        """
        with self._lock:
            self._refresh_locked()
            by_day: Dict[date_cls, float] = defaultdict(float)
            for order in self.order_history:
                if order.status != OrderStatus.FILLED:
                    continue
                day = datetime.fromtimestamp(order.created_at).date()
                by_day[day] += order.realized_pnl - order.fee
            return dict(by_day)

    def close(self):
        """Clean up the broker"""
        self._running = False
        if self._data_thread.is_alive():
            self._data_thread.join(timeout=3)
        if self._store is not None:
            self._store.close()