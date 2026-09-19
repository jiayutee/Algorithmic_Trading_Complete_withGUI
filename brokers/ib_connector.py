from ib_insync import IB, MarketOrder, Contract
from typing import Optional, Union


from brokers.execution_guard import guarded_live_order

class IBKRConnector:
    """
    Interactive Brokers Connector for live trading
    Requires IBKR TWS or Gateway running with API access enabled
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 7497, client_id: int = 1):
        """
        Initialize connection to TWS/Gateway
        :param host: IP address where TWS/Gateway is running
        :param port: IB defaults -- TWS: 7496 live / 7497 paper; IB Gateway: 4001 live / 4002 paper.
            Ports are user-configurable, so this connector never *assumes* a port means paper: orders always
            go through the live-order guard (brokers/execution_guard.py).
        :param client_id: Client ID for this connection (must be unique per connection)
        """
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.connect()

    def connect(self) -> None:
        """Establish connection to TWS/Gateway"""
        if not self.ib.isConnected():
            self.ib.connect(self.host, self.port, self.client_id)

    def disconnect(self) -> None:
        """Disconnect from TWS/Gateway"""
        if self.ib.isConnected():
            self.ib.disconnect()

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

    @guarded_live_order("IBKR")
    def submit_order(
            self,
            symbol: str,
            qty: Union[int, float],
            side: str,
            order_type: str = 'MKT',
            sec_type: str = 'STK',
            currency: str = 'USD',
            exchange: str = 'SMART'
    ) -> dict:
        """
        Submit an order to IBKR
        :param symbol: Ticker symbol
        :param qty: Quantity to trade (positive for buy, negative for sell)
        :param side: 'long' or 'short' (for position) or 'buy'/'sell'
        :param order_type: Order type (MKT, LMT, etc.)
        :param sec_type: Security type (STK, FUT, OPT, etc.)
        :param currency: Currency of the trade
        :param exchange: Exchange where the order should be routed
        :return: Dictionary with order details
        """
        # Create contract
        contract = Contract(
            symbol=symbol,
            secType=sec_type,
            currency=currency,
            exchange=exchange
        )

        # Resolve side to quantity
        if isinstance(side, str):
            side = side.lower()
            if side in ['long', 'buy']:
                qty = abs(qty)
            elif side in ['short', 'sell']:
                qty = -abs(qty)

        # Create and place order
        order = MarketOrder('BUY' if qty > 0 else 'SELL', abs(qty))
        trade = self.ib.placeOrder(contract, order)

        # Wait for execution
        self.ib.sleep(1)  # Give it a moment to process

        return {
            'order_id': trade.order.orderId,
            'symbol': symbol,
            'quantity': qty,
            'side': 'BUY' if qty > 0 else 'SELL',
            'status': trade.orderStatus.status,
            'filled': trade.orderStatus.filled,
            'remaining': trade.orderStatus.remaining,
            'avg_fill_price': trade.orderStatus.avgFillPrice
        }

    def get_position(self, symbol: str, sec_type: str = 'STK', currency: str = 'USD') -> Optional[dict]:
        """
        Get current position for a symbol
        :param symbol: Ticker symbol
        :param sec_type: Security type
        :param currency: Currency
        :return: Dictionary with position details or None if no position
        """
        positions = self.ib.positions()

        for position in positions:
            contract = position.contract
            if (contract.symbol == symbol and
                    contract.secType == sec_type and
                    contract.currency == currency):
                return {
                    'symbol': symbol,
                    'position': position.position,
                    'avg_cost': position.avgCost,
                    'contract': {
                        'sec_type': sec_type,
                        'currency': currency,
                        'exchange': contract.exchange
                    }
                }
        return None

    def get_positions(self) -> dict:
        """All open positions as {symbol: {qty, avg_cost, sec_type, currency, exchange}} (same keyed shape as the
        other brokers' portfolio entries). Multiple contracts on one symbol (e.g. option legs) are keyed
        ``SYMBOL:secType`` after the first so none is silently overwritten."""
        out: dict = {}
        for position in self.ib.positions():
            c = position.contract
            key = c.symbol if c.symbol not in out else f"{c.symbol}:{c.secType}"
            out[key] = {"qty": float(position.position), "avg_cost": float(position.avgCost),
                        "sec_type": c.secType, "currency": c.currency, "exchange": c.exchange}
        return out

    def get_account_info(self) -> dict:
        """Get basic account information"""
        account = self.ib.accountSummary()
        return {
            'net_liquidation': next((a.value for a in account if a.tag == 'NetLiquidation'), None),
            'buying_power': next((a.value for a in account if a.tag == 'BuyingPower'), None),
            'leverage': next((a.value for a in account if a.tag == 'GrossPositionValue'), None),
            'available_funds': next((a.value for a in account if a.tag == 'AvailableFunds'), None)
        }