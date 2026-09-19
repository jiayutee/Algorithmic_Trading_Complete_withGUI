#Live Trading

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide


from brokers.execution_guard import guarded_live_order

class AlpacaConnector:
    def __init__(self, api_key, secret_key, paper=True):
        self.client = TradingClient(api_key, secret_key, paper=paper)
        # A paper account never risks real money, so the live-order guard passes it through.
        self.paper_mode = bool(paper)

    @guarded_live_order("Alpaca")
    def submit_order(self, symbol, qty, side, order_type='market'):
        # The UI sends "buy"/"sell"; only the literal "long" used to map to BUY, so a
        # "buy" order was silently submitted as a SELL.
        normalized = str(side).lower()
        if normalized in ('buy', 'long'):
            order_side = OrderSide.BUY
        elif normalized in ('sell', 'short'):
            order_side = OrderSide.SELL
        else:
            raise ValueError(f"Unknown order side: {side!r}")
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            type=order_type
        )
        return self.client.submit_order(order)

    def get_position(self, symbol):
        return self.client.get_open_position(symbol)