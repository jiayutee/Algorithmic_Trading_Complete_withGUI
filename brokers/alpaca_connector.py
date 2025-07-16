#Live Trading

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide


class AlpacaConnector:
    def __init__(self, api_key, secret_key, paper=True):
        self.client = TradingClient(api_key, secret_key, paper=paper)

    def submit_order(self, symbol, qty, side, order_type='market'):
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side.lower() == 'long' else OrderSide.SELL,
            type=order_type
        )
        return self.client.submit_order(order)

    def get_position(self, symbol):
        return self.client.get_open_position(symbol)