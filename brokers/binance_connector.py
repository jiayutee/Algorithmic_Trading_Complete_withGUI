from binance.exceptions import BinanceAPIException
from binance.client import Client

class BinanceConnector:
    def __init__(self, api_key, secret_key, paper=True):
        try:
            self.client = Client(api_key, secret_key, testnet=paper)
        except Exception as e:
            raise RuntimeError(f"Failed to initialise Binance client: {e}") from e

    def submit_order(self, symbol, qty, side, order_type='MARKET', futures=True):
        order_side = "BUY" if side.lower() in ('long', 'buy') else "SELL"
        if futures:
            return self.client.futures_create_order(
                symbol=symbol,
                quantity=qty,
                side=order_side,
                type=order_type
            )
        else:
            return self.client.create_order(
                symbol=symbol,
                quantity=qty,
                side=order_side,
                type=order_type
            )

    def get_position(self, symbol, futures=True):
        if futures:
            positions = self.client.futures_position_information()
            position = next((p for p in positions if p['symbol'] == symbol), None)

            if not position or float(position['positionAmt']) == 0:
                return None

            return {
                'symbol': position['symbol'],
                'qty': float(position['positionAmt']),
                'side': 'long' if float(position['positionAmt']) > 0 else 'short',
                'avg_entry_price': float(position['entryPrice']),
                'current_price': float(position['markPrice']),
                'unrealized_pl': float(position['unRealizedProfit']),
                'leverage': int(position['leverage'])
            }
        else:
            balances = self.client.get_account()['balances']
            asset = symbol.replace('USDT', '') if symbol.endswith('USDT') else symbol
            balance = next((b for b in balances if b['asset'] == asset), None)

            if not balance or (float(balance['free']) == 0 and float(balance['locked']) == 0):
                return None

            return {
                'symbol': asset,
                'qty': float(balance['free']) + float(balance['locked']),
                'side': 'long',  # Spot is always long
                'avg_entry_price': None,  # Binance spot doesn't provide this
                'current_price': float(self.client.get_symbol_ticker(symbol=symbol)['price']),
                'unrealized_pl': None  # Not directly available for spot
            }

    def get_historical_klines(self, symbol, interval, start_str, end_str=None):
        """Get historical klines from Binance"""
        return self.client.get_historical_klines(symbol, interval, start_str, end_str)