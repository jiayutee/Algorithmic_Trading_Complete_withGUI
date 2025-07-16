#Unified live/historical data
import websockets
import asyncio
import json
import numpy as np
import pandas as pd
import ta
import configparser
from time import sleep
from binance.exceptions import BinanceAPIException
from binance.client import Client

class BinanceConnector:
    def __init__(self, api_key, secret_key, paper=True):
        self.client = Client(api_key, secret_key, testnet=paper)

    def submit_order(self, symbol, qty, side, order_type='MARKET', futures=True):
        if futures:
            order = self.client.futures_create_order(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side.lower() == 'long' else OrderSide.SELL,
                type=order_type
            )
        else:
            order = self.client.create_order(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side.lower() == 'long' else OrderSide.SELL,
                type=order_type
            )
        return self.client.submit_order(order)

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