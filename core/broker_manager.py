from brokers.alpaca_connector import AlpacaConnector
from brokers.binance_connector import BinanceConnector
from brokers.simulatedbroker import SimulatedBroker


class BrokerManager:
    def __init__(self, alpaca_key=None, alpaca_secret=None,
                 binance_key=None, binance_secret=None, binance_testnet_key=None, binance_testnet_secret=None):
        self.brokers = {
            "Simulator": SimulatedBroker(),
            "Alpaca": AlpacaConnector(alpaca_key, alpaca_secret) if alpaca_key and alpaca_secret else None,
            "Binance": BinanceConnector(binance_key, binance_secret, paper=False) if binance_key and binance_secret else None,
            "Binance_testnet": BinanceConnector(binance_testnet_key, binance_testnet_secret, paper=True) if binance_testnet_key and binance_testnet_secret else None
        }

    def get_broker(self, name):
        return self.brokers.get(name)