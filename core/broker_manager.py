from brokers.alpaca_connector import AlpacaConnector
from brokers.binance_connector import BinanceConnector
from brokers.simulatedbroker import SimulatedBroker


class BrokerManager:
    def __init__(self, alpaca_key=None, alpaca_secret=None,
                 binance_key=None, binance_secret=None, binance_testnet_key=None, binance_testnet_secret=None):
        self.brokers = {
            "Simulator": SimulatedBroker(),
            "Alpaca": AlpacaConnector(alpaca_key, alpaca_secret) if alpaca_key and alpaca_secret else None,
        }
        
        # Initialize Binance with error handling
        try:
            if binance_key and binance_secret:
                self.brokers["Binance"] = BinanceConnector(binance_key, binance_secret, paper=False)
            else:
                self.brokers["Binance"] = None
        except Exception as e:
            print(f"⚠️ Failed to connect to Binance: {e}")
            self.brokers["Binance"] = None
            
        try:
            if binance_testnet_key and binance_testnet_secret:
                self.brokers["Binance_testnet"] = BinanceConnector(binance_testnet_key, binance_testnet_secret, paper=True)
            else:
                self.brokers["Binance_testnet"] = None
        except Exception as e:
            print(f"⚠️ Failed to connect to Binance Testnet: {e}")
            self.brokers["Binance_testnet"] = None

    def get_broker(self, name):
        broker = self.brokers.get(name)
        if broker is None:
            # For Simulator, always return it even if it's None (shouldn't happen)
            if name == "Simulator":
                return self.brokers["Simulator"]
            raise ValueError(f"Broker '{name}' is not configured properly. Please check API keys in config/settings.py.")
        return broker
    def get_availabele_brokers(self):
        return [name for name, broker in self.brokers.items() if broker is not None]