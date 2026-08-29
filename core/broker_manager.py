import logging

try:
    from brokers.alpaca_connector import AlpacaConnector
    _ALPACA_AVAILABLE = True
except ImportError:
    AlpacaConnector = None
    _ALPACA_AVAILABLE = False

try:
    from brokers.binance_connector import BinanceConnector
    _BINANCE_AVAILABLE = True
except ImportError:
    BinanceConnector = None
    _BINANCE_AVAILABLE = False

try:
    from brokers.kucoin_connector import KuCoinConnector
    _KUCOIN_AVAILABLE = True
except ImportError:
    KuCoinConnector = None
    _KUCOIN_AVAILABLE = False

try:
    from brokers.mexc_connector import MexcConnector
    _MEXC_AVAILABLE = True
except ImportError:
    MexcConnector = None
    _MEXC_AVAILABLE = False

from brokers.simulatedbroker import SimulatedBroker

logger = logging.getLogger(__name__)


class BrokerManager:
    def __init__(self, alpaca_key=None, alpaca_secret=None,
                 binance_key=None, binance_secret=None, binance_testnet_key=None, binance_testnet_secret=None,
                 kucoin_key=None, kucoin_secret=None, kucoin_password=None,
                 mexc_key=None, mexc_secret=None):
        self.brokers = {
            "Simulator": SimulatedBroker(),
        }

        # Alpaca connector (requires alpaca-trade-api / alpaca-py package)
        try:
            if _ALPACA_AVAILABLE and alpaca_key and alpaca_secret:
                self.brokers["Alpaca"] = AlpacaConnector(alpaca_key, alpaca_secret)
            else:
                self.brokers["Alpaca"] = None
        except Exception as e:
            logger.warning("Failed to connect to Alpaca: %s", e)
            self.brokers["Alpaca"] = None

        # Initialize Binance with error handling
        try:
            if _BINANCE_AVAILABLE and binance_key and binance_secret:
                self.brokers["Binance"] = BinanceConnector(binance_key, binance_secret, paper=False)
            else:
                self.brokers["Binance"] = None
        except Exception as e:
            logger.warning("Failed to connect to Binance: %s", e)
            self.brokers["Binance"] = None

        try:
            if _BINANCE_AVAILABLE and binance_testnet_key and binance_testnet_secret:
                self.brokers["Binance_testnet"] = BinanceConnector(binance_testnet_key, binance_testnet_secret, paper=True)
            else:
                self.brokers["Binance_testnet"] = None
        except Exception as e:
            logger.warning("Failed to connect to Binance Testnet: %s", e)
            self.brokers["Binance_testnet"] = None

        # Initialize KuCoin with error handling
        try:
            if _KUCOIN_AVAILABLE and kucoin_key and kucoin_secret:
                self.brokers["KuCoin"] = KuCoinConnector(
                    kucoin_key, kucoin_secret, kucoin_password or "", paper_mode=False
                )
            else:
                self.brokers["KuCoin"] = None
        except Exception as e:
            logger.warning("Failed to connect to KuCoin: %s", e)
            self.brokers["KuCoin"] = None

        # Initialize MEXC with error handling
        try:
            if _MEXC_AVAILABLE and mexc_key and mexc_secret:
                self.brokers["MEXC"] = MexcConnector(mexc_key, mexc_secret, paper_mode=False)
            else:
                self.brokers["MEXC"] = None
        except Exception as e:
            logger.warning("Failed to connect to MEXC: %s", e)
            self.brokers["MEXC"] = None

    def get_broker(self, name):
        broker = self.brokers.get(name)
        if broker is None:
            # For Simulator, always return it even if it's None (shouldn't happen)
            if name == "Simulator":
                return self.brokers["Simulator"]
            raise ValueError(f"Broker '{name}' is not configured properly. Please check API keys in config/settings.py.")
        return broker

    def get_available_brokers(self):
        """Return names of all configured (non-None) brokers."""
        return [name for name, broker in self.brokers.items() if broker is not None]

    # Keep the old misspelled name as a backward-compatible alias
    def get_availabele_brokers(self):
        """Deprecated alias for get_available_brokers (typo kept for backward compat)."""
        return self.get_available_brokers()

    def get_portfolio(self) -> dict:
        """Aggregate portfolio data from all active brokers.

        Returns a dict keyed by broker name, each value containing at minimum:
            {
                "cash": float,
                "positions": {symbol: {...position fields...}},
            }

        If a broker raises during the query, its entry will contain an "error"
        key instead of cash/positions so that one broken broker never prevents
        the others from being reported.
        """
        result: dict = {}
        active_brokers = self.get_available_brokers()

        for name in active_brokers:
            broker = self.brokers[name]
            if broker is None:
                continue
            try:
                entry = _extract_portfolio(name, broker)
            except Exception as exc:
                logger.error("get_portfolio: broker %s raised an unexpected error: %s", name, exc)
                entry = {"error": str(exc)}
            result[name] = entry

        return result


# ---------------------------------------------------------------------------
# Internal helpers — kept outside the class to stay testable in isolation
# ---------------------------------------------------------------------------

def _extract_portfolio(broker_name: str, broker) -> dict:
    """Pull cash + positions from a single broker connector.

    Each connector has a slightly different API surface:
    - SimulatedBroker  → get_account_info() + .positions dict
    - AlpacaConnector  → TradingClient; no account-level helper yet
    - BinanceConnector → no account-level helper yet
    - KuCoinConnector / MexcConnector → ccxt client; unified fetch_balance()
    - IBKRConnector    → get_account_info() (not currently wired into BrokerManager)

    For connectors that don't expose an account method we return what we can
    and mark the rest as None rather than raising.
    """
    entry: dict = {"cash": None, "positions": {}}

    # --- SimulatedBroker ---
    if hasattr(broker, "get_account_info") and hasattr(broker, "positions"):
        try:
            info = broker.get_account_info()
            entry["cash"] = float(info.get("cash", info.get("balance", 0.0)))
            entry["portfolio_value"] = float(info.get("portfolio_value", entry["cash"]))
            entry["pnl"] = info.get("pnl")
            # Convert Position dataclass objects to plain dicts
            positions_raw = broker.positions
            for symbol, pos in positions_raw.items():
                if hasattr(pos, "__dict__"):
                    entry["positions"][symbol] = vars(pos)
                else:
                    entry["positions"][symbol] = pos
        except Exception as exc:
            logger.warning("get_portfolio: %s get_account_info failed: %s", broker_name, exc)
            entry["error"] = str(exc)
        return entry

    # --- IBKRConnector (has get_account_info but no .positions dict) ---
    if hasattr(broker, "get_account_info"):
        try:
            info = broker.get_account_info()
            entry["cash"] = info.get("available_funds") or info.get("buying_power")
            entry["account_info"] = info
        except Exception as exc:
            logger.warning("get_portfolio: %s get_account_info failed: %s", broker_name, exc)
            entry["error"] = str(exc)
        return entry

    # --- AlpacaConnector ---
    if hasattr(broker, "client") and hasattr(broker.client, "get_all_positions"):
        try:
            raw_positions = broker.client.get_all_positions()
            for pos in raw_positions:
                sym = getattr(pos, "symbol", None) or str(pos)
                entry["positions"][sym] = {
                    "qty": float(getattr(pos, "qty", 0)),
                    "avg_entry_price": float(getattr(pos, "avg_entry_price", 0)),
                    "current_price": float(getattr(pos, "current_price", 0) or 0),
                    "unrealized_pl": float(getattr(pos, "unrealized_pl", 0) or 0),
                    "market_value": float(getattr(pos, "market_value", 0) or 0),
                }
        except Exception as exc:
            logger.warning("get_portfolio: %s get_all_positions failed: %s", broker_name, exc)
            entry["error"] = str(exc)
        # Alpaca TradingClient exposes account info via get_account()
        try:
            acct = broker.client.get_account()
            entry["cash"] = float(getattr(acct, "cash", 0) or 0)
            entry["portfolio_value"] = float(getattr(acct, "portfolio_value", 0) or 0)
        except Exception as exc:
            logger.warning("get_portfolio: %s get_account failed: %s", broker_name, exc)
        return entry

    # --- BinanceConnector (spot account balance) ---
    if hasattr(broker, "client") and hasattr(broker.client, "get_account"):
        try:
            acct = broker.client.get_account()
            balances = acct.get("balances", []) if isinstance(acct, dict) else []
            non_zero = {
                b["asset"]: {"free": float(b["free"]), "locked": float(b["locked"])}
                for b in balances
                if float(b.get("free", 0)) > 0 or float(b.get("locked", 0)) > 0
            }
            entry["positions"] = non_zero
            usdt = non_zero.get("USDT", {})
            entry["cash"] = usdt.get("free", 0.0)
        except Exception as exc:
            logger.warning("get_portfolio: %s get_account failed: %s", broker_name, exc)
            entry["error"] = str(exc)
        return entry

    # --- KuCoinConnector / MexcConnector (ccxt-based; unified fetch_balance) ---
    if hasattr(broker, "client") and hasattr(broker.client, "fetch_balance"):
        try:
            bal = broker.client.fetch_balance()
            totals = bal.get("total", {}) or {}
            frees = bal.get("free", {}) or {}
            useds = bal.get("used", {}) or {}
            non_zero = {
                asset: {
                    "free": float(frees.get(asset, 0.0) or 0.0),
                    "locked": float(useds.get(asset, 0.0) or 0.0),
                }
                for asset, total in totals.items()
                if (total or 0.0) > 0
            }
            entry["positions"] = non_zero
            entry["cash"] = float(frees.get("USDT", 0.0) or 0.0)
        except Exception as exc:
            logger.warning("get_portfolio: %s fetch_balance failed: %s", broker_name, exc)
            entry["error"] = str(exc)
        return entry

    # Fallback for unknown connector types
    logger.warning("get_portfolio: broker %s has no known portfolio method", broker_name)
    entry["error"] = "no_portfolio_method"
    return entry