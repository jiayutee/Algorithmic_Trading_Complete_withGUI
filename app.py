# app.py
import sys
import importlib

try:
    from dotenv import load_dotenv
    # Load environment variables from .env file before other imports
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on environment variables set externally

try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView  # MUST come before QApplication — Qt ordering requirement
except ImportError:
    QWebEngineView = None  # PyQtWebEngine not installed; chart view will be disabled
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt
from core.data_loader import DataLoader
from core.strategy_manager import StrategyManager
from core.broker_manager import BrokerManager
from ui.main_window import MainWindow
from config.settings import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY,
    KUCOIN_API_KEY, KUCOIN_SECRET_KEY,
    BINANCE_API_KEY, BINANCE_SECRET_KEY,
    BINANCE_TESTNET_API_KEY, BINANCE_TESTNET_SECRET_KEY,
)

# Optional heavy dependencies — imported lazily so missing packages don't
# prevent the app from starting. Each entry is (pip_name, import_name).
_OPTIONAL_DEPS = [
    ("tensorflow", "tensorflow"),
    ("torch", "torch"),
    ("finrl", "finrl"),
    ("ib_insync", "ib_insync"),
    ("stable_baselines3", "stable_baselines3"),
]


def _check_optional_deps() -> list[str]:
    """Return a list of optional packages that are not installed."""
    missing = []
    for pip_name, import_name in _OPTIONAL_DEPS:
        if importlib.util.find_spec(import_name) is None:
            missing.append(pip_name)
    return missing


class TradingApp:
    def __init__(self):
        missing = _check_optional_deps()
        if missing:
            print(
                f"[app] Optional packages not installed (some strategies will be unavailable): "
                f"{', '.join(missing)}"
            )

        # Initialize core components
        self.data_loader = DataLoader(
            live_api_key=ALPACA_API_KEY,
            live_secret_key=ALPACA_SECRET_KEY,
            kucoin_key=KUCOIN_API_KEY,
            kucoin_secret=KUCOIN_SECRET_KEY,
            binance_key=BINANCE_API_KEY,
            binance_secret=BINANCE_SECRET_KEY,
        )

        self.strategy_manager = StrategyManager()
        self.broker_manager = BrokerManager(
            alpaca_key=ALPACA_API_KEY,
            alpaca_secret=ALPACA_SECRET_KEY,
            binance_key=BINANCE_API_KEY,
            binance_secret=BINANCE_SECRET_KEY,
            binance_testnet_key=BINANCE_TESTNET_API_KEY,
            binance_testnet_secret=BINANCE_TESTNET_SECRET_KEY,
        )

        # Create main window
        self.window = MainWindow(
            data_loader=self.data_loader,
            strategy_manager=self.strategy_manager,
            broker_manager=self.broker_manager,
            missing_deps=missing,
        )


def main():
    # Must be set before QApplication is created
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Initialize and show main window
    trading_app = TradingApp()
    trading_app.window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()