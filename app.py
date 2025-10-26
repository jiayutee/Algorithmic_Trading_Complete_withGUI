# app.py
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from core.data_loader import DataLoader
from core.strategy_manager import StrategyManager
from core.broker_manager import BrokerManager
from core.ai_analyzer import DeepSeekAnalyzer
from ui.main_window import MainWindow
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, KUCOIN_API_KEY, KUCOIN_SECRET_KEY, BINANCE_API_KEY, BINANCE_SECRET_KEY, BINANCE_TESTNET_API_KEY, BINANCE_TESTNET_SECRET_KEY, DEEPSEEK_API_KEY


class TradingApp:
    def __init__(self):
        # Get DeepSeek API key first
        self.deepseek_api_key = DEEPSEEK_API_KEY
        self.ai_analyzer = None
        
        # Initialize AI analyzer if API key provided
        if self.deepseek_api_key:
            try:
                self.ai_analyzer = DeepSeekAnalyzer(self.deepseek_api_key)
                print("✅ DeepSeek AI Analyzer initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize DeepSeek AI: {e}")
                self.show_error_message("DeepSeek AI Error", f"Failed to initialize AI analyzer: {str(e)}")
        else:
            print("⚠️ DeepSeek API key not provided - AI features disabled")

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
            binance_testnet_secret=BINANCE_TESTNET_SECRET_KEY
        )

        # Create main window
        self.window = MainWindow(
            data_loader=self.data_loader,
            strategy_manager=self.strategy_manager,
            broker_manager=self.broker_manager,
            ai_analyzer = self.ai_analyzer
        )


def main():
    # Configure high DPI scaling
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