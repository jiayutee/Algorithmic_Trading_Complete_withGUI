# test_gui.py
"""
Headless tests for GUI combo box population and import correctness.

All tests instantiate Qt widgets without calling show(), so no display
is required.  A minimal QApplication is created once per session via the
`qapp` fixture below.
"""
import sys
import random
import pytest

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QComboBox, QLabel
)
from PyQt5.QtCore import Qt, QDateTime
from PyQt5.QtChart import QChart, QChartView, QLineSeries


# ---------------------------------------------------------------------------
# Session-scoped QApplication fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    """Create a single QApplication for the entire test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


# ---------------------------------------------------------------------------
# Legacy demo window (kept for reference, not tested by pytest)
# ---------------------------------------------------------------------------

class MockTradingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading GUI Test")
        self.resize(800, 600)

        # Create controls
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Historical", "Live Simulation"])

        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(["AAPL", "TSLA", "BTC-USD", "GOLD"])

        self.fetch_btn = QPushButton("Generate Test Data")
        self.fetch_btn.clicked.connect(self.load_mock_data)

        self.status_label = QLabel("Ready to test")

        # Chart setup
        self.chart = QChart()
        self.chart_view = QChartView(self.chart)

        # Layout
        central = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Data Source:"))
        layout.addWidget(self.source_combo)
        layout.addWidget(QLabel("Symbol:"))
        layout.addWidget(self.symbol_combo)
        layout.addWidget(self.fetch_btn)
        layout.addWidget(self.status_label)
        layout.addWidget(self.chart_view)

        central.setLayout(layout)
        self.setCentralWidget(central)

    def load_mock_data(self):
        """Generates fake price data for testing"""
        self.chart.removeAllSeries()

        series = QLineSeries()
        base_price = random.uniform(100, 200)

        for i in range(50):
            timestamp = QDateTime.currentDateTime().addSecs(i * 86400)
            price = base_price + random.uniform(-5, 5)
            series.append(timestamp.toMSecsSinceEpoch(), price)

        self.chart.addSeries(series)
        self.chart.createDefaultAxes()
        self.chart.axisX().setFormat("dd MMM")
        self.status_label.setText(
            f"Displaying {self.symbol_combo.currentText()} "
            f"({self.source_combo.currentText()})"
        )


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------

def test_import_main_window():
    """MainWindow should be importable without errors."""
    from ui.main_window import MainWindow  # noqa: F401


def test_import_app_module():
    """app module should be importable without errors."""
    import app  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers to build a MainWindow with all mocks (no display)
# ---------------------------------------------------------------------------

def _make_main_window(qapp):
    """Instantiate MainWindow headlessly using lightweight mock collaborators."""
    from unittest.mock import MagicMock
    from ui.main_window import MainWindow

    # Minimal strategy manager mock
    strategy_manager = MagicMock()
    strategy_manager.get_available_strategies.return_value = [
        "MACD/RSI", "EMA Crossover", "Stochastic", "LSTM Predictor"
    ]

    # Minimal broker manager mock
    broker_manager = MagicMock()

    # Minimal data loader mock
    data_loader = MagicMock()

    win = MainWindow(
        data_loader=data_loader,
        strategy_manager=strategy_manager,
        broker_manager=broker_manager,
        missing_deps=[],
    )
    return win


# ---------------------------------------------------------------------------
# Combo box population tests
# ---------------------------------------------------------------------------

class TestComboBoxes:
    """All six combo boxes must be non-empty and contain the expected items."""

    @pytest.fixture(autouse=True)
    def window(self, qapp):
        self.win = _make_main_window(qapp)
        yield
        self.win.destroy()

    # 1. Symbol combo
    def test_symbol_combo_not_empty(self):
        assert self.win.symbol_combo.count() > 0, "Symbol combo is empty"

    def test_symbol_combo_contains_crypto(self):
        items = [self.win.symbol_combo.itemText(i) for i in range(self.win.symbol_combo.count())]
        assert "BTCUSDT" in items, f"BTCUSDT not in symbol combo: {items}"

    def test_symbol_combo_contains_equity(self):
        items = [self.win.symbol_combo.itemText(i) for i in range(self.win.symbol_combo.count())]
        has_equity = any(s in items for s in ("AAPL", "TSLA", "SPY"))
        assert has_equity, f"No equity symbol in combo: {items}"

    # 2. Interval combo
    def test_interval_combo_not_empty(self):
        assert self.win.interval_combo.count() > 0, "Interval combo is empty"

    def test_interval_combo_has_standard_intervals(self):
        items = [self.win.interval_combo.itemText(i) for i in range(self.win.interval_combo.count())]
        for expected in ("1d", "1h", "1m"):
            assert expected in items, f"'{expected}' not in interval combo: {items}"

    # 3. Days input (QLineEdit, not a combo but still a selector)
    def test_days_input_has_default_value(self):
        assert self.win.days_input.text() != "", "Days input is empty"
        assert int(self.win.days_input.text()) > 0, "Days default should be positive"

    # 4. Data source combo
    def test_data_source_combo_not_empty(self):
        assert self.win.data_source_combo.count() > 0, "Data source combo is empty"

    def test_data_source_combo_has_historical(self):
        items = [self.win.data_source_combo.itemText(i) for i in range(self.win.data_source_combo.count())]
        assert "Historical" in items, f"'Historical' not in data source combo: {items}"

    def test_data_source_combo_has_live(self):
        items = [self.win.data_source_combo.itemText(i) for i in range(self.win.data_source_combo.count())]
        assert "Live" in items, f"'Live' not in data source combo: {items}"

    # 5. Strategy combo
    def test_strategy_combo_not_empty(self):
        assert self.win.strategy_combo.count() > 0, "Strategy combo is empty"

    def test_strategy_combo_has_none_option(self):
        items = [self.win.strategy_combo.itemText(i) for i in range(self.win.strategy_combo.count())]
        assert "None" in items, f"'None' not in strategy combo: {items}"

    def test_strategy_combo_has_macd_rsi(self):
        items = [self.win.strategy_combo.itemText(i) for i in range(self.win.strategy_combo.count())]
        assert "MACD/RSI" in items, f"'MACD/RSI' not in strategy combo: {items}"

    def test_strategy_combo_has_ema_crossover(self):
        items = [self.win.strategy_combo.itemText(i) for i in range(self.win.strategy_combo.count())]
        assert "EMA Crossover" in items, f"'EMA Crossover' not in strategy combo: {items}"

    def test_strategy_combo_no_phantom_strategies(self):
        """'DDPG Strategy' was in the UI but never in the strategy manager — ensure it's gone."""
        items = [self.win.strategy_combo.itemText(i) for i in range(self.win.strategy_combo.count())]
        assert "DDPG Strategy" not in items, f"Stale 'DDPG Strategy' entry still present: {items}"

    # 6. Broker combo
    def test_broker_combo_not_empty(self):
        assert self.win.broker_combo.count() > 0, "Broker combo is empty"

    def test_broker_combo_has_simulator(self):
        items = [self.win.broker_combo.itemText(i) for i in range(self.win.broker_combo.count())]
        assert "Simulator" in items, f"'Simulator' not in broker combo: {items}"

    def test_broker_combo_has_alpaca(self):
        items = [self.win.broker_combo.itemText(i) for i in range(self.win.broker_combo.count())]
        assert "Alpaca" in items, f"'Alpaca' not in broker combo: {items}"

    def test_broker_combo_has_binance(self):
        items = [self.win.broker_combo.itemText(i) for i in range(self.win.broker_combo.count())]
        assert "Binance" in items, f"'Binance' not in broker combo: {items}"

    # Order type combo (bonus — used by order entry panel)
    def test_order_type_combo_not_empty(self):
        assert self.win.order_type_combo.count() > 0, "Order type combo is empty"

    def test_order_type_combo_has_market(self):
        items = [self.win.order_type_combo.itemText(i) for i in range(self.win.order_type_combo.count())]
        assert "Market" in items, f"'Market' not in order type combo: {items}"

    def test_order_type_combo_has_limit(self):
        items = [self.win.order_type_combo.itemText(i) for i in range(self.win.order_type_combo.count())]
        assert "Limit" in items, f"'Limit' not in order type combo: {items}"


# ---------------------------------------------------------------------------
# Strategy manager population tests
# ---------------------------------------------------------------------------

class TestStrategyManagerPopulation:
    """Strategy manager must provide the expected non-ML strategies."""

    def test_get_available_strategies_returns_list(self):
        from core.strategy_manager import StrategyManager
        sm = StrategyManager()
        strategies = sm.get_available_strategies()
        assert isinstance(strategies, list)
        assert len(strategies) > 0

    def test_macd_rsi_available(self):
        from core.strategy_manager import StrategyManager
        sm = StrategyManager()
        assert "MACD/RSI" in sm.get_available_strategies()

    def test_ema_crossover_available(self):
        from core.strategy_manager import StrategyManager
        sm = StrategyManager()
        assert "EMA Crossover" in sm.get_available_strategies()

    def test_stochastic_available(self):
        from core.strategy_manager import StrategyManager
        sm = StrategyManager()
        assert "Stochastic" in sm.get_available_strategies()


# ---------------------------------------------------------------------------
# Legacy entry point (run the demo window manually)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MockTradingGUI()
    window.show()
    sys.exit(app.exec_())
