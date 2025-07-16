# test_gui.py
import sys
import random
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QComboBox, QLabel
)
from PyQt5.QtChart import QChart, QChartView, QLineSeries
from PyQt5.QtCore import Qt, QDateTime


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

        # Create mock data
        series = QLineSeries()
        base_price = random.uniform(100, 200)

        for i in range(50):
            timestamp = QDateTime.currentDateTime().addSecs(i * 86400)
            price = base_price + random.uniform(-5, 5)
            series.append(timestamp.toMSecsSinceEpoch(), price)

        # Display data
        self.chart.addSeries(series)
        self.chart.createDefaultAxes()
        self.chart.axisX().setFormat("dd MMM")
        self.status_label.setText(
            f"Displaying {self.symbol_combo.currentText()} "
            f"({self.source_combo.currentText()})"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MockTradingGUI()
    window.show()
    sys.exit(app.exec_())