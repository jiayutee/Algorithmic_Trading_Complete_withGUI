import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTextEdit
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class StatisticsWindow(QMainWindow):
    def __init__(self, stats_data=None):
        super().__init__()
        self.setWindowTitle("Trade Statistics")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.layout.addWidget(self.stats_text)

        self.plot_canvas = PlotCanvas(self, width=8, height=6)
        self.layout.addWidget(self.plot_canvas)

        if stats_data:
            self.update_data(stats_data)

    def update_data(self, stats_data):
        # Format and display summary statistics
        summary = stats_data.get("summary", {})
        summary_text = ""
        for key, value in summary.items():
            summary_text += f"{key}: {value}\n"
        self.stats_text.setText(summary_text)

        # Update plots
        self.plot_canvas.plot(stats_data)

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

    def plot(self, stats_data):
        self.figure.clear()

        # Example data for plotting
        cumulative_pnl = stats_data.get("cumulative_pnl", [])
        total_asset_value = stats_data.get("total_asset_value", [])
        profit_per_trade = stats_data.get("profit_per_trade", [])

        # Subplot 1: Cumulative PnL
        ax1 = self.figure.add_subplot(311)
        ax1.plot(cumulative_pnl, label="Cumulative PnL")
        ax1.set_title("Performance Metrics")
        ax1.set_ylabel("PnL")
        ax1.legend()

        # Subplot 2: Total Asset Value
        ax2 = self.figure.add_subplot(312)
        ax2.plot(total_asset_value, label="Total Asset Value")
        ax2.set_ylabel("Asset Value")
        ax2.legend()

        # Subplot 3: Profit per Trade
        ax3 = self.figure.add_subplot(313)
        ax3.bar(range(len(profit_per_trade)), profit_per_trade, label="Profit per Trade")
        ax3.set_ylabel("Profit")
        ax3.set_xlabel("Trade Number")
        ax3.legend()

        self.figure.tight_layout()
        self.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Example usage with dummy data
    dummy_stats = {
        "summary": {
            "Sharpe Ratio": 1.5,
            "Alpha": 0.05,
            "Beta": 1.2,
            "Number of Closed Trades": 50,
            "Win Rate": "60%",
        },
        "cumulative_pnl": np.random.rand(100).cumsum(),
        "total_asset_value": np.random.rand(100).cumsum() + 10000,
        "profit_per_trade": np.random.randn(50) * 100
    }
    main_win = StatisticsWindow(dummy_stats)
    main_win.show()
    sys.exit(app.exec_())