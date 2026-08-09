# ui/main_window.py
import os
import tempfile
from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
                             QComboBox, QPushButton, QLabel, QGroupBox, QLineEdit,
                             QTextEdit, QTabWidget, QSplitter, QTableWidget,
                             QTableWidgetItem, QHeaderView, QApplication, QFormLayout,
                             QFrame, QSizePolicy)
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QColor
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QUrl
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    _WEBENGINE_AVAILABLE = True
except ImportError:
    QWebEngineView = None
    _WEBENGINE_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from plotly.utils import PlotlyJSONEncoder
    _PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    make_subplots = None
    PlotlyJSONEncoder = None
    _PLOTLY_AVAILABLE = False

import json
import pandas as pd
import numpy as np
from queue import Empty
from datetime import datetime, timedelta

try:
    from ui.statistics_window import StatisticsWindow
    _STATS_WINDOW_AVAILABLE = True
except ImportError:
    StatisticsWindow = None
    _STATS_WINDOW_AVAILABLE = False

from typing import Dict, List, Optional
from core.news_scraper import scrape_and_analyze_finviz_news
from core.logger import logger


class NewsWorker(QThread):
    """Fetches news + sentiment for a symbol off the main thread."""
    results_ready = pyqtSignal(object)   # emits a pd.DataFrame
    error = pyqtSignal(str)

    def __init__(self, symbol: str):
        super().__init__()
        self.symbol = symbol

    def run(self):
        try:
            from core.news_pipeline import NewsPipeline
            from core.sentiment import SentimentAnalyzer
            pipeline = NewsPipeline.from_env()
            pipeline.sentiment_analyzer = SentimentAnalyzer(force_rule_based=True)
            df = pipeline.fetch_news_dataframe(self.symbol, limit=25)
            self.results_ready.emit(df)
        except Exception as e:
            self.error.emit(str(e))


class DataLoadWorker(QThread):
    """Fetches price candles (+ the news/sentiment merge embedded in
    data_loader.load_data) off the main thread. Without this, the initial
    load froze the whole window for 45-90s while rate-limited news sources
    retried, since network I/O ran directly on the Qt event-loop thread."""
    results_ready = pyqtSignal(object)   # emits a pd.DataFrame
    error = pyqtSignal(str)

    def __init__(self, data_loader, symbol: str, source: str, live: bool, days: int, interval: str):
        super().__init__()
        self.data_loader = data_loader
        self.symbol = symbol
        self.source = source
        self.live = live
        self.days = days
        self.interval = interval

    def run(self):
        try:
            df = self.data_loader.load_data(
                symbol=self.symbol,
                source=self.source,
                live=self.live,
                days=self.days,
                interval=self.interval,
            )
            self.results_ready.emit(df)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self, data_loader, strategy_manager, broker_manager,
                 missing_deps: Optional[List[str]] = None):
        super().__init__()
        self.data_loader = data_loader
        self.strategy_manager = strategy_manager
        self.broker_manager = broker_manager
        self._missing_deps = missing_deps or []
        self._supervisor = None
        self.setWindowTitle("Algorithmic Trading Terminal")

        # Screen-aware sizing
        screen = QApplication.primaryScreen().availableGeometry()
        w = int(screen.width() * 0.92)
        h = int(screen.height() * 0.92)
        self.resize(w, h)
        self.move(screen.center() - self.rect().center())

        # Global dark stylesheet
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #0d1117; color: #e6edf3; }
            QGroupBox { border: 1px solid #30363d; border-radius: 6px; margin-top: 8px;
                        font-size: 11px; font-weight: 600; color: #8b949e; padding: 6px 4px 4px 4px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
            QComboBox { background: #161b22; border: 1px solid #30363d; border-radius: 4px;
                        padding: 3px 6px; color: #e6edf3; font-size: 12px; min-height: 22px; }
            QComboBox::drop-down { border: none; width: 18px; }
            QComboBox QAbstractItemView { background: #161b22; border: 1px solid #30363d; color: #e6edf3; }
            QLineEdit { background: #161b22; border: 1px solid #30363d; border-radius: 4px;
                        padding: 3px 6px; color: #e6edf3; font-size: 12px; min-height: 22px; }
            QLineEdit:focus { border-color: #58a6ff; }
            QPushButton { background: #21262d; border: 1px solid #30363d; border-radius: 4px;
                          padding: 4px 10px; color: #e6edf3; font-size: 12px; min-height: 24px; }
            QPushButton:hover { background: #30363d; border-color: #58a6ff; }
            QPushButton:pressed { background: #161b22; }
            QPushButton:disabled { color: #484f58; border-color: #21262d; }
            QTabWidget::pane { border: 1px solid #30363d; border-radius: 4px; }
            QTabBar::tab { background: #161b22; border: 1px solid #30363d; padding: 5px 12px;
                           color: #8b949e; font-size: 11px; border-bottom: none; border-radius: 4px 4px 0 0; }
            QTabBar::tab:selected { background: #0d1117; color: #e6edf3; border-bottom: 2px solid #58a6ff; }
            QTableWidget { background: #0d1117; gridline-color: #21262d; color: #e6edf3;
                           border: none; font-size: 11px; }
            QTableWidget::item:selected { background: #1f6feb33; }
            QHeaderView::section { background: #161b22; color: #8b949e; border: none;
                                   border-bottom: 1px solid #30363d; padding: 4px 6px; font-size: 10px;
                                   font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
            QTextEdit { background: #161b22; border: 1px solid #30363d; border-radius: 4px;
                        color: #e6edf3; font-family: 'SF Mono', 'Consolas', monospace; font-size: 11px; }
            QScrollBar:vertical { background: #0d1117; width: 8px; }
            QScrollBar::handle:vertical { background: #30363d; border-radius: 4px; min-height: 20px; }
            QSplitter::handle { background: #30363d; width: 1px; height: 1px; }
            QStatusBar { background: #161b22; border-top: 1px solid #30363d; color: #8b949e; font-size: 11px; }
            QLabel { color: #e6edf3; font-size: 12px; }
            QPushButton#liveBtn { background: #1a4731; border-color: #2ea043; color: #3fb950; font-weight: 600; }
            QPushButton#liveBtn:hover { background: #1f5e3a; }
        """)

        # Broker state
        self.current_broker = None
        self.current_broker_name = "Simulator"

        # Initialize state (timers, streaming flags, etc.)
        self._init_state()

        # Build UI
        central_widget = QWidget()
        central_widget.setContentsMargins(0, 0, 0, 0)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Top bar
        root_layout.addWidget(self._build_topbar())

        # Content splitter
        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.addWidget(self._build_left_panel())

        if _WEBENGINE_AVAILABLE:
            self.plotly_view = QWebEngineView()
            empty_html = """
                <html>
                <head>
                    <meta charset="utf-8"/>
                    <style>
                    body {
                        background-color: #0d1117;
                        color: #e6edf3;
                        margin: 0;
                        padding: 0;
                    }
                    </style>
                </head>
                <body></body>
                </html>
                """
            self.plotly_view.setHtml(empty_html)
        else:
            from PyQt5.QtWidgets import QLabel
            self.plotly_view = QLabel("Chart view unavailable (PyQtWebEngine not installed)")
            self.plotly_view.setAlignment(Qt.AlignCenter)
            self.plotly_view.setStyleSheet("color: #8b949e; font-size: 13px;")
        content_splitter.addWidget(self.plotly_view)
        content_splitter.addWidget(self._build_right_panel())

        # Set initial sizes: left 220, chart stretch, right 200
        content_splitter.setSizes([220, w - 220 - 200, 200])
        content_splitter.setStretchFactor(0, 0)
        content_splitter.setStretchFactor(1, 1)
        content_splitter.setStretchFactor(2, 0)

        root_layout.addWidget(content_splitter, stretch=1)

        # Bottom panel
        root_layout.addWidget(self._build_bottom_panel())

        self.setCentralWidget(central_widget)

        # Status bar
        self.statusBar().showMessage("Ready")

        # Connect signals
        self.load_btn.clicked.connect(self.load_data)
        self.backtest_btn.clicked.connect(self.run_backtest)
        self.trade_btn.clicked.connect(self.start_trading)
        self.reset_btn.clicked.connect(self.reset_chart_zoom)
        self.simulate_btn.clicked.connect(self.start_simulation)
        self.play_btn.clicked.connect(self.play_simulation)
        self.pause_btn.clicked.connect(self.pause_simulation)
        self.symbol_combo.currentTextChanged.connect(self.load_data)
        self.symbol_combo.currentTextChanged.connect(lambda _: self._fetch_news())
        self.order_type_combo.currentTextChanged.connect(self.on_order_type_changed)
        self.refresh_account_btn.clicked.connect(self.refresh_account_info)
        self.buy_btn.clicked.connect(lambda: self.place_order("buy"))
        self.sell_btn.clicked.connect(lambda: self.place_order("sell"))

        # Initial order type UI state
        self.on_order_type_changed("Market")

        # Broker monitoring timer
        self.broker_timer.start(5000)

        # Defer initial data load and news fetch
        QTimer.singleShot(200, self.load_data)
        QTimer.singleShot(3000, self._fetch_news)

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------

    def _init_state(self):
        """Initialise all runtime state variables and timers."""
        # Realtime streaming
        self.realtime_timer = QTimer()
        self.realtime_timer.timeout.connect(self.process_realtime_updates)
        self.is_streaming = False
        self.realtime_df = pd.DataFrame()
        self.current_interval = '1m'
        self.max_candles = 130

        # Simulation
        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self.update_simulation_chart)
        self.simulation_data = None
        self.simulation_index = 0
        self.buy_signal_plotted = False
        self.sell_signal_plotted = False

        # Broker timer (started in __init__ after widgets exist)
        self.broker_timer = QTimer()
        self.broker_timer.timeout.connect(self.refresh_account_info)

        # News
        self.news_timer = QTimer()
        self.news_timer.timeout.connect(self.update_live_news)
        self.last_seen_headline = ""
        self.latest_sentiment = {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}

    # ------------------------------------------------------------------
    # Layout builders
    # ------------------------------------------------------------------

    def _build_topbar(self):
        bar = QWidget()
        bar.setFixedHeight(44)
        bar.setStyleSheet("background: #161b22; border-bottom: 1px solid #30363d;")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(6)

        # Brand label
        brand = QLabel("◈ AlgoTrader")
        brand.setStyleSheet("color: #58a6ff; font-size: 13px; font-weight: bold;")
        layout.addWidget(brand)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #30363d;")
        layout.addWidget(sep)

        # Symbol
        layout.addWidget(self._muted_label("Symbol"))
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "AAPL", "TSLA", "GOLD", "SPY", "QQQ"])
        self.symbol_combo.setFixedWidth(110)
        layout.addWidget(self.symbol_combo)

        # Interval
        layout.addWidget(self._muted_label("Interval"))
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(['1d', '1h', '15m', '5m', '1m'])
        self.interval_combo.setFixedWidth(70)
        layout.addWidget(self.interval_combo)

        # Days
        layout.addWidget(self._muted_label("Days"))
        self.days_input = QLineEdit("365")
        self.days_input.setValidator(QIntValidator(1, 10000))
        self.days_input.setFixedWidth(50)
        layout.addWidget(self.days_input)

        # Source
        layout.addWidget(self._muted_label("Source"))
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems(["Historical", "Live", "Realtime Stream", "FinRL-Yahoo"])
        self.data_source_combo.setFixedWidth(120)
        layout.addWidget(self.data_source_combo)

        # Strategy — populated dynamically from StrategyManager so the list
        # always reflects what is actually available (respects missing deps).
        layout.addWidget(self._muted_label("Strategy"))
        self.strategy_combo = QComboBox()
        strategy_items = ["None"]
        if hasattr(self, 'strategy_manager') and self.strategy_manager is not None:
            try:
                strategy_items += self.strategy_manager.get_available_strategies()
            except Exception:
                # Fallback to known strategies if manager is unavailable
                strategy_items += [
                    "MACD/RSI", "EMA Crossover", "Stochastic",
                    "LSTM Predictor", "TD3 Strategy"
                ]
        else:
            strategy_items += [
                "MACD/RSI", "EMA Crossover", "Stochastic",
                "LSTM Predictor", "TD3 Strategy"
            ]
        self.strategy_combo.addItems(strategy_items)
        self.strategy_combo.setFixedWidth(130)
        layout.addWidget(self.strategy_combo)

        # Broker
        layout.addWidget(self._muted_label("Broker"))
        self.broker_combo = QComboBox()
        self.broker_combo.addItems(["Simulator", "Alpaca", "Interactive Brokers", "Binance"])
        self.broker_combo.setFixedWidth(130)
        layout.addWidget(self.broker_combo)

        layout.addStretch()

        # Action buttons
        self.load_btn = QPushButton("Load")
        self.backtest_btn = QPushButton("Backtest")
        self.simulate_btn = QPushButton("Simulate")
        self.play_btn = QPushButton("▶")
        self.pause_btn = QPushButton("⏸")
        self.trade_btn = QPushButton("Go Live")
        self.trade_btn.setObjectName("liveBtn")
        self.reset_btn = QPushButton("↺")
        self.reset_btn.setToolTip("Reset zoom")
        self.reset_btn.setFixedWidth(28)

        self.play_btn.hide()
        self.pause_btn.hide()

        for btn in [self.load_btn, self.backtest_btn, self.simulate_btn,
                    self.play_btn, self.pause_btn, self.trade_btn, self.reset_btn]:
            layout.addWidget(btn)

        return bar

    def _muted_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #8b949e; font-size: 11px;")
        return lbl

    def _build_left_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(180)  # allow splitter to shrink below the initial 220px default
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Parameters group
        params_group = QGroupBox("Parameters")
        form = QFormLayout()
        form.setSpacing(4)

        self.cash_input = QLineEdit("100000")
        self.cash_input.setValidator(QIntValidator(1000, 10000000))
        form.addRow("Cash ($):", self.cash_input)

        self.market_fee_input = QLineEdit("0.1")
        self.market_fee_input.setValidator(QDoubleValidator(0, 10, 4))
        self.market_fee_input.setFixedWidth(60)
        form.addRow("Mkt Fee %:", self.market_fee_input)

        self.limit_fee_input = QLineEdit("0.05")
        self.limit_fee_input.setValidator(QDoubleValidator(0, 10, 4))
        self.limit_fee_input.setFixedWidth(60)
        form.addRow("Lim Fee %:", self.limit_fee_input)

        params_group.setLayout(form)
        layout.addWidget(params_group)

        # Order Entry group
        order_group = QGroupBox("Order Entry")
        order_layout = QVBoxLayout()
        order_layout.setSpacing(4)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type"))
        self.order_type_combo = QComboBox()
        self.order_type_combo.addItems(["Market", "Limit", "Stop"])
        type_row.addWidget(self.order_type_combo)
        order_layout.addLayout(type_row)

        qty_row = QHBoxLayout()
        qty_row.addWidget(QLabel("Qty"))
        self.order_qty_input = QLineEdit("1")
        self.order_qty_input.setFixedWidth(70)
        qty_row.addWidget(self.order_qty_input)
        order_layout.addLayout(qty_row)

        self.limit_price_input = QLineEdit()
        self.limit_price_input.setPlaceholderText("Limit / Stop price")
        self.limit_price_input.hide()
        order_layout.addWidget(self.limit_price_input)

        trade_row = QHBoxLayout()
        self.buy_btn = QPushButton("▲ BUY")
        self.buy_btn.setStyleSheet(
            "QPushButton { background: #1a4731; border: 1px solid #2ea043; color: #3fb950; font-weight: bold; }"
            "QPushButton:hover { background: #1f5e3a; }"
        )
        self.sell_btn = QPushButton("▼ SELL")
        self.sell_btn.setStyleSheet(
            "QPushButton { background: #3d1a1a; border: 1px solid #da3633; color: #f85149; font-weight: bold; }"
            "QPushButton:hover { background: #4d2020; }"
        )
        trade_row.addWidget(self.buy_btn)
        trade_row.addWidget(self.sell_btn)
        order_layout.addLayout(trade_row)

        order_group.setLayout(order_layout)
        layout.addWidget(order_group)

        layout.addStretch()
        return panel

    def _build_right_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(160)  # allow splitter to shrink below the initial 200px default
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Account group
        account_group = QGroupBox("Account")
        account_layout = QVBoxLayout()
        account_layout.setSpacing(4)

        self.account_label = QLabel("Simulator\n$100,000.00")
        self.account_label.setWordWrap(True)
        self.account_label.setStyleSheet("color: #3fb950;")
        account_layout.addWidget(self.account_label)

        self.refresh_account_btn = QPushButton("↻ Refresh")
        account_layout.addWidget(self.refresh_account_btn)

        account_group.setLayout(account_layout)
        layout.addWidget(account_group)

        # P&L group
        pnl_group = QGroupBox("P & L")
        pnl_layout = QVBoxLayout()

        self.pnl_label = QLabel("$0.00")
        self.pnl_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #3fb950;")
        self.pnl_label.setAlignment(Qt.AlignCenter)
        pnl_layout.addWidget(self.pnl_label)

        pnl_group.setLayout(pnl_layout)
        layout.addWidget(pnl_group)

        # Backtest Results group
        results_group = QGroupBox("Backtest Results")
        results_layout = QFormLayout()
        results_layout.setSpacing(4)

        self.bt_sharpe_label = QLabel("—")
        self.bt_sharpe_label.setStyleSheet("color: #58a6ff;")
        results_layout.addRow("Sharpe:", self.bt_sharpe_label)

        self.bt_winrate_label = QLabel("—")
        self.bt_winrate_label.setStyleSheet("color: #3fb950;")
        results_layout.addRow("Win Rate:", self.bt_winrate_label)

        self.bt_maxdd_label = QLabel("—")
        self.bt_maxdd_label.setStyleSheet("color: #f85149;")
        results_layout.addRow("Max DD:", self.bt_maxdd_label)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        # Positions group
        positions_group = QGroupBox("Positions")
        positions_layout = QVBoxLayout()

        self.positions_text = QTextEdit()
        self.positions_text.setReadOnly(True)
        positions_layout.addWidget(self.positions_text)

        positions_group.setLayout(positions_layout)
        layout.addWidget(positions_group, stretch=1)

        return panel

    def _build_bottom_panel(self):
        self.bottom_tabs = QTabWidget()
        self.bottom_tabs.setMinimumHeight(220)
        self.bottom_tabs.setMaximumHeight(320)

        self._setup_orders_tab()
        self._setup_news_tab()
        self._setup_agent_monitor_tab()

        if self._missing_deps:
            self._setup_missing_deps_tab()

        return self.bottom_tabs

    # ------------------------------------------------------------------
    # Orders tab
    # ------------------------------------------------------------------

    def _setup_orders_tab(self):
        """Build the Orders tab showing all submitted orders for the session."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Toolbar row
        toolbar = QHBoxLayout()
        self._orders_status_label = QLabel("Orders: none yet")
        self._orders_status_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        toolbar.addWidget(self._orders_status_label)
        toolbar.addStretch()
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(60)
        clear_btn.clicked.connect(self._clear_orders_tab)
        toolbar.addWidget(clear_btn)
        layout.addLayout(toolbar)

        # Table: Time | Symbol | Side | Type | Qty | Fill Price | Status
        self._orders_table = QTableWidget(0, 7)
        self._orders_table.setHorizontalHeaderLabels(
            ["Time", "Symbol", "Side", "Type", "Qty", "Fill Price", "Status"]
        )
        self._orders_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._orders_table.horizontalHeader().setStretchLastSection(True)
        self._orders_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._orders_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._orders_table.setAlternatingRowColors(True)
        self._orders_table.verticalHeader().setVisible(False)
        layout.addWidget(self._orders_table)

        self.bottom_tabs.addTab(tab, "Orders")

    def _refresh_orders_tab(self):
        """Repopulate the Orders table from current_broker.order_history."""
        if not self.current_broker or not hasattr(self.current_broker, 'order_history'):
            return

        history = self.current_broker.order_history
        self._orders_table.setRowCount(len(history))

        _SIDE_COLORS = {
            "buy":  ("#1a4731", "#3fb950"),
            "sell": ("#3d1a1a", "#f85149"),
        }
        _STATUS_COLORS = {
            "filled":   "#3fb950",
            "pending":  "#f0883e",
            "rejected": "#f85149",
            "canceled": "#8b949e",
        }

        for row, order in enumerate(history):
            ts = datetime.fromtimestamp(order.created_at).strftime("%H:%M:%S")
            side_str = order.side.value if hasattr(order.side, 'value') else str(order.side)
            type_str = order.order_type.value if hasattr(order.order_type, 'value') else str(order.order_type)
            status_str = order.status.value if hasattr(order.status, 'value') else str(order.status)
            fill_price = f"${order.filled_avg_price:,.4f}" if order.filled_avg_price else "—"

            items = [
                QTableWidgetItem(ts),
                QTableWidgetItem(order.symbol),
                QTableWidgetItem(side_str.upper()),
                QTableWidgetItem(type_str.capitalize()),
                QTableWidgetItem(f"{order.filled_qty:.4f}"),
                QTableWidgetItem(fill_price),
                QTableWidgetItem(status_str.capitalize()),
            ]

            # Colour the Side cell
            bg_hex, fg_hex = _SIDE_COLORS.get(side_str.lower(), ("#1c2128", "#e6edf3"))
            items[2].setBackground(QColor(bg_hex))
            items[2].setForeground(QColor(fg_hex))

            # Colour the Status cell
            status_color = _STATUS_COLORS.get(status_str.lower(), "#e6edf3")
            items[6].setForeground(QColor(status_color))

            for col, item in enumerate(items):
                item.setTextAlignment(Qt.AlignCenter)
                self._orders_table.setItem(row, col, item)

        count = len(history)
        filled = sum(
            1 for o in history
            if (o.status.value if hasattr(o.status, 'value') else str(o.status)) == "filled"
        )
        self._orders_status_label.setText(f"Orders: {count} total, {filled} filled")

        # Switch to Orders tab so the user sees the result immediately
        self.bottom_tabs.setCurrentIndex(0)

    def _clear_orders_tab(self):
        """Clear the orders display (does NOT cancel broker orders)."""
        self._orders_table.setRowCount(0)
        self._orders_status_label.setText("Orders: cleared")

    # ------------------------------------------------------------------
    # News tab
    # ------------------------------------------------------------------

    def _setup_news_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Toolbar row
        toolbar = QHBoxLayout()
        self._news_status_label = QLabel("News: —")
        self._news_status_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        toolbar.addWidget(self._news_status_label)
        toolbar.addStretch()
        self._news_refresh_btn = QPushButton("↻ Refresh")
        self._news_refresh_btn.setFixedWidth(80)
        self._news_refresh_btn.clicked.connect(self._fetch_news)
        toolbar.addWidget(self._news_refresh_btn)
        layout.addLayout(toolbar)

        # Table: Time | Headline | Source | Sentiment | Score
        self._news_table = QTableWidget(0, 5)
        self._news_table.setHorizontalHeaderLabels(["Time", "Headline", "Source", "Sentiment", "Score"])
        self._news_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._news_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._news_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._news_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._news_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self._news_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._news_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._news_table.setAlternatingRowColors(True)
        self._news_table.verticalHeader().setVisible(False)
        self._news_table.setWordWrap(False)
        layout.addWidget(self._news_table)

        self.bottom_tabs.addTab(tab, "News")

        # Worker state
        self._news_worker: Optional[NewsWorker] = None

        # Auto-refresh every 5 minutes
        self._news_auto_timer = QTimer()
        self._news_auto_timer.timeout.connect(self._fetch_news)
        self._news_auto_timer.start(5 * 60 * 1000)

    def _fetch_news(self):
        symbol = self.symbol_combo.currentText()
        if self._news_worker and self._news_worker.isRunning():
            return  # already fetching
        self._news_status_label.setText(f"Fetching news for {symbol}…")
        self._news_refresh_btn.setEnabled(False)
        self._news_worker = NewsWorker(symbol)
        self._news_worker.results_ready.connect(self._on_news_ready)
        self._news_worker.error.connect(self._on_news_error)
        self._news_worker.start()

    def _on_news_ready(self, df):
        self._news_refresh_btn.setEnabled(True)
        now = datetime.now().strftime("%H:%M:%S")

        if df.empty:
            self._news_status_label.setText(f"No news found — {now}")
            self._news_table.setRowCount(0)
            return

        _SENTIMENT_COLORS = {
            "positive": ("#1a4731", "#3fb950"),   # bg, fg
            "negative": ("#3d1a1a", "#f85149"),
            "neutral":  ("#1c2128", "#8b949e"),
        }

        self._news_table.setRowCount(len(df))
        for row, (_, article) in enumerate(df.iterrows()):
            # Time
            dt = article.get("datetime", "")
            time_str = pd.Timestamp(dt).strftime("%m-%d %H:%M") if dt else "—"
            self._news_table.setItem(row, 0, QTableWidgetItem(time_str))

            # Headline
            headline = str(article.get("headline", ""))
            self._news_table.setItem(row, 1, QTableWidgetItem(headline))

            # Source
            source = str(article.get("source", ""))
            self._news_table.setItem(row, 2, QTableWidgetItem(source))

            # Sentiment label + colouring
            label = str(article.get("sentiment_label", "neutral")).lower()
            bg_hex, fg_hex = _SENTIMENT_COLORS.get(label, _SENTIMENT_COLORS["neutral"])
            sent_item = QTableWidgetItem(label.capitalize())
            sent_item.setBackground(QColor(bg_hex))
            sent_item.setForeground(QColor(fg_hex))
            self._news_table.setItem(row, 3, sent_item)

            # Confidence score
            score = article.get("sentiment_confidence", 0.0)
            score_str = f"{float(score):.0%}" if score else "—"
            score_item = QTableWidgetItem(score_str)
            score_item.setTextAlignment(Qt.AlignCenter)
            self._news_table.setItem(row, 4, score_item)

        self._news_status_label.setText(
            f"{len(df)} articles for {self.symbol_combo.currentText()} — updated {now}"
        )

    def _on_news_error(self, err: str):
        self._news_refresh_btn.setEnabled(True)
        self._news_status_label.setText(f"News error: {err[:80]}")
        logger.warning("News tab fetch failed: %s", err)

    # ------------------------------------------------------------------
    # Agent Monitor tab
    # ------------------------------------------------------------------

    def _setup_agent_monitor_tab(self):
        """Build the Agent Monitor tab and wire up a refresh timer."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Top row: status label + start/stop buttons
        controls = QHBoxLayout()
        self._agent_status_label = QLabel("Agents: stopped")
        self._agent_status_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        self._agent_start_btn = QPushButton("Start Agents")
        self._agent_stop_btn = QPushButton("Stop Agents")
        self._agent_stop_btn.setEnabled(False)
        self._agent_start_btn.clicked.connect(self._start_supervisor)
        self._agent_stop_btn.clicked.connect(self._stop_supervisor)
        controls.addWidget(self._agent_status_label)
        controls.addStretch()
        controls.addWidget(self._agent_start_btn)
        controls.addWidget(self._agent_stop_btn)
        layout.addLayout(controls)

        # Table: one row per agent
        self._agent_table = QTableWidget(0, 3)
        self._agent_table.setHorizontalHeaderLabels(["Agent", "Status", "Summary"])
        self._agent_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self._agent_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._agent_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._agent_table.setAlternatingRowColors(True)
        self._agent_table.setMaximumHeight(110)
        layout.addWidget(self._agent_table)

        # LLM summary line
        self._llm_summary_label = QLabel("LLM summary: —")
        self._llm_summary_label.setWordWrap(True)
        self._llm_summary_label.setStyleSheet("color: #cccccc; font-size: 10px; font-style: italic;")
        layout.addWidget(self._llm_summary_label)

        self.bottom_tabs.addTab(tab, "Agent Monitor")

        # Poll supervisor snapshot every 3 s (only active when supervisor is running)
        self._agent_timer = QTimer()
        self._agent_timer.timeout.connect(self._refresh_agent_table)

    def _start_supervisor(self):
        try:
            from core.runtime.supervisor import Supervisor
            self._supervisor = Supervisor()
            self._supervisor.start(loop_delay=5.0)
            self._agent_start_btn.setEnabled(False)
            self._agent_stop_btn.setEnabled(True)
            self._agent_status_label.setText("Agents: running")
            self._agent_status_label.setStyleSheet("color: #2ecc71; font-size: 11px;")
            self._agent_timer.start(3000)
            logger.info("Runtime supervisor started from UI.")
        except Exception as e:
            self._agent_status_label.setText(f"Agents: failed to start — {e}")
            logger.error("Failed to start supervisor: %s", e)

    def _stop_supervisor(self):
        if self._supervisor:
            self._supervisor.stop()
            self._supervisor = None
        self._agent_timer.stop()
        self._agent_start_btn.setEnabled(True)
        self._agent_stop_btn.setEnabled(False)
        self._agent_status_label.setText("Agents: stopped")
        self._agent_status_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        self._agent_table.setRowCount(0)
        self._llm_summary_label.setText("LLM summary: —")

    def _refresh_agent_table(self):
        if self._supervisor is None:
            return
        try:
            snap = self._supervisor.snapshot()
        except Exception:
            return

        meta = snap.pop("__meta__", {})
        llm_text = meta.get("last_summary", "—")
        self._llm_summary_label.setText(f"LLM summary: {llm_text[:200]}")

        _STATUS_COLORS = {
            "ok": "#2ecc71",
            "warning": "#f39c12",
            "error": "#e74c3c",
            "no_data": "#aaaaaa",
        }

        agents = sorted(snap.keys())
        self._agent_table.setRowCount(len(agents))
        for row, name in enumerate(agents):
            info = snap[name]
            latest = info.get("latest")
            status = latest.status if latest else "—"
            summary = str(latest.summary)[:120] if latest else "—"

            name_item = QTableWidgetItem(name)
            status_item = QTableWidgetItem(status)
            summary_item = QTableWidgetItem(summary)

            color = _STATUS_COLORS.get(str(status).lower(), "#ffffff")
            status_item.setForeground(QColor(color))

            self._agent_table.setItem(row, 0, name_item)
            self._agent_table.setItem(row, 1, status_item)
            self._agent_table.setItem(row, 2, summary_item)

    def _setup_missing_deps_tab(self):
        """Show a warning tab listing optional packages that are not installed."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        label = QLabel(
            "<b>Some optional packages are not installed.</b> "
            "The following strategies/brokers will be unavailable until you install them:<br><br>"
            + "".join(f"&nbsp;&nbsp;• <code>pip install {d}</code><br>" for d in self._missing_deps)
        )
        label.setTextFormat(Qt.RichText)
        label.setWordWrap(True)
        label.setStyleSheet("color: #f39c12; font-size: 11px; padding: 8px;")
        layout.addWidget(label)
        layout.addStretch()
        self.bottom_tabs.addTab(tab, "⚠ Deps")

    def closeEvent(self, event):
        if self._supervisor:
            self._supervisor.stop()
        if self.is_streaming:
            self.stop_realtime_stream()
        self.broker_timer.stop()
        self.realtime_timer.stop()
        self.simulation_timer.stop()
        self.news_timer.stop()
        self._news_auto_timer.stop()

        # Best-effort graceful stop. This does NOT reliably work: NewsWorker
        # and DataLoadWorker both do synchronous blocking network I/O inside
        # run() (HTTP requests with retry/backoff, sometimes 30s+ under rate
        # limiting), and QThread.quit() only tells a thread's *event loop* to
        # exit -- it has no effect on code that's blocked in a network call
        # and isn't processing events at all. wait() below will very often
        # time out having done nothing. Left in place because it's free and
        # harmless, not because it's sufficient -- the real fix is the
        # os._exit() below.
        for worker in (getattr(self, "_news_worker", None), getattr(self, "_data_load_worker", None)):
            if worker and worker.isRunning():
                worker.quit()
                worker.wait(200)

        super().closeEvent(event)

        # Python will not fully exit the process until every non-daemon
        # thread finishes -- including any of the above still blocked in
        # network I/O. Without this, the window disappears immediately but
        # the underlying process (and its Dock icon) lingers until that
        # blocked call eventually completes on its own, up to ~30s. Hard-exit
        # immediately instead: no in-memory state here needs a graceful
        # flush, and this is what "closing the window" should actually mean.
        os._exit(0)

    # ------------------------------------------------------------------
    # Data & chart methods
    # ------------------------------------------------------------------

    def calculate_technical_indicators(self):
        """Calculate indicators on native timeframe (no resampling)."""
        df = self.df
        # Moving Averages
        self.df['MA20'] = df['Close'].rolling(window=20).mean()
        self.df['MA50'] = df['Close'].rolling(window=50).mean()
        self.df['MA200'] = df['Close'].rolling(window=200).mean()
        # EMA
        self.df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        self.df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        # MACD
        self.df['MACD'] = self.df['EMA12'] - self.df['EMA26']
        self.df['Signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        # Stochastic
        low14 = df['Low'].rolling(14).min()
        high14 = df['High'].rolling(14).max()
        self.df['K'] = 100 * ((df['Close'] - low14) / (high14 - low14))
        self.df['D'] = self.df['K'].rolling(3).mean()

    def check_strategy_signal(self, data):
        if len(data) < 2:
            return 0
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        strategy = self.strategy_combo.currentText()

        if strategy == "MACD/RSI":
            if latest['RSI'] > 30 and latest['MACD'] > latest['Signal']:
                return 1
            elif latest['RSI'] > 70 or latest['MACD'] < latest['Signal']:
                return -1
        elif strategy == "EMA Crossover":
            if latest['EMA12'] > latest['EMA26'] and prev['EMA12'] <= prev['EMA26']:
                return 1
            elif latest['EMA12'] < latest['EMA26'] and prev['EMA12'] >= prev['EMA26']:
                return -1
        elif strategy == "Stochastic":
            if latest['K'] > latest['D'] and prev['K'] <= prev['D'] and latest['K'] < 20:
                return 1
            elif latest['K'] < latest['D'] and prev['K'] >= prev['D'] and latest['K'] > 80:
                return -1
        return 0

    def start_simulation(self):
        if not hasattr(self, 'df') or self.df.empty:
            self.statusBar().showMessage("Load data before simulation.")
            return
        self.initial_simulation_cash = float(self.cash_input.text())
        self.sim_portfolio = {
            'cash': float(self.cash_input.text()),
            'position': 0,
            'position_value': 0,
            'total_value': float(self.cash_input.text()),
            'pnl': 0
        }
        self.pnl_label.setText("$0.00")
        self.buy_signal_plotted = False
        self.sell_signal_plotted = False
        self.simulation_data = self.df.copy()
        self.simulation_index = 0

        if len(self.simulation_data) > 0:
            display_start = max(0, len(self.simulation_data) - 250)
            initial_display_data = self.simulation_data.iloc[display_start:]
            self.fig = go.Figure(data=[go.Candlestick(
                x=initial_display_data.index.astype(str),
                open=initial_display_data['Open'],
                high=initial_display_data['High'],
                low=initial_display_data['Low'],
                close=initial_display_data['Close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            )])

            if len(initial_display_data) > 0:
                low_min = initial_display_data['Low'].min()
                high_max = initial_display_data['High'].max()
                price_range = high_max - low_min
                padding = price_range * 0.125 if price_range > 0 else high_max * 0.01
                y_range = [low_min - padding, high_max + padding]
                self.fig.update_layout(
                    height=600,
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis=dict(type='date', rangeslider_visible=False),
                    yaxis=dict(title='Price', side='right', range=y_range),
                    hovermode='x unified',
                    template='plotly_dark'
                )

            self.update_plotly_view()

        self.play_btn.show()
        self.pause_btn.show()
        self.simulate_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.play_btn.setEnabled(True)
        self.statusBar().showMessage("Simulation ready. Press Play to start.")

    def play_simulation(self):
        if self.simulation_data is None:
            return
        if self.simulation_index == 0:
            self.simulation_index = min(200, len(self.simulation_data))
        self.simulation_timer.start(250)
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.statusBar().showMessage("Simulation playing...")

    def pause_simulation(self):
        self.simulation_timer.stop()
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.statusBar().showMessage("Simulation paused.")

    def update_simulation_chart(self):
        if self.simulation_index >= len(self.simulation_data):
            self.simulation_timer.stop()
            self.statusBar().showMessage("Simulation finished.")
            self.play_btn.hide()
            self.pause_btn.hide()
            self.simulate_btn.setEnabled(True)
            self.simulation_index = 0
            return

        current_data = self.simulation_data.iloc[:self.simulation_index + 1]
        new_candle = self.simulation_data.iloc[self.simulation_index]
        strategy = self.strategy_combo.currentText()

        if strategy not in ("None", "False"):
            signal = self.check_strategy_signal(current_data)
            if signal == 1 and self.sim_portfolio['position'] == 0:
                self.sim_portfolio['position'] = self.sim_portfolio['cash'] / new_candle['Close']
                self.sim_portfolio['cash'] = 0
                self.fig.add_trace(go.Scatter(
                    x=[str(new_candle.name)], y=[new_candle['Close']],
                    mode='markers', marker=dict(symbol='triangle-up', size=15, color='green'),
                    name='Buy Signal', showlegend=not self.buy_signal_plotted
                ))
                self.buy_signal_plotted = True
            elif signal == -1 and self.sim_portfolio['position'] > 0:
                self.sim_portfolio['cash'] = self.sim_portfolio['position'] * new_candle['Close']
                self.sim_portfolio['position'] = 0
                self.fig.add_trace(go.Scatter(
                    x=[str(new_candle.name)], y=[new_candle['Close']],
                    mode='markers', marker=dict(symbol='triangle-down', size=15, color='red'),
                    name='Sell Signal', showlegend=not self.sell_signal_plotted
                ))
                self.sell_signal_plotted = True

            self.sim_portfolio['position_value'] = self.sim_portfolio['position'] * new_candle['Close']
            self.sim_portfolio['total_value'] = self.sim_portfolio['cash'] + self.sim_portfolio['position_value']
            self.sim_portfolio['pnl'] = self.sim_portfolio['total_value'] - self.initial_simulation_cash
            self.pnl_label.setText(f"${self.sim_portfolio['pnl']:,.2f}")

        window_size = 200
        start_idx = max(0, self.simulation_index - window_size + 1)
        view_data = self.simulation_data.iloc[start_idx:self.simulation_index + 1]

        if len(view_data) == 0:
            return

        x_range = [str(view_data.index[0]), str(view_data.index[-1])]

        low_min = view_data['Low'].min()
        high_max = view_data['High'].max()
        price_range = high_max - low_min

        padding = price_range * 0.125 if price_range > 0 else high_max * 0.01
        y_min = low_min - padding
        y_max = high_max + padding

        with self.fig.batch_update():
            self.fig.data[0].x = current_data.index.astype(str)
            self.fig.data[0].open = current_data['Open']
            self.fig.data[0].high = current_data['High']
            self.fig.data[0].low = current_data['Low']
            self.fig.data[0].close = current_data['Close']
            self.fig.update_layout(
                xaxis_range=x_range,
                yaxis_range=[y_min, y_max]
            )

        self.update_plotly_view()
        self.simulation_index += 1

    def reset_chart_zoom(self):
        """Reset the chart zoom to the initial view."""
        if hasattr(self, 'fig'):
            self.plot_candles()
            self.statusBar().showMessage("Chart zoom reset.")

    def load_data(self):
        source = self.data_source_combo.currentText()
        symbol = self.symbol_combo.currentText()
        interval = self.interval_combo.currentText()
        logger.debug("Loading data — source: %s, symbol: %s, interval: %s", source, symbol, interval)

        if source == "Realtime Stream":
            self.start_realtime_stream(symbol)
            return

        if self.is_streaming:
            self.stop_realtime_stream()

        if getattr(self, "_data_load_worker", None) and self._data_load_worker.isRunning():
            self.statusBar().showMessage("Already loading data — please wait...")
            return

        try:
            days = int(self.days_input.text())
        except Exception as e:
            self.statusBar().showMessage(f"Error: {str(e)}")
            return

        self.statusBar().showMessage(f"Loading {symbol}...")
        self._data_load_worker = DataLoadWorker(
            data_loader=self.data_loader,
            symbol=symbol,
            source=source,
            live=(source == "Live"),
            days=days,
            interval=interval,
        )
        self._data_load_worker.results_ready.connect(
            lambda df: self._on_data_loaded(df, symbol)
        )
        self._data_load_worker.error.connect(self._on_data_load_error)
        self._data_load_worker.start()

    def _on_data_loaded(self, df, symbol):
        try:
            self.df = df
            assert not self.df.empty, "Loaded empty DataFrame"
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            assert all(col in self.df.columns for col in required_cols), f"Missing columns: {required_cols}"
            assert pd.api.types.is_datetime64_any_dtype(self.df.index), "Index must be datetime"

            # Compute indicators so simulation/strategy-signal checks never hit a KeyError
            self.calculate_technical_indicators()
            self.plot_candles()
            self.statusBar().showMessage(f"Loaded {len(self.df)} candles for {symbol}")

        except Exception as e:
            self.statusBar().showMessage(f"Error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def _on_data_load_error(self, message):
        self.statusBar().showMessage(f"Error: {message}")
        logger.error("Data load failed: %s", message)

    def _run_backtest_logic(self):
        """Unified backtest entry point called by background thread worker usually."""
        return self._run_unified_backtest()

    def _run_backtest_logic_with_broker(self, broker):
        """Unified backtest entry point with broker called by background thread worker."""
        return self._run_unified_backtest(broker=broker)

    def _run_unified_backtest(self, broker=None):
        strategy_name = self.strategy_combo.currentText()
        logger.debug("Strategy name: %s", strategy_name)

        if strategy_name in ("None", "False"):
            self.statusBar().showMessage("No strategy selected!")
            return False

        if not hasattr(self, 'df') or self.df.empty:
            self.statusBar().showMessage("Please load data before backtest.")
            return False

        try:
            kwargs = {}
            if strategy_name == "LSTM Predictor":
                kwargs = {'ticker': self.symbol_combo.currentText(), 'sequence_length': 60}

            strategy_wrapper = self.strategy_manager.get_strategy(strategy_name, **kwargs)
            if not strategy_wrapper:
                 self.statusBar().showMessage(f"Failed to load strategy: {strategy_name}")
                 return False

            broker_mode = "simulated"
            real_broker = None

            current_broker_name = self.broker_combo.currentText()
            if current_broker_name != "Simulator":
                 if broker:
                     broker_mode = "real"
                     real_broker = broker

            try:
                market_fee = float(self.market_fee_input.text()) / 100.0
                limit_fee = float(self.limit_fee_input.text()) / 100.0
            except ValueError:
                market_fee = 0.001
                limit_fee = 0.0005

            initial_cash = float(self.cash_input.text())

            results = self.strategy_manager.run_backtest(
                strategy_wrapper=strategy_wrapper,
                data=self.df,
                cash=initial_cash,
                broker_mode=broker_mode,
                broker=real_broker,
                market_fee=market_fee,
                limit_fee=limit_fee
            )

            return results

        except Exception as e:
            self.statusBar().showMessage(f"Backtest error: {str(e)}")
            logger.error("Backtest error: %s", e)
            import traceback
            logger.error(traceback.format_exc())
            return False

    def run_backtest(self):
        try:
            backtest_broker = self.broker_manager.get_broker("Simulator")

            if not backtest_broker:
                self.statusBar().showMessage("Simulator not available")
                return

            initial_cash = float(self.cash_input.text())

            backtest_broker = self.broker_manager.get_broker("Simulator")
            backtest_broker.balance = initial_cash
            backtest_broker.initial_balance = initial_cash
            backtest_broker.portfolio_value = initial_cash
            backtest_broker.positions = {}
            backtest_broker.order_history = []
            backtest_broker.closed_positions = []
            backtest_broker.orders = {}

            try:
                backtest_broker.market_fee = float(self.market_fee_input.text()) / 100.0
                backtest_broker.limit_fee = float(self.limit_fee_input.text()) / 100.0
            except ValueError:
                pass

            logger.debug("Broker reset: Balance=$%.2f, Orders=%d", backtest_broker.balance, len(backtest_broker.order_history))

            results = self._run_backtest_logic_with_broker(backtest_broker)
            if results is False:
                return

            summary = results.get('summary', {})
            final_value = summary.get('Final Value', 0)
            total_pnl = summary.get('P&L', 0)
            sharpe_ratio = summary.get('Sharpe Ratio', results.get('sharpe', 0))
            max_drawdown = summary.get('Max Drawdown (%)', results.get('max_drawdown', 0))
            win_rate = summary.get('Win Rate', f"{results.get('win_rate', 0):.2f}%")
            total_trades = summary.get('Number of Closed Trades', 0)

            sharpe_str = f"{sharpe_ratio:.2f}" if isinstance(sharpe_ratio, (int, float)) else "N/A"
            dd_str = f"{max_drawdown:.2f}%" if isinstance(max_drawdown, (int, float)) else "N/A"

            msg = (f"Backtest complete | "
                f"Final: ${final_value:,.2f} | "
                f"P&L: ${total_pnl:+,.2f} | "
                f"Sharpe: {sharpe_str} | "
                f"MaxDD: {dd_str} | "
                f"Win Rate: {win_rate} | "
                f"Trades: {total_trades}")

            self.statusBar().showMessage(msg)

            # Update persistent results panel labels (right panel)
            self.bt_sharpe_label.setText(sharpe_str)
            self.bt_winrate_label.setText(win_rate if isinstance(win_rate, str) else f"{win_rate:.2f}%")
            self.bt_maxdd_label.setText(dd_str)

            self.plot_signals(results.get('signals', []))

            if 'Final Value' in summary:
                 backtest_broker.balance = float(summary['Final Value'])
                 backtest_broker.portfolio_value = float(summary['Final Value'])

            self.current_broker = backtest_broker
            self.refresh_account_info()

            logger.info("BACKTEST RESULTS: %s", summary)

        except Exception as e:
            self.statusBar().showMessage(f"Backtest error: {str(e)}")
            logger.error("Backtest error: %s", e)
            import traceback
            logger.error(traceback.format_exc())

    def plot_signals(self, signals):
        # 'buy' = open long; 'buy_cover' = close short (both rendered as green up-triangles)
        buy_signals = [s for s in signals if s.get('type') in ('buy', 'buy_cover')]
        # 'sell' = close long; 'sell_short' = open short (both rendered as red down-triangles)
        sell_signals = [s for s in signals if s.get('type') in ('sell', 'sell_short')]
        if buy_signals:
            self.fig.add_trace(go.Scatter(
                x=[s['date'] for s in buy_signals],
                y=[s['price'] for s in buy_signals],
                mode='markers',
                marker=dict(symbol='triangle-up', size=15, color='green'),
                name='Buy Signal'
            ))
        if sell_signals:
            self.fig.add_trace(go.Scatter(
                x=[s['date'] for s in sell_signals],
                y=[s['price'] for s in sell_signals],
                mode='markers',
                marker=dict(symbol='triangle-down', size=15, color='red'),
                name='Sell Signal'
            ))
        self.update_plotly_view()

    def on_order_type_changed(self, order_type):
        """Update UI based on selected order type"""
        if order_type == "Limit":
            self.limit_price_input.setVisible(True)
            self.limit_price_input.setPlaceholderText("Required")
        elif order_type == "Stop":
            self.limit_price_input.setVisible(True)
            self.limit_price_input.setPlaceholderText("Stop Price")
        else:  # Market
            self.limit_price_input.setVisible(False)

    def refresh_account_info(self):
        """Refresh and display account information"""
        try:
            if self.current_broker:
                account_info = self.current_broker.get_account_info()

                logger.debug("Broker: balance=$%.2f pv=$%.2f pnl=$%.2f", account_info.get('balance', 0), account_info.get('portfolio_value', 0), account_info.get('pnl', 0))

                if hasattr(self.current_broker, 'positions'):
                    for symbol, position in self.current_broker.positions.items():
                        logger.debug("  Position %s: %.6f @ $%.2f", symbol, position.qty, position.avg_price)

                if hasattr(self.current_broker, 'order_history'):
                    logger.debug("Order history: %d orders", len(self.current_broker.order_history))

                broker_name = self.broker_combo.currentText()
                balance = account_info.get('balance', 0)
                pnl = account_info.get('pnl', 0)
                pnl_color = "#3fb950" if pnl >= 0 else "#f85149"

                self.account_label.setText(f"{broker_name}\n${balance:,.2f}")
                self.account_label.setStyleSheet(f"color: {pnl_color};")

                self.pnl_label.setText(f"${pnl:+,.2f}")
                self.pnl_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {pnl_color};")

                self.update_positions_display()

        except Exception as e:
            logger.warning("Error refreshing account: %s", e)

    def update_positions_display(self):
        """Update positions display"""
        if not self.current_broker:
            return

        try:
            positions_text = "CURRENT POSITIONS:\n"
            positions_text += "-" * 30 + "\n"

            has_positions = False
            if hasattr(self.current_broker, 'positions'):
                for symbol, position in self.current_broker.positions.items():
                    if position.qty != 0:
                        has_positions = True
                        sign = "+" if position.pnl >= 0 else ""
                        positions_text += (
                            f"{symbol}: {position.qty:+.2f} @ ${position.avg_price:.2f}\n"
                            f"  PnL: {sign}${position.pnl:,.2f}\n"
                        )

            if not has_positions:
                positions_text += "No active positions\n"

            self.positions_text.setPlainText(positions_text)

        except Exception as e:
            self.positions_text.setPlainText(f"Error loading positions: {str(e)}")

    def place_order(self, side):
        """Place an order through the current broker"""
        try:
            symbol = self.symbol_combo.currentText()
            qty = float(self.order_qty_input.text())
            order_type = self.order_type_combo.currentText().lower()

            limit_price = None
            stop_price = None

            if order_type == "limit" and self.limit_price_input.text():
                limit_price = float(self.limit_price_input.text())
            elif order_type == "stop" and self.limit_price_input.text():
                stop_price = float(self.limit_price_input.text())

            if not self.current_broker:
                self.current_broker = self.broker_manager.get_broker(self.broker_combo.currentText())

            order = self.current_broker.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price
            )

            if order.status.value == "filled":
                self.statusBar().showMessage(
                    f"{side.upper()} order filled for {qty} {symbol} @ ${order.filled_avg_price:.2f}"
                )
            elif order.status.value == "pending":
                self.statusBar().showMessage(
                    f"{side.upper()} order pending for {qty} {symbol}"
                )
            else:
                self.statusBar().showMessage(
                    f"Order {order.status.value}: {qty} {symbol}"
                )

            self.refresh_account_info()
            self._refresh_orders_tab()

        except Exception as e:
            self.statusBar().showMessage(f"Order error: {str(e)}")

    def start_trading(self):
        broker_name = self.broker_combo.currentText()
        strategy_name = self.strategy_combo.currentText()

        if strategy_name in ("None", "False"):
            self.statusBar().showMessage("No strategy selected!")
            return
        try:
            broker = self.broker_manager.get_broker(broker_name)
            self.current_broker = broker

            if not self.current_broker:
                self.statusBar().showMessage(f"Broker {broker_name} not configured!")
                return

            if broker_name == "Simulator":
                try:
                    self.current_broker.market_fee = float(self.market_fee_input.text()) / 100.0
                    self.current_broker.limit_fee = float(self.limit_fee_input.text()) / 100.0
                except ValueError:
                    pass

            initial_cash = float(self.cash_input.text())
            if hasattr(self.current_broker, 'balance'):
                self.current_broker.balance = initial_cash
                self.current_broker.initial_balance = initial_cash
                self.current_broker.portfolio_value = initial_cash

            if strategy_name == "LSTM Predictor":
                symbol = self.symbol_combo.currentText()
                strategy = self.strategy_manager.get_strategy(strategy_name, ticker=symbol, sequence_length=60)
            else:
                strategy = self.strategy_manager.get_strategy(strategy_name)

            self.refresh_account_info()

            self.statusBar().showMessage(f"Live trading started with {broker_name} using {strategy_name}")

        except Exception as e:
            self.statusBar().showMessage(f"Trading error: {str(e)}")

    # === REALTIME STREAMING (Plotly-only) ===
    def start_realtime_stream(self, symbol):
        if self.is_streaming:
            self.stop_realtime_stream()

        interval = self.interval_combo.currentText()
        self.current_interval = interval
        self.realtime_df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close'])

        minutes_map = {'1m':1,'5m':5,'15m':15,'30m':30,'1h':60,'1d':1440}
        minutes_needed = 130 * minutes_map.get(interval, 1)
        days_needed = max(1, minutes_needed / (6.5 * 60))
        try:
            hist_df = self.data_loader.load_data(symbol=symbol, live=True, interval=interval, days=days_needed)
            self.realtime_df = hist_df.tail(self.max_candles).copy()
        except Exception as e:
            self.statusBar().showMessage(f"Error loading historical: {e}")
            return

        self.data_loader.start_realtime_stream(symbol=symbol, callback=self.handle_realtime_data)
        self.is_streaming = True
        self.realtime_timer.start(100)
        self.statusBar().showMessage(f"Started {interval} stream for {symbol}")
        self.update_realtime_chart()

    def handle_realtime_data(self, data):
        self.data_loader.realtime_queue.put(data)

    def process_realtime_updates(self):
        try:
            while True:
                data = self.data_loader.realtime_queue.get_nowait()
                self.process_realtime_data(data)
        except Empty:
            pass

    def process_realtime_data(self, data):
        ts = pd.Timestamp(data['timestamp'])

        if 'bids' in data and 'asks' in data and data['bids'] and data['asks']:
            best_bid = float(data['bids'][0][0])
            best_ask = float(data['asks'][0][0])
            price = (best_bid + best_ask) / 2
        elif 'price' in data:
            price = data['price']
        else:
            logger.warning(f"Real-time data missing 'price' or 'bids'/'asks' for candlestick: {data}")
            return

        if self.current_interval.endswith('m'):
            freq = self.current_interval.replace('m', 'min')
        elif self.current_interval.endswith('h'):
            freq = self.current_interval.replace('h', 'H')
        elif self.current_interval.endswith('d'):
            freq = self.current_interval.replace('d', 'D')
        else:
            freq = self.current_interval

        if self.realtime_df.empty:
            new_row = pd.DataFrame([{'Open': price, 'High': price, 'Low': price, 'Close': price}], index=[ts])
            self.realtime_df = pd.concat([self.realtime_df, new_row])
        else:
            last_ts = self.realtime_df.index[-1].floor(freq)
            current_ts = ts.floor(freq)

            if current_ts == last_ts:
                self.realtime_df.loc[self.realtime_df.index[-1], 'High'] = max(self.realtime_df.iloc[-1]['High'], price)
                self.realtime_df.loc[self.realtime_df.index[-1], 'Low'] = min(self.realtime_df.iloc[-1]['Low'], price)
                self.realtime_df.loc[self.realtime_df.index[-1], 'Close'] = price
            else:
                new_row = pd.DataFrame([{'Open': price, 'High': price, 'Low': price, 'Close': price}], index=[ts])
                self.realtime_df = pd.concat([self.realtime_df, new_row])
                if len(self.realtime_df) > self.max_candles:
                    self.realtime_df = self.realtime_df.iloc[-self.max_candles:]

        self.update_realtime_chart()

    def update_realtime_chart(self):
        if self.realtime_df.empty:
            return

        if len(self.fig.data) == 0:
            self.fig.add_trace(go.Candlestick(
                x=self.realtime_df.index,
                open=self.realtime_df['Open'],
                high=self.realtime_df['High'],
                low=self.realtime_df['Low'],
                close=self.realtime_df['Close'],
                name='Price'
            ))
        else:
            self.fig.data[0].x = self.realtime_df.index
            self.fig.data[0].open = self.realtime_df['Open']
            self.fig.data[0].high = self.realtime_df['High']
            self.fig.data[0].low = self.realtime_df['Low']
            self.fig.data[0].close = self.realtime_df['Close']

        self.fig.update_layout(
            height=600,
            xaxis=dict(type='date', rangeslider_visible=False),
            yaxis=dict(title='Price', autorange=True),
            template='plotly_dark'
        )
        self.update_plotly_view()

    def update_plotly_view(self):
        """Render Plotly figure into QWebEngineView via a temp file.

        QWebEngineView.setHtml() base64-encodes its argument internally and
        silently fails to render anything above ~2MB (no exception raised —
        it just shows a blank view). to_html(include_plotlyjs=True) embeds
        the full Plotly.js bundle inline, which alone exceeds that limit.
        Writing to disk and using load(QUrl.fromLocalFile(...)) has no such
        size restriction.
        """
        if not _WEBENGINE_AVAILABLE or not _PLOTLY_AVAILABLE:
            return
        if not hasattr(self, 'fig'):
            return
        html = self.fig.to_html(include_plotlyjs=True, full_html=True)
        chart_path = os.path.join(tempfile.gettempdir(), "algotrader_chart.html")
        with open(chart_path, "w", encoding="utf-8") as f:
            f.write(html)
        self.plotly_view.load(QUrl.fromLocalFile(chart_path))

    def stop_realtime_stream(self):
        if self.is_streaming:
            self.data_loader.stop_realtime_stream()
            self.realtime_timer.stop()
            self.is_streaming = False
            self.realtime_df = pd.DataFrame()

    def update_live_news(self):
        symbol = self.symbol_combo.currentText()
        logger.debug("Checking news for %s", symbol)

        news_df = scrape_and_analyze_finviz_news(symbol)

        if not news_df.empty:
            latest_headline = news_df.iloc[0]['headline']
            if hasattr(self, 'last_seen_headline') and latest_headline != self.last_seen_headline:
                logger.debug("New news: %s", latest_headline)
                self.last_seen_headline = latest_headline
                if hasattr(self, 'latest_sentiment') and self.latest_sentiment is not None:
                    self.latest_sentiment['positive'] = news_df.iloc[0]['positive']
                    self.latest_sentiment['negative'] = news_df.iloc[0]['negative']
                    self.latest_sentiment['neutral'] = news_df.iloc[0]['neutral']
                self.statusBar().showMessage(f"New News: {latest_headline}", 5000)

    def show_statistics(self):
        """Run backtest and show the statistics window"""
        results = self._run_backtest_logic()
        if results is False:
            return

        if not _STATS_WINDOW_AVAILABLE:
            self.statusBar().showMessage("Statistics window unavailable (matplotlib not installed)")
            return

        self.stats_window = StatisticsWindow(results)
        self.stats_window.show()

    def plot_candles(self):
        if not hasattr(self, 'df') or self.df.empty:
            return

        self.fig = go.Figure(data=[go.Candlestick(
            x=self.df.index,
            open=self.df['Open'],
            high=self.df['High'],
            low=self.df['Low'],
            close=self.df['Close'],
            name='Price'
        )])

        self.fig.update_layout(
            height=600,
            xaxis=dict(type='date', rangeslider_visible=False),
            yaxis=dict(title='Price'),
            template='plotly_dark'
        )
        self.update_plotly_view()
