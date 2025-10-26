# ui/main_window.py
from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
                             QComboBox, QPushButton, QLabel, QGroupBox, QLineEdit, QTextEdit)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWebEngineWidgets import QWebEngineView
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder
import json
import pandas as pd
import numpy as np
from queue import Empty
from datetime import datetime, timedelta
from ui.statistics_window import StatisticsWindow
from typing import Dict
from core.news_scraper import scrape_and_analyze_finviz_news


class MainWindow(QMainWindow):
    def __init__(self, data_loader, strategy_manager, broker_manager, ai_analyzer):
        super().__init__()
        self.data_loader = data_loader
        self.strategy_manager = strategy_manager
        self.broker_manager = broker_manager
        self.ai_analyzer = ai_analyzer
        self.setWindowTitle("Algorithmic Trading Terminal")
        self.resize(1400, 800)

        # Central Widget
        # create empty container
        central_widget = QWidget()
        # add vertical shelves/organizer to the container
        self.main_layout = QVBoxLayout(central_widget)
        # place items on the shelves (in order)
        # 1. Control Panel (top)
        self.control_layout = QHBoxLayout()
        self.control_layout.setObjectName("controlPanelLayout")

        # Add control widgets
        # data source
        self.control_layout.addWidget(QLabel("Data Source:"))
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems(["Historical", "Live", "Realtime Stream", "FinRL-Yahoo"])
        self.control_layout.addWidget(self.data_source_combo)
        # symbol selection
        self.control_layout.addWidget(QLabel("Symbol:"))
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "AAPL", "TSLA", "GOLD", "SPY", "QQQ"])
        self.symbol_combo.currentTextChanged.connect(self.load_data)
        self.control_layout.addWidget(self.symbol_combo)
        # strategy selection
        self.control_layout.addWidget(QLabel("Strategy:"))
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "False",
            "MACD/RSI",
            "EMA Crossover",
            "Stochastic",
            "LSTM Predictor",
            "FinRL Strategy",
            "DDPG Strategy"
        ])
        self.control_layout.addWidget(self.strategy_combo)
        # broker selection
        self.control_layout.addWidget(QLabel("Broker:"))
        self.broker_combo = QComboBox()
        self.broker_combo.addItems(["Simulator", "Alpaca", "Interactive Brokers", "Binance"])
        self.control_layout.addWidget(self.broker_combo)

        # Store broker manager
        self.current_broker = None # the actual working object that does the trading
        self.current_broker_name = "Simulator" # the display name shown in the combo box

        # Action Buttons
        self.load_btn = QPushButton("Load Data")
        self.backtest_btn = QPushButton("Run Backtest")
        self.trade_btn = QPushButton("Start Live Trading")
        self.reset_btn = QPushButton("Reset Zoom")
        self.simulate_btn = QPushButton("Simulate")
        self.play_btn = QPushButton("Play")
        self.pause_btn = QPushButton("Pause")
        self.control_layout.addWidget(self.load_btn)
        self.control_layout.addWidget(self.backtest_btn)
        self.control_layout.addWidget(self.trade_btn)
        self.control_layout.addWidget(self.reset_btn)
        self.control_layout.addWidget(self.simulate_btn)
        self.control_layout.addWidget(self.play_btn)
        self.control_layout.addWidget(self.pause_btn)
        self.play_btn.hide()
        self.pause_btn.hide()

        # Add interval, days, and cash controls
        self.setup_interval_controls()
        self.setup_days_input()
        self.setup_cash_input()

        self.main_layout.addLayout(self.control_layout)

        # 2. Plotly Chart View (middle-top)
        self.plotly_view = QWebEngineView()
        empty_html = """
            <html>
            <head>
                <meta charset="utf-8"/>
                <style>
                body {
                    background-color: #121212;
                    color: #ffffff;
                    margin: 0;
                    padding: 0;
                }
                </style>
            </head>
            <body></body>
            </html>
            """
        self.plotly_view.setHtml(empty_html)
        self.main_layout.addWidget(self.plotly_view)

        # 3. Broker Controls (middle)
        self.setup_broker_ui()
        
        # Broker monitoring timer
        self.broker_timer = QTimer()
        self.broker_timer.timeout.connect(self.refresh_account_info)
        self.broker_timer.start(5000)

        # 4. DeepSeek AI Analysis Controls (middle-bottom)
        if self.ai_analyzer:
            self.setup_ai_analysis_ui()
        else:
            print("AI Analyzer not initialized; AI features disabled.")

        # 5. Status Bar & P&L (bottom)
        self.statusBar().showMessage("Ready")
        self.pnl_label = QLabel("P&L: $0.00")
        self.pnl_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #2ecc71;")
        self.pnl_label.setFixedHeight(20)
        self.main_layout.addWidget(self.pnl_label, alignment=Qt.AlignRight)

        self.setCentralWidget(central_widget)

        # Connect Signals
        self.load_btn.clicked.connect(self.load_data)
        self.backtest_btn.clicked.connect(self.run_backtest)
        self.trade_btn.clicked.connect(self.start_trading)
        self.reset_btn.clicked.connect(self.reset_chart_zoom)
        self.simulate_btn.clicked.connect(self.start_simulation)
        self.play_btn.clicked.connect(self.play_simulation)
        self.pause_btn.clicked.connect(self.pause_simulation)

        # Realtime & Simulation Timers
        self.realtime_timer = QTimer()
        self.realtime_timer.timeout.connect(self.process_realtime_updates)
        self.is_streaming = False

        # Initialize with sample data
        self.load_data()
        self.realtime_df = pd.DataFrame()
        self.current_interval = '1m'
        self.max_candles = 130

        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self.update_simulation_chart)
        self.simulation_data = None
        self.simulation_index = 0
        self.buy_signal_plotted = False
        self.sell_signal_plotted = False

        # News & AI
        self.news_timer = QTimer()
        self.news_timer.timeout.connect(self.update_live_news)
        self.last_seen_headline = ""
        self.latest_sentiment = {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}

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
        # STORE INITIAL CASH FOR P&L CALCULATION
        self.initial_simulation_cash = float(self.cash_input.text())
        self.sim_portfolio = {
            'cash': float(self.cash_input.text()),
            'position': 0,  # will be populated as trades happen
            'position_value': 0,
            'total_value': float(self.cash_input.text()),
            'pnl': 0
        }
        self.pnl_label.setText("P&L: $0.00")
        self.buy_signal_plotted = False
        self.sell_signal_plotted = False
        self.simulation_data = self.df.copy()
        self.simulation_index = 0

        # window_size = min(200, len(self.df))
        # initial_data = self.simulation_data.iloc[:window_size]
        # if len(initial_data) == 0:
        #     return

        if len(self.simulation_data) > 0:
            # Start from the beginning but you might want to start later
            display_start = max(0, len(self.simulation_data) - 250)  # Show last 250 candles
            initial_display_data = self.simulation_data.iloc[display_start:]
            # Create initial chart with reasonable view
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
            
            # Set initial view to show reasonable range
            if len(initial_display_data) > 0:
                low_min = initial_display_data['Low'].min()
                high_max = initial_display_data['High'].max()
                price_range = high_max - low_min
                padding = price_range * 0.125 if price_range > 0 else high_max * 0.01
                y_range = [low_min - padding, high_max + padding]
                # === DYNAMIC Y-AXIS FOR INITIAL VIEW ===
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

        if strategy != "False":
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
            self.pnl_label.setText(f"P&L: ${self.sim_portfolio['pnl']:,.2f}")

        # === DYNAMIC Y-AXIS SCALING (80% height) ===
        window_size = 200
        start_idx = max(0, self.simulation_index - window_size + 1)
        view_data = self.simulation_data.iloc[start_idx:self.simulation_index + 1]

        if len(view_data) == 0:
            return

        x_range = [str(view_data.index[0]), str(view_data.index[-1])]

        # Calculate price range in visible window
        low_min = view_data['Low'].min()
        high_max = view_data['High'].max()
        price_range = high_max - low_min

        # Add padding to fill ~80% of chart height
        # Padding = (price_range / 0.8) * 0.2 / 2 = price_range * 0.125
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

    def load_data(self):
        source = self.data_source_combo.currentText()
        symbol = self.symbol_combo.currentText()
        interval = self.interval_combo.currentText()
        print(f"[DEBUG] Loading data for source: {source}, symbol: {symbol}, interval: {interval}")

        try:
            if source == "Realtime Stream":
                self.start_realtime_stream(symbol)
                return

            if self.is_streaming:
                self.stop_realtime_stream()

            days = int(self.days_input.text())
            self.df = self.data_loader.load_data(
                symbol=symbol,
                source=source,
                live=(source == "Live"),
                days=days,
                interval=interval
            )

            assert not self.df.empty, "Loaded empty DataFrame"
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            assert all(col in self.df.columns for col in required_cols), f"Missing columns: {required_cols}"
            assert pd.api.types.is_datetime64_any_dtype(self.df.index), "Index must be datetime"

            self.plot_candles()
            self.statusBar().showMessage(f"Loaded {len(self.df)} candles for {symbol}")

        except Exception as e:
            self.statusBar().showMessage(f"Error: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def plot_candles(self):
        if not hasattr(self, 'df') or self.df.empty:
            return

        if isinstance(self.df.index, pd.MultiIndex):
            self.df = self.df.droplevel('Ticker')
        if 'Ticker' in self.df.columns.names:
            self.df.columns = self.df.columns.droplevel('Ticker')
        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.df.index = pd.to_datetime(self.df.index)
        if self.df.index.tz is None:
            self.df.index = self.df.index.tz_localize('UTC')
        else:
            self.df.index = self.df.index.tz_convert('UTC')

        self.calculate_technical_indicators()

        self.fig = make_subplots(
            rows=5, cols=1, shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=('Price', 'Volume', 'MACD', 'RSI', 'Stochastic'),
            row_width=[0.15, 0.15, 0.15, 0.10, 0.75]
        )

        self.fig.add_trace(go.Candlestick(
            x=self.df.index,
            open=self.df['Open'],
            high=self.df['High'],
            low=self.df['Low'],
            close=self.df['Close'],
            name='Price'
        ), row=1, col=1)

        colors = np.where(self.df['Close'] >= self.df['Open'], 'green', 'red')
        self.fig.add_trace(go.Bar(
            x=self.df.index,
            y=self.df['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.4,
            yaxis='y2'
        ), row=1, col=1)

        self.add_technical_indicators()

        # Create buttons for indicator toggles
        # buttons arrangement:    
        #    * Trace 0 (`go.Candlestick`): The main price data
        #    * Trace 1 (`go.Bar`): Volume bars
        #    * Trace 2 (`go.Scatter`): Volume line xxxx
        #    * Trace 3 (`go.Scatter`): Moving Average 20 (MA20)
        #    * Trace 4 (`go.Scatter`): Moving Average 50 (MA50)
        #    * Trace 5 (`go.Scatter`): Moving Average 200 (MA200)
        #    * Trace 6 (`go.Scatter`): Exponential Moving Average 12 (EMA12)
        #    * Trace 7 (`go.Scatter`): Exponential Moving Average 26 (EMA26)
        #    * Trace 8 (`go.Scatter`): MACD Line
        #    * Trace 9 (`go.Scatter`): MACD Signal Line
        #    * Trace 10 (`go.Scatter`): Relative Strength Index (RSI)
        #    * Trace 11 (`go.Scatter`): Stochastic Oscillator %K
        #    * Trace 12 (`go.Scatter`): Stochastic Oscillator %D
        buttons = [
            dict(label="MA", method="restyle", args=["visible", [True, True, True, True, True, False, False, False, False, False, False, False, False]],
                args2=["visible", [True, True, False, False, False, False, False, False, False, False, False, False, False]]),
            dict(label="EMA", method="restyle", args=["visible", [True, True, False, False, False, True, True, False, False, False, False, False, False]],
                args2=["visible", [True, True, False, False, False, False, False, False, False, False, False, False, False]]),
            dict(label="MACD", method="restyle", args=["visible", [True, True, False, False, False, False, False, True, True, False, False, False, False]],
                args2=["visible", [True, True, False, False, False, False, False, False, False, False, False, False, False]]),
            dict(label="RSI", method="restyle", args=["visible", [True, True, False, False, False, False, False, False, False, True, False, False, False]],
                args2=["visible", [True, True, False, False, False, False, False, False, False, False, False, False, False]]),
            dict(label="Stochastic", method="restyle", args=["visible", [True, True, False, False, False, False, False, False, False, False, True, True, False]],
                args2=["visible", [True, True, False, False, False, False, False, False, False, False, False, False, False]]),
            dict(label="All Off", method="restyle", args=[
                {"visible": [True, True, False, False, False, False, False, False, False, False, False, False, False]}],args2=[
                {"visible": [True, True, False, False, False, False, False, False, False, False, False, False, False]}]),
        ]

        initial_range = [self.df.index[-250], self.df.index[-1]] if len(self.df) > 250 else [self.df.index[0], self.df.index[-1]]
        is_crypto = "USD" in self.symbol_combo.currentText().upper()

        # Calculate y-range for initial view
        view_df = self.df.loc[initial_range[0]:initial_range[1]]
        low_min = view_df['Low'].min()
        high_max = view_df['High'].max()
        price_range = high_max - low_min
        padding = price_range * 0.125 if price_range > 0 else high_max * 0.01
        y_range = [low_min - padding, high_max + padding]

        xaxis_config = dict(
            type='date',
            rangeslider=dict(visible=True),
            rangeselector=dict(buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ]),
            range=initial_range
        )
        if not is_crypto:
            xaxis_config["rangebreaks"] = [dict(bounds=["sat", "mon"]), dict(bounds=[16, 9.5], pattern="hour")]

        self.fig.update_layout(
            height=800,
            margin=dict(l=20, r=20, t=40, b=20),
            updatemenus=[dict(
                type="buttons", direction="right", x=0.5, y=1.15,
                xanchor="center", yanchor="top", buttons=buttons
            )],
            xaxis=xaxis_config,
            yaxis=dict(title='Price', side='left', range=y_range),
            # yaxis2=dict(title='Volume', side='right', overlaying='y', range=[0, self.df['Volume'].max() * 3]),
            # yaxis=dict(title='Price', side='left', fixedrange=False),
            yaxis2=dict(title='Volume', side='right', overlaying='y', rangemode='tozero', showgrid=False, showline=False, zeroline=False, tickformat=".2s"),
            hovermode='x unified',
            template='plotly_dark'
        )

        self.update_plotly_view()

    def add_technical_indicators(self):
        self.fig.add_trace(go.Scatter(x=self.df.index, y=self.df['MA20'], name='MA20', line=dict(color='blue'),visible=False), row=1, col=1)
        self.fig.add_trace(go.Scatter(x=self.df.index, y=self.df['MA50'], name='MA50', line=dict(color='orange'),visible=False), row=1, col=1)
        self.fig.add_trace(go.Scatter(x=self.df.index, y=self.df['MA200'], name='MA200', line=dict(color='purple'),visible=False), row=1, col=1)
        self.fig.add_trace(go.Scatter(x=self.df.index, y=self.df['EMA12'], name='EMA12', line=dict(color='cyan'),visible=False), row=1, col=1)
        self.fig.add_trace(go.Scatter(x=self.df.index, y=self.df['EMA26'], name='EMA26', line=dict(color='magenta'),visible=False), row=1, col=1)
        self.fig.add_trace(go.Scatter(x=self.df.index, y=self.df['MACD'], name='MACD', line=dict(color='blue'),visible=False), row=3, col=1)
        self.fig.add_trace(go.Scatter(x=self.df.index, y=self.df['Signal'], name='Signal', line=dict(color='red'),visible=False), row=3, col=1)
        self.fig.add_trace(go.Scatter(x=self.df.index, y=self.df['RSI'], name='RSI', line=dict(color='purple'),visible=False), row=4, col=1)
        self.fig.add_trace(go.Scatter(x=self.df.index, y=self.df['K'], name='K', line=dict(color='blue'),visible=False), row=5, col=1)
        self.fig.add_trace(go.Scatter(x=self.df.index, y=self.df['D'], name='D', line=dict(color='red'),visible=False), row=5, col=1)

    def update_plotly_view(self):
        try:
            if self.fig and self.fig.data:
                print(f"[DEBUG] Updating plot view. Number of points in first trace: {len(self.fig.data[0].x)}")
            else:
                print("[DEBUG] Updating plot view, but no data in figure.")
            fig_dict = self.fig.to_dict()
            raw_html = f'''
                <html>
                <head><meta charset="utf-8"/>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
                <style>#plotly-div {{ width: 100%; height: 600px; }} body {{ margin:0; padding:0; }}</style>
                </head>
                <body><div id="plotly-div"></div>
                <script>
                    try {{
                        var graph = {json.dumps(fig_dict, cls=PlotlyJSONEncoder)};
                        Plotly.newPlot('plotly-div', graph);
                    }} catch (e) {{ console.error(e); }}
                </script></body></html>'''
            self.plotly_view.setHtml(raw_html)
        except Exception as e:
            print("Error in update_plotly_view:", str(e))

    def reset_chart_zoom(self):
        if hasattr(self, 'fig') and hasattr(self, 'df') and not self.df.empty:
            self.fig.update_xaxes(autorange=True)
            self.fig.update_yaxes(autorange=True)
            self.update_plotly_view()

    def setup_interval_controls(self):
        interval_label = QLabel("Candle Interval:")
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(['1m', '5m', '15m', '30m', '1h', '1d'])
        self.interval_combo.setCurrentText('1d')
        self.control_layout.insertWidget(4, interval_label)
        self.control_layout.insertWidget(5, self.interval_combo)
        self.interval_combo.currentTextChanged.connect(self.on_interval_changed)
        self.data_source_combo.currentTextChanged.connect(self.update_interval_states)

    def on_interval_changed(self, interval):
        symbol = self.symbol_combo.currentText()
        source = self.data_source_combo.currentText()
        is_crypto = "USDT" in symbol.upper()
        if source == "Historical" and not is_crypto and interval in ["1m", "5m", "15m", "30m"]:
            self.statusBar().showMessage(f"Yahoo Finance doesn't support {interval} for stocks")
            self.interval_combo.setCurrentText("1h")
            return
        self.statusBar().showMessage(f"Loading {interval} data for {symbol}")
        self.load_data()

    def update_interval_states(self):
        source = self.data_source_combo.currentText()
        for i in range(self.interval_combo.count()):
            interval = self.interval_combo.itemText(i)
            enabled = not (source == "Historical" and interval in ['1m', '5m', '15m', '30m'])
            self.interval_combo.model().item(i).setEnabled(enabled)
        if not self.interval_combo.currentData(Qt.UserRole + 1):
            self.interval_combo.setCurrentText('1d')

    def setup_days_input(self):
        days_label = QLabel("Days to Plot:")
        self.days_input = QLineEdit("365")
        self.days_input.setFixedWidth(60)
        self.days_input.setValidator(QIntValidator(1, 365*5))
        days_layout = QHBoxLayout()
        days_layout.addWidget(days_label)
        days_layout.addWidget(self.days_input)
        self.control_layout.insertLayout(self.control_layout.count() - 4, days_layout)
        self.days_input.returnPressed.connect(self.load_data)

    def setup_cash_input(self):
        cash_label = QLabel("Initial Cash:")
        self.cash_input = QLineEdit("100000")
        self.cash_input.setFixedWidth(100)
        self.cash_input.setValidator(QIntValidator(100, 100000000))
        cash_layout = QHBoxLayout()
        cash_layout.addWidget(cash_label)
        cash_layout.addWidget(self.cash_input)
        self.control_layout.insertLayout(self.control_layout.count() - 4, cash_layout)

    def _run_backtest_logic(self):
        strategy_name = self.strategy_combo.currentText()
        print(f"DEBUG: Strategy name: {strategy_name}")

        if strategy_name == "False":
            self.statusBar().showMessage("No strategy selected!")
            return False
        if not hasattr(self, 'df') or self.df.empty:
            self.statusBar().showMessage("Please load data before backtest.")
            return False
        if strategy_name == "FinRL Strategy":
            self.statusBar().showMessage("FinRL strategies not supported for single-symbol backtest.")
            return False
        try:
            if strategy_name == "LSTM Predictor":
                symbol = self.symbol_combo.currentText()
                strategy = self.strategy_manager.get_strategy(strategy_name, ticker=symbol, sequence_length=60)
            else:
                strategy = self.strategy_manager.get_strategy(strategy_name)
            results = self.strategy_manager.run_backtest(strategy, self.df, float(self.cash_input.text()))
            # Check what the strategy manager returns
            print(f"DEBUG: Strategy type: {type(strategy)}")
            print(f"DEBUG: Strategy attributes: {dir(strategy)}")
            return results
        except Exception as e:
            self.statusBar().showMessage(f"Backtest error: {str(e)}")
            print(f"Backtest error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False

    def _run_backtest_logic_with_broker(self, broker):
        """Run backtest logic but execute trades through the broker"""
        strategy_name = self.strategy_combo.currentText()
        print(f"DEBUG: Strategy name: {strategy_name}")
        
        if strategy_name == "False":
            self.statusBar().showMessage("No strategy selected!")
            return False
            
        if not hasattr(self, 'df') or self.df.empty:
            self.statusBar().showMessage("Please load data before backtest.")
            return False
            
        try:
            # Get strategy - HANDLE BACKTRADER STRATEGIES DIFFERENTLY
            if strategy_name == "LSTM Predictor":
                symbol = self.symbol_combo.currentText()
                strategy = self.strategy_manager.get_strategy(strategy_name, ticker=symbol, sequence_length=60)
                # For LSTM, use normal approach
                results = self.strategy_manager.run_backtest(strategy, self.df, float(self.cash_input.text()))
                
            else:
                # For backtrader strategies (MACD/RSI, EMA Crossover, Stochastic)
                # Get the strategy CLASS, not instance
                strategy_class = self.strategy_manager.get_strategy(strategy_name)
                print(f"🔍 Strategy class: {strategy_class}")
                
                # Run backtest using backtrader's Cerebro
                results = self._run_backtrader_backtest(strategy_class, self.df, float(self.cash_input.text()), broker)
            
            # # EXECUTE TRADES THROUGH BROKER (only if we have signals)
            # if results and 'signals' in results:
            #     signals = results.get('signals', [])
            #     print(f"🔍 Executing {len(signals)} signals through broker")
                
            #     for signal in signals:
            #         if signal['type'] == 'buy':
            #             broker.submit_order(
            #                 symbol=self.symbol_combo.currentText(),
            #                 qty=signal.get('qty', 1),
            #                 side='buy',
            #                 order_type='market'
            #             )
            #         elif signal['type'] == 'sell':
            #             broker.submit_order(
            #                 symbol=self.symbol_combo.currentText(),
            #                 qty=signal.get('qty', 1),
            #                 side='sell', 
            #                 order_type='market'
            #             )
            # else:
            #     print("🔍 No signals found in results")
                
            return results
            
        except Exception as e:
            self.statusBar().showMessage(f"Backtest error: {str(e)}")
            print(f"❌ Backtest error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False

    def _run_backtrader_backtest(self, strategy_class, data, initial_cash, broker):
        """Run backtest for backtrader strategies WITH analytics"""
        import backtrader as bt
        import numpy as np
        
        print("🔍 Running backtrader backtest with analytics...")
        
        # Create Cerebro engine
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(initial_cash)
        
        # ADD ANALYZERS
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        
        # Add strategy
        cerebro.addstrategy(strategy_class)
        
        # Convert DataFrame to backtrader data format
        bt_data = bt.feeds.PandasData(dataname=data)
        cerebro.adddata(bt_data)
        
        # Run backtest
        print("🔍 Starting cerebro run with analyzers...")
        strat_list = cerebro.run()
        print(f"🔍 Cerebro completed. Strategies returned: {len(strat_list)}")
        
        # Extract results AND execute trades in broker
        if strat_list:
            strategy_instance = strat_list[0]
            return self._extract_enhanced_backtrader_results(strategy_instance, initial_cash, broker)
        else:
            return {'signals': [], 'summary': {}}

    def _extract_enhanced_backtrader_results(self, strategy_instance, initial_cash, broker):
        """Extract results with backtrader's built-in analytics and enhanced summary"""
        import numpy as np
        
        final_value = strategy_instance.broker.getvalue()
        total_pnl = final_value - initial_cash
        
        # EXTRACT ANALYZER RESULTS
        sharpe_ratio = 0.0
        total_closed_trades = 0
        winning_trades = 0
        profit_per_trade = []
        
        if hasattr(strategy_instance, 'analyzers'):
            # Sharpe Ratio
            if hasattr(strategy_instance.analyzers.sharpe, 'get_analysis'):
                try:
                    sharpe_analysis = strategy_instance.analyzers.sharpe.get_analysis()
                    sharpe_ratio = sharpe_analysis.get('sharperatio', 0.0)
                    sharpe_ratio = float(sharpe_ratio) if sharpe_ratio is not None else 0.0
                    print(f"🔍 Sharpe ratio: {sharpe_ratio}")
                except Exception as e:
                    print(f"❌ Sharpe analyzer error: {e}")
                    sharpe_ratio = 0.0
            
            # Trade Analysis
            if hasattr(strategy_instance.analyzers.trades, 'get_analysis'):
                try:
                    trade_analysis = strategy_instance.analyzers.trades.get_analysis()
                    print(f"🔍 Trade analysis available: {bool(trade_analysis)}")
                    
                    if trade_analysis:
                        # Get total closed trades
                        if 'total' in trade_analysis and hasattr(trade_analysis['total'], 'closed'):
                            total_closed_trades = trade_analysis['total'].closed
                        elif 'total' in trade_analysis and isinstance(trade_analysis['total'], dict):
                            total_closed_trades = trade_analysis['total'].get('closed', 0)
                        
                        # Get winning trades
                        if 'won' in trade_analysis and hasattr(trade_analysis['won'], 'total'):
                            winning_trades = trade_analysis['won'].total
                        elif 'won' in trade_analysis and isinstance(trade_analysis['won'], dict):
                            winning_trades = trade_analysis['won'].get('total', 0)
                        
                        profit_per_trade = self._extract_pnl_from_trade_analysis(trade_analysis)
                        
                    print(f"🔍 From analyzer: {total_closed_trades} closed trades, {winning_trades} winning trades")
                        
                except Exception as e:
                    print(f"❌ Trade analyzer error: {e}")
        
        # METHOD 1: Extract from cerebro's trade records
        if not profit_per_trade:
            profit_per_trade = self._extract_trades_from_cerebro(strategy_instance)
            print(f"🔍 From cerebro: {len(profit_per_trade)} trades")
        
        # METHOD 2: Use strategy's signal-based P&L calculation
        if not profit_per_trade and hasattr(strategy_instance, 'signals'):
            profit_per_trade = self._calculate_pnl_from_signals(strategy_instance)
            print(f"🔍 From signals: {len(profit_per_trade)} trades")
        
        # METHOD 3: Use cumulative P&L from strategy
        if not profit_per_trade and hasattr(strategy_instance, 'cumulative_pnl'):
            cumulative = strategy_instance.cumulative_pnl
            cumulative_float = float(cumulative) if cumulative is not None else 0.0
            if cumulative_float != 0:
                profit_per_trade = [cumulative_float]
                if total_closed_trades == 0:
                    total_closed_trades = 1
                    winning_trades = 1 if cumulative_float > 0 else 0
                print(f"🔍 Using cumulative P&L: ${cumulative_float:+,.2f}")
        
        # METHOD 4: Use total P&L as single trade
        if not profit_per_trade and total_pnl != 0:
            profit_per_trade = [total_pnl]
            if total_closed_trades == 0:
                total_closed_trades = 1
                winning_trades = 1 if total_pnl > 0 else 0
            print(f"🔍 Using total P&L as single trade: ${total_pnl:+,.2f}")
        
        # SANITIZE profit_per_trade - CONVERT ALL VALUES TO FLOATS
        sanitized_profit_per_trade = []
        for pnl in profit_per_trade:
            try:
                if pnl is None:
                    sanitized_pnl = 0.0
                elif isinstance(pnl, str):
                    clean_str = pnl.replace('$', '').replace(',', '').strip()
                    sanitized_pnl = float(clean_str) if clean_str else 0.0
                else:
                    sanitized_pnl = float(pnl)
                sanitized_profit_per_trade.append(sanitized_pnl)
            except (ValueError, TypeError) as e:
                print(f"⚠️ Could not convert P&L value '{pnl}' to float: {e}")
                sanitized_profit_per_trade.append(0.0)
        
        print(f"🔍 Sanitized P&L values: {[f'${x:+.2f}' for x in sanitized_profit_per_trade]}")
        
        # Calculate win rate
        if total_closed_trades == 0:
            total_closed_trades = len(sanitized_profit_per_trade)
            winning_trades = len([p for p in sanitized_profit_per_trade if p > 0])
        
        win_rate = (winning_trades / total_closed_trades * 100) if total_closed_trades > 0 else 0
        
        # CALCULATE METRICS WITH SANITIZED NUMBERS
        try:
            avg_profit = np.mean(sanitized_profit_per_trade) if sanitized_profit_per_trade else 0
            median_profit = np.median(sanitized_profit_per_trade) if sanitized_profit_per_trade else 0
            largest_win = max(sanitized_profit_per_trade) if sanitized_profit_per_trade else 0
            largest_loss = min(sanitized_profit_per_trade) if sanitized_profit_per_trade else 0
        except Exception as e:
            print(f"❌ Error calculating metrics: {e}")
            avg_profit = total_pnl
            median_profit = total_pnl
            largest_win = total_pnl if total_pnl > 0 else 0
            largest_loss = total_pnl if total_pnl < 0 else 0
        
        # Create enhanced summary
        summary = {
            'Final Value': float(final_value),
            'P&L': float(total_pnl),
            'Return %': float((total_pnl / initial_cash) * 100),
            'Sharpe Ratio': float(sharpe_ratio),
            'Total Trades': int(total_closed_trades),
            'Winning Trades': int(winning_trades),
            'Win Rate': f"{win_rate:.2f}%",
            'Average Profit per Trade': float(avg_profit),
            'Median Profit per Trade': float(median_profit),
            'Largest Win': float(largest_win),
            'Largest Loss': float(largest_loss),
        }
        
        print(f"🔍 Enhanced results: {total_closed_trades} trades, Win Rate: {win_rate:.1f}%, Sharpe: {sharpe_ratio:.2f}")
        
        # Extract and execute signals
        signals = []
        if hasattr(strategy_instance, 'signals'):
            signals = strategy_instance.signals
            print(f"🔍 Found {len(signals)} trading signals")
        
        # Execute trades in broker
        self._execute_historical_trades_in_broker(signals, broker)
        
        return {
            'signals': signals,
            'summary': summary,
            'profit_per_trade': sanitized_profit_per_trade
        }

    def _extract_trades_from_cerebro(self, strategy_instance):
        """Extract trades from backtrader's cerebro instance"""
        profit_per_trade = []
        
        try:
            # Method 1: Check cerebro's trade records
            if hasattr(strategy_instance, 'env') and hasattr(strategy_instance.env, 'trades'):
                for trade in strategy_instance.env.trades:
                    if hasattr(trade, 'pnl') and hasattr(trade, 'isclosed') and trade.isclosed:
                        profit_per_trade.append(trade.pnl)
                print(f"🔍 Found {len(profit_per_trade)} trades in cerebro.env.trades")
            
            # Method 2: Check strategy's trade records
            elif hasattr(strategy_instance, 'trades'):
                for trade in strategy_instance.trades:
                    if hasattr(trade, 'pnl') and hasattr(trade, 'isclosed') and trade.isclosed:
                        profit_per_trade.append(trade.pnl)
                print(f"🔍 Found {len(profit_per_trade)} trades in strategy.trades")
                
        except Exception as e:
            print(f"❌ Trade extraction from cerebro error: {e}")
        
        return profit_per_trade

    def _extract_pnl_from_trade_analysis(self, trade_analysis):
        """Safely extract P&L values from trade analysis"""
        profit_per_trade = []
        
        try:
            # Method 1: Try to get from pnl.net
            if 'pnl' in trade_analysis and 'net' in trade_analysis['pnl']:
                pnl_data = trade_analysis['pnl']['net']
                if hasattr(pnl_data, '__iter__') and not isinstance(pnl_data, str):
                    for pnl in pnl_data:
                        try:
                            profit_per_trade.append(float(pnl))
                        except (ValueError, TypeError):
                            pass
                else:
                    try:
                        profit_per_trade.append(float(pnl_data))
                    except (ValueError, TypeError):
                        pass
            
            # Method 2: Try to get from individual trades
            if not profit_per_trade and 'trades' in trade_analysis:
                trades_list = trade_analysis['trades']
                if hasattr(trades_list, '__iter__'):
                    for trade in trades_list:
                        if hasattr(trade, 'pnl'):
                            try:
                                profit_per_trade.append(float(trade.pnl))
                            except (ValueError, TypeError):
                                pass
            
            print(f"🔍 Extracted {len(profit_per_trade)} P&L values from trade analysis")
            
        except Exception as e:
            print(f"❌ P&L extraction from trade analysis error: {e}")
        
        return profit_per_trade

    def _calculate_pnl_from_signals(self, strategy_instance):
        """Calculate P&L from buy/sell signals with type safety"""
        profit_per_trade = []
        
        try:
            signals = strategy_instance.signals
            buy_signals = [s for s in signals if s['type'] == 'buy']
            sell_signals = [s for s in signals if s['type'] == 'sell']
            
            # Match buy and sell signals to calculate P&L
            min_trades = min(len(buy_signals), len(sell_signals))
            for i in range(min_trades):
                buy = buy_signals[i]
                sell = sell_signals[i]
                
                buy_price = float(buy['price']) if buy['price'] is not None else 0.0
                sell_price = float(sell['price']) if sell['price'] is not None else 0.0
                qty = float(buy.get('qty', 1)) if buy.get('qty') is not None else 1.0
                
                trade_pnl = (sell_price - buy_price) * qty
                profit_per_trade.append(trade_pnl)
            
            print(f"🔍 Calculated {len(profit_per_trade)} trades from signal matching")
            
        except Exception as e:
            print(f"❌ P&L calculation from signals error: {e}")
        
        return profit_per_trade

    def _execute_historical_trades_in_broker(self, signals, broker):
        """Execute trades in broker - WITH FORCE CLOSE"""
        print(f"🔍 Executing {len(signals)} historical trades in broker")
        
        executed_count = 0
        current_position = 0
        
        # Process signals sequentially
        i = 0
        while i < len(signals):
            signal = signals[i]
            
            try:
                execution_price = signal.get('price')
                if execution_price is None:
                    i += 1
                    continue
                
                # Use consistent position sizing
                available_cash = broker.balance
                risk_percent = 0.1  # Same as your strategy's risk_per_trade
                position_value = available_cash * risk_percent
                qty = position_value / execution_price if execution_price > 0 else 0
                
                if qty <= 0.0001:
                    i += 1
                    continue
                
                print(f"🔍 Processing signal {i}: {signal['type']} @ ${execution_price:.2f}, qty={qty:.6f}")
                
                if signal['type'] == 'sell_short' and current_position == 0:
                    # ENTER SHORT
                    broker.submit_order(
                        symbol=self.symbol_combo.currentText(),
                        qty=qty,
                        side='sell',
                        order_type='market',
                        execution_price=execution_price
                    )
                    current_position = -qty
                    executed_count += 1
                    print(f"📉 ENTER SHORT: {qty:.6f} @ ${execution_price:.2f}")
                    
                elif signal['type'] == 'buy' and current_position < 0:
                    # EXIT SHORT (buy to cover)
                    broker.submit_order(
                        symbol=self.symbol_combo.currentText(),
                        qty=abs(current_position),  # Cover entire short position
                        side='buy',
                        order_type='market',
                        execution_price=execution_price
                    )
                    current_position = 0
                    executed_count += 1
                    print(f"🔚 EXIT SHORT: {abs(current_position):.6f} @ ${execution_price:.2f}")
                    
                elif signal['type'] == 'buy' and current_position == 0:
                    # ENTER LONG
                    broker.submit_order(
                        symbol=self.symbol_combo.currentText(),
                        qty=qty,
                        side='buy',
                        order_type='market',
                        execution_price=execution_price
                    )
                    current_position = qty
                    executed_count += 1
                    print(f"📈 ENTER LONG: {qty:.6f} @ ${execution_price:.2f}")
                    
                elif signal['type'] == 'sell' and current_position > 0:
                    # EXIT LONG
                    broker.submit_order(
                        symbol=self.symbol_combo.currentText(),
                        qty=current_position,  # Sell entire long position
                        side='sell',
                        order_type='market',
                        execution_price=execution_price
                    )
                    current_position = 0
                    executed_count += 1
                    print(f"🔚 EXIT LONG: {current_position:.6f} @ ${execution_price:.2f}")
                
                elif signal['type'] == 'sell_short' and current_position > 0:
                    # We have a long position but got a short signal - CLOSE LONG FIRST
                    print(f"🔄 Switching from LONG to SHORT: Closing long position first")
                    broker.submit_order(
                        symbol=self.symbol_combo.currentText(),
                        qty=current_position,
                        side='sell',
                        order_type='market',
                        execution_price=execution_price
                    )
                    current_position = 0
                    executed_count += 1
                    print(f"🔚 EXIT LONG (switch): {current_position:.6f} @ ${execution_price:.2f}")
                    
                    # Then ENTER SHORT with the same signal
                    broker.submit_order(
                        symbol=self.symbol_combo.currentText(),
                        qty=qty,
                        side='sell',
                        order_type='market',
                        execution_price=execution_price
                    )
                    current_position = -qty
                    executed_count += 1
                    print(f"📉 ENTER SHORT (switch): {qty:.6f} @ ${execution_price:.2f}")
                
                i += 1
                    
            except Exception as e:
                print(f"❌ Error executing signal: {e}")
                i += 1
        
        # DEBUG: Check position synchronization before force close
        print(f"🔍 PRE-FORCE CLOSE DEBUG:")
        print(f"   current_position tracking: {current_position:.6f}")
        
        broker_position_qty = 0
        if hasattr(broker, 'positions') and self.symbol_combo.currentText() in broker.positions:
            broker_position_qty = broker.positions[self.symbol_combo.currentText()].qty
            print(f"   broker actual position: {broker_position_qty:.6f}")
        else:
            print(f"   broker actual position: No position found")
        
        # Check if positions are synchronized
        if abs(current_position - broker_position_qty) > 0.000001:
            print(f"⚠️  POSITION DESYNC: tracking={current_position:.6f}, broker={broker_position_qty:.6f}")
        
        # CRITICAL: FORCE CLOSE ANY REMAINING POSITIONS
        self._force_close_all_positions(broker, current_position)
        
        print(f"🔍 Successfully executed {executed_count} trades")

    def _force_close_all_positions(self, broker, tracked_position=0):
        """Force close all remaining positions at current market price with enhanced debugging"""
        try:
            current_symbol = self.symbol_combo.currentText()
            current_price = self.df['Close'].iloc[-1] if hasattr(self, 'df') and not self.df.empty else 0
            
            print(f"🔍 FORCE CLOSE ANALYSIS:")
            print(f"   Tracked position: {tracked_position:.6f}")
            print(f"   Current price: ${current_price:.2f}")
            
            # Method 1: Close based on tracked position (your original logic)
            if abs(tracked_position) > 0.000001:
                print(f"🔄 Closing based on tracked position...")
                if tracked_position > 0:  # Long position
                    broker.submit_order(
                        symbol=current_symbol,
                        qty=tracked_position,
                        side='sell',
                        order_type='market',
                        execution_price=current_price
                    )
                    print(f"🔄 FORCE CLOSE LONG (tracked): {tracked_position:.6f} @ ${current_price:.2f}")
                else:  # Short position
                    broker.submit_order(
                        symbol=current_symbol,
                        qty=abs(tracked_position),
                        side='buy',
                        order_type='market',
                        execution_price=current_price
                    )
                    print(f"🔄 FORCE CLOSE SHORT (tracked): {abs(tracked_position):.6f} @ ${current_price:.2f}")
            
            # Method 2: Close based on actual broker positions (backup safety)
            if hasattr(broker, 'positions'):
                for symbol, position in list(broker.positions.items()):
                    if abs(position.qty) > 0.000001:  # Has position
                        print(f"🔄 Closing based on broker position...")
                        if position.qty > 0:  # Long position
                            broker.submit_order(
                                symbol=symbol,
                                qty=position.qty,
                                side='sell',
                                order_type='market',
                                execution_price=current_price
                            )
                            print(f"🔄 FORCE CLOSE LONG (broker): {position.qty:.6f} @ ${current_price:.2f}")
                        else:  # Short position
                            broker.submit_order(
                                symbol=symbol,
                                qty=abs(position.qty),
                                side='buy',
                                order_type='market',
                                execution_price=current_price
                            )
                            print(f"🔄 FORCE CLOSE SHORT (broker): {abs(position.qty):.6f} @ ${current_price:.2f}")
            
            # Final verification
            final_broker_position = 0
            if hasattr(broker, 'positions') and current_symbol in broker.positions:
                final_broker_position = broker.positions[current_symbol].qty
            
            if abs(final_broker_position) < 0.000001:
                print(f"✅ FORCE CLOSE SUCCESS: All positions closed")
            else:
                print(f"❌ FORCE CLOSE WARNING: Position still exists: {final_broker_position:.6f}")
                
        except Exception as e:
            print(f"❌ Error force closing positions: {e}")

    
    def run_backtest(self):
        # For backtesting, always use simulator
        try:
            backtest_broker = self.broker_manager.get_broker("Simulator")
            
            if not backtest_broker:
                self.statusBar().showMessage("❌ Simulator not available")
                return
            
            # RESET BROKER COMPLETELY BEFORE BACKTEST
            initial_cash = float(self.cash_input.text())
            
            # Create a FRESH broker instance to avoid residual orders
            backtest_broker = self.broker_manager.get_broker("Simulator")
            backtest_broker.balance = initial_cash
            backtest_broker.initial_balance = initial_cash
            backtest_broker.portfolio_value = initial_cash
            backtest_broker.positions = {}
            backtest_broker.order_history = []  # Clear ALL previous orders
            backtest_broker.closed_positions = []  # Clear closed positions
            backtest_broker.orders = {}  # Clear pending orders
            
            print(f"🔍 Broker reset: Balance=${backtest_broker.balance:.2f}, Orders={len(backtest_broker.order_history)}")

            # Run backtest WITH enhanced analytics
            results = self._run_backtest_logic_with_broker(backtest_broker)
            if results is False:
                return
            
            # Get enhanced summary
            summary = results.get('summary', {})
            final_value = summary.get('Final Value', 0)
            total_pnl = summary.get('P&L', 0)
            sharpe_ratio = summary.get('Sharpe Ratio', 0)
            win_rate = summary.get('Win Rate', '0%')
            total_trades = summary.get('Total Trades', 0)
            
            # Create comprehensive status message
            sharpe_str = f"{sharpe_ratio:.2f}" if isinstance(sharpe_ratio, (int, float)) else "N/A"
            
            msg = (f"Backtest complete | "
                f"Final: ${final_value:,.2f} | "
                f"P&L: ${total_pnl:+,.2f} | "
                f"Sharpe: {sharpe_str} | "
                f"Win Rate: {win_rate} | "
                f"Trades: {total_trades}")
            
            self.statusBar().showMessage(msg)
            
            # Plot trading signals
            self.plot_signals(results.get('signals', []))
            
            # Show broker positions in UI
            self.current_broker = backtest_broker
            self.refresh_account_info()
            
            # Print detailed results to console
            print("\n" + "="*60)
            print("📊 BACKTEST RESULTS SUMMARY")
            print("="*60)
            for key, value in summary.items():
                print(f"  {key}: {value}")
            print("="*60)

        except Exception as e:
            self.statusBar().showMessage(f"Backtest error: {str(e)}")
            print(f"Backtest error: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def update_positions_display(self):
        """Enhanced positions display with backtest metrics"""
        if not self.current_broker:
            return
            
        try:
            positions_text = "📊 BACKTEST RESULTS\n"
            positions_text += "━" * 50 + "\n"
            
            # Broker positions
            has_positions = False
            if hasattr(self.current_broker, 'positions'):
                for symbol, position in self.current_broker.positions.items():
                    if position.qty != 0:
                        has_positions = True
                        pnl_color = "🟢" if position.pnl >= 0 else "🔴"
                        positions_text += (
                            f"{pnl_color} {symbol}: {position.qty:+.2f} @ ${position.avg_price:.2f} | "
                            f"PnL: ${position.pnl:+,.2f}\n"
                        )
            
            if not has_positions:
                positions_text += "No open positions\n"
                
            # Add strategy performance summary
            positions_text += "\n📈 STRATEGY PERFORMANCE\n"
            positions_text += "━" * 50 + "\n"
            
            # You could store the last backtest results and display them here
            # positions_text += f"Sharpe Ratio: {self.last_sharpe_ratio}\n"
            # positions_text += f"Total Trades: {self.last_trade_count}\n"
            # positions_text += f"Win Rate: {self.last_win_rate}%\n"
                
            self.positions_text.setPlainText(positions_text)
            
        except Exception as e:
            self.positions_text.setPlainText(f"Error loading results: {str(e)}")

    def plot_signals(self, signals):
        buy_signals = [s for s in signals if s['type'] == 'buy']
        sell_signals = [s for s in signals if s['type'] == 'sell']
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

    def setup_broker_ui(self):
        """Add broker-specific controls to your existing UI"""
        # Broker control group - add this to your existing control_layout
        broker_group = QGroupBox("📊 Broker Controls")
        broker_layout = QVBoxLayout()
        
        # Account info display
        account_layout = QHBoxLayout()
        account_layout.addWidget(QLabel("Account:"))
        self.account_label = QLabel("Simulator - $100,000.00")
        self.account_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
        account_layout.addWidget(self.account_label)
        account_layout.addStretch()
        
        # Refresh account button
        self.refresh_account_btn = QPushButton("🔄 Refresh")
        self.refresh_account_btn.clicked.connect(self.refresh_account_info)
        account_layout.addWidget(self.refresh_account_btn)
        broker_layout.addLayout(account_layout)
        
        # Order controls
        order_layout = QHBoxLayout()
        
        # Order type
        order_layout.addWidget(QLabel("Order Type:"))
        self.order_type_combo = QComboBox()
        self.order_type_combo.addItems(["Market", "Limit", "Stop"])
        order_layout.addWidget(self.order_type_combo)
        
        # Quantity
        order_layout.addWidget(QLabel("Qty:"))
        self.order_qty_input = QLineEdit("1")
        self.order_qty_input.setFixedWidth(60)
        order_layout.addWidget(self.order_qty_input)
        
        # Limit price (visible for limit orders)
        order_layout.addWidget(QLabel("Limit Price:"))
        self.limit_price_input = QLineEdit()
        self.limit_price_input.setFixedWidth(80)
        self.limit_price_input.setPlaceholderText("Optional")
        order_layout.addWidget(self.limit_price_input)
        
        # Buy/Sell buttons
        self.buy_btn = QPushButton("🟢 BUY")
        self.buy_btn.clicked.connect(lambda: self.place_order("buy"))
        self.buy_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                font-weight: bold;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        order_layout.addWidget(self.buy_btn)
        
        self.sell_btn = QPushButton("🔴 SELL") 
        self.sell_btn.clicked.connect(lambda: self.place_order("sell"))
        self.sell_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-weight: bold;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        order_layout.addWidget(self.sell_btn)
        
        broker_layout.addLayout(order_layout)
        
        # Positions display
        self.positions_text = QTextEdit()
        self.positions_text.setMaximumHeight(120)
        self.positions_text.setReadOnly(True)
        self.positions_text.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #ffffff;
                font-family: 'Courier New', monospace;
                font-size: 10px;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        broker_layout.addWidget(self.positions_text)
        
        broker_group.setLayout(broker_layout)
        
        # Add to your existing main layout (adjust position as needed)
        self.main_layout.insertWidget(2, broker_group)  # Insert after control panel
        
        # Connect order type changes
        self.order_type_combo.currentTextChanged.connect(self.on_order_type_changed)
        
        # Initial UI state
        self.on_order_type_changed("Market")
        self.refresh_account_info()

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
                
                print(f"🔍 BROKER DEBUG:")
                print(f"   Balance: ${account_info.get('balance', 0):,.2f}")
                print(f"   Portfolio Value: ${account_info.get('portfolio_value', 0):,.2f}")
                print(f"   Initial Balance: ${account_info.get('initial_balance', 0):,.2f}")
                print(f"   P&L: ${account_info.get('pnl', 0):,.2f}")
                
                # Check positions
                if hasattr(self.current_broker, 'positions'):
                    print(f"   Positions: {self.current_broker.positions}")
                    for symbol, position in self.current_broker.positions.items():
                        print(f"     {symbol}: {position.qty} @ ${position.avg_price:.2f}")
                
                # Check order history WITH PRICES
                if hasattr(self.current_broker, 'order_history'):
                    print(f"   Order History: {len(self.current_broker.order_history)} orders")
                    for i, order in enumerate(self.current_broker.order_history[-10:]):  # Last 10 orders
                        price_display = f"@ ${order.filled_avg_price:.2f}" if order.filled_avg_price else "@ MARKET"
                        print(f"     {i+1:2d}. {order.side.value:4} {order.qty:4} {order.symbol} {price_display}")

                # Update account label
                broker_name = self.broker_combo.currentText()
                balance = account_info.get('balance', 0)
                pnl = account_info.get('pnl', 0)
                pnl_color = "#2ecc71" if pnl >= 0 else "#e74c3c"
                
                self.account_label.setText(
                    f"{broker_name} - ${balance:,.2f} | P&L: "
                    f"<span style='color: {pnl_color}'>${pnl:+,.2f}</span>"
                )
                
                # Update positions display
                self.update_positions_display()
                
        except Exception as e:
            print(f"Error refreshing account: {str(e)}")

    def update_positions_display(self):
        """Update positions display"""
        if not self.current_broker:
            return
            
        try:
            positions_text = "📊 CURRENT POSITIONS:\n"
            positions_text += "━" * 40 + "\n"
            
            has_positions = False
            # Access positions from the simulated broker
            if hasattr(self.current_broker, 'positions'):
                for symbol, position in self.current_broker.positions.items():
                    if position.qty != 0:  # Only show non-zero positions
                        has_positions = True
                        pnl_color = "🟢" if position.pnl >= 0 else "🔴"
                        positions_text += (
                            f"{pnl_color} {symbol}: {position.qty:+.2f} @ ${position.avg_price:.2f} | "
                            f"PnL: ${position.pnl:+,.2f}\n"
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
            
            # Get optional price parameters
            limit_price = None
            stop_price = None
            
            if order_type == "limit" and self.limit_price_input.text():
                limit_price = float(self.limit_price_input.text())
            elif order_type == "stop" and self.limit_price_input.text():
                stop_price = float(self.limit_price_input.text())
            
            # Ensure we have a broker instance
            if not self.current_broker:
                self.current_broker = self.broker_manager.get_broker(self.broker_combo.currentText())
            
            # Submit order
            order = self.current_broker.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price
            )
            
            # Show order result
            if order.status.value == "filled":
                self.statusBar().showMessage(
                    f"✅ {side.upper()} order filled for {qty} {symbol} @ ${order.filled_avg_price:.2f}"
                )
            elif order.status.value == "pending":
                self.statusBar().showMessage(
                    f"⏳ {side.upper()} order pending for {qty} {symbol}"
                )
            else:
                self.statusBar().showMessage(
                    f"❌ Order {order.status.value}: {qty} {symbol}"
                )
            
            # Refresh account info
            self.refresh_account_info()
            
        except Exception as e:
            self.statusBar().showMessage(f"❌ Order error: {str(e)}")

    def start_trading(self):
        broker_name = self.broker_combo.currentText()
        strategy_name = self.strategy_combo.currentText()

        if strategy_name == "False":
            self.statusBar().showMessage("No strategy selected!")
            return
        try:
            # get broker instance
            broker = self.broker_manager.get_broker(broker_name)
            self.current_broker = broker_name

            if not self.current_broker:
                self.statusBar().showMessage(f"Broker {broker_name} not configured!")
                return

            # SET INITIAL CASH FROM USER INPUT
            initial_cash = float(self.cash_input.text())
            if hasattr(self.current_broker, 'balance'):
                self.current_broker.balance = initial_cash
                self.current_broker.initial_balance = initial_cash
                self.current_broker.portfolio_value = initial_cash
            # Get strategy instance
            if strategy_name == "LSTM Predictor":
                symbol = self.symbol_combo.currentText()
                strategy = self.strategy_manager.get_strategy(strategy_name, ticker=symbol, sequence_length=60)
            else:
                strategy = self.strategy_manager.get_strategy(strategy_name)

            # Update UI for live trading
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
        ts = pd.to_datetime(data['timestamp'], unit='s', utc=True)
        price = data['price']

        if self.realtime_df.empty:
            new_row = pd.DataFrame([{'Open': price, 'High': price, 'Low': price, 'Close': price}], index=[ts])
            self.realtime_df = pd.concat([self.realtime_df, new_row])
        else:
            last_ts = self.realtime_df.index[-1].floor(self.current_interval)
            current_ts = ts.floor(self.current_interval)

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

        # Update existing trace instead of recreating figure
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
            yaxis=dict(title='Price', autorange=True),  # ← Key: autorange=True
            template='plotly_dark'
        )
        self.update_plotly_view()
        # if self.realtime_df.empty:
        #     return

        # self.fig = go.Figure(data=[go.Candlestick(
        #     x=self.realtime_df.index,
        #     open=self.realtime_df['Open'],
        #     high=self.realtime_df['High'],
        #     low=self.realtime_df['Low'],
        #     close=self.realtime_df['Close'],
        #     name='Price'
        # )])
        # self.fig.update_layout(
        #     height=600,
        #     xaxis=dict(type='date', rangeslider_visible=False),
        #     yaxis=dict(title='Price'),
        #     template='plotly_dark'
        # )
        # self.update_plotly_view()

    def stop_realtime_stream(self):
        if self.is_streaming:
            self.data_loader.stop_realtime_stream()
            self.realtime_timer.stop()
            self.is_streaming = False
            self.realtime_df = pd.DataFrame()

    def closeEvent(self, event):
        if self.is_streaming:
            self.stop_realtime_stream()
        super().closeEvent(event)

    def setup_ai_analysis_ui(self):
        """Add AI analysis controls to UI - NO API KEY INPUT NEEDED"""
        # Create a group box for better organization
        ai_group = QGroupBox("🤖 DeepSeek AI Analysis")
        ai_layout = QVBoxLayout()
        
        # Status label showing AI is ready
        ai_status_layout = QHBoxLayout()
        ai_status_layout.addWidget(QLabel("Status:"))
        self.ai_status_label = QLabel("✅ Ready")
        self.ai_status_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
        ai_status_layout.addWidget(self.ai_status_label)
        ai_status_layout.addStretch()
        ai_layout.addLayout(ai_status_layout)
        
        # Analyze Button - NO API KEY INPUT
        self.analyze_btn = QPushButton("Analyze Current Symbol")
        self.analyze_btn.clicked.connect(self.run_ai_analysis)
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:disabled {
                background-color: #7f8c8d;
            }
        """)
        ai_layout.addWidget(self.analyze_btn)

        # Add debug buttons (remove after testing)
        debug_layout = QHBoxLayout()
        
        test_analyzer_btn = QPushButton("Test AI")
        test_analyzer_btn.clicked.connect(self.test_ai_analyzer_directly)
        debug_layout.addWidget(test_analyzer_btn)
        
        test_api_btn = QPushButton("Test API")
        test_api_btn.clicked.connect(self.test_deepseek_connection)
        debug_layout.addWidget(test_api_btn)
        
        ai_layout.addLayout(debug_layout)
        
        # Multi-symbol analysis button
        self.multi_analyze_btn = QPushButton("Analyze All Major Cryptos")
        self.multi_analyze_btn.clicked.connect(self.run_multi_ai_analysis)
        ai_layout.addWidget(self.multi_analyze_btn)
        
        # AI Results Display
        self.ai_results_text = QTextEdit()
        self.ai_results_text.setMaximumHeight(250)
        self.ai_results_text.setReadOnly(True)
        self.ai_results_text.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #00ff00;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        ai_layout.addWidget(self.ai_results_text)
        
        ai_group.setLayout(ai_layout)
        
        # Add to main layout (adjust position as needed in your layout)
        self.main_layout.addWidget(ai_group)
    
    def run_ai_analysis(self):
        """Run AI analysis - with comprehensive debugging"""
        if not self.ai_analyzer:
            self.statusBar().showMessage("❌ AI Analyzer not available")
            return
            
        symbol = self.symbol_combo.currentText()
        interval = self.interval_combo.currentText()
        days = int(self.days_input.text())
        
        print(f"[AI DEBUG] Starting analysis for {symbol}, {interval}, {days} days")
        
        # Show analyzing message
        self.ai_results_text.setPlainText(
            f"🤖 AI ANALYSIS STARTED\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Symbol: {symbol}\n"
            f"Interval: {interval}\n"
            f"Period: {days} days\n"
            f"Status: Initializing...\n\n"
            f"Please wait...\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        
        # Disable button during analysis
        self.analyze_btn.setEnabled(False)
        self.ai_status_label.setText("🔄 Analyzing...")
        self.ai_status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
        
        from threading import Thread
        def analyze_thread():
            try:
                print(f"[AI DEBUG] Thread started")
                
                # Step 1: Load data
                print(f"[AI DEBUG] Step 1: Loading data")
                self._update_ai_display("🔄 Loading market data...")
                
                analysis_data = None
                if (hasattr(self, 'df') and not self.df.empty and 
                    len(self.df) > 10 and 
                    self.symbol_combo.currentText() == symbol):
                    
                    print(f"[AI DEBUG] Using existing data: {len(self.df)} candles")
                    analysis_data = self.df
                else:
                    print(f"[AI DEBUG] Loading fresh data from platform")
                    self._update_ai_display("🌐 Fetching data from exchange...")
                    
                    analysis_data = self.data_loader.load_data(
                        symbol=symbol,
                        source="Historical",
                        live=False,
                        days=days,
                        interval=interval
                    )
                    print(f"[AI DEBUG] Fresh data loaded: {len(analysis_data)} candles")
                
                if analysis_data is None or analysis_data.empty:
                    error_msg = "❌ No data available for analysis"
                    print(f"[AI DEBUG] {error_msg}")
                    self.display_ai_results({'error': error_msg})
                    return
                
                # Step 2: Run AI analysis
                print(f"[AI DEBUG] Step 2: Calling AI analyzer")
                self._update_ai_display("🤖 Analyzing with DeepSeek AI...")
                
                # Test if analyzer is working with a simple call first
                print(f"[AI DEBUG] Testing AI analyzer...")
                try:
                    # Try a simple analysis first
                    test_result = self.ai_analyzer.analyze_crypto_data(symbol, analysis_data)
                    print(f"[AI DEBUG] AI analysis completed successfully")
                    print(f"[AI DEBUG] Result keys: {test_result.keys() if isinstance(test_result, dict) else 'Not a dict'}")
                    
                    self.display_ai_results(test_result)
                    
                except Exception as ai_error:
                    print(f"[AI DEBUG] AI analysis failed: {ai_error}")
                    self.display_ai_results({'error': f"AI analysis error: {str(ai_error)}"})
                
            except Exception as e:
                print(f"[AI DEBUG] Thread error: {e}")
                import traceback
                print(f"[AI DEBUG] Traceback: {traceback.format_exc()}")
                
                error_msg = f"❌ Analysis failed: {str(e)}"
                self.display_ai_results({'error': error_msg})
            finally:
                print(f"[AI DEBUG] Thread completed, re-enabling UI")
                # Re-enable button
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(0, lambda: self.analyze_btn.setEnabled(True))
                QTimer.singleShot(0, lambda: self.ai_status_label.setText("✅ Ready"))
                QTimer.singleShot(0, lambda: self.ai_status_label.setStyleSheet("color: #2ecc71; font-weight: bold;"))
        
        print(f"[AI DEBUG] Starting analysis thread")
        Thread(target=analyze_thread, daemon=True).start()
        print(f"[AI DEBUG] Thread started successfully")

    def run_multi_ai_analysis(self):
        """Run AI analysis on all major crypto symbols - direct data loading"""
        if not self.ai_analyzer:
            self.statusBar().showMessage("❌ AI Analyzer not available")
            return
        
        crypto_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "BNBUSDT"]
        interval = self.interval_combo.currentText()
        days = int(self.days_input.text())
        
        self.ai_results_text.setPlainText(
            f"🤖 Starting Multi-Crypto AI Analysis...\n"
            f"📈 Analyzing {len(crypto_symbols)} cryptocurrencies:\n"
            f"   {', '.join(crypto_symbols)}\n"
            f"   Interval: {interval}, Period: {days} days\n\n"
            f"⏳ This may take 1-2 minutes...\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )
        
        self.multi_analyze_btn.setEnabled(False)
        self.ai_status_label.setText("🔄 Multi-Analysis...")
        self.ai_status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
        
        from threading import Thread
        def multi_analyze_thread():
            try:
                results = {}
                for i, symbol in enumerate(crypto_symbols):
                    try:
                        # Update progress in UI
                        progress_msg = f"📥 Loading {interval} data for {symbol}... ({i+1}/{len(crypto_symbols)})"
                        self.debug_ai_analysis(progress_msg)
                        
                        # Load data directly from platform
                        df = self.data_loader.load_data(
                            symbol=symbol,
                            source="Historical",
                            live=False,
                            days=days,
                            interval=interval
                        )
                        
                        if df.empty:
                            results[symbol] = {'error': f"No data available for {symbol}"}
                            self.debug_ai_analysis(f"❌ No data for {symbol}")
                        else:
                            # Run AI analysis
                            analysis_msg = f"🤖 Analyzing {symbol}... ({i+1}/{len(crypto_symbols)})"
                            self.debug_ai_analysis(analysis_msg)
                            
                            analysis = self.ai_analyzer.analyze_crypto_data(symbol, df)
                            results[symbol] = analysis
                            self.debug_ai_analysis(f"✅ Analysis complete for {symbol}")
                            
                            # Update results progressively
                            self.update_multi_analysis_results(results)
                            
                    except Exception as e:
                        error_msg = f"❌ Failed to analyze {symbol}: {str(e)}"
                        self.debug_ai_analysis(error_msg)
                        results[symbol] = {'error': error_msg}
                        self.update_multi_analysis_results(results)
                
                self.debug_ai_analysis("✅ Multi-crypto analysis completed")
                
            except Exception as e:
                error_msg = f"❌ Multi-analysis failed: {str(e)}"
                self.debug_ai_analysis(error_msg)
                self.display_ai_results({'error': error_msg})
            finally:
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(0, lambda: self.multi_analyze_btn.setEnabled(True))
                QTimer.singleShot(0, lambda: self.ai_status_label.setText("✅ Ready"))
                QTimer.singleShot(0, lambda: self.ai_status_label.setStyleSheet("color: #2ecc71; font-weight: bold;"))
        
        Thread(target=multi_analyze_thread, daemon=True).start()

    def update_multi_analysis_results(self, results: dict):
        """Update multi-analysis results progressively"""
        result_text = "🤖 Multi-Crypto AI Analysis Results\n"
        result_text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        for symbol, analysis in results.items():
            if 'error' in analysis:
                result_text += f"❌ {symbol}: {analysis['error']}\n\n"
            else:
                rec = analysis.get('recommendation', {})
                action = rec.get('action', 'HOLD')
                confidence = rec.get('confidence', 50)
                current_price = analysis.get('current_price', 0)
                
                # Color code the action
                if action == 'STRONG_BUY':
                    action_icon = "🟢"
                elif action == 'BUY':
                    action_icon = "🟡" 
                elif action == 'STRONG_SELL':
                    action_icon = "🔴"
                elif action == 'SELL':
                    action_icon = "🟠"
                else:
                    action_icon = "⚪"
                
                result_text += f"{action_icon} {symbol}: {action} ({confidence}%)\n"
                result_text += f"   💰 Price: ${current_price:,.2f}\n"
                
                # Add price targets if available
                targets = rec.get('price_targets', {})
                if targets.get('short_term'):
                    result_text += f"   🎯 Short-term: ${targets['short_term']:,.2f}\n"
                if targets.get('medium_term'):
                    result_text += f"   🎯 Medium-term: ${targets['medium_term']:,.2f}\n"
                    
                result_text += "\n"
        
        result_text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        result_text += f"📊 {len(results)}/{5} cryptocurrencies analyzed"
        
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, lambda: self.ai_results_text.setPlainText(result_text))
    
    def display_ai_results(self, analysis: Dict):
        """Display AI analysis results"""
        if 'error' in analysis:
            result_text = f"❌ {analysis['error']}"
        else:
            result_text = self._format_ai_analysis(analysis)
        
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, lambda: self._update_ai_display(result_text))
    
    def _update_ai_display(self, text: str):
        """Update AI results display"""
        self.ai_results_text.setPlainText(text)
        self.statusBar().showMessage("AI analysis completed")
    
    def _format_ai_analysis(self, analysis: Dict) -> str:
        """Format AI analysis for display"""
        symbol = analysis.get('symbol', 'Unknown')
        current_price = analysis.get('current_price', 0)
        price_change = analysis.get('price_change', 0)
        recommendation = analysis.get('recommendation', {})
        
        text = f"""
            🤖 DEEPSEEK AI ANALYSIS - {symbol}
            📊 Current Price: ${current_price:,.2f} ({price_change:+.2f}%)

            🎯 TRADING RECOMMENDATION: {recommendation.get('action', 'HOLD')}
            📈 Confidence: {recommendation.get('confidence', 50)}%

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            📊 TECHNICAL ANALYSIS:
            {analysis.get('technical_analysis', 'No technical analysis available')}

            ⚠️ RISK ASSESSMENT:
            {analysis.get('risk_assessment', 'No risk assessment available')}

            💡 ADDITIONAL INSIGHTS:
            {analysis.get('additional_insights', 'No additional insights available')}

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            🕒 Analysis Time: {analysis.get('timestamp', 'Unknown')}
        """
        
        return text.strip()
    
    def _update_ai_display(self, message: str):
        """Update AI display with progress message"""
        from PyQt5.QtCore import QTimer
        
        def update_display():
            current_text = self.ai_results_text.toPlainText()
            lines = current_text.split('\n')
            
            # Find and replace the status line
            for i, line in enumerate(lines):
                if line.startswith("Status:") or line.startswith("🔄") or line.startswith("🌐") or line.startswith("🤖"):
                    lines[i] = message
                    break
            else:
                # Add new status line if not found
                lines.insert(4, message)  # Insert after the header
            
            new_text = '\n'.join(lines)
            self.ai_results_text.setPlainText(new_text)
            
            # Auto-scroll to show latest message
            cursor = self.ai_results_text.textCursor()
            cursor.movePosition(cursor.End)
            self.ai_results_text.setTextCursor(cursor)
        
        QTimer.singleShot(0, update_display)

    def test_ai_analyzer_directly(self):
        """Test the AI analyzer directly to see if it's working"""
        if not self.ai_analyzer:
            print("❌ AI Analyzer not available")
            return
        
        print("🧪 Testing AI Analyzer Directly...")
        
        # Create simple test data
        import pandas as pd
        from datetime import datetime, timedelta
        
        test_data = pd.DataFrame({
            'Open': [40000, 41000, 41500, 42000, 42500],
            'High': [40500, 41500, 42000, 42500, 43000],
            'Low': [39500, 40500, 41000, 41500, 42000],
            'Close': [40200, 41200, 41700, 42200, 42700],
            'Volume': [1000, 1200, 1500, 1800, 2000]
        })
        
        # Create datetime index
        dates = [datetime.now() - timedelta(days=i) for i in range(4, -1, -1)]
        test_data.index = dates
        
        print(f"🧪 Test data created: {len(test_data)} rows")
        
        try:
            print("🧪 Calling AI analyzer...")
            result = self.ai_analyzer.analyze_crypto_data("TESTBTC", test_data)
            print(f"🧪 AI Analysis Result: {result}")
            
            if 'error' in result:
                print(f"❌ AI Error: {result['error']}")
            else:
                print(f"✅ AI Success: {result.get('recommendation', 'No recommendation')}")
                
        except Exception as e:
            print(f"❌ AI Test Failed: {e}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")

    def test_deepseek_connection(self):
        """Test if we can connect to DeepSeek API"""
        if not self.ai_analyzer:
            print("❌ No AI Analyzer")
            return
        
        print("🔌 Testing DeepSeek API Connection...")
        
        try:
            # Try to import the OpenAI client directly
            import openai
            
            # Test a simple API call
            client = openai.OpenAI(
                api_key=self.ai_analyzer.client.api_key,
                base_url="https://api.deepseek.com"
            )
            
            print("🔌 Testing simple chat completion...")
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "Say 'Hello World'"}],
                max_tokens=10
            )
            
            print(f"✅ API Connection Successful: {response.choices[0].message.content}")
            return True
            
        except Exception as e:
            print(f"❌ API Connection Failed: {e}")
            return False
        
    def update_live_news(self):
        symbol = self.symbol_combo.currentText()
        print(f"Checking for new news for {symbol}...")
        
        news_df = scrape_and_analyze_finviz_news(symbol)
        
        if not news_df.empty:
            latest_headline = news_df.iloc[0]['headline']
            if latest_headline != self.last_seen_headline:
                print(f"New news found: {latest_headline}")
                self.last_seen_headline = latest_headline
                self.latest_sentiment['positive'] = news_df.iloc[0]['positive']
                self.latest_sentiment['negative'] = news_df.iloc[0]['negative']
                self.latest_sentiment['neutral'] = news_df.iloc[0]['neutral']
                self.statusBar().showMessage(f"New News: {latest_headline}", 5000)

    def show_statistics(self):
        """Run backtest and show the statistics window"""
        results = self._run_backtest_logic()
        if results is False:
            return

        self.stats_window = StatisticsWindow(results)
        self.stats_window.show()