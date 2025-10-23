# ui/main_window.py
from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
                             QComboBox, QPushButton, QLabel, QScrollArea, QGraphicsSimpleTextItem, QGroupBox, QDateTimeEdit, QLineEdit)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtChart import QChart, QChartView, QCandlestickSeries, QCandlestickSet, QBarCategoryAxis, QValueAxis, QDateTimeAxis
from PyQt5.QtCore import Qt, QDateTime, QEvent, QMargins, QTimer
from PyQt5.QtGui import QPainter, QFont, QColor
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
from core.news_scraper import scrape_and_analyze_finviz_news


class MainWindow(QMainWindow):
    def __init__(self, data_loader, strategy_manager, broker_manager):
        super().__init__()
        self.data_loader = data_loader
        self.strategy_manager = strategy_manager
        self.broker_manager = broker_manager
        self.setWindowTitle("Algorithmic Trading Terminal")
        self.resize(1400, 800)

        # Central Widget
        central_widget = QWidget()
        self.main_layout = QVBoxLayout(central_widget)

        # Control Panel
        self.control_layout = QHBoxLayout()
        self.control_layout.setObjectName("controlPanelLayout")  # set the name of the control layout

        # Data Source
        self.control_layout.addWidget(QLabel("Data Source:"))
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems(["Historical", "Live", "Realtime Stream", "FinRL-Yahoo"])
        self.control_layout.addWidget(self.data_source_combo)

        # Symbol Selection
        self.control_layout.addWidget(QLabel("Symbol:"))
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "AAPL", "TSLA", "GOLD", "SPY", "QQQ"])
        self.control_layout.addWidget(self.symbol_combo)
        self.symbol_combo.currentTextChanged.connect(self.load_data)

        # Strategy Selection
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

        # Broker Selection
        self.control_layout.addWidget(QLabel("Broker:"))
        self.broker_combo = QComboBox()
        self.broker_combo.addItems(["Simulator", "Alpaca", "Interactive Brokers", "Binance"])
        self.control_layout.addWidget(self.broker_combo)

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

        # Add interval selection
        self.setup_interval_controls()

        # Add days input
        self.setup_days_input()

        # Add initial cash input
        self.setup_cash_input()

        self.main_layout.addLayout(self.control_layout)

        # # Chart Area (Scrollable)
        # self.chart_scroll = QScrollArea()
        # self.chart_scroll.setWidgetResizable(True)
        #
        # self.chart = QChart()
        # self.chart.setAnimationOptions(QChart.SeriesAnimations)
        # self.chart.legend().setVisible(True)
        #
        # self.chart_view = QChartView(self.chart)
        # self.chart_view.setRenderHint(QPainter.Antialiasing)
        # # Set the chart view as the scroll area's widget
        # self.chart_scroll.setWidget(self.chart_view)
        #
        # self.main_layout.addWidget(self.chart_scroll)
        #
        # # Enable scroll bars
        # self.chart_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        # self.chart_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        # self.chart_view.setRubberBand(QChartView.HorizontalRubberBand)

        # Remove all QChart-related code and replace with:
        self.plotly_view = QWebEngineView()
        empty_html = """
            <html>
            <head>
                <meta charset=\"utf-8\"/>
                <style>
                body {
                    background-color: #121212;  /* Dark background */
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

        # Initialize Plotly figure
        self.fig = make_subplots(rows=1, cols=1)
        self.fig.update_layout(
            height=600,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            template='plotly_dark'
        )

        # Status Bar
        self.statusBar().showMessage("Ready")

        # P&L Label
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


        # Initialize with sample data
        self.load_data()

        # # Add hover tracking
        # self.chart_view.setMouseTracking(True)
        # self.chart_view.scene().installEventFilter(self)
        # self.chart_view.viewport().installEventFilter(self)

        # Price display tooltip
        self.price_label = QGraphicsSimpleTextItem()
        self.price_label.setFont(QFont("Arial", 10))
        self.price_label.setBrush(QColor(Qt.white))
        self.price_label.setPen(QColor(Qt.black))
        self.price_label.setZValue(100)
        # self.chart_view.scene().addItem(self.price_label)
        self.price_label.hide()

        # Add realtime streaming attributes
        self.realtime_timer = QTimer()
        self.realtime_timer.timeout.connect(self.process_realtime_updates)
        self.is_streaming = False
        self.historical_candles = []
        self.current_candle = False
        self.current_interval = '1m'
        self.last_candle_time = False
        self.max_candles = 130 # Max candles to display in real-time chart

        # Add simulation attributes
        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self.update_simulation_chart)
        self.simulation_data = False
        self.simulation_index = 0
        self.buy_signal_plotted = False
        self.sell_signal_plotted = False

        # Add news scraping attributes
        self.news_timer = QTimer()
        self.news_timer.timeout.connect(self.update_live_news)
        self.last_seen_headline = ""
        self.latest_sentiment = {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}


    def check_strategy_signal(self, data):
        strategy_name = self.strategy_combo.currentText()
        
        if len(data) < 2:
            return 0
            
        latest_data = data.iloc[-1]
        previous_data = data.iloc[-2]

        if strategy_name == "MACD/RSI":
            if latest_data['RSI'] > 30 and latest_data['MACD'] > latest_data['Signal']:
                return 1  # Buy
            elif latest_data['RSI'] > 70 or latest_data['MACD'] < latest_data['Signal']:
                return -1  # Sell
            
        elif strategy_name == "EMA Crossover":
            if latest_data['EMA12'] > latest_data['EMA26'] and previous_data['EMA12'] <= previous_data['EMA26']:
                return 1
            elif latest_data['EMA12'] < latest_data['EMA26'] and previous_data['EMA12'] >= previous_data['EMA26']:
                return -1

        elif strategy_name == "Stochastic":
            if latest_data['K'] > latest_data['D'] and previous_data['K'] <= previous_data['D'] and latest_data['K'] < 20:
                return 1
            elif latest_data['K'] < latest_data['D'] and previous_data['K'] >= previous_data['D'] and latest_data['K'] > 80:
                return -1
                
        return 0 # Hold

    def start_simulation(self):
        """Start historical data simulation"""
        strategy_name = self.strategy_combo.currentText()

        # Initialize portfolio and signal flags
        self.sim_portfolio = {
            'cash': float(self.cash_input.text()),
            'position': 0,
            'position_value': 0,
            'total_value': float(self.cash_input.text()),
            'pnl': 0
        }
        self.pnl_label.setText("P&L: $0.00")
        self.buy_signal_plotted = False
        self.sell_signal_plotted = False

        self.simulation_data = self.df.copy()
        self.simulation_index = 0
        self.play_btn.show()
        self.pause_btn.show()
        self.simulate_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.play_btn.setEnabled(True)

        # Define the window size
        window_size = 200
        
        # Get the initial window of data
        initial_data = self.simulation_data.iloc[0:window_size]
        
        # Calculate initial y-axis range
        min_low = initial_data['Low'].min()
        max_high = initial_data['High'].max()
        y_padding = (max_high - min_low) * 0.1
        initial_y_range = [min_low - y_padding, max_high + y_padding]

        # Setup the chart for simulation, plotting only the initial data
        self.fig = go.Figure(data=[go.Candlestick(
            x=initial_data.index.astype(str),
            open=initial_data['Open'],
            high=initial_data['High'],
            low=initial_data['Low'],
            close=initial_data['Close'],
            name='Price',
            increasing_line_color='green',
            decreasing_line_color='red'
        )])
        
        self.fig.update_layout(
            height=600,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(
                type='date',
                rangeslider=dict(visible=True), # Hide rangeslider during simulation
            ),
            yaxis=dict(
                title='Price',
                side='right',
                range=initial_y_range
            ),
            hovermode='x unified'
        )

        self.update_plotly_view()
        self.statusBar().showMessage("Simulation ready. Press Play to start.")

    def play_simulation(self):
        """Play the simulation"""
        if self.simulation_data is False:
            return
        # Start the index from the end of the initial window
        if self.simulation_index == 0:
            self.simulation_index = 200
            
        self.simulation_timer.start(250)
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.statusBar().showMessage("Simulation playing...")

    def pause_simulation(self):
        """Pause the simulation"""
        self.simulation_timer.stop()
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.statusBar().showMessage("Simulation paused.")

    def update_simulation_chart(self):
        """Update the chart with the next candle, sliding the window"""
        if self.simulation_index >= len(self.simulation_data):
            self.simulation_timer.stop()
            self.statusBar().showMessage("Simulation finished.")
            self.play_btn.hide()
            self.pause_btn.hide()
            self.simulate_btn.setEnabled(True)
            self.simulation_index = 0 # Reset for next run
            return

        # Get all data revealed so far
        current_data = self.simulation_data.iloc[:self.simulation_index + 1]
        new_candle = self.simulation_data.iloc[self.simulation_index]

        strategy_name = self.strategy_combo.currentText()

        # Execute strategy if one is selected
        if strategy_name != "False":
            signal = self.check_strategy_signal(current_data)
            
            if signal == 1 and self.sim_portfolio['position'] == 0: # Buy
                self.sim_portfolio['position'] = self.sim_portfolio['cash'] / new_candle['Close']
                self.sim_portfolio['cash'] = 0
                self.fig.add_trace(go.Scatter(x=[str(new_candle.name)], y=[new_candle['Close']], mode='markers', marker=dict(symbol='triangle-up', size=15, color='green'), name='Buy Signal', showlegend=not self.buy_signal_plotted))
                if not self.buy_signal_plotted:
                    self.buy_signal_plotted = True
            elif signal == -1 and self.sim_portfolio['position'] > 0: # Sell
                self.sim_portfolio['cash'] = self.sim_portfolio['position'] * new_candle['Close']
                self.sim_portfolio['position'] = 0
                self.fig.add_trace(go.Scatter(x=[str(new_candle.name)], y=[new_candle['Close']], mode='markers', marker=dict(symbol='triangle-down', size=15, color='red'), name='Sell Signal', showlegend=not self.sell_signal_plotted))
                if not self.sell_signal_plotted:
                    self.sell_signal_plotted = True

            # Update portfolio value and P&L
            self.sim_portfolio['position_value'] = self.sim_portfolio['position'] * new_candle['Close']
            self.sim_portfolio['total_value'] = self.sim_portfolio['cash'] + self.sim_portfolio['position_value']
            self.sim_portfolio['pnl'] = self.sim_portfolio['total_value'] - float(self.cash_input.text())
            self.pnl_label.setText(f"P&L: ${self.sim_portfolio['pnl']:,.2f}")

        # Define the sliding window for the main view
        window_size = 200
        start_index = max(0, self.simulation_index - window_size + 1)
        view_window_data = self.simulation_data.iloc[start_index : self.simulation_index + 1]

        # Define the new visible range for the x-axis
        new_x_range = [str(view_window_data.index[0]), str(view_window_data.index[-1])]
        
        # Calculate y-axis range for the current window
        min_low = view_window_data['Low'].min()
        max_high = view_window_data['High'].max()
        y_padding = (max_high - min_low) * 0.1
        new_y_range = [min_low - y_padding, max_high + y_padding]

        with self.fig.batch_update():
            # Update the trace data with all revealed candles
            self.fig.data[0].x = current_data.index.astype(str)
            self.fig.data[0].open = current_data['Open']
            self.fig.data[0].high = current_data['High']
            self.fig.data[0].low = current_data['Low']
            self.fig.data[0].close = current_data['Close']
            
            # Update the layout to slide the view window
            self.fig.update_layout(
                xaxis_range=new_x_range,
                yaxis_range=new_y_range
            )

        self.update_plotly_view()
        self.simulation_index += 1
    def load_data(self):
        """Load data based on current selections"""
        source = self.data_source_combo.currentText()
        symbol = self.symbol_combo.currentText()
        interval = self.interval_combo.currentText()
        print(f"[DEBUG] Loading data for source: {source}, symbol: {symbol}, interval: {interval}")

        try:
            if source == "Realtime Stream":
                self.start_realtime_stream(symbol)
                return

            # Stop any existing stream
            if self.is_streaming:
                self.stop_realtime_stream()

            # Original loading logic
            live = (source == "Live")
            days = int(self.days_input.text()) # Get days from user input
            self.df = self.data_loader.load_data(
                symbol=symbol,
                source=source,
                live=live,
                days=days,
                interval=interval
            )

            # Critical data validation
            assert not self.df.empty, "Loaded empty DataFrame"
            required_cols = ['Open', 'High', 'Low', 'Close']
            assert all(col in self.df.columns for col in required_cols), f"Missing columns: {required_cols}"
            assert pd.api.types.is_datetime64_any_dtype(self.df.index), "Index must be datetime"
            print(f"Historical data sample:\n{self.df.head()}")
            print(f"Index type: {type(self.df.index[0])}")
            print(f"Index tz: {self.df.index.tz}")
            print(f"OHLC dtypes:\n{self.df.dtypes}")

            self.plot_candles()
            self.statusBar().showMessage(f"Loaded {len(self.df)} candles for {symbol}")

        except Exception as e:
            self.statusBar().showMessage(f"Error: {str(e)}")


    def plot_candles(self):
        """Plot candlestick chart with volume using Plotly"""
        if not hasattr(self, 'df') or self.df.empty:
            return

        # Convert MultiIndex to single index (drop Ticker)
        if isinstance(self.df.index, pd.MultiIndex):
            self.df = self.df.droplevel('Ticker')
        # Drop Ticker from columns if present (like in your Price column)
        if 'Ticker' in self.df.columns.names:
            self.df.columns = self.df.columns.droplevel('Ticker')

        # Ensure timezone consistency (convert all to America/New_York or UTC)
        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.df.index = pd.to_datetime(self.df.index)

        # Handle timezones properly
        if self.df.index.tz is False:
            # If no timezone info, assume UTC for historical data
            self.df.index = self.df.index.tz_localize('America/New_York')
        else:
            # If has timezone info, convert to UTC
            self.df.index = self.df.index.tz_convert('America/New_York')

        # DEBUG: Print critical data info
        print("DataFrame head:\n", self.df.head())
        print("Index type:", type(self.df.index[0]))
        print("OHLC dtypes:\n", self.df[['Open', 'High', 'Low', 'Close']].dtypes)
        print("NaN counts:\n", self.df.isna().sum())

        # Create figure with subplots
        self.fig = make_subplots(rows=5, cols=1, shared_xaxes=True,
                                 vertical_spacing=0.08, subplot_titles=('Price', 'Volume', 'MACD', 'RSI', 'Stochastic'),
                                 row_width=[0.15, 0.15, 0.15, 0.10, 0.75],
                                 specs=[[{"secondary_y": True}], [{}], [{}], [{}], [{}]])  # Add secondary_y for the first subplot

        # Add candlestick trace
        self.fig.add_trace(go.Candlestick(
            x=self.df.index,
            open=self.df['Open'],
            high=self.df['High'],
            low=self.df['Low'],
            close=self.df['Close'],
            name='Price',
            increasing_line_color='green',
            decreasing_line_color='red'
        ), row=1, col=1)

        colors = np.where(self.df['Close'] >= self.df['Open'], 'green', 'red')
        # Add volume trace
        self.fig.add_trace(go.Bar(
            x=self.df.index,
            y=self.df['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.4
        ), row=1, col=1, secondary_y=True)

        self.fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['Volume'],
            name='Volume Line',
            line=dict(color='orange'),
            opacity=0.4
        ), row=1, col=1)

        self.fig.update_yaxes(title_text="Volume", row=1, col=1)
        self.fig.update_xaxes(rangeslider_visible=False)


        # Calculate all technical indicators upfront
        self.calculate_technical_indicators()
        # Add all technical indicators (initially hidden)
        self.add_technical_indicators()

        # Create buttons for indicator toggles
        # buttons arrangement:    
        #    * Trace 0 (`go.Candlestick`): The main price data
        #    * Trace 1 (`go.Bar`): Volume bars
        #    * Trace 2 (`go.Scatter`): Volume line
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
        indicator_buttons = [
            dict(label="MA", method="restyle", args=["visible", [True, True, True, True, True, True, False, False, False, False, False, False, False]],
                args2=["visible", [True, True, True, False, False, False, False, False, False, False, False, False, False]]),
            dict(label="EMA", method="restyle", args=["visible", [True, True, True, False, False, False, True, True, False, False, False, False, False]],
                args2=["visible", [True, True, True, False, False, False, False, False, False, False, False, False, False]]),
            dict(label="MACD", method="restyle", args=["visible", [True, True, True, False, False, False, False, False, True, True, False, False, False]],
                args2=["visible", [True, True, True, False, False, False, False, False, False, False, False, False, False]]), 
            dict(label="RSI", method="restyle", args=["visible", [True, True, True, False, False, False, False, False, False, False, True, False, False]],
                args2=["visible", [True, True, True, False, False, False, False, False, False, False, False, False, False]]), 
            dict(label="Stochastic", method="restyle", args=["visible", [True, True, True, False, False, False, False, False, False, False, False, True, True]],
                args2=["visible", [True, True, True, False, False, False, False, False, False, False, False, False, False]]), 
            dict(label="All Off", method="restyle", args=[
                {"visible": [True, True, True, False, False, False, False, False, False, False, False, False, False]}],args2=[
                {"visible": [True, True, True, False, False, False, False, False, False, False, False, False, False]}]),
        ]

        if len(self.df.index) > 200:
            initial_range = [self.df.index[-200], self.df.index[-1]]
        else:
            initial_range = [self.df.index[0], self.df.index[-1]]

        self.is_crypto = "USD" in self.symbol_combo.currentText().upper() if hasattr(self, 'symbol_combo') else False
        
        xaxis_config = dict(
                            type='date',
                            rangeslider=dict(visible=True),
                            rangeselector=dict(
                                buttons=list([
                                    dict(count=1, label="1m", step="month", stepmode="backward"),
                                    dict(count=3, label="3m", step="month", stepmode="backward"),
                                    dict(count=6, label="6m", step="month", stepmode="backward"),
                                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                                    dict(count=1, label="1y", step="year", stepmode="backward"),
                                    dict(step="all")
                                ])
                            ),
                            range=initial_range,
                            tickmode='auto',
                            nticks=10
                        )
        # Only add rangebreaks for non-crypto assets
        if not self.is_crypto:
            xaxis_config["rangebreaks"] = [dict(bounds=["sat", "mon"]),dict(bounds=[16, 9.5], pattern="hour")] # skip weekends and outside trading hours

        # Update layout with buttons
        self.fig.update_layout(
            height=800,
            margin=dict(l=20, r=20, t=40, b=20),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0.5,
                    y=1.15,
                    xanchor="center",
                    yanchor="top",
                    buttons=indicator_buttons,
                    pad={"r": 10, "t": 10}
                )
            ],
            xaxis=xaxis_config,
            yaxis=dict(title='Price', side='left', fixedrange=False, showgrid=True),
            yaxis2=dict(title='Volume', side='right', overlaying='y', showgrid=True, range=[0, self.df['Volume'].max() * 3]), # Secondary y-axis for volume
            xaxis2=dict(title='Date', showgrid=True),
            hovermode='x unified',
            dragmode='pan',  # Enable panning
            template='plotly_dark'
        )


        # Update the view
        self.update_plotly_view()

    def update_plotly_view(self):
        try:
            # Convert figure to dict
            fig_dict = self.fig.to_dict()

            # Generate HTML with error handling
            raw_html = f'''
                <html>
                    <head>
                        <meta charset="utf-8"/>
                        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                        <style>
                            #plotly-div {{
                                width: 100%;
                                height: 600px;
                            }}
                            body {{
                                margin: 0;
                                padding: 0;
                            }}
                        </style>
                    </head>
                    <body>
                        <div id="plotly-div"></div>
                        <script>
                            try {{
                                var graph = {json.dumps(fig_dict, cls=PlotlyJSONEncoder)};
                                Plotly.newPlot('plotly-div', graph);
                            }} catch (e) {{
                                console.error('JavaScript error:', e);
                                document.body.innerHTML = '<pre>' + e.stack + '</pre>';
                            }}
                        </script>
                    </body>
                </html>
                '''

            self.plotly_view.setHtml(raw_html)
            self.plotly_view.show()  # Force show if hidden

        except Exception as e:
            print("Error in update_plotly_view:", str(e))

    def calculate_technical_indicators(self):
        """Calculate all technical indicators"""
        # Resample to ensure calculations are based on consistent intervals
        resampled_df = self.df.resample('D').ffill() # Resample to daily, forward-filling gaps

        # Moving Averages
        self.df['MA20'] = resampled_df['Close'].rolling(window=20).mean()
        self.df['MA50'] = resampled_df['Close'].rolling(window=50).mean()
        self.df['MA200'] = resampled_df['Close'].rolling(window=200).mean()

        # Exponential Moving Averages
        self.df['EMA12'] = resampled_df['Close'].ewm(span=12, adjust=False).mean()
        self.df['EMA26'] = resampled_df['Close'].ewm(span=26, adjust=False).mean()

        # MACD
        ema12 = resampled_df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = resampled_df['Close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = ema12 - ema26
        self.df['Signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()

        # RSI
        delta = resampled_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))

        # Stochastic Oscillator
        low14 = resampled_df['Low'].rolling(window=14).min()
        high14 = resampled_df['High'].rolling(window=14).max()
        self.df['K'] = 100 * ((resampled_df['Close'] - low14) / (high14 - low14))
        self.df['D'] = self.df['K'].rolling(window=3).mean()

    def add_technical_indicators(self):
        """Add all technical indicator traces to figure"""
        # MA Traces
        self.fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df['MA20'],
            name='MA20', line=dict(color='blue'),
            visible=False
        ), row=1, col=1)
        self.fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df['MA50'],
            name='MA50', line=dict(color='orange'),
            visible=False
        ), row=1, col=1)
        self.fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df['MA200'],
            name='MA200', line=dict(color='purple'),
            visible=False
        ), row=1, col=1)

        # EMA Traces
        self.fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df['EMA12'],
            name='EMA12', line=dict(color='cyan'),
            visible=False
        ), row=1, col=1)
        self.fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df['EMA26'],
            name='EMA26', line=dict(color='magenta'),
            visible=False
        ), row=1, col=1)

        # MACD Traces
        self.fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df['MACD'],
            name='MACD', line=dict(color='blue'),
            visible=False
        ), row=3, col=1)
        self.fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df['Signal'],
            name='Signal', line=dict(color='red'),
            visible=False
        ), row=3, col=1)

        # RSI Trace
        self.fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df['RSI'],
            name='RSI', line=dict(color='purple'),
            visible=False
        ), row=4, col=1)

        # Stochastic Traces
        self.fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df['K'],
            name='K', line=dict(color='blue'),
            visible=False
        ), row=5, col=1)
        self.fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df['D'],
            name='D', line=dict(color='red'),
            visible=False
        ), row=5, col=1)

    def reset_chart_zoom(self):
        """Reset the chart to its default zoom level."""
        if hasattr(self, 'fig') and hasattr(self, 'df') and not self.df.empty:
            # Setting autorange to True resets the zoom
            self.fig.update_xaxes(autorange=True)
            self.fig.update_yaxes(autorange=True)
            self.update_plotly_view()

    def eventFilter(self, obj, event):
        # print(f"Event from: {'viewport' if obj is self.chart_view.viewport() else 'scene'}")
        if obj is self.chart_view.scene():
            if event.type() == QEvent.GraphicsSceneMouseMove:
                #Get mouse position in scene coordinates
                pos = event.scenePos()
                #Convert to chart value coordinates
                chart_pos = self.chart.mapToValue(pos)
                x_val = chart_pos.x() # Extract X value (timestamp in ms)
                y_val = chart_pos.y() # Price value

                # print(f"[DEBUG] Mapped chart value: {chart_pos.x()}, {chart_pos.y()}")

                # Find nearest candle
                closest = False
                min_dist = float('inf')

                for candle_set in self.candle_sets:
                    # compare using candle_index
                    candle_index = int(round(x_val))
                    if 0 <= candle_index < len(self.candle_sets):
                        closest = self.candle_sets[candle_index]

                #Update price label if candle found
                if closest:
                    # print(f"DEBUG: closest found")
                    # print(f"DEBUG: O: {closest.open():.2f}  H: {closest.high():.2f}\n"
                    #     f"L: {closest.low():.2f}  C: {closest.close():.2f}")
                    self.price_label.setText(
                        f"O: {closest.open():.2f}  H: {closest.high():.2f}\n"
                        f"L: {closest.low():.2f}  C: {closest.close():.2f}"
                    )
                    self.price_label.setPos(pos.x() + 10, pos.y() - 30)
                    self.price_label.show()
                else:
                    # print(f"hide")
                    self.price_label.hide()

        return super().eventFilter(obj, event)

    def setup_interval_controls(self):
        """Add interval selection UI"""
        # Create interval selection widgets
        interval_layout = QHBoxLayout()
        interval_label = QLabel("Candle Interval:")
        interval_layout.addWidget(interval_label)

        self.interval_combo = QComboBox()
        intervals = ['1m', '5m', '15m', '30m', '1h', '1d']
        self.interval_combo.addItems(intervals)
        self.interval_combo.setCurrentText('1d')

        # interval_layout.addWidget(self.interval_combo)
        #
        # self.control_layout.insertLayout(self.control_layout.count() - 3, interval_layout)  # Adjust index as needed
        # # Add to existing controls
        # self.control_layout = self.findChild(QHBoxLayout)
        # self.control_layout.insertLayout(2, interval_layout)

        # Insert into control panel (after data source/symbol controls)
        # Count how many widgets are before where we want to insert
        insert_position = 4  # Adjust based on your existing controls
        self.control_layout.insertWidget(insert_position, interval_label)
        self.control_layout.insertWidget(insert_position + 1, self.interval_combo)

        # Connect signals
        self.interval_combo.currentTextChanged.connect(self.on_interval_changed)
        self.data_source_combo.currentTextChanged.connect(self.update_interval_states)

    def on_interval_changed(self, interval):
        """Handle interval changes"""
        # Validate the interval first
        if self.data_source_combo.currentText() == "Historical" and interval in ["1m", "5m", "15m", "30m"]:
            self.statusBar().showMessage(f"Yahoo Finance doesn't support {interval} interval")
            self.interval_combo.setCurrentText("1h")  # Reset to supported interval
            return

        # Reload data with new interval
        self.load_data()

    def update_interval_states(self):
        """Enable/disable interval options based on data source"""
        source = self.data_source_combo.currentText()

        for i in range(self.interval_combo.count()):
            interval = self.interval_combo.itemText(i)
            if source == "Historical":
                # Only enable supported Yahoo intervals
                self.interval_combo.model().item(i).setEnabled(interval in ['1h', '1d'])
            else:  # Live
                self.interval_combo.model().item(i).setEnabled(True)

        # Ensure current selection is valid
        if not self.interval_combo.currentData(Qt.UserRole + 1):  # Check enabled state
            self.interval_combo.setCurrentText('1d')

    def setup_days_input(self):
        """Add UI for selecting the number of days for historical data"""
        
        days_layout = QHBoxLayout()
        days_label = QLabel("Days to Plot:")
        days_layout.addWidget(days_label)

        self.days_input = QLineEdit("365")  # Default to 365 days
        self.days_input.setFixedWidth(60)
        self.days_input.setValidator(QIntValidator(1, 365*5)) # Allow 1 to 5 years of data
        days_layout.addWidget(self.days_input)

        # Insert into control panel (after interval controls)
        insert_position = self.control_layout.count() - 4 # Adjust based on your existing controls
        self.control_layout.insertLayout(insert_position, days_layout)

        # Connect returnPressed signal to load_data
        self.days_input.returnPressed.connect(self.load_data)

    def setup_cash_input(self):
        """Add UI for setting the initial cash for backtesting"""
        cash_layout = QHBoxLayout()
        cash_label = QLabel("Initial Cash:")
        cash_layout.addWidget(cash_label)

        self.cash_input = QLineEdit("100000")  # Default to 100,000
        self.cash_input.setFixedWidth(100)
        self.cash_input.setValidator(QIntValidator(100, 100000000)) # Allow from 100 to 100,000,000
        cash_layout.addWidget(self.cash_input)

        # Insert into control panel
        insert_position = self.control_layout.count() - 4 # Adjust based on your existing controls
        self.control_layout.insertLayout(insert_position, cash_layout)

    

    

    

    def _run_backtest_logic(self):
        strategy_name = self.strategy_combo.currentText()
        if strategy_name == "False":
            self.statusBar().showMessage("No strategy selected!")
            return False

        if not hasattr(self, 'df') or self.df.empty:
            self.statusBar().showMessage("Please load data before running a backtest.")
            return False

        if strategy_name == "FinRL Strategy":
            self.statusBar().showMessage("FinRL strategies are designed for portfolio management and cannot be backtested on a single stock.")
            return False

        try:
            if strategy_name == "LSTM Predictor":
                symbol = self.symbol_combo.currentText()
                sequence_length = 60  # Hardcoded sequence length for LSTM
                strategy = self.strategy_manager.get_strategy(strategy_name, ticker=symbol, sequence_length=sequence_length)
            else:
                strategy = self.strategy_manager.get_strategy(strategy_name)

            print(f"[DEBUG] DataFrame columns before backtest: {self.df.columns}")
            results = self.strategy_manager.run_backtest(
                strategy=strategy,
                data=self.df,
                cash=float(self.cash_input.text())
            )
            return results

        except Exception as e:
            self.statusBar().showMessage(f"Backtest error: {str(e)}")
            print(f"Backtest error: {str(e)}")
            return False

    def run_backtest(self):
        """Run backtest with selected strategy"""
        results = self._run_backtest_logic()
        if results is False:
            return

        self.plot_signals(results.get('signals', []))

        summary = results.get('summary', {})
        final_value = summary.get('Final Value', 0)
        sharpe_ratio = summary.get('Sharpe Ratio', 'N/A')
        realized_pnl = np.sum(results.get('profit_per_trade', []))
        
        sharpe_str = f"{sharpe_ratio:.2f}" if isinstance(sharpe_ratio, float) else "N/A"

        msg = (f"Backtest complete | "
               f"Final Value: ${final_value:,.2f} | "
               f"Sharpe: {sharpe_str} | "
               f"Total Realized P&L: ${realized_pnl:,.2f}")
        self.statusBar().showMessage(msg)
        print(msg)

    def show_statistics(self):
        """Run backtest and show the statistics window"""
        results = self._run_backtest_logic()
        if results is False:
            return

        self.stats_window = StatisticsWindow(results)
        self.stats_window.show()

    def plot_signals(self, signals):
        """Plot buy/sell signals on chart"""
        buy_signals = [s for s in signals if s['type'] == 'buy']
        sell_signals = [s for s in signals if s['type'] == 'sell']

        if buy_signals:
            buy_dates = [s['date'] for s in buy_signals]
            buy_prices = [s['price'] for s in buy_signals]
            self.fig.add_trace(go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode='markers',
                marker=dict(symbol='triangle-up', size=15, color='green'),
                name='Buy Signal',
                hoverinfo='text',
                text=[f'Buy @ {p:.2f}' for p in buy_prices],
                textposition='top center'
            ))

        if sell_signals:
            sell_dates = [s['date'] for s in sell_signals]
            sell_prices = [s['price'] for s in sell_signals]
            self.fig.add_trace(go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode='markers',
                marker=dict(symbol='triangle-down', size=15, color='red'),
                name='Sell Signal',
                hoverinfo='text',
                text=[f'Sell @ {p:.2f}' for p in sell_prices],
                textposition='bottom center'
            ))

        self.update_plotly_view()

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

    def start_trading(self):
        """Start live trading with selected broker"""
        broker_name = self.broker_combo.currentText()
        strategy_name = self.strategy_combo.currentText()

        if strategy_name == "False":
            self.statusBar().showMessage("No strategy selected!")
            return

        try:
            broker = self.broker_manager.get_broker(broker_name)
            if strategy_name == "LSTM Predictor":
                symbol = self.symbol_combo.currentText()
                sequence_length = 60  # Hardcoded sequence length for LSTM
                strategy = self.strategy_manager.get_strategy(strategy_name, ticker=symbol, sequence_length=sequence_length)
            else:
                strategy = self.strategy_manager.get_strategy(strategy_name)

            # Start live trading thread
            self.statusBar().showMessage(
                f"Live trading started with {broker_name} using {strategy_name}"
            )

        except Exception as e:
            self.statusBar().showMessage(f"Trading error: {str(e)}")

    # NEW METHODS FOR REALTIME STREAMING
    def start_realtime_stream(self, symbol):
        """Initialize realtime streaming with historical context"""
        if self.is_streaming:
            self.stop_realtime_stream()

        interval = self.interval_combo.currentText()
        self.current_interval = interval

        # Calculate days needed to get approximately 130 candles
        interval_to_minutes = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '1d': 1440
        }
        minutes_needed = 130 * interval_to_minutes.get(interval, 1)
        days_needed = max(1, minutes_needed / (6.5 * 60))  # 6.5 trading hours per day

        try:
            # Load historical data for context
            self.df = self.data_loader.load_data(
                symbol=symbol,
                live=True,
                interval=interval,
                days=days_needed
            )
        except Exception as e:
            self.statusBar().showMessage(f"Error loading historical data: {str(e)}")
            return

        # Initialize chart
        self.chart.removeAllSeries()
        self.realtime_series = QCandlestickSeries()
        self.realtime_series.setName(f"{symbol} ({interval})")
        self.realtime_series.setIncreasingColor(Qt.green)
        self.realtime_series.setDecreasingColor(Qt.red)
        self.chart.addSeries(self.realtime_series)

        # Add historical candles (limited to max_candles)
        self.historical_candles = []
        for i, row in self.df.tail(self.max_candles).iterrows():
            ts = int(i.timestamp() * 1000) if isinstance(i, pd.Timestamp) else int(pd.to_datetime(i).timestamp() * 1000)
            candle = QCandlestickSet(
                row['Open'],
                row['High'],
                row['Low'],
                row['Close'],
                ts
            )
            self.historical_candles.append(candle)
            self.realtime_series.append(candle)

        # Setup axes
        self.setup_realtime_axes()

        # Initialize realtime tracking
        self.current_candle = False
        self.current_candle_set = False
        self.last_candle_time = False

        # Start streaming
        print("[DEBUG] Starting data stream...")
        self.data_loader.start_realtime_stream(
            symbol=symbol,
            callback=self.handle_realtime_data
        )
        self.is_streaming = True
        self.realtime_timer.start(100)  # Update every 100ms
        print("[DEBUG] Realtime stream started successfully")
        self.statusBar().showMessage(f"Started {interval} stream for {symbol}")

    def setup_realtime_axes(self):
        """Configure axes for realtime chart"""
        # Remove old axes if they exist
        for axis in self.chart.axes():
            self.chart.removeAxis(axis)

        # Create new axes
        self.x_axis = QDateTimeAxis()
        self.x_axis.setTitleText("Time")

        # Set appropriate time format based on interval
        if self.current_interval == '1d':
            self.x_axis.setFormat("MM-dd")
        elif self.current_interval == '1h':
            self.x_axis.setFormat("MM-dd HH:mm")
        else:  # For minute intervals
            self.x_axis.setFormat("HH:mm:ss")
        self.chart.addAxis(self.x_axis, Qt.AlignBottom)

        self.y_axis = QValueAxis()
        self.y_axis.setTitleText("Price")
        self.chart.addAxis(self.y_axis, Qt.AlignLeft)

        # Attach axes to series if it exists
        if hasattr(self, 'realtime_series') and self.realtime_series:
            self.realtime_series.attachAxis(self.x_axis)
            self.realtime_series.attachAxis(self.y_axis)

    def update_axes_range(self):
        """Adjust axes to show all candles with padding"""
        all_candles = self.historical_candles + ([self.current_candle_set] if self.current_candle_set else [])
        if not all_candles:
            return

        timestamps = [c.timestamp() for c in all_candles]
        min_ts = min(timestamps)
        max_ts = max(timestamps)

        # Add 5% time padding to the right
        time_range = max_ts - min_ts
        max_ts += time_range * 0.05

        self.x_axis.setRange(
            QDateTime.fromMSecsSinceEpoch(int(min_ts)),
            QDateTime.fromMSecsSinceEpoch(int(max_ts))
        )

        highs = [c.high() for c in all_candles]
        lows = [c.low() for c in all_candles]
        padding = (max(highs) - min(lows)) * 0.05  # 5% price padding

        self.y_axis.setRange(
            min(lows) - padding,
            max(highs) + padding
        )

    def stop_realtime_stream(self):
        """Clean up realtime streaming resources"""
        if self.is_streaming:
            try:
                self.data_loader.stop_realtime_stream()
                self.realtime_timer.stop()

                # # Clear axes
                # for axis in self.chart.axes():
                #     self.chart.removeAxis(axis)

                # Clear series if it exists
                if hasattr(self, 'realtime_series'):
                    self.realtime_series.clear()
                    self.chart.removeSeries(self.realtime_series)
                    self.realtime_series = False

                self.historical_candles = []
                self.current_candle = False
                self.current_candle_set = False
                self.is_streaming = False

            except Exception as e:
                print(f"Error stopping stream: {e}")
                self.is_streaming = False

    def handle_realtime_data(self, data):
        """Callback for incoming realtime data"""
        print(f"[DEBUG] Received realtime data: {data}")  # Verify data is coming in
        self.data_loader.realtime_queue.put(data)

    def process_realtime_updates(self):
        """Process all waiting realtime updates"""
        print("[DEBUG] Timer triggered")  # Verify timer is working
        try:
            while True:
                data = self.data_loader.realtime_queue.get_nowait()
                print(f"[DEBUG] Processing: {data}")  # Verify queue processing
                self.process_realtime_data(data)
        except Empty:
            print("[DEBUG] Queue empty")
            pass

    def process_realtime_data(self, data):
        """Process individual realtime tick"""
        print(f"[DEBUG] Processing tick: {data}")
        timestamp = datetime.fromtimestamp(data['timestamp'])
        price = data['price']

        # Initialize first candle if needed
        if self.current_candle is False:
            print("[DEBUG] Creating first candle")
            self.current_candle = {
                'timestamp': timestamp,
                'open': price,
                'high': price,
                'low': price,
                'close': price
            }
            self.last_candle_time = timestamp
            self.update_realtime_chart()
            return

        # Check if we should start a new candle based on interval
        interval_min = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '1d': 1440
        }.get(self.current_interval, 1)

        elapsed_min = (timestamp - self.last_candle_time).total_seconds() / 60

        if elapsed_min >= interval_min:
            self.finalize_realtime_candle()
            self.last_candle_time = timestamp
            self.current_candle = {
                'timestamp': timestamp,
                'open': price,
                'high': price,
                'low': price,
                'close': price
            }
        else:
            # Update current candle
            self.current_candle['high'] = max(self.current_candle['high'], price)
            self.current_candle['low'] = min(self.current_candle['low'], price)
            self.current_candle['close'] = price

        self.update_realtime_chart()

    def finalize_realtime_candle(self):
        """Complete the current candle and add to series"""
        if not self.current_candle:
            return

        candle = QCandlestickSet(
            self.current_candle['open'],
            self.current_candle['high'],
            self.current_candle['low'],
            self.current_candle['close'],
            int(self.current_candle['timestamp'].timestamp() * 1000)
        )

        self.realtime_series.append(candle)
        self.historical_candles.append(candle)

        # Remove oldest candle if we exceed our limit
        if len(self.historical_candles) > self.max_candles:
            oldest = self.historical_candles.pop(0)
            self.realtime_series.remove(oldest)

        # Reset current candle visualization
        self.current_candle_set = False

    def update_realtime_chart(self):
        """Update the Plotly chart with streaming data"""
        if not self.current_candle:
            return

        # Get all candles
        candles = self.historical_candles

        # Convert to DataFrame for easier handling
        df = pd.DataFrame([{
            'Open': c.open(),
            'High': c.high(),
            'Low': c.low(),
            'Close': c.close(),
            'Date': pd.to_datetime(c.timestamp() / 1000, unit='s')
        } for c in candles])

        # Add current candle if it exists
        if self.current_candle_set:
            df = df.append({
                'Open': self.current_candle['open'],
                'High': self.current_candle['high'],
                'Low': self.current_candle['low'],
                'Close': self.current_candle['close'],
                'Date': self.current_candle['timestamp']
            }, ignore_index=True)

        # Update figure
        self.fig.data = []  # Clear existing data
        candlestick = go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        )
        self.fig.add_trace(candlestick)

        # Auto-range
        self.fig.update_xaxes(autorange=True)
        self.fig.update_yaxes(autorange=True)

        # Update the view
        self.update_plotly_view()

    def closeEvent(self, event):
        """Clean up when closing - modified to stop streaming"""
        if hasattr(self, 'is_streaming') and self.is_streaming:
            self.stop_realtime_stream()
        super().closeEvent(event)