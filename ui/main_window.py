# ui/main_window.py
from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
                             QComboBox, QPushButton, QLabel, QGroupBox, QLineEdit, QTextEdit)
from PyQt5.QtGui import QIntValidator, QDoubleValidator
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
from core.logger import logger


class MainWindow(QMainWindow):
    def __init__(self, data_loader, strategy_manager, broker_manager):
        super().__init__()
        self.data_loader = data_loader
        self.strategy_manager = strategy_manager
        self.broker_manager = broker_manager
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
        self.setup_fee_inputs()

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

    def setup_interval_controls(self):
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(['1d', '1h', '15m', '5m', '1m'])
        self.control_layout.addWidget(QLabel("Interval:"))
        self.control_layout.addWidget(self.interval_combo)

    def setup_days_input(self):
        self.days_input = QLineEdit("365")
        self.days_input.setValidator(QIntValidator(1, 10000))
        self.days_input.setFixedWidth(50)
        self.control_layout.addWidget(QLabel("Days:"))
        self.control_layout.addWidget(self.days_input)

    def setup_cash_input(self):
        self.cash_input = QLineEdit("100000")
        self.cash_input.setValidator(QIntValidator(1000, 10000000))
        self.cash_input.setFixedWidth(80)
        self.control_layout.addWidget(QLabel("Cash:"))
        self.control_layout.addWidget(self.cash_input)
        
    def setup_fee_inputs(self):
        from PyQt5.QtGui import QDoubleValidator
        self.market_fee_input = QLineEdit("0.1") # Default 0.1%
        self.market_fee_input.setValidator(QDoubleValidator(0, 10, 4))
        self.market_fee_input.setFixedWidth(50)
        self.control_layout.addWidget(QLabel("Mkt Fee %:"))
        self.control_layout.addWidget(self.market_fee_input)

        self.limit_fee_input = QLineEdit("0.05") # Default 0.05%
        self.limit_fee_input.setValidator(QDoubleValidator(0, 10, 4))
        self.limit_fee_input.setFixedWidth(50)
        self.control_layout.addWidget(QLabel("Lim Fee %:"))
        self.control_layout.addWidget(self.limit_fee_input)
        
        self.pause_btn.clicked.connect(self.pause_simulation)

        # Realtime & Simulation Timers
        self.realtime_timer = QTimer()
        self.realtime_timer.timeout.connect(self.process_realtime_updates)
        self.is_streaming = False

        # Initialize with sample data - Defer to allow UI to show
        # self.load_data()
        QTimer.singleShot(100, self.load_data)
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

    def reset_chart_zoom(self):
        """Reset the chart zoom to the initial view."""
        if hasattr(self, 'fig'):
            self.plot_candles()
            self.statusBar().showMessage("Chart zoom reset.")

    # ... (previous methods) ...

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
            # Map 'Live' from UI to 'live' arg in DataLoader (which now fetches recent data)
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

    # ... (plot_candles, add_technical_indicators, update_plotly_view, reset_chart_zoom, setup_interval_controls, on_interval_changed, update_interval_states, setup_days_input, setup_cash_input) ...

    def _run_backtest_logic(self):
        """Unified backtest entry point called by background thread worker usually."""
        return self._run_unified_backtest()

    def _run_backtest_logic_with_broker(self, broker):
        """Unified backtest entry point with broker called by background thread worker."""
        # For the unified system, we pass the broker mode and object to the manager
        return self._run_unified_backtest(broker=broker)

    def _run_unified_backtest(self, broker=None):
        strategy_name = self.strategy_combo.currentText()
        print(f"DEBUG: Strategy name: {strategy_name}")

        if strategy_name == "False":
            self.statusBar().showMessage("No strategy selected!")
            return False
            
        if not hasattr(self, 'df') or self.df.empty:
            self.statusBar().showMessage("Please load data before backtest.")
            return False

        try:
            # 1. Get standardized strategy wrapper
            kwargs = {}
            if strategy_name == "LSTM Predictor":
                kwargs = {'ticker': self.symbol_combo.currentText(), 'sequence_length': 60}

            strategy_wrapper = self.strategy_manager.get_strategy(strategy_name, **kwargs)
            if not strategy_wrapper:
                 self.statusBar().showMessage(f"Failed to load strategy: {strategy_name}")
                 return False

            # 2. Determine broker mode
            broker_mode = "simulated"
            real_broker = None
            
            current_broker_name = self.broker_combo.currentText()
            if current_broker_name != "Simulator":
                 if broker:
                     broker_mode = "real"
                     real_broker = broker
            
            # 3. Extract fees from GUI
            try:
                market_fee = float(self.market_fee_input.text()) / 100.0  # Convert % to decimal
                limit_fee = float(self.limit_fee_input.text()) / 100.0    # Convert % to decimal
            except ValueError:
                market_fee = 0.001  # Fallback to 0.1%
                limit_fee = 0.0005  # Fallback to 0.05%

            # 4. Run Backtest
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
            print(f"❌ Backtest error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False

    # Deprecated/Removed methods



                        

        




    # Legacy methods removed during refactoring


    
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
            
            # Apply user-defined fees to the broker instance
            try:
                backtest_broker.market_fee = float(self.market_fee_input.text()) / 100.0
                backtest_broker.limit_fee = float(self.limit_fee_input.text()) / 100.0
            except ValueError:
                pass
            
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
            
            # Manually update backtest_broker state from results for UI display
            # (Since the actual backtest ran in a separate engine loop)
            if 'Final Value' in summary:
                 backtest_broker.balance = float(summary['Final Value'])
                 backtest_broker.portfolio_value = float(summary['Final Value'])
                 # Note: Positions are not easily reconstructible here without replay, 
                 # but balance helps the user see the result in the 'Account' label.

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
            self.current_broker = broker

            if not self.current_broker:
                self.statusBar().showMessage(f"Broker {broker_name} not configured!")
                return

            # Apply user-defined fees if using Simulator
            if broker_name == "Simulator":
                try:
                    self.current_broker.market_fee = float(self.market_fee_input.text()) / 100.0
                    self.current_broker.limit_fee = float(self.limit_fee_input.text()) / 100.0
                except ValueError:
                    pass

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
        ts = pd.Timestamp(data['timestamp'])
        
        # Check if the data is an order book depth update
        if 'bids' in data and 'asks' in data and data['bids'] and data['asks']:
            # Extract best bid and best ask
            best_bid = float(data['bids'][0][0])
            best_ask = float(data['asks'][0][0])
            price = (best_bid + best_ask) / 2
        elif 'price' in data: # Fallback for other data types if they have 'price'
            price = data['price']
        else:
            logger.warning(f"Real-time data missing 'price' or 'bids'/'asks' for candlestick: {data}")
            return # Skip processing if no price information

        # Map interval to unambiguous pandas frequency string
        if self.current_interval.endswith('m'):
            freq = self.current_interval.replace('m', 'min') # Use 'min' for minute
        elif self.current_interval.endswith('h'):
            freq = self.current_interval.replace('h', 'H') # Use 'H' for hour
        elif self.current_interval.endswith('d'):
            freq = self.current_interval.replace('d', 'D') # Use 'D' for day
        else:
            freq = self.current_interval

        if self.realtime_df.empty:
            new_row = pd.DataFrame([{'Open': price, 'High': price, 'Low': price, 'Close': price}], index=[ts])
            self.realtime_df = pd.concat([self.realtime_df, new_row])
        else:
            # Existing logic for updating/adding candles
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
        self.update_plotly_view()

    def update_plotly_view(self):
        """Convert Plotly figure to HTML and display in QWebEngineView."""
        if hasattr(self, 'fig'):
            import os
            from PyQt5.QtCore import QUrl

            # Create a temporary file to store the HTML
            file_path = os.path.join(os.getcwd(), "live_price_chart.html")
            self.fig.write_html(file_path, include_plotlyjs='cdn')
            
            # Load the HTML from the temporary file
            self.plotly_view.setUrl(QUrl.fromLocalFile(file_path))
        else:
            pass

    def stop_realtime_stream(self):
        if self.is_streaming:
            self.data_loader.stop_realtime_stream()
            self.realtime_timer.stop()
            self.is_streaming = False
            self.realtime_df = pd.DataFrame()

    def closeEvent(self, event):
        if self.is_streaming:
            self.stop_realtime_stream()
        
        self.broker_timer.stop()
        self.realtime_timer.stop()
        self.simulation_timer.stop()
        self.news_timer.stop()
        
        super().closeEvent(event)


    def update_live_news(self):
        symbol = self.symbol_combo.currentText()
        print(f"Checking for new news for {symbol}...")
        
        # Scrape news using the scraper logic
        news_df = scrape_and_analyze_finviz_news(symbol)
        
        if not news_df.empty:
            latest_headline = news_df.iloc[0]['headline']
            if hasattr(self, 'last_seen_headline') and latest_headline != self.last_seen_headline:
                print(f"New news found: {latest_headline}")
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
