import backtrader as bt
import pandas as pd


class CustomPandasData(bt.feeds.PandasData):
    # Add parameters for the indicators
    lines = (
        'MA20', 'MA50', 'MA200',
        'EMA12', 'EMA26',
        'MACD', 'Signal',
        'RSI',
        'K', 'D',
    )

    # Set the corresponding column names in the DataFrame
    params = (
        ('datetime', None),
        ('open', -1),
        ('high', -1),
        ('low', -1),
        ('close', -1),
        ('volume', -1),
        ('openinterest', -1),
        ('MA20', -1), # -1 means not present in the data. If present, Backtrader will use it.
        ('MA50', -1),
        ('MA200', -1),
        ('EMA12', -1),
        ('EMA26', -1),
        ('MACD', -1),
        ('Signal', -1),
        ('RSI', -1),
        ('K', -1),
        ('D', -1),
    )


class Backtester:
    def __init__(self):
        self.cerebro = bt.Cerebro()

    def add_data(self, df):
        """Convert pandas DataFrame to Backtrader feed"""
        # Ensure datetime index is named 'datetime' for Backtrader
        df.index.name = 'datetime'
        data = CustomPandasData(dataname=df)
        self.cerebro.adddata(data)

    def add_strategy(self, strategy_class, **params):
        self.cerebro.addstrategy(strategy_class, **params)

    def run_backtest(self, cash=100000.0):
        self.cerebro.broker.setcash(cash)
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe') #risk free rate is 1% by default
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
        results = self.cerebro.run()
        strategy = results[0]  # Get the strategy instance
        signals = []
        # Check if the strategy has a 'signals' attribute
        if hasattr(strategy, 'signals'):
            signals = strategy.signals

        trade_analysis = strategy.analyzers.trade_analyzer.get_analysis()

        return {
            'final_value': self.cerebro.broker.getvalue(),
            'sharpe': results[0].analyzers.sharpe.get_analysis(),
            'signals': signals,
            'trade_analysis': trade_analysis,
            'cumulative_pnl': strategy.cumulative_pnl
        }