import backtrader as bt
import pandas as pd
import numpy as np
import yfinance as yf

class CustomPandasData(bt.feeds.PandasData):
    lines = (
        'MA20', 'MA50', 'MA200',
        'EMA12', 'EMA26',
        'MACD', 'Signal',
        'RSI',
        'K', 'D',
    )
    params = (
        ('datetime', None),
        ('open', -1),
        ('high', -1),
        ('low', -1),
        ('close', -1),
        ('volume', -1),
        ('openinterest', -1),
        ('MA20', -1),
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
        self.df = None

    def add_data(self, df):
        self.df = df
        df.index.name = 'datetime'
        data = CustomPandasData(dataname=df)
        self.cerebro.adddata(data)

    def add_strategy(self, strategy_class, **params):
        self.cerebro.addstrategy(strategy_class, **params)

    def run_backtest(self, cash=100000.0):
        self.cerebro.broker.setcash(cash)
        self.cerebro.broker.setcommission(commission=0.001)
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, compression=1)
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
        self.cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

        results = self.cerebro.run()
        strategy = results[0]
        
        pyfolio_analysis = strategy.analyzers.pyfolio.get_analysis()
        returns = pd.Series(pyfolio_analysis['returns'])
        
        # Fetch benchmark data
        start_date = self.df.index[0].strftime('%Y-%m-%d')
        end_date = self.df.index[-1].strftime('%Y-%m-%d')
        benchmark_data = yf.download('SPY', start=start_date, end=end_date)
        benchmark_returns = benchmark_data['Adj Close'].pct_change().dropna()
        
        # Align returns
        returns.index = returns.index.tz_localize(None)
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')

        # Alpha and Beta calculation
        cov = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        var = np.var(aligned_benchmark)
        beta = cov / var if var != 0 else 0
        
        risk_free_rate = 0.01 # Assuming 1% risk-free rate
        avg_return = np.mean(aligned_returns) * 252 # Annualized
        avg_benchmark_return = np.mean(aligned_benchmark) * 252 # Annualized
        alpha = avg_return - (risk_free_rate + beta * (avg_benchmark_return - risk_free_rate))

        trade_analysis = strategy.analyzers.trade_analyzer.get_analysis()
        total_closed_trades = trade_analysis.total.closed if trade_analysis and trade_analysis.total else 0
        win_rate = (trade_analysis.won.total / total_closed_trades) * 100 if total_closed_trades > 0 else 0
        
        pnl_per_trade = [trade.pnl for trade in strategy.closed_trades] if hasattr(strategy, 'closed_trades') else []
        
        # Prepare summary
        summary = {
            "Sharpe Ratio": strategy.analyzers.sharpe.get_analysis().get('sharperatio', 0),
            "Alpha": alpha,
            "Beta": beta,
            "Number of Closed Trades": total_closed_trades,
            "Win Rate": f"{win_rate:.2f}%",
            "Average Profit per Trade": np.mean(pnl_per_trade) if pnl_per_trade else 0,
            "Median Profit per Trade": np.median(pnl_per_trade) if pnl_per_trade else 0,
            "Final Value": self.cerebro.broker.getvalue()
        }
        
        portfolio_values = pd.Series(pyfolio_analysis['portfolio_value'])

        return {
            "summary": summary,
            "cumulative_pnl": np.cumsum(pnl_per_trade).tolist(),
            "total_asset_value": portfolio_values.tolist(),
            "profit_per_trade": pnl_per_trade
        }

    def get_signals(self):
        # This part might need adjustment depending on how signals are generated and stored
        if self.cerebro.strats and hasattr(self.cerebro.strats[0][0], 'signals'):
            return self.cerebro.strats[0][0].signals
        return []