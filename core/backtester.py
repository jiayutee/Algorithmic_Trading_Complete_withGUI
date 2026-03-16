import backtrader as bt
import pandas as pd
import numpy as np
import yfinance as yf
import os
from core.logger import logger

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

class MakerTakerCommission(bt.CommInfoBase):
    """
    Custom commission scheme to distinguish between Maker (Limit) and Taker (Market) fees.
    """
    params = (
        ('maker_fee', 0.0005),  # Default 0.05%
        ('taker_fee', 0.001),   # Default 0.1%
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size, price, pseudoexec):
        # Access the order being processed via the internal broker state
        order = self._broker._order
        
        # Treat Limit orders as Maker, others (Market, Stop) as Taker
        if order.exectype == bt.Order.Limit:
            fee = self.p.maker_fee
        else:
            fee = self.p.taker_fee
            
        return abs(size) * price * fee

class Backtester:
    """
    Manages backtesting execution using Backtrader.
    """
    _benchmark_cache = {} # Class-level cache for benchmark data

    def __init__(self):
        self.cerebro = bt.Cerebro()
        self.df = None

    def add_data(self, df):
        """Add data to Cerebro"""
        self.df = df
        if df.index.name != 'datetime':
            df.index.name = 'datetime'
        
        data = CustomPandasData(dataname=df)
        self.cerebro.adddata(data)

    def add_strategy(self, strategy_class, **params):
        """Add strategy to Cerebro"""
        self.cerebro.addstrategy(strategy_class, **params)

    def run_backtest(self, cash=100000.0, broker_mode="simulated", broker=None, benchmark_ticker="SPY", market_fee=0.001, limit_fee=0.0005):
        """
        Run the backtest.
        
        Args:
            cash (float): Initial cash.
            broker_mode (str): 'simulated' or 'real'.
            broker (object): Real broker instance (for fee structure mimicking).
            benchmark_ticker (str): Ticker for Alpha/Beta calculation (e.g., 'SPY', 'BTC-USD').
            market_fee (float): Fee for market orders (percentage as decimal).
            limit_fee (float): Fee for limit orders (percentage as decimal).
            
        Returns:
            dict: Backtest results with metrics.
        """
        logger.info(f"Running backtest... Cash: {cash}, Mode: {broker_mode}, Benchmark: {benchmark_ticker}, Mkt Fee: {market_fee}, Lim Fee: {limit_fee}")
        
        # Configure Broker
        self.cerebro.broker.setcash(cash)
        
        # Apply custom commission scheme
        comminfo = MakerTakerCommission(maker_fee=limit_fee, taker_fee=market_fee)
        self.cerebro.broker.addcommissioninfo(comminfo)

        # Add Analyzers
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, compression=1)
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
        self.cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

        # Run
        try:
            results = self.cerebro.run()
            if not results:
                logger.warning("Backtest returned no results.")
                return {}
            
            strategy = results[0]
            return self._generate_report(strategy, benchmark_ticker)
            
        except Exception as e:
            logger.error(f"Backtest execution failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    def _generate_report(self, strategy, benchmark_ticker):
        """Generate performance report"""
        pyfolio_analysis = strategy.analyzers.pyfolio.get_analysis()
        returns = pd.Series(pyfolio_analysis['returns'])
        
        # Calculate Alpha/Beta using cached benchmark
        alpha, beta = self._calculate_alpha_beta(returns, benchmark_ticker)

        # Trade Analysis
        trade_analysis = strategy.analyzers.trade_analyzer.get_analysis()
        total_closed_trades = trade_analysis.total.closed if trade_analysis and trade_analysis.total else 0
        win_rate = (trade_analysis.won.total / total_closed_trades) * 100 if total_closed_trades > 0 else 0
        
        # PnL per trade (requires strategy to track this or extracting from trade_analyzer)
        # Using a safer extraction method if direct list not available
        pnl_per_trade = []
        if total_closed_trades > 0:
            # Backtrader TradeAnalyzer structure is complex, simplified extraction:
            # This is a placeholder since extracting per-trade PnL list from TradeAnalyzer dict is non-trivial without iteration
            # We'll rely on the strategy having 'closed_trades' list if implemented, else empty
            if hasattr(strategy, 'closed_trades'):
                 pnl_per_trade = [trade.pnl for trade in strategy.closed_trades]

        # Prepare summary
        sharpe = strategy.analyzers.sharpe.get_analysis().get('sharperatio', 0)
        
        summary = {
            "Sharpe Ratio": sharpe if sharpe is not None else 0,
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
            "cumulative_pnl": np.cumsum(pnl_per_trade).tolist() if pnl_per_trade else [],
            "total_asset_value": portfolio_values.tolist(),
            "profit_per_trade": pnl_per_trade,
            "signals": getattr(strategy, 'signals', [])
        }

    def _calculate_alpha_beta(self, returns, benchmark_ticker):
        """Calculate Alpha and Beta against a benchmark with caching"""
        if self.df is None or len(self.df) == 0:
            return 0, 0
            
        try:
            start_date = self.df.index[0].strftime('%Y-%m-%d')
            end_date = self.df.index[-1].strftime('%Y-%m-%d')
            cache_key = f"{benchmark_ticker}_{start_date}_{end_date}"

            if cache_key in self._benchmark_cache:
                benchmark_returns = self._benchmark_cache[cache_key]
            else:
                logger.info(f"Downloading benchmark data ({benchmark_ticker})...")
                # Suppress yfinance progress
                benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date, progress=False)
                if benchmark_data.empty:
                    logger.warning(f"Benchmark data for {benchmark_ticker} is empty.")
                    return 0, 0
                    
                # Handle MultiIndex if present
                if isinstance(benchmark_data.columns, pd.MultiIndex):
                    # Try to find 'Adj Close' or 'Close'
                    if 'Adj Close' in benchmark_data.columns.get_level_values(0):
                         benchmark_vals = benchmark_data['Adj Close']
                    else:
                         benchmark_vals = benchmark_data.xs('Close', axis=1, level=0, drop_level=True)
                else:
                    benchmark_vals = benchmark_data['Adj Close'] if 'Adj Close' in benchmark_data.columns else benchmark_data['Close']
                
                # If it's a DataFrame (multiple symbols?), take first column
                if isinstance(benchmark_vals, pd.DataFrame):
                    benchmark_vals = benchmark_vals.iloc[:, 0]
                    
                benchmark_returns = benchmark_vals.pct_change().dropna()
                self._benchmark_cache[cache_key] = benchmark_returns

            # Align returns
            returns.index = returns.index.tz_localize(None)
            benchmark_returns.index = benchmark_returns.index.tz_localize(None)
            
            aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
            
            if len(aligned_returns) < 2:
                return 0, 0

            # Calculation
            cov = np.cov(aligned_returns, aligned_benchmark)[0, 1]
            var = np.var(aligned_benchmark)
            beta = cov / var if var != 0 else 0
            
            risk_free_rate = 0.01 
            avg_return = np.mean(aligned_returns) * 252 
            avg_benchmark_return = np.mean(aligned_benchmark) * 252 
            alpha = avg_return - (risk_free_rate + beta * (avg_benchmark_return - risk_free_rate))
            
            return alpha, beta
            
        except Exception as e:
            logger.error(f"Alpha/Beta calculation failed: {e}")
            return 0, 0

    def get_signals(self):
        if self.cerebro.strats and hasattr(self.cerebro.strats[0][0], 'signals'):
            return self.cerebro.strats[0][0].signals
        return []