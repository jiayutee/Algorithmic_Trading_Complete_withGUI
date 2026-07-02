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
    Uses a fixed taker fee for all orders (order type detection via broker internals is
    version-dependent and fragile; a flat commission is more reliable).
    """
    params = (
        ('maker_fee', 0.0005),  # Default 0.05%
        ('taker_fee', 0.001),   # Default 0.1%
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size, price, pseudoexec):
        # Use taker fee conservatively; distinguishing order type via broker internals
        # is unreliable across backtrader versions.
        return abs(size) * price * self.p.taker_fee

class Backtester:
    """
    Manages backtesting execution using Backtrader.
    """
    _benchmark_cache = {} # Class-level cache for benchmark data

    def __init__(self):
        self.cerebro = bt.Cerebro()
        self.df = None

    def add_data(self, df):
        """Add data to Cerebro.

        Accepts either a full CustomPandasData-compatible DataFrame (with indicator
        columns) or a minimal OHLCV DataFrame — in which case it uses PandasData.
        """
        self.df = df
        if df.index.name != 'datetime':
            df.index.name = 'datetime'

        # Detect whether indicator columns are present
        indicator_cols = {'MA20', 'MA50', 'MA200', 'EMA12', 'EMA26', 'MACD', 'Signal', 'RSI', 'K', 'D'}
        if indicator_cols.issubset(set(df.columns)):
            data = CustomPandasData(dataname=df)
        else:
            data = bt.feeds.PandasData(dataname=df)
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
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

        # Run
        try:
            results = self.cerebro.run()
            if not results:
                logger.warning("Backtest returned no results.")
                return {}

            strategy = results[0]
            return self._generate_report(strategy, benchmark_ticker, initial_cash=cash)

        except Exception as e:
            logger.error(f"Backtest execution failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    def _generate_report(self, strategy, benchmark_ticker, initial_cash=100000.0):
        """Generate performance report"""
        # PyFolio returns
        try:
            pyfolio_analysis = strategy.analyzers.pyfolio.get_analysis()
            returns = pd.Series(pyfolio_analysis['returns'])
            portfolio_values = pd.Series(pyfolio_analysis['portfolio_value'])
        except Exception:
            returns = pd.Series([], dtype=float)
            portfolio_values = pd.Series([], dtype=float)

        # Calculate Alpha/Beta using cached benchmark
        alpha, beta = self._calculate_alpha_beta(returns, benchmark_ticker)

        # Trade Analysis
        trade_analysis = strategy.analyzers.trade_analyzer.get_analysis()
        try:
            total_closed_trades = trade_analysis.total.closed
        except (AttributeError, KeyError):
            total_closed_trades = 0
        try:
            won_trades = trade_analysis.won.total
        except (AttributeError, KeyError):
            won_trades = 0
        win_rate = (won_trades / total_closed_trades) * 100 if total_closed_trades > 0 else 0.0

        # Sharpe ratio
        sharpe = strategy.analyzers.sharpe.get_analysis().get('sharperatio', 0)
        if sharpe is None:
            sharpe = 0.0

        # Max drawdown from DrawDown analyzer
        dd_analysis = strategy.analyzers.drawdown.get_analysis()
        max_drawdown_pct = dd_analysis.get('max', {}).get('drawdown', 0.0)
        if max_drawdown_pct is None:
            max_drawdown_pct = 0.0

        # PnL per trade
        pnl_per_trade = []
        if total_closed_trades > 0:
            if hasattr(strategy, 'closed_trades'):
                pnl_per_trade = [trade.pnl for trade in strategy.closed_trades]

        # Final portfolio value
        final_value = self.cerebro.broker.getvalue()

        summary = {
            "Sharpe Ratio": round(sharpe, 4),
            "Max Drawdown (%)": round(max_drawdown_pct, 2),
            "Win Rate": f"{win_rate:.2f}%",
            "Alpha": round(alpha, 4),
            "Beta": round(beta, 4),
            "Number of Closed Trades": total_closed_trades,
            "Average Profit per Trade": round(np.mean(pnl_per_trade), 2) if pnl_per_trade else 0,
            "Median Profit per Trade": round(np.median(pnl_per_trade), 2) if pnl_per_trade else 0,
            "Final Value": round(final_value, 2),
            "P&L": round(final_value - initial_cash, 2),
        }

        return {
            # Top-level shorthand keys used by tests and UI
            "sharpe": sharpe,
            "max_drawdown": max_drawdown_pct,
            "win_rate": win_rate,
            # Detailed summary for GUI display
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
            returns_tz_naive = returns.copy()
            if hasattr(returns_tz_naive.index, 'tz') and returns_tz_naive.index.tz is not None:
                returns_tz_naive.index = returns_tz_naive.index.tz_localize(None)
            benchmark_returns_tz_naive = benchmark_returns.copy()
            if hasattr(benchmark_returns_tz_naive.index, 'tz') and benchmark_returns_tz_naive.index.tz is not None:
                benchmark_returns_tz_naive.index = benchmark_returns_tz_naive.index.tz_localize(None)

            aligned_returns, aligned_benchmark = returns_tz_naive.align(benchmark_returns_tz_naive, join='inner')

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
