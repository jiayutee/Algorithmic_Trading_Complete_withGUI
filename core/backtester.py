import backtrader as bt
import pandas as pd
import numpy as np
import yfinance as yf
import os
import json
import csv
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
        # PyFolio returns.
        #
        # NOTE: backtrader's PyFolio analyzer (bt.analyzers.PyFolio) only ever
        # exposes 'returns', 'positions', 'transactions' and 'gross_lev' keys
        # from get_analysis() -- there is no 'portfolio_value' key. Looking one
        # up used to raise a KeyError that was silently swallowed by a bare
        # `except Exception`, which meant the equity curve (total_asset_value)
        # and Alpha/Beta were *always* empty/zero regardless of the data,
        # since `returns` was set to an empty Series every single time.
        #
        # Fix: read the 'returns' key (which does exist) and reconstruct the
        # equity curve ourselves by compounding those per-bar returns against
        # the strategy's starting cash.
        try:
            pyfolio_analysis = strategy.analyzers.pyfolio.get_analysis()
            returns = pd.Series(pyfolio_analysis['returns'])
        except (KeyError, AttributeError) as e:
            logger.warning(f"PyFolio analyzer produced no usable 'returns' data: {e}")
            returns = pd.Series([], dtype=float)

        if len(returns) > 0:
            portfolio_values = initial_cash * (1.0 + returns).cumprod()
        else:
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

        # PnL per trade (net of commission, matching the final portfolio value)
        pnl_per_trade = []
        if total_closed_trades > 0:
            if hasattr(strategy, 'closed_trades'):
                pnl_per_trade = [trade.pnlcomm for trade in strategy.closed_trades]

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
                benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
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


# ---------------------------------------------------------------------------
# Export helper — module-level so it can be imported independently of the
# Backtester class and called from UI code or scripts.
# ---------------------------------------------------------------------------

def export_report(report: dict, filepath: str, format: str = 'csv') -> dict:
    """Export a completed backtest report dict to disk as CSV or JSON.

    Format design decisions
    -----------------------
    JSON (format='json'):
        A single file is written containing every section of the report in a
        naturally nested structure.  This is the recommended format when the
        caller needs the full report in one file and wants to round-trip it
        back into Python without extra parsing.

        Top-level keys in the JSON file:
          "summary"          dict — all summary metrics plus top-level shorthand
                             keys (sharpe, max_drawdown, win_rate, final_value,
                             total_return).
          "equity_curve"     list[float] — portfolio value for each bar.
          "profit_per_trade" list[float] — net P&L (after commission) for each
                             closed trade, in chronological order.
          "cumulative_pnl"   list[float] — running cumulative sum of
                             profit_per_trade.
          "signals"          list[dict]  — each entry has "date", "type",
                             "price" (the same structure used by the chart overlay).

    CSV (format='csv'):
        CSV cannot cleanly represent nested data, so three separate files are
        written.  The caller's filepath is treated as a *stem* — any trailing
        file extension is stripped and the following suffixes are appended:

          <stem>_summary.csv    Two columns: "metric", "value".
                                Rows: every key from summary + top-level
                                shorthand (sharpe, max_drawdown, win_rate,
                                final_value, total_return).
          <stem>_trades.csv     One row per closed trade.
                                Columns: "trade_index", "pnl", "cumulative_pnl".
          <stem>_equity.csv     One row per bar.
                                Columns: "bar_index", "portfolio_value".

        Signals are not written to a separate CSV by default because their
        "date" field may already be represented in string form and the three
        files above cover the metrics required by the roadmap item.

    Args:
        report:   The dict returned by ``Backtester.run_backtest()``.
        filepath: Output path.
                  - For JSON: written as-is; ".json" is appended if the path
                    has no file extension.
                  - For CSV: treated as a stem; three files are created from it.
        format:   ``'csv'`` (default) or ``'json'`` (case-insensitive).

    Returns:
        A dict mapping logical section names to the absolute file path(s)
        that were written.

        JSON:  ``{"json": "/abs/path/to/report.json"}``
        CSV:   ``{"summary": "..._summary.csv",
                  "trades":  "..._trades.csv",
                  "equity":  "..._equity.csv"}``

    Raises:
        ValueError: if ``format`` is not ``'csv'`` or ``'json'``.
    """
    fmt = format.lower()
    if fmt not in ('csv', 'json'):
        raise ValueError(
            f"Unsupported format: {format!r}. Must be 'csv' or 'json'."
        )

    # Ensure the destination directory exists.
    output_dir = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(output_dir, exist_ok=True)

    # Build the flat summary dict that is common to both formats.
    # It merges the nested summary sub-dict with the top-level shorthand keys
    # so callers get everything in one place.
    final_value = report.get("summary", {}).get("Final Value", None)
    initial_cash = (
        final_value - report.get("summary", {}).get("P&L", 0)
        if final_value is not None
        else None
    )
    total_return = (
        round((report.get("summary", {}).get("P&L", 0) / initial_cash) * 100, 4)
        if initial_cash and initial_cash != 0
        else None
    )

    flat_summary = {
        **report.get("summary", {}),
        "sharpe": report.get("sharpe"),
        "max_drawdown": report.get("max_drawdown"),
        "win_rate": report.get("win_rate"),
        "final_value": final_value,
        "total_return_pct": total_return,
    }

    equity_curve = report.get("total_asset_value", [])
    profit_per_trade = report.get("profit_per_trade", [])
    cumulative_pnl = report.get("cumulative_pnl", [])
    signals = report.get("signals", [])

    if fmt == 'json':
        # Normalise filepath: add .json if the path has no extension.
        if not os.path.splitext(filepath)[1]:
            filepath = filepath + '.json'

        payload = {
            "summary": flat_summary,
            "equity_curve": equity_curve,
            "profit_per_trade": profit_per_trade,
            "cumulative_pnl": cumulative_pnl,
            "signals": signals,
        }

        with open(filepath, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=2, default=str)

        logger.info(f"Backtest report exported (JSON): {filepath}")
        return {"json": os.path.abspath(filepath)}

    else:  # csv
        # Strip any extension from filepath to derive the common stem.
        stem = os.path.splitext(os.path.abspath(filepath))[0]

        summary_path = stem + '_summary.csv'
        trades_path = stem + '_trades.csv'
        equity_path = stem + '_equity.csv'

        # --- Summary CSV ---
        with open(summary_path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(['metric', 'value'])
            for key, val in flat_summary.items():
                writer.writerow([key, val])

        # --- Trades CSV ---
        with open(trades_path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(['trade_index', 'pnl', 'cumulative_pnl'])
            for i, pnl in enumerate(profit_per_trade):
                cum = cumulative_pnl[i] if i < len(cumulative_pnl) else ''
                writer.writerow([i + 1, pnl, cum])

        # --- Equity curve CSV ---
        with open(equity_path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(['bar_index', 'portfolio_value'])
            for i, val in enumerate(equity_curve):
                writer.writerow([i, val])

        logger.info(
            f"Backtest report exported (CSV): {summary_path}, {trades_path}, {equity_path}"
        )
        return {
            "summary": summary_path,
            "trades": trades_path,
            "equity": equity_path,
        }
