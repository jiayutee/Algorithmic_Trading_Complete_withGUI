import backtrader as bt
import numpy as np
import pandas as pd
from stable_baselines3 import TD3

class TD3Strategy(bt.Strategy):
    params = (
        ('model', None),
        ('stats_update_signal', None),
    )

    def __init__(self):
        self.model = self.params.model
        self.trades = []
        self.stats_update_signal = self.p.stats_update_signal

    def next(self):
        if self.model is None:
            return

        # Create a simple observation from the current data
        # This is a placeholder and should be replaced with your actual environment's observation space
        observation = np.array([
            self.data.open[0],
            self.data.high[0],
            self.data.low[0],
            self.data.close[0],
            self.data.volume[0]
        ])

        # stable_baselines3 models expect a batch of observations, so reshape
        action, _states = self.model.predict(observation.reshape(1, -1), deterministic=True)
        action = action[0] # Get the single action from the batch

        # Simple trading logic based on the action
        if action > 0.5: # Example threshold for buying
            self.buy()
        elif action < -0.5: # Example threshold for selling
            self.sell()

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades.append(trade)
            if self.stats_update_signal:
                self._calculate_and_emit_stats()

    def _calculate_and_emit_stats(self):
        # This method will calculate and emit partial statistics
        # It needs access to analyzers, which are available after cerebro.run()
        # For real-time updates, we can only get basic info from broker and trades list
        
        # Ensure analyzers are available
        if not hasattr(self, 'analyzers') or not hasattr(self.analyzers, 'trade_analyzer'):
            return

        trade_analyzer = self.analyzers.trade_analyzer.get_analysis()
        total_closed_trades = trade_analyzer.total.closed if trade_analyzer and trade_analyzer.total else 0
        win_rate = (trade_analyzer.won.total / total_closed_trades) * 100 if total_closed_trades > 0 else 0
        
        pnl_per_trade = [t.pnlcomm for t in self.trades]

        summary = {
            "Number of Closed Trades": total_closed_trades,
            "Win Rate": f"{win_rate:.2f}%",
            "Average Profit per Trade": np.mean(pnl_per_trade) if pnl_per_trade else 0,
            "Median Profit per Trade": np.median(pnl_per_trade) if pnl_per_trade else 0,
            "Final Value": self.broker.getvalue()
        }
        stats = {
            "summary": summary,
            "cumulative_pnl": np.cumsum(pnl_per_trade).tolist(),
            "profit_per_trade": pnl_per_trade,
            "total_asset_value": [self.broker.getvalue()]
        }
        self.stats_update_signal.emit(stats)