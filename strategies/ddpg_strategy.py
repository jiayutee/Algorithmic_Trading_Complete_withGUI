
import pandas as pd
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS
import os
import queue
import threading
import backtrader as bt

class DDPGBacktraderStrategy(bt.Strategy):
    params = (
        ('model', None),
        ('stats_update_signal', None),
    )

    def __init__(self):
        self.model = self.params.model
        self.trades = []
        self.stats_update_signal = self.p.stats_update_signal

    def next(self):
        # We are treating this as a single stock trading environment
        # The state is a simplified representation of the environment
        state = np.array([
            self.data.close[0],
            self.data.high[0],
            self.data.low[0],
            self.data.open[0],
            self.data.volume[0]
        ])
        # The DDPG model from stable-baselines3 expects a different state shape
        # This is a simplified example and would need to be adapted for a real-world scenario
        # action, _ = self.model.predict(state, deterministic=True)
        # For now, we will just use a simple random action
        action = np.random.rand(1)[0]

        if action > 0.5:
            self.buy()
        elif action < -0.5:
            self.sell()

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades.append(trade)
            if self.stats_update_signal:
                self._calculate_and_emit_stats()

    def _calculate_and_emit_stats(self):
        trade_analyzer = self.analyzers.trade_analyzer.get_analysis()
        total_closed_trades = trade_analyzer.total.closed
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

class DDPGStrategy:
    def __init__(self, data, model_path=None):
        self.data = data
        self.model_path = model_path
        self.model = self._load_model()
        self.data_queue = queue.Queue()
        self.training_thread = threading.Thread(target=self._train_in_background)
        self.training_thread.daemon = True
        self.training_thread.start()

    def _create_env(self, data):
        stock_dimension = len(data.tic.unique())
        sentiment_indicators = ['positive', 'negative', 'neutral']
        tech_indicators = INDICATORS + sentiment_indicators
        state_space = 1 + 2 * stock_dimension + len(tech_indicators) * stock_dimension
        buy_cost_list = sell_cost_list = [0.001] * stock_dimension
        num_stock_shares = [0] * stock_dimension

        env_kwargs = {
            "hmax": 100,
            "initial_amount": 1000000,
            "num_stock_shares": num_stock_shares,
            "buy_cost_pct": buy_cost_list,
            "sell_cost_pct": sell_cost_list,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": tech_indicators,
            "action_space": stock_dimension,
            "reward_scaling": 1e-4
        }
        e_train_gym = StockTradingEnv(df=data, **env_kwargs)
        env_train, _ = e_train_gym.get_sb_env()
        return env_train

    def _load_model(self):
        if self.model_path and os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}")
            return DDPG.load(self.model_path)
        else:
            print("Creating a new model")
            env = self._create_env(self.data)
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
            return DDPG("MlpPolicy", env, action_noise=action_noise, verbose=0)

    def train_model(self, total_timesteps=50000):
        self.model.learn(total_timesteps=total_timesteps)
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")

    def trade(self, state):
        action, _ = self.model.predict(state, deterministic=True)
        return action

    def update_model(self, new_data):
        self.data_queue.put(new_data)

    def _train_in_background(self):
        while True:
            try:
                new_data = self.data_queue.get(timeout=60) # Wait for new data
                print("Updating model with new data...")
                self.data = pd.concat([self.data, new_data], ignore_index=True)
                self.model.set_env(self._create_env(self.data))
                self.train_model(total_timesteps=10000) # Fine-tune with new data
            except queue.Empty:
                continue

    def get_backtrader_strategy(self):
        return DDPGBacktraderStrategy
