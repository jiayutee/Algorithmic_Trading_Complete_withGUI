# from strategies.technical.macd_rsi import MACD_RSI_Strategy
# from strategies.technical.ema_crossover import EMA_Crossover_Strategy
# from strategies.ml.lstm_predictor import LSTMPredictor
from strategies.simple_strategies import MACD_RSI_Strategy, EMACrossoverStrategy, StochasticStrategy
from strategies.ml_strategies import LSTMPredictor
from strategies.FinRL_strategy import FinRLStrategy
from strategies.TD3_strategy import TD3Strategy
from stable_baselines3 import TD3
import os
from core.backtester import Backtester
import backtrader as bt


class StrategyManager:
    def __init__(self):
        self.strategies = {
            "MACD/RSI": MACD_RSI_Strategy,
            "EMA Crossover": EMACrossoverStrategy,
            "Stochastic": StochasticStrategy,
            "LSTM Predictor": LSTMPredictor,
            "FinRL Strategy": FinRLStrategy,
            "TD3 Strategy": TD3Strategy
        }
        self.backtester = Backtester()

    def get_strategy(self, name, **kwargs):
        strategy_class = self.strategies.get(name)
        
        # Check if it's a backtrader strategy
        if issubclass(strategy_class, bt.Strategy):
            # For backtrader strategies, return the CLASS, not instance
            print(f"✅ Returning backtrader strategy class: {strategy_class}")
            return strategy_class
        else:
            # For custom strategies (like LSTM), return instance
            print(f"✅ Creating custom strategy instance: {strategy_class}")
            # if name == "TD3 Strategy":
            #     model_path = "trained_models/td3_model.zip"
            #     if os.path.exists(model_path):
            #         model = TD3.load(model_path)
            #         kwargs['model'] = model
            #     else:
            #         print(f"Warning: TD3 model not found at {model_path}. Please train the model first.")
            #         return None # Or raise an error
            return strategy_class(**kwargs)
    
        # if strategy_class:
        #     if name == "TD3 Strategy":
        #         model_path = "trained_models/td3_model.zip"
        #         if os.path.exists(model_path):
        #             model = TD3.load(model_path)
        #             kwargs['model'] = model
        #         else:
        #             print(f"Warning: TD3 model not found at {model_path}. Please train the model first.")
        #             return None # Or raise an error
        #     return strategy_class(**kwargs)
        # return None

    def run_backtest(self, strategy, data, cash=100000.0):
        self.backtester.add_data(data)
        self.backtester.add_strategy(strategy)
        results = self.backtester.run_backtest(cash=cash)
        return results