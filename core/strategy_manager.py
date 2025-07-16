# from strategies.technical.macd_rsi import MACD_RSI_Strategy
# from strategies.technical.ema_crossover import EMA_Crossover_Strategy
# from strategies.ml.lstm_predictor import LSTMPredictor
from strategies.simple_strategies import MACD_RSI_Strategy, EMACrossoverStrategy, StochasticStrategy
from strategies.ml_strategies import LSTMPredictor
from strategies.FinRL_strategy import FinRLStrategy
from core.backtester import Backtester


class StrategyManager:
    def __init__(self):
        self.strategies = {
            "MACD/RSI": MACD_RSI_Strategy,
            "EMA Crossover": EMACrossoverStrategy,
            "Stochastic": StochasticStrategy,
            "LSTM Predictor": LSTMPredictor,
            "FinRL Strategy": FinRLStrategy
        }
        self.backtester = Backtester()

    def get_strategy(self, name, **kwargs):
        strategy_class = self.strategies.get(name)
        if strategy_class and kwargs:
            return strategy_class(**kwargs)
        return strategy_class

    def run_backtest(self, strategy, data, cash=100000.0):
        self.backtester.add_data(data)
        self.backtester.add_strategy(strategy)
        results = self.backtester.run_backtest(cash=cash)
        return results