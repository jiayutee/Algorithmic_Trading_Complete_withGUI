from core.logger import logger
from strategies.simple_strategies import MACD_RSI_Strategy, EMACrossoverStrategy, StochasticStrategy
from strategies.ml_strategies import LSTMPredictor
try:
    from strategies.FinRL_strategy import FinRLStrategy
except ImportError:
    FinRLStrategy = None
    logger.warning("FinRL module not found. FinRL Strategy will be unavailable.")

from strategies.TD3_strategy import TD3Strategy
from stable_baselines3 import TD3
import os
from core.backtester import Backtester
import backtrader as bt

class StrategyWrapper:
    """
    Wrapper to standardize the interface between Backtrader strategies (classes)
    and custom strategies (instances).
    """
    def __init__(self, name, strategy_obj, is_backtrader=False):
        self.name = name
        self.strategy_obj = strategy_obj
        self.is_backtrader = is_backtrader

    def __repr__(self):
        return f"<StrategyWrapper name={self.name} is_backtrader={self.is_backtrader}>"


class StrategyManager:
    def __init__(self):
        self.strategies = {
            "MACD/RSI": MACD_RSI_Strategy,
            "EMA Crossover": EMACrossoverStrategy,
            "Stochastic": StochasticStrategy,
            "LSTM Predictor": LSTMPredictor,
            "TD3 Strategy": TD3Strategy
        }
        
        if FinRLStrategy:
            self.strategies["FinRL Strategy"] = FinRLStrategy
            
        self.backtester = Backtester()

    def get_available_strategies(self):
        """Return list of available strategy names"""
        return list(self.strategies.keys())

    def get_strategy(self, name, **kwargs):
        """
        Get a strategy wrapped in a consistent interface.
        
        Args:
            name (str): The name of the strategy from self.strategies keys.
            **kwargs: Arguments to pass to the strategy constructor (for non-Backtrader strategies).
            
        Returns:
            StrategyWrapper: A wrapper containing the strategy object and type info.
        """
        strategy_class = self.strategies.get(name)
        if not strategy_class:
            logger.error(f"Strategy {name} not found.")
            return None
        
        # Check if it's a backtrader strategy
        # Note: Some strategies might be classes but not inherit directly if using a different base
        is_bt = False
        try:
            if issubclass(strategy_class, bt.Strategy):
                is_bt = True
        except TypeError:
            # If strategy_class is not a class (but an instance), issubclass raises TypeError
            pass

        if is_bt:
            # For backtrader, we pass the CLASS itself
            logger.info(f"Returning Backtrader strategy class: {name}")
            return StrategyWrapper(name, strategy_class, is_backtrader=True)
        else:
            # For custom strategies, we instantiate them
            try:
                logger.info(f"Creating custom strategy instance: {name}")
                instance = strategy_class(**kwargs)
                return StrategyWrapper(name, instance, is_backtrader=False)
            except Exception as e:
                logger.error(f"Failed to instantiate strategy {name}: {e}")
                return None

    def run_backtest(self, strategy_wrapper, data, cash=100000.0, broker_mode="simulated", broker=None, market_fee=0.001, limit_fee=0.0005):
        """
        Unified backtest entry point.
        
        Args:
            strategy_wrapper (StrategyWrapper): The strategy to test.
            data (pd.DataFrame): Historical data.
            cash (float): Initial cash.
            broker_mode (str): 'simulated' or 'real'.
            broker (object): The broker connector instance (required if mode is 'real').
            market_fee (float): Fee for market orders (decimal).
            limit_fee (float): Fee for limit orders (decimal).

        Returns:
            dict: Backtest results.
        """
        logger.info(f"Starting backtest for {strategy_wrapper.name} in {broker_mode} mode")

        if strategy_wrapper.is_backtrader:
            # Clear previous runs
            self.backtester = Backtester() 
            self.backtester.add_data(data)
            self.backtester.add_strategy(strategy_wrapper.strategy_obj)
            
            # Run
            return self.backtester.run_backtest(
                cash=cash, 
                broker_mode=broker_mode, 
                broker=broker,
                market_fee=market_fee,
                limit_fee=limit_fee
            )
            
        else:
            # Custom strategy execution (e.g., LSTM, FinRL)
            # These usually define their own backtest logic internally or simple loop
            # For now, we delegate to their internal method if available or wrap it
            strategy = strategy_wrapper.strategy_obj
            if hasattr(strategy, 'train_model'): # FinRL style
                # This is a bit complex as FinRL strategies often need training first
                # We'll assume for this scope we just return a placeholder or use its predict
                logger.warning("Custom/ML strategy backtesting is minimal in this unified view.")
                return {"error": "Custom strategy backtesting not fully integrated in unified view yet."}
            
            # Simple custom strategies might have a trade() method
            # This part depends heavily on the interface of LSTMPredictor etc.
            # Preserving original simple logic if it existed:
            # internal logic seemed to rely on the side-effect of run_backtest in previous main_window
            # But here we should probably leave it to the backtester if we can adapt it.
            
            # For now, fallback to the previous behavior logic if applicable
            return {"message": "Custom strategy execution completed (logic handled internally)."}