from core.data_loader import DataLoader
from core.strategy_manager import StrategyManager
import pandas as pd
import os

def test_integration():
    print("Testing Data Loader...")
    dl = DataLoader()
    # Test valid symbol
    df = dl.load_data("BTC-USD", source="Yahoo", days=30, interval="1d")
    if df is not None and not df.empty:
        print(f"✅ Data loaded: {len(df)} rows")
    else:
        print("❌ Data loading failed")
        return

    print("\nTesting Strategy Manager & Backtester...")
    # Mocking config for backtrader (assuming default config exists or providing minimal)
    # StrategyManager uses DataLoader? No, it takes df.
    
    sm = StrategyManager()
    
    # Check available strategies
    strategies = sm.get_available_strategies()
    print(f"Available strategies: {strategies}")
    
    print("Running Backtest (Simulated)...")
    strategy_name = "MACD/RSI"
    if strategy_name not in strategies:
        print(f"❌ {strategy_name} not found in available strategies")
        return

    wrapper = sm.get_strategy(strategy_name)
    if not wrapper:
        print(f"❌ Failed to get wrapper for {strategy_name}")
        return

    results = sm.run_backtest(
        strategy_wrapper=wrapper,
        data=df,
        cash=100000,
        broker_mode="simulated"
    )
    
    if results and isinstance(results, dict):
        summary = results.get('summary', {})
        print("✅ Backtest successful!")
        print("Summary:", summary)
    else:
        print("❌ Backtest failed or returned invalid results:", results)

if __name__ == "__main__":
    test_integration()
