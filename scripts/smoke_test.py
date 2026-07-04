#!/usr/bin/env python3
"""QA smoke test — Day 5 T6e.

Exercises the two end-to-end slices that the pytest suite covers with
synthetic data, but here against REAL market data (yfinance) and by calling
the SimulatedBroker order-entry logic directly (no Qt GUI involved):

  1. Backtest end-to-end for all 3 strategies (MACD+RSI, EMA Crossover,
     Stochastic) against a real symbol/date range, confirming each produces
     a complete result: trades (signals), an equity curve
     (total_asset_value), and a results dict with the expected keys.

  2. Order entry smoke test: SimulatedBroker.submit_order for both buy and
     sell sides, confirming fills, position updates, and P&L — the same
     code path ui/main_window.py's place_order() calls
     (self.current_broker.submit_order(...)), exercised directly.

This script is meant to be run ad hoc (`python scripts/smoke_test.py`) as a
quick, human-readable sanity check; it is NOT collected by pytest (see
pytest.ini norecursedirs = ... scripts ...) and intentionally prints a
pass/fail summary rather than using assert/raise, so a single failing area
doesn't stop the rest of the smoke test from running.
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import yfinance as yf

from core.backtester import Backtester
from strategies.simple_strategies import (
    MACD_RSI_Strategy,
    EMACrossoverStrategy,
    StochasticStrategy,
)
from brokers.simulatedbroker import SimulatedBroker, OrderStatus

SYMBOL = "AAPL"
PERIOD = "1y"

results_log = []


def log(area, ok, detail=""):
    results_log.append((area, ok, detail))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {area}" + (f" — {detail}" if detail else ""))


def fetch_sample_data(symbol=SYMBOL, period=PERIOD):
    df = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


def smoke_test_strategy(name, strategy_cls, df):
    try:
        b = Backtester()
        b.add_data(df.copy())
        b.add_strategy(strategy_cls)
        res = b.run_backtest(cash=100_000, benchmark_ticker="SPY")

        if "error" in res:
            log(f"backtest:{name}", False, f"Backtester returned error: {res['error']}")
            return

        required_keys = {"sharpe", "max_drawdown", "win_rate", "summary",
                          "total_asset_value", "signals"}
        missing = required_keys - set(res.keys())
        if missing:
            log(f"backtest:{name}", False, f"Missing result keys: {missing}")
            return

        n_signals = len(res["signals"])
        n_equity_points = len(res["total_asset_value"])
        final_value = res["summary"].get("Final Value")

        log(
            f"backtest:{name}", True,
            f"{n_signals} signal(s), {n_equity_points} equity points, "
            f"final value ${final_value:,.2f}" if final_value is not None
            else f"{n_signals} signal(s), {n_equity_points} equity points"
        )
    except Exception as e:
        log(f"backtest:{name}", False, f"Exception: {e}\n{traceback.format_exc()}")


def smoke_test_order_entry():
    """Exercise SimulatedBroker.submit_order directly (the same call
    ui/main_window.py's place_order() makes), for both buy and sell sides."""
    try:
        broker = SimulatedBroker(initial_balance=10_000.0, market_fee=0.001, limit_fee=0.0005)
        broker.market_data["AAPL"] = 150.0

        # BUY
        buy_order = broker.submit_order("AAPL", qty=10.0, side="buy",
                                         order_type="market", execution_price=150.0)
        if buy_order.status != OrderStatus.FILLED:
            log("order_entry:buy", False, f"Buy order not filled, status={buy_order.status}")
        else:
            pos = broker.get_position("AAPL")
            ok = pos is not None and abs(pos.qty - 10.0) < 1e-6
            log("order_entry:buy", ok,
                f"filled {buy_order.filled_qty} @ ${buy_order.filled_avg_price:.2f}, "
                f"position={pos.qty if pos else None}, balance=${broker.balance:.2f}")

        # SELL (partial, then full) to exercise P&L update
        sell_order = broker.submit_order("AAPL", qty=10.0, side="sell",
                                          order_type="market", execution_price=160.0)
        account = broker.get_account_info()
        if sell_order.status != OrderStatus.FILLED:
            log("order_entry:sell", False, f"Sell order not filled, status={sell_order.status}")
        else:
            pos_after = broker.get_position("AAPL")
            pnl = account["pnl"]
            # Expect ~ (160-150)*10 - fees = ~100 - fees, so pnl should be positive
            ok = pos_after is None and pnl > 0
            log("order_entry:sell", ok,
                f"filled {sell_order.filled_qty} @ ${sell_order.filled_avg_price:.2f}, "
                f"position_after={pos_after}, account_pnl=${pnl:.2f}")

        broker.close()
    except Exception as e:
        log("order_entry", False, f"Exception: {e}\n{traceback.format_exc()}")


def main():
    print(f"=== QA Smoke Test — Day 5 T6e — symbol={SYMBOL}, period={PERIOD} ===\n")

    print("--- Fetching real market data ---")
    try:
        df = fetch_sample_data()
        print(f"Fetched {len(df)} rows for {SYMBOL} ({df.index[0].date()} to {df.index[-1].date()})\n")
    except Exception as e:
        log("data_fetch", False, f"Exception: {e}\n{traceback.format_exc()}")
        df = None

    print("--- Backtest smoke tests (3 strategies) ---")
    if df is not None and len(df) > 60:
        smoke_test_strategy("MACD+RSI", MACD_RSI_Strategy, df)
        smoke_test_strategy("EMA Crossover", EMACrossoverStrategy, df)
        smoke_test_strategy("Stochastic", StochasticStrategy, df)
    else:
        log("backtest", False, "Skipped — no usable market data fetched")

    print("\n--- Order entry smoke test (SimulatedBroker, no GUI) ---")
    smoke_test_order_entry()

    print("\n=== Summary ===")
    n_pass = sum(1 for _, ok, _ in results_log if ok)
    n_total = len(results_log)
    for area, ok, detail in results_log:
        print(f"  [{'PASS' if ok else 'FAIL'}] {area}")
    print(f"\n{n_pass}/{n_total} checks passed")

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
