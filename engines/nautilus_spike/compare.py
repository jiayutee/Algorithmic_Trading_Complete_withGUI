"""Run the SAME daily dataset through backtrader (existing engine) and NautilusTrader; compare on identical metrics.

    ~/.venvs/algotrader311/bin/python -m engines.nautilus_spike.compare [--symbol BTCUSDT --days 1500 --repeats 5]

Both equity curves are scored by the same function (core.risk_sizing.perf_stats) so the comparison isolates the ENGINES
(fill model, indicator seeding, accounting), not two different metric definitions. Wall-clock is the median of N runs.
A second, clearly synthetic benchmark scales the bar count to show runtime growth.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from decimal import Decimal

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

CASH = 100_000.0
FEE = 0.001


def load_dataset(symbol: str = "BTCUSDT", days: int = 1500) -> pd.DataFrame:
    from core.order_flow_data import fetch_klines
    kl = fetch_klines(symbol, days=days)
    return kl[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def score(equity: pd.Series) -> dict:
    from core.risk_sizing import perf_stats
    r = equity.pct_change().dropna()
    s = perf_stats(r)
    return {"final_value": float(equity.iloc[-1]), "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
            "sharpe": float(s["sharpe"]), "max_drawdown": float(s["max_drawdown"]), "bars": int(len(equity))}


# ----------------------------------------------------------------------------------------------- backtrader

def run_backtrader(df: pd.DataFrame, cash: float = CASH, fee: float = FEE) -> dict:
    from core.backtester import Backtester
    from strategies.simple_strategies import EMACrossoverStrategy
    logging.disable(logging.CRITICAL)
    t0 = time.perf_counter()
    b = Backtester()
    b.add_data(df.copy())
    b.add_strategy(EMACrossoverStrategy)
    rep = b.run_backtest(cash=cash, benchmark_ticker=None, market_fee=fee, limit_fee=fee)
    wall = time.perf_counter() - t0
    logging.disable(logging.NOTSET)
    eq = pd.Series(rep["total_asset_value"], index=pd.to_datetime(rep["dates"]))
    strat = b.cerebro.runstrats[0][0]
    fills = [(pd.Timestamp(s["date"]), s["type"], float(s["price"]), float(s["qty"])) for s in strat.signals]
    return {"equity": eq, "fills": fills, "wall_s": wall, "orders": strat.order_count}


def run_backtrader_bare(df: pd.DataFrame, cash: float = CASH, fee: float = FEE) -> float:
    """Wall-clock of a plain ``cerebro.run()`` (no analyzers, no report) -- the fair engine-only runtime."""
    import backtrader as bt
    from strategies.simple_strategies import EMACrossoverStrategy
    logging.disable(logging.CRITICAL)
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(bt.feeds.PandasData(dataname=df.copy()))
    cerebro.addstrategy(EMACrossoverStrategy)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=fee)
    t0 = time.perf_counter()
    cerebro.run()
    wall = time.perf_counter() - t0
    logging.disable(logging.NOTSET)
    return wall


# ------------------------------------------------------------------------------------------------ nautilus

def run_nautilus(df: pd.DataFrame, cash: float = CASH) -> dict:
    from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.config import LoggingConfig
    from nautilus_trader.model.currencies import USDT
    from nautilus_trader.model.data import BarType
    from nautilus_trader.model.enums import AccountType, OmsType
    from nautilus_trader.model.identifiers import Venue
    from nautilus_trader.model.objects import Money
    from nautilus_trader.persistence.wranglers import BarDataWrangler
    from nautilus_trader.test_kit.providers import TestInstrumentProvider
    from engines.nautilus_spike.nautilus_ema import EMACrossConfig, EMACrossNautilus

    t0 = time.perf_counter()
    instrument = TestInstrumentProvider.btcusdt_binance()           # maker/taker fee 0.1%, matches the backtrader fee
    bar_type = BarType.from_str(f"{instrument.id}-1-DAY-LAST-EXTERNAL")
    nd = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
    nd.index = pd.to_datetime(nd.index).tz_localize("UTC") + pd.Timedelta(days=1)   # Binance index = bar OPEN; Nautilus wants bar CLOSE
    bars = BarDataWrangler(bar_type, instrument).process(nd)
    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
    engine.add_venue(venue=Venue("BINANCE"), oms_type=OmsType.NETTING, account_type=AccountType.MARGIN,
                     base_currency=USDT, starting_balances=[Money(cash, USDT)], default_leverage=Decimal(1))
    engine.add_instrument(instrument)
    engine.add_data(bars)
    strat = EMACrossNautilus(EMACrossConfig(instrument_id=instrument.id, bar_type=bar_type))
    engine.add_strategy(strat)
    setup_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    engine.run()
    run_s = time.perf_counter() - t1
    fills = engine.trader.generate_order_fills_report()
    fills_out = []
    if fills is not None and len(fills):
        for _, row in fills.iterrows():
            t = pd.Timestamp(row["ts_last"])
            t = t.tz_convert(None) if t.tzinfo is not None else t
            fills_out.append((t, str(row["side"]), float(row["avg_px"]), float(row["filled_qty"])))
    idx = pd.to_datetime([pd.Timestamp(ts, unit="ns") for ts, _ in strat.equity]).tz_localize(None) - pd.Timedelta(days=1)
    eq = pd.Series([e for _, e in strat.equity], index=idx)
    out = {"equity": eq, "fills": fills_out, "wall_s": setup_s + run_s, "setup_s": setup_s, "run_s": run_s, "orders": strat.order_count}
    engine.dispose()
    return out


# ------------------------------------------------------------------------------------------------ comparison

def compare(df: pd.DataFrame, repeats: int = 5) -> dict:
    bt_runs = [run_backtrader(df) for _ in range(repeats)]
    nt_runs = [run_nautilus(df) for _ in range(repeats)]
    bt, nt = bt_runs[-1], nt_runs[-1]
    common = bt["equity"].index.intersection(nt["equity"].index)
    bt_eq, nt_eq = bt["equity"].reindex(common), nt["equity"].reindex(common)
    # start both scorecards from the first bar where either engine has left the initial cash, so warm-up isn't double counted
    res = {"bars": int(len(common)), "window": [str(common[0].date()), str(common[-1].date())],
           "backtrader": {**score(bt_eq), "wall_s_median": statistics.median(r["wall_s"] for r in bt_runs),
                          "bare_cerebro_run_s_median": statistics.median(run_backtrader_bare(df) for _ in range(repeats)), "orders": bt["orders"],
                          "fills": len(bt["fills"])},
           "nautilus": {**score(nt_eq), "wall_s_median": statistics.median(r["wall_s"] for r in nt_runs),
                        "run_only_s_median": statistics.median(r["run_s"] for r in nt_runs), "orders": nt["orders"], "fills": len(nt["fills"])},
           "equity_max_abs_diff": float((bt_eq - nt_eq).abs().max()),
           "equity_corr_of_returns": float(bt_eq.pct_change().corr(nt_eq.pct_change())),
           "first_fills": {"backtrader": [(str(t.date()), k, round(p, 2), round(q, 6)) for t, k, p, q in bt["fills"][:4]],
                           "nautilus": [(str(t.date()), k, round(p, 2), round(q, 6)) for t, k, p, q in nt["fills"][:4]]}}
    return res


def synthetic(n: int, seed: int = 0) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(r.normal(0, 0.01, n)))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.002, "Low": np.minimum(open_, close) * 0.998,
                         "Close": close, "Volume": 1e6}, index=pd.date_range("2000-01-01", periods=n, freq="D"))


def scaling(sizes=(1_500, 10_000, 50_000)) -> list:
    out = []
    for n in sizes:
        d = synthetic(n)
        b, nn = run_backtrader(d), run_nautilus(d)
        out.append({"bars": n, "backtrader_s": b["wall_s"], "backtrader_bare_s": run_backtrader_bare(d), "nautilus_s": nn["wall_s"], "nautilus_run_only_s": nn["run_s"]})
        print(f"  {n:>7} daily bars: backtrader {b['wall_s']:.2f}s (bare cerebro {out[-1]['backtrader_bare_s']:.2f}s) | nautilus {nn['wall_s']:.2f}s (engine.run {nn['run_s']:.2f}s)", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--no-scaling", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"))
    args = ap.parse_args(argv)
    df = load_dataset(args.symbol, args.days)
    print(f"dataset {args.symbol}: {len(df)} daily bars {df.index[0].date()} -> {df.index[-1].date()}", flush=True)
    res = compare(df, args.repeats)
    res["dataset"] = {"symbol": args.symbol, "bars": int(len(df))}
    if not args.no_scaling:
        print("scaling (synthetic random walk):", flush=True)
        res["scaling_synthetic"] = scaling()
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "scaling_synthetic"}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
