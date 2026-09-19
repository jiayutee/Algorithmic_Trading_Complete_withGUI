# Phase 8.0 spike: NautilusTrader vs backtrader

**This is an evaluation, not a migration decision.** It changes nothing outside `engines/nautilus_spike/`.
Whether to adopt Nautilus is a separate owner decision.

Reproduce (Python 3.11 env from `scripts/setup_py311_env.sh`):

```bash
~/.venvs/algotrader311/bin/python -m engines.nautilus_spike.compare        # writes results.json
~/.venvs/algotrader311/bin/python -m pytest engines/nautilus_spike -q
```

## What was compared
`EMACrossoverStrategy` (EMA 12/26, long and short, 10% of cash per trade, exit-only on the opposite cross) ported
rule-for-rule to Nautilus's `on_bar` model (`nautilus_ema.py`), run on the same data: Binance BTCUSDT daily,
1,499 bars, 2022-08-12 -> 2026-09-18, start cash $100,000, 0.1% taker fee in both.
Both equity curves are scored by the *same* function (`core.risk_sizing.perf_stats`), so this compares engines, not
two metric definitions. Runtimes are the median of 5 runs.

## Result on the real dataset

| | backtrader (existing) | NautilusTrader 1.221.0 |
|---|---|---|
| Final portfolio value | $115,378.81 | $115,380.75 |
| Total return | 15.379% | 15.381% |
| Sharpe (daily, annualised) | 0.8666 | 0.8667 |
| Max drawdown | -4.1162% | -4.1162% |
| Orders / fills | 49 / 49 | 49 / 49 |
| Wall-clock, whole run | 0.18 s (via the project's `Backtester`, includes its analyzers/report) | 0.09 s (setup + run) |
| Wall-clock, engine only | 0.12 s (bare `cerebro.run()`) | 0.08 s (`engine.run()`) |

Equity curves differ by at most **$1.98** (0.002%); correlation of daily returns 0.99999999.
**On this strategy and dataset the two engines produce the same answer.**

## Runtime as bars grow (synthetic random walk, same strategy)

| daily bars | backtrader via `Backtester` | backtrader bare `cerebro.run()` | Nautilus setup + run |
|---|---|---|---|
| 1,500 | 0.21 s | 0.12 s | 0.13 s |
| 10,000 | 1.29 s | 0.87 s | 0.38 s |
| 50,000 | 6.22 s | 4.27 s | 2.03 s |

Nautilus is about 1.5x faster on the real dataset and about 2.2x faster at 50,000 bars against the *fair* baseline
(bare `cerebro.run()`). That is a modest gain, not an order of magnitude, and backtest speed is not currently a bottleneck here.

## Differences found (each is a real behavioural difference between the engines)
1. **Fill price.** backtrader fills a market order at the **next bar's open**. Nautilus filled at the **signal bar's close**
   (first BTC trade: 20,771.61 vs 20,771.59). For 24/7 crypto on daily bars these are within cents; for instruments with
   overnight gaps (equities) Nautilus's default is more optimistic, because you cannot trade at the close you just observed.
   Encoded in `test_nautilus_spike.py` using a gapped fixture.
2. **Indicator seeding.** The engines seed the EMA differently, so the *first* cross can land a day apart (seen on a synthetic
   fixture; on the real data all 49 orders matched). Every later fill agreed.
3. **Timestamps.** Nautilus expects a bar's timestamp to be its *close*. Binance klines are indexed by *open*, so the spike shifts the
   index by one bar. Getting this wrong gives lookahead. Easy to get wrong when porting.
4. **Account model.** Shorting a spot pair needs a Nautilus *margin* account (leverage 1); a cash account refuses shorts.

## What this does NOT show
- One strategy, one dataset, daily bars, one crypto pair. Nothing about intraday, multiple assets, limit/stop orders or partial fills.
- Nautilus's **live** side (order lifecycle, reconnection, reconciliation with a venue) was not exercised. That, not backtest
  parity, is the real argument for or against it.
- Backtest agreement says nothing about whether the strategy makes money (see Phases 6.5-6.9: it does not show an edge).

## What migrating would actually cost (context for the decision, not a recommendation)
- Nautilus needs Python >= 3.11; the app, orchestrator and collectors run on 3.9.
- The report pipeline (pyfolio, alpha/beta), the broker layer, the live-order guard, the rationale hooks and both UIs are built
  around backtrader strategies. All six strategies would need porting and re-verifying.
- Where Nautilus could genuinely help is the *execution* gaps in the platform review (durable order lifecycle, reconnect
  recovery, reconciliation, one event-driven path from signal to fill). A follow-up spike aimed there (Binance testnet or IB paper) would
  be more informative than more backtest parity.
