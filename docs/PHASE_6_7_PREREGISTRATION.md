# Phase 6.7 pre-registration: crypto momentum (cross-sectional and time-series)

Written 2026-09-19, **before any code for this experiment existed or was run**. Momentum is the best-documented
non-ML effect in crypto, needs no model to overfit, and reuses the Phase 6.5/7.1 data. Same rules as before: fixed
parameters, nothing tuned, anything changed after seeing results is labelled exploratory.

## Setup (fixed)
- Universe / data: the Phase 6.5 universe (8 Binance USDT pairs, up to 1,500 daily bars, aligned on common dates).
- Decisions use only returns strictly before the day they are applied to (rebalance at the close of day i-1).
- Rebalance every **7 days**. Trailing window **L = 28 days**. Costs: 0.1% of one-way turnover, weights drift between rebalances.
- **XSMOM (cross-sectional):** rank the 8 symbols by trailing-28-day return; hold the top **3** at equal weight (1/3 each).
- **TSMOM (time-series):** each symbol gets weight 1/8 if its own trailing-28-day return > 0, else 0 (cash earns 0).
- **Benchmark EW:** 1/8 in every symbol, same 7-day rebalance and fees.
- Evaluated from the first day a full 28-day window exists; identical dates for all three.

## Comparisons (K = 2, Bonferroni -> 97.5% intervals)
| id | question | statistic |
|----|----------|-----------|
| M1 | Does XSMOM beat equal weight? | Sharpe(XSMOM) - Sharpe(EW) |
| M2 | Does TSMOM beat equal weight? | Sharpe(TSMOM) - Sharpe(EW) |

Interval: paired moving-block bootstrap over dates (10-day blocks, 2,000 resamples).

## A comparison counts as a finding only if ALL hold
1. Lower bound of the 97.5% interval on the Sharpe difference > 0.
2. Difference > 0 in both halves of the evaluated period.
3. Max drawdown of the strategy no worse than the benchmark's by more than 5 percentage points.
Otherwise "no evidence". For context only (NOT used to pick anything): the same two strategies at L = 14 and L = 56.

## Limits stated in advance
- **Survivorship bias:** the 8 symbols are large caps that exist today. This flatters every arm, the equal-weight
  benchmark most of all, but cross-sectional selection among survivors can also mislead in either direction.
- One regime (2022-2026), heavily driven by BTC beta; a strategy that is merely less exposed can look better on
  Sharpe in a falling market and worse in a rising one.
- 8 assets is a small cross-section; 3 of 8 is a coarse portfolio.
