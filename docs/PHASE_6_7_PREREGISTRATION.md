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

---

# Results (run 2026-09-19; protocol exactly as above, nothing tuned)

`training_ground/results/phase_6_7.json`. Reproduce: `python training_ground/experiments_6_7.py`.
Window 2022-09-10 -> 2026-09-18 (1,470 days, 8 symbols), 7-day rebalance, 0.1% fees.

| strategy | Sharpe | ann. return | ann. vol | max drawdown | turnover / yr |
|---|---|---|---|---|---|
| Equal weight (benchmark) | 0.68 | 25.5% | 60.6% | -67.0% | 2.3 |
| XSMOM (top 3 of 8 by 28-day return) | 0.84 | 39.8% | 64.1% | -72.5% | 31.7 |
| TSMOM (long if own 28-day return > 0) | 0.84 | 28.4% | 38.3% | -37.5% | 13.4 |

| comparison | Sharpe difference | 97.5% interval | halves | verdict |
|---|---|---|---|---|
| M1 XSMOM vs equal weight | +0.16 | [-0.24, +0.53] | +0.28 / +0.04 | **no evidence** |
| M2 TSMOM vs equal weight | +0.17 | [-0.59, +0.85] | -0.08 / +0.40 | **no evidence** |

**Neither met the first criterion.** Both point estimates are positive, but the intervals comfortably include zero, and
TSMOM's edge is entirely in the second half.

## Reading it
- **TSMOM did what a trend filter does, not what an alpha does:** volatility 38% vs 61% and max drawdown -37.5% vs -67.0%
  for about the same Sharpe as XSMOM. That is risk reduction by sitting in cash during down-trends. The Sharpe
  difference is not distinguishable from noise, but the drawdown halving is large and mechanically expected. Whether that
  is worth having depends on the goal: it does not make more money per unit of risk *provably*, it makes the ride smoother.
  (Phase 6.6's vol-targeting did not de-risk; a trend filter did. Worth a separately pre-registered hypothesis, not a claim.)
- **XSMOM's higher return came with higher risk** (drawdown -72.5%) and 14x the turnover of the benchmark.
- **Context, not tested, do not act on it:** other lookbacks (L = 14: XSMOM Sharpe 1.09; L = 56: 0.94; benchmark 0.70-0.71) point the same
  direction as L = 28 for XSMOM. Picking the best lookback after seeing this would be exactly the tuning this protocol forbids.
- **Caveats stated in advance still apply:** survivorship (today's large caps), one BTC-dominated regime, only 8 assets.
- Natural next *pre-registered* hypothesis if pursued: does a trend filter reduce drawdown at equal Sharpe (a drawdown
  claim, tested on more assets or a longer history), rather than another Sharpe comparison.
