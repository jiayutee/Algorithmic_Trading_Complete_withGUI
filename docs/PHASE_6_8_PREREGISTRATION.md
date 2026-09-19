# Phase 6.8 pre-registration: does a trend filter cut drawdown on assets it was NOT observed on?

Written 2026-09-19, **before the 8 symbols below were fetched or any 6.8 code existed**. Phase 6.7 found no
significant Sharpe edge for time-series momentum (TSMOM) but *observed* a large drawdown reduction (-67% -> -37.5%).
That observation was made after looking at the data, so by itself it is a hypothesis, not a result. This tests it
on a different, fixed set of assets, with the identical TSMOM rule and no parameter choices.

## Setup (fixed)
- **Universe (chosen now, not from data):** LINKUSDT, DOTUSDT, AVAXUSDT, ATOMUSDT, TRXUSDT, NEARUSDT, UNIUSDT, FILUSDT.
  Symbols with fewer than 1,100 daily bars are dropped and reported; if fewer than 5 remain the experiment is void.
- **Strategies (identical to Phase 6.7):** TSMOM = weight 1/N on each symbol whose trailing-28-day return is > 0, else cash;
  benchmark EW = 1/N in every symbol. 7-day rebalance, 0.1% fees, weights drift, decisions use only past returns.
- Same evaluated dates for both. Up to 1,500 daily bars.

## Hypotheses (K = 2, Bonferroni -> 97.5% intervals; paired moving-block bootstrap over dates, 30-day blocks, 2,000 resamples)
Blocks are 30 days (not 10) because drawdown is a path statistic and short blocks understate it.
| id | claim | statistic |
|----|-------|-----------|
| D1 | TSMOM's maximum drawdown is shallower than equal weight | maxDD(TSMOM) - maxDD(EW) (drawdowns are negative numbers, so > 0 means shallower) |
| D2 | TSMOM does not give up risk-adjusted return (non-inferiority) | Sharpe(TSMOM) - Sharpe(EW) |

## Success criteria -- "drawdown reduction confirmed" only if ALL hold
1. D1: lower bound of the 97.5% interval > 0.
2. D1: shallower drawdown in BOTH halves of the period.
3. D2: lower bound of the 97.5% interval > -0.50 (TSMOM is not materially worse on Sharpe).
Otherwise "not confirmed". A confirmed result means "a trend filter makes the ride smoother at similar Sharpe",
**not** "it makes more money".

## Limits stated in advance
- Same calendar period as Phase 6.7 and crypto assets are highly correlated: this is out-of-sample in the
  cross-section, not in time. It cannot rule out that the result is specific to the 2022-2026 regime (a long bull-bear cycle).
- Survivorship: all 8 symbols exist today.
- Drawdown is a noisy statistic with essentially one or two big episodes in the sample; intervals will be wide.

---

# Results (run 2026-09-19; protocol exactly as above, nothing tuned)

`training_ground/results/phase_6_8.json`. Reproduce: `python training_ground/experiments_6_8.py`.
All 8 pre-registered symbols had enough history. Window 2022-09-10 -> 2026-09-18 (1,470 days).

| | Sharpe | ann. return | ann. vol | max drawdown |
|---|---|---|---|---|
| Equal weight | 0.30 | -3.5% | 69.6% | -78.3% |
| TSMOM (28-day trend filter) | 0.53 | 14.6% | 44.3% | -63.7% |

| test | value | 97.5% interval | halves |
|---|---|---|---|
| D1 maxDD(TSMOM) - maxDD(EW) | **+14.6 pp shallower** | [+1.2, +41.4] pp | +24.3 / +14.6 pp |
| D2 Sharpe(TSMOM) - Sharpe(EW) | +0.23 | [-0.40, +0.72] (bound -0.50) | - |

**Verdict: drawdown reduction CONFIRMED** on all three pre-registered criteria (D1 interval above 0, shallower in both halves,
Sharpe not materially worse).

## Reading it -- what this does and does not say
- **Confirmed, but the margin is thin:** the lower bound of the D1 interval is +1.2 pp. The point estimate (14.6 pp) is
  half of what Phase 6.7 showed on the original 8 (-67% -> -37.5%), so the honest effect size is "meaningfully shallower,
  somewhere between marginal and large", not "halves the drawdown".
- **It is a smoother ride, not proven extra return.** D2's interval [-0.40, +0.72] includes zero: TSMOM's higher Sharpe is not
  distinguishable from noise. On these alts equal weight lost money over the window (-3.5%/yr), so being in cash during
  down-trends was worth a lot here; in a window dominated by a rally the same rule would likely lag.
- **A -63.7% drawdown is still enormous.** The filter reduces it; it does not make these assets safe.
- **Same period, correlated assets:** out-of-sample in the cross-section only. Survivorship applies. Two prior 6.x results
  (6.6 vol-targeting) showed a *different* de-risking rule did not help, so "trend filter de-risks" is specific to trend, not to any de-risking.
- Reasonable next steps (each needs its own plan): a longer history including 2018 and 2020, and whether a filter on BTC alone
  gates the whole book (one signal instead of eight).
