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
