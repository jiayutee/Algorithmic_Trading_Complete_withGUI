# Phase 7.1 pre-registration: does risk-based allocation beat equal weight?

Written 2026-09-19, **before any experiment below was run**. Phases 6.5/6.6 found no predictive
edge to trade. What is left is *how the same assets are combined*. This asks the narrow, testable
question: across the 8 crypto assets, does allocating by risk (instead of 1/N) improve
risk-adjusted returns out of sample, after costs? Same rules as 6.5/6.6: fixed protocol, nothing
tuned, results reported whatever they are.

## Setup (fixed)

- **Universe / data:** the Phase 6.5 universe (BTC, ETH, BNB, XRP, ADA, LTC, DOGE, SOL vs USDT), daily
  closes, last 1,500 days. Long-only, fully invested (weights >= 0, sum to 1).
- **Method:** rebalance every 30 days using ONLY the trailing 250 days of returns; between rebalances
  weights drift with returns. First rebalance after 250 days, so every method is judged on the same dates.
- **Allocators** (`core/portfolio_optimizer.py`, default settings, no tuning): `equal` (benchmark),
  `inverse_vol`, `min_variance`, `risk_parity`, `hrp`, `max_sharpe`.
- **Costs:** 0.1% of turnover (sum of absolute weight changes at each rebalance).

## Comparisons (K = 5, each allocator vs `equal`; Bonferroni -> 99% intervals)

Statistic: difference in annualised Sharpe ratio, paired moving-block bootstrap over dates
(10-day blocks, 2,000 resamples).

## An allocator counts as a finding only if ALL hold

1. Lower bound of the 99% interval on the Sharpe difference > 0.
2. Difference > 0 in both halves of the evaluation period.
3. Max drawdown no worse than `equal`'s.

Otherwise "no evidence". Also reported (not tested): annual return, volatility, max drawdown,
average number of effective assets, annual turnover.

## Not tested

Leverage, shorting, other rebalance frequencies or lookbacks, Black-Litterman, adding
strategy return streams as assets. Changing any of these after seeing results is a new hypothesis.
