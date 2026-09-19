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

---

# Results (run 2026-09-19; protocol exactly as above, nothing tuned)

Raw numbers: `training_ground/results/phase_7_1.json`. Reproduce: `python training_ground/experiments_7_1.py`.
Window 2023-04-20 -> 2026-09-18 (1,248 days, 8 assets), 0.1% cost on turnover, monthly rebalance.

| method | Sharpe | ann. return | ann. vol | max drawdown | effective assets | turnover / yr |
|--------|--------|-------------|----------|--------------|------------------|---------------|
| equal (benchmark) | 0.69 | 26.5% | 59.6% | -67.1% | 8.0 | 1.3 |
| inverse_vol | 0.76 | 30.7% | 56.9% | -64.9% | 7.6 | 1.5 |
| min_variance | 0.92 | 40.8% | 51.9% | -60.2% | 5.3 | 2.7 |
| risk_parity | 0.77 | 31.7% | 56.7% | -64.6% | 7.5 | 1.5 |
| hrp | 0.85 | 36.8% | 53.7% | -61.9% | 6.2 | 1.8 |
| max_sharpe | 0.65 | 23.2% | 55.4% | -70.1% | 2.1 | 7.6 |

| vs equal | Sharpe difference | 99% interval | halves | verdict |
|----------|-------------------|--------------|--------|---------|
| inverse_vol | +0.064 | [-0.026, +0.144] | +0.04 / +0.06 | no evidence |
| min_variance | +0.228 | [-0.152, +0.615] | +0.09 / +0.19 | no evidence |
| risk_parity | +0.078 | [-0.025, +0.183] | +0.06 / +0.07 | no evidence |
| hrp | +0.161 | [-0.045, +0.384] | +0.10 / +0.15 | no evidence |
| max_sharpe | -0.040 | [-0.840, +0.658] | -0.21 / +0.10 | no evidence |

**No allocator met the pre-registered first criterion** (interval lower bound > 0 at the 99%
Bonferroni level), so by the rules set in advance there is no finding.

## Reading it (context, not a change to the verdict)

- **Consistent direction:** inverse-vol, risk parity, HRP and min-variance all beat equal weight in
  point estimate, in *both* halves, with lower volatility and shallower drawdown -- what the theory
  predicts when lower-volatility assets are given more weight. It is suggestive, not proven: 3.4 years
  of a single, highly correlated asset class is a small sample for a 99% bar.
- **Max-Sharpe (mean-variance) is the one to avoid:** it concentrated in ~2 assets, turned over 7.6x a
  year and did worse than equal weight -- the classic estimation-error failure.
- **Practical reading:** risk-based weights (inverse-vol, risk parity, HRP) appear to be a low-cost,
  low-risk default over 1/N for this universe, but this experiment does not prove it.
