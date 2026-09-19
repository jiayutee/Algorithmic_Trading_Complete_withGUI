# Phase 6.6 pre-registration: volatility-targeted position sizing

Written 2026-09-19, **before any experiment below was run**. Phase 6.5 found that next-day
*direction* is not predictable but next-day *volatility* is (GBM AUC 0.730, +0.03 over a naive
rule). This tests whether that is worth money: does sizing exposure by forecast volatility
improve risk-adjusted returns? Same rules as 6.5: fixed protocol, nothing tuned, anything
changed after seeing results is labelled exploratory.

## Setup (fixed)

- **Data / validation:** the Phase 6.5 universe (8 Binance USDT pairs, 1,500 days), the same
  walk-forward (first training window 400 dates, retrain every 20, pooled across symbols).
  Everything is evaluated on the same out-of-sample dates.
- **Forecast target:** next-day Parkinson volatility `ln(High/Low)` of the following bar.
- **Volatility forecasts compared**
  - `NAIVE`: EWMA (span 20) of the squared daily `ln(High/Low)`, square-rooted (needs only the past).
  - `GBM`: LightGBM *regressor* (the fixed default parameters used in 6.2/6.5, no tuning) on the
    Phase 6.0 features, predicting `log` of next-day `ln(High/Low)`; forecast = `exp(prediction)`.
- **Exposure rule (identical for both forecasts, long-only, no leverage):**
  `w_t = min(1, k * median(sigma_hat[<=t]) / sigma_hat_t)` with `k = 0.7`, where the median is
  *expanding* (past only). Comparing a forecast to its own history makes the rule scale-free, so a
  forecast that is merely biased high or low cannot win or lose on bias.
- **Portfolio:** equal weight across the 8 symbols, decided at each close for the next day.
- **Costs:** 0.1% of the change in exposure, with a no-trade band: exposure is only changed when the
  target moves by more than 0.10 from the current exposure.
- **Benchmark for each variant: `FIXED-matched`** = constant exposure equal to that variant's own
  average exposure, so a win cannot come from simply holding less. That isolates the value of *timing*.

## Comparisons (K = 3, Bonferroni -> 98.3% intervals)

| id | question | statistic |
|----|----------|-----------|
| C1 | Does vol targeting with the naive forecast beat matched fixed exposure? | Sharpe(NAIVE-VT) - Sharpe(FIXED matched) |
| C2 | Does vol targeting with the GBM forecast beat matched fixed exposure? | Sharpe(GBM-VT) - Sharpe(FIXED matched) |
| C3 | Does the GBM forecast beat the naive forecast for sizing? | Sharpe(GBM-VT) - Sharpe(NAIVE-VT) |

Intervals: paired moving-block bootstrap over dates (10-day blocks, 2,000 resamples).

## A comparison counts as a finding only if ALL hold

1. Lower bound of the 98.3% interval on the Sharpe difference > 0.
2. Same sign (difference > 0) in both halves of the out-of-sample period.
3. Max drawdown of the first strategy is no worse than the second's.

Otherwise: "no evidence" -- reported as such. Also reported for context (not tested): annualised
return, volatility, Calmar ratio, average exposure, annual turnover, and net-of-cost figures.

## Out of scope

Leverage above 1, per-symbol tuning of `k`, shorting, other forecast horizons or models. If C2/C3
fail, the next step is a *new* pre-registered hypothesis, not a tweak of `k`.
