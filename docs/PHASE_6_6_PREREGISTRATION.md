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

---

# Results (run 2026-09-19; protocol exactly as above, nothing tuned)

Raw numbers: `training_ground/results/phase_6_6.json`. Reproduce: `python training_ground/experiments_6_6.py`.
Window 2023-10-16 -> 2026-08-31 (1,051 days, 8 symbols), costs and no-trade band as pre-registered.

| strategy | Sharpe | ann. return | ann. vol | max drawdown | Calmar | avg exposure | turnover / yr |
|----------|--------|-------------|----------|--------------|--------|--------------|---------------|
| GBM-VT | 0.40 | 8.3% | 38.7% | -54.6% | 0.15 | 0.70 | 32.2 |
| NAIVE-VT | 0.65 | 19.5% | 38.9% | -56.3% | 0.35 | 0.67 | 4.2 |
| FIXED, matched to GBM-VT | 0.79 | 27.6% | 42.5% | -52.2% | 0.53 | 0.70 | 0 |
| FIXED, matched to NAIVE-VT | 0.79 | 26.8% | 40.8% | -50.7% | 0.53 | 0.67 | 0 |
| Buy and hold (100%) | 0.79 | 34.1% | 61.0% | -66.7% | 0.51 | 1.00 | 0 |

| comparison | Sharpe difference | 98.3% interval | halves | verdict |
|------------|-------------------|----------------|--------|---------|
| C1 NAIVE-VT vs FIXED matched | -0.13 | [-0.51, +0.29] | -0.12 / +0.07 | no evidence |
| C2 GBM-VT vs FIXED matched | -0.39 | [-0.75, +0.01] | -0.45 / -0.19 | no evidence |
| C3 GBM-VT vs NAIVE-VT | -0.25 | [-0.55, +0.05] | -0.34 / -0.26 | no evidence |

**No comparison met the first criterion**, and all point estimates are negative.

## Reading it

- **The AUC gain did not turn into a sizing gain.** Phase 6.5's +0.03 AUC was about *ranking* days by
  volatility relative to a 60-day median. Sizing needs a *level* forecast that is stable from day to day.
  The GBM level forecasts are noisy (log-correlation with the naive EWMA only 0.45), so exposure
  churned (32 turnovers a year vs 4.2) and fees consumed the benefit.
- **Even the naive version barely de-risked.** Volatility 38.9% vs 40.8% for the matched fixed
  benchmark and no drawdown improvement: crypto drawdowns come from sudden jumps that trailing
  volatility does not anticipate.
- **Sample caveat:** this window was a strong, mostly-up market for these assets, which penalises any
  rule that holds less. It is one regime, not a universal verdict.
- Not tuned, not re-run. The natural *new* hypothesis (pre-register first) is a more stable forecast
  (e.g. GBM combined with the EWMA, or a longer smoothing), not a different `k`.
