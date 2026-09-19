# Phase 6.5 pre-registration

Written 2026-09-19, **before any experiment below was run**. Purpose: with enough
tries, some configuration always looks profitable by luck. Fixing the questions and
the pass/fail rules first is what makes a "yes" mean something. Anything changed or
added after seeing results is labelled *exploratory* and does not count as a finding.

> **Amendment 1 (2026-09-19, still before any experiment was run):** H1a/H1b were first
> written as "BTC only", which can never satisfy criterion 2 (>= 6 of 8 symbols). They now
> use one model per symbol on all 8 symbols; the overall AUC is computed on the pooled
> out-of-sample predictions. Also added: criterion 3 for horizon h > 1 uses non-overlapping
> decisions every h bars, so overlapping returns are not double counted.

Baseline already known (Phase 6.2): BTCUSDT, daily, next-bar direction, price + calendar
features -> OOS AUC 0.491, 95% CI [0.456, 0.523] (no skill).

## Protocol (identical for every experiment)

- **Data:** Binance spot daily bars, last 1,500 days, symbols BTC, ETH, BNB, XRP, ADA, LTC,
  DOGE, SOL (USDT pairs). A symbol with < 1,100 bars is dropped and the drop reported.
- **Model:** LightGBM with the fixed default parameters in `strategies/gbm_strategy.py`.
  **No tuning, on any experiment.**
- **Validation:** walk-forward (`core/ml_validation.py`), first training window 400 bars,
  retrain every 20 bars, purge gap = label horizon. Pooled experiments split on *dates*, so
  a date is never on both sides of a boundary.
- **Metric:** out-of-sample AUC of P(up). Confidence interval by moving-block bootstrap over
  dates (10-day blocks, 2,000 resamples; all symbols on a date resampled together, which
  respects both autocorrelation and cross-symbol correlation).
- **Multiple testing:** 7 directional experiments (H1a, H1b, H2a, H2b, H4a, H4b, H4c) ->
  Bonferroni: a directional result needs the **99.3% interval** (alpha = 0.05 / 7).
- **Economic check:** long if P(up) >= 0.55, short if <= 0.45, else flat; 0.1% fee per side;
  Sharpe compared with buy-and-hold **on the same out-of-sample bars**.

## A directional hypothesis counts as a finding only if ALL hold

1. Lower bound of the 99.3% AUC interval > 0.50.
2. AUC > 0.50 in at least 6 of the 8 symbols (consistency, not one lucky coin).
3. Fee-aware rule Sharpe > buy-and-hold Sharpe on the same bars.
4. Replication: AUC > 0.50 in both the first and the second half of the out-of-sample period.

Otherwise the verdict is "no evidence", and that is a valid, reported result.

## Hypotheses

| id | question | change vs baseline |
|----|----------|--------------------|
| H1a | Is a 5-day direction more predictable than 1-day? (fees matter less too) | separate per-symbol models for all 8 symbols, horizon 5 |
| H1b | Same at 10 days | separate per-symbol models for all 8 symbols, horizon 10 |
| H2a | Does pooling 8 symbols (8x the rows) help? | pooled, horizon 1 |
| H2b | Pooled, horizon 5 | pooled, horizon 5 |
| H3 | Accumulated news sentiment | **deferred**: runs only when >= 300 days each have >= 3 headlines for a symbol (news store had 131 BTC headlines over 18 months on 2026-09-19 -> not testable) |
| H4a | Order flow: does the taker-buy share of volume (buyers lifting offers) predict direction? | pooled, h1, + taker-flow features |
| H4b | Do perpetual-futures funding rates (crowding/leverage) predict direction? | pooled, h1, + funding features |
| H4c | Both | pooled, h1, + both |
| H5 | Volatility: is "tomorrow's range is above its trailing 60-day median" predictable, and does the GBM beat the naive rule "score = today's range"? | pooled, h1, target = high-vol day; criterion = paired AUC difference (GBM - naive) 95% CI lower bound > 0 |

H5 is not directional and not part of the Bonferroni family; volatility persistence is
well documented, so the informative question is whether the model adds anything over a
one-line rule.

## Not testable / out of scope here

- Open-interest history (free endpoint keeps ~30 days).
- CPI / labour macro (needs a FRED key); VIX / yields are available but weakly relevant to crypto.
- Any parameter search. If a hypothesis fails, the next step is a *new* pre-registered
  hypothesis, not a tweak of this one.

---

# Results (run 2026-09-19; protocol exactly as above, nothing tuned)

Raw numbers: `training_ground/results/phase_6_5.json`. Reproduce:
`python training_ground/experiments_6_5.py`. All 8 symbols had >= 1,100 bars.

## Directional hypotheses (AUC; interval = 99.3% Bonferroni block bootstrap)

| id | AUC | 99.3% interval | symbols > 0.5 | 1st / 2nd half | rule Sharpe vs buy-and-hold | verdict |
|----|-----|----------------|---------------|----------------|-----------------------------|---------|
| H1a per-symbol, 5-day | 0.515 | [0.477, 0.553] | 7 / 8 | 0.524 / 0.506 | -0.29 vs 0.83 | no evidence |
| H1b per-symbol, 10-day | 0.501 | [0.458, 0.549] | 5 / 8 | 0.528 / 0.473 | -0.15 vs 0.79 | no evidence |
| H2a pooled, 1-day | 0.499 | [0.469, 0.525] | 4 / 8 | 0.502 / 0.497 | -1.34 vs 0.80 | no evidence |
| H2b pooled, 5-day | 0.500 | [0.453, 0.553] | 3 / 8 | 0.508 / 0.491 | 0.20 vs 0.83 | no evidence |
| H4a + taker flow | 0.498 | [0.470, 0.526] | 5 / 8 | 0.495 / 0.505 | -1.66 vs 0.80 | no evidence |
| H4b + funding rate | 0.502 | [0.474, 0.531] | 5 / 8 | 0.497 / 0.510 | -1.08 vs 0.81 | no evidence |
| H4c + both | 0.501 | [0.471, 0.530] | 5 / 8 | 0.499 / 0.504 | -1.16 vs 0.81 | no evidence |
| H3 news sentiment | -- | -- | -- | -- | -- | deferred (not enough history) |

**No directional hypothesis met even the first criterion.** Every interval contains 0.50 and
no rule beat buy-and-hold after fees. The nearest miss is H1a (5-day horizon: AUC 0.515,
positive in 7 of 8 symbols) -- but with 7 experiments run, a 0.515 whose interval reaches 0.477
is what luck looks like. Not a finding; it may be worth a *new* pre-registered follow-up, not a tweak.

## H5 volatility (not directional; 95% intervals)

| quantity | value |
|----------|-------|
| GBM AUC for "tomorrow's range above its 60-day median" | **0.730** [0.708, 0.754] |
| naive rule (score = today's range / 60-day median) | 0.701 |
| GBM minus naive, paired | **+0.03**, interval [0.009, 0.048] (excludes 0) |

**Volatility is predictable and the model adds a small but real amount over the one-line rule.**
This matches the well-documented persistence of volatility and is the one thing these features
can do. It is a *risk* tool, not a return source: it can size positions or set stops, it cannot
tell you which way price will move.

## What this means

- Price, calendar, taker-flow and funding features do not predict next-day (or 5/10-day) direction
  for these 8 large crypto assets over 2022-2026. That now has a proper, hard-to-fool test behind it.
- The practical use of the model is **volatility forecasting for risk sizing**. A follow-up
  (pre-register first) is whether volatility-targeted sizing improves risk-adjusted returns.
- Still open: news sentiment (H3) once months of history accumulate; other data (order-book depth,
  liquidations, on-chain) would be *new* hypotheses.
