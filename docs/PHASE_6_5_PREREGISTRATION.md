# Phase 6.5 pre-registration

Written 2026-09-19, **before any experiment below was run**. Purpose: with enough
tries, some configuration always looks profitable by luck. Fixing the questions and
the pass/fail rules first is what makes a "yes" mean something. Anything changed or
added after seeing results is labelled *exploratory* and does not count as a finding.

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
| H1a | Is a 5-day direction more predictable than 1-day? (fees matter less too) | BTC only, horizon 5 |
| H1b | Same at 10 days | BTC only, horizon 10 |
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
