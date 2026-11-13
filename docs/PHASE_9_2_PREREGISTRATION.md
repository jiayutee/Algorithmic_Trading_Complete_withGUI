# Phase 9.2 pre-registration (DRAFT for owner review): does a recalibration model beat the Kalshi price as a probability?

Written 2026-11-14, **before any 9.2 code exists and before any 9.2 number has been computed.** This file is the protocol;
the model code is written only after this file is merged, and any change after that is an amendment, labelled as one.
Same rules as 6.5 and 9.3: fixed protocol, nothing tuned to pass, deviations labelled exploratory.

Status: **DRAFT. The owner reviews and merges it; nothing runs until then.** No execution path exists or is implied.

## What is already known (disclosed so the test is honest about its own history)
- 9.1: no riskless arbitrage in liquid books. 9.3a/9.3 re-runs (2026-09-26, 7 days of data): longshots (L): no evidence;
  favourites (F, ask >= 0.90): a pre-registered FINDING, pre-fee edge +0.023, CI [+0.019, +0.027], hit rate 123/123.
  That finding was seen on snapshots from 2026-09-19 to 2026-09-26 and has not been re-run on the longer history.
- Consequence for this study: a model could "discover" F because we already know it is there. To keep that from
  counting as new evidence, **every test fold below starts on or after 2026-10-10**; the 9.3 window is training data only.
- The data hold from the Sprint Board asked for about four weeks of resolved snapshots. The collector now has 52 days
  (6,109 snapshots, 5,726 resolved markets, 2026-09-19 to 2026-11-13) with gaps on days the Mac slept, which are not filled.

## Data (fixed)
- `training_ground/datasets/kalshi_snapshots.sqlite3`, opened read-only. **Cut-off: markets with `close_time` <= 2026-11-13 23:59 UTC.**
  The window is frozen here; it is not extended after seeing results. Later data is a separate replication, not a top-up.
- Unit: one snapshot per resolved market, the latest snapshot at least 2 hours before `close_time` (as in 9.3).
- Filters, identical to 9.3 Amendment 2: two-sided book (`yes_bid >= 0.01`, spread <= 0.10). Dropped counts are reported.
- Outcome: the market's `result` (YES = 1).
- Market probability `p_mkt` = mid of `yes_bid` and `yes_ask` at the snapshot, clipped to [0.01, 0.99].

## The one primary hypothesis (K = 1, so no multiple-comparison correction; the interval is 95%)
**H1:** a regularised logistic recalibration of the market price has a lower out-of-sample log-loss than the market price itself.

- Model (fixed, no search): `logit P(YES) = b0 + b1*logit(p_mkt) + b2*lead_hours + b3*spread + category dummies`
  - category = series-ticker family (the same grouping as the 9.3 category-mix table); families with < 30 training markets are pooled into "other"
  - L2 penalty, `C = 1.0`, `class_weight=None`, scikit-learn defaults otherwise. **No hyper-parameter is tuned on any data.**
  - No gradient-boosting, no neural nets: with this many events a richer model would mostly fit noise.
- Validation: expanding-window walk-forward by calendar week of `close_time`. Train on all weeks strictly before the test week,
  minus a 1-day purge gap; predict the test week. The first test week is the first week starting on or after 2026-10-10.
- Statistic: per-market difference `d_i = logloss(model_i) - logloss(p_mkt_i)`, pooled over all test folds. Negative = model better.
- Interval: bootstrap resampling **events** (markets in one event are correlated), 5,000 resamples, 95% two-sided, seed 0.

### H1 counts as a finding only if ALL hold
1. The upper bound of the 95% interval for mean `d_i` is < 0.
2. The sign of the mean `d_i` is negative in both halves of the pooled test set, split by `close_time`.
3. At least 300 test markets and 100 distinct test events.
Otherwise: **"no evidence that the model beats the market price."** That outcome is expected to be common, since prediction-market
prices are usually hard to beat and the effect we already know about (F) is small.

## Reported, but not tested (exploratory, no claims)
- Brier score for model and market; a 10-bucket calibration table; `d_i` by category; the F bucket (ask >= 0.90) on its own.
- The fitted coefficients, so a reader can see whether the model is just re-learning the favourite-longshot slope.

## What a pass would and would not mean
- A pass means the **probability estimates** are better calibrated than the price on later, unseen weeks of this venue and window.
- It does **not** mean a profit. Turning probabilities into trades needs a separate pre-registered rule with the assumed taker fee
  (`core.kalshi_arbitrage.taker_fee`), a liquidity check at the snapshot ask and a position-size cap. None of that is part of 9.2.
- No order path is added. Kalshi stays read-only and is not a broker in this codebase.

## Limits stated in advance
- One window of about eight weeks, dominated by city temperature, commodity and sports markets; it cannot generalise to politics or economics.
- Collection gaps (days with 0 or 1 of 3 slots) mean some days are under-sampled. They are reported, not corrected.
- The test folds hold roughly five weeks, so the interval will be wide; "no evidence" can simply mean "not enough data yet".
- The category dummies make the model partly a lookup of which families are mispriced in this window. That is allowed, but it is
  the first thing to doubt if H1 passes: the planned replication is the same protocol on the next 8 weeks of collected data.

## Implementation plan (after this file is merged, not before)
`training_ground/experiments_9_2.py`, reading the same database as `experiments_9_3_collected.py`, writing
`training_ground/results/phase_9_2_<date>.json`, with offline tests on a synthetic database where the answer is known
(one calibrated market, one with a planted bias).
