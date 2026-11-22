# Phase 10.0 pre-registration (DRAFT): how news sentiment gets priced in over time

Written 2026-11-17 by the overnight work-loop, **before any model code exists and before any real-data number has been computed.**
Status: **DRAFT (revised 2026-11-22 after owner-requested review, see Revision history) — awaiting owner approval.** This file is a plan, not a result. Phase 10.0 is a LEARNING exercise: the bar is that the owner
understands the model after reading it, not only that the tests pass (Sprint Board acceptance criteria).

## The question in plain language
When a strongly positive or negative headline about a coin is published, does its price keep moving in that direction for some hours afterwards
(the news "diffuses" into the price), and if so, what does the shape of that drift look like? The model is

    expected cumulative abnormal return at h hours after headline i  =  beta * s_i * K(h)

- `s_i` is the headline's signed sentiment shock: `positive - negative` from the stored sentiment record, in [-1, 1].
- `K(h)` is the **kernel**: how much of the eventual move has happened by hour h. K(0) = 0, and K rises towards 1 if the effect is permanent.
- `beta` is the size of the full move per unit of sentiment.

The task's rule is **fit K, do not assume it**. So the first estimate of K is non-parametric (one average per hour bucket). Parametric shapes are
then compared against it and against "no effect":

| Kernel | Formula | What it would mean |
|---|---|---|
| Null | K(h) = 0 | Headlines carry no information that is not already in the price. |
| Instant | K(h) = 1 for h >= 1 | Priced in within the first hour bar (efficient market). |
| Exponential | K(h) = 1 - exp(-h / tau) | Steady absorption with half-life tau·ln2. |
| Bass | K(h) = (1 - e^{-(p+q)h}) / (1 + (q/p) e^{-(p+q)h}) | Slow start, then an acceleration as more traders react (imitation q), then saturation. |

The Bass curve is what new-product adoption looks like: early "innovators" (p) react to the news itself, later "imitators" (q) react to the
price moving. The companion doc `docs/diffusion_sentiment_explained.md` (slice 3) will explain why each parameter has that reading.

## Data (fixed)
**Eligibility (added 2026-11-22):** a headline is only usable if its stored time is a genuine **publication** time. Rows stamped with the fetch clock are
excluded: `pub_time_is_fetch_time = 1` where that column exists, and for older rows any timestamp with fractional seconds (the fetch clock writes
microseconds; real publication times do not). Reason: t0 would otherwise be "when the collector ran" (05:00-08:00 UTC daily), often hours after the
price reacted. Timestamps are parsed with `format="ISO8601"` (the column mixes formats; pandas' default silently turned 84% of them into NaT in the
2026-11-22 profile).

**Expected sample today (counts only, no returns), profiled 2026-11-22:** about 221 eligible rows -> about 101 de-clustered events, almost all from September,
about 41 in a 40% holdout. That is far below the 200-holdout requirement, so **running now would return INSUFFICIENT DATA**. Prerequisite: the `openbb_news`
source (the only one with real timestamps) was dead from about 2026-10-01 because of a yfinance fault; the fix is PR #57. **No slice-2 run until at least 500
eligible de-clustered events exist** (500 x 40% = 200 holdout), re-profiled by count only. That is a gate, not a target to tune towards.

- **Headlines (as profiled 2026-11-17, before the eligibility rule above):** `news_store.sqlite3`, table `news`. Profiled read-only on 2026-11-17 (counts only, no returns computed): 4,680 rows; dense
  coverage starts 2026-09 (Sep 1,205 / Oct 2,033 / Nov to date 922); 3,078 rows since 2026-09-01 carry a `*USDT` ticker, 2,114 of them with a
  non-neutral label. Labeller mix in that subset: deepseek-chat 2,836, rule-based-headline-v1 242.
- **Universe:** the crypto tickers that appear in the headlines and have Binance spot hourly klines (BTC, ETH, XRP, SOL, ADA, DOGE, ...).
- **Prices:** Binance 1h klines via the existing `core.data_loader` path. Completed bars only.
- **Event window:** eligible headlines published from 2026-09-01 to the run date. Earlier rows are too sparse to use.

## Event construction (fixed)
1. Keep a headline only if it is **eligible** (genuine publication time, see Data), has exactly one `*USDT` ticker, a non-neutral label, and a timestamp that is not exactly midnight (00:00:00 is
   taken to mean date-only precision; 2 such rows exist since 09-01).
2. Event time t0 = the first hourly bar that **opens after** the publication time. Nothing at or before the publication time counts as reaction.
3. **De-clustering:** for the same ticker, a headline within 6 h of an earlier kept headline is merged into that event, and the event's `s` is
   the mean of the merged shocks. This stops one story told by five outlets from counting five times.
4. **Abnormal return:** for BTC, the raw log return. For every other coin, the log return minus beta_mkt × BTC's log return, where beta_mkt is
   estimated on the 30 days of hourly bars ending at t0 (past data only).
5. **Horizons:** h = 1 … 48 hours. **Pre-event placebo window:** h = -24 … -1.

## Definitions used below (added 2026-11-22)
- `CAR(h)` = cumulative abnormal log return from t0's open to the close of the bar h hours later.
- **Normalisation:** `beta` is the full move at the end of the window, so K(48) = 1 for the Instant and non-parametric kernels; for Exponential and Bass,
  `beta` is the asymptote and K(48) < 1 is allowed. The criterion (a) quantity is always the direct, kernel-free `beta48` = least-squares slope of CAR(48) on s.
  The non-parametric K(h) is slope_h / slope_48, so it is reported on the same scale.
- **Pre-event drift** `beta_pre` = least-squares slope of the abnormal return over the 24 hours *before* t0 (open of t0-24 to open of t0) on s.

## Evaluation (fixed now, before any data is looked at)
- **Split by time, not at random:** events in the first 60% of the event-time range are used to fit; the last 40% are the holdout. All verdict
  numbers come from the holdout only.
- **Minimum sample:** at least 200 de-clustered holdout events, otherwise the verdict is **INSUFFICIENT DATA** and no shape is reported as a
  finding.
- **Kernel selection happens on the fit set only:** each kernel is fitted on the fit set (least squares on CAR(h) / s_i over all events and horizons) and
  ranked by day-blocked 5-fold cross-validated MSE inside the fit set. The single winner is then scored **once** on the holdout against Null. The
  holdout MSE of the other kernels is reported descriptively and is never used to pick the winner (picking the best of three on the data that then
  tests it would be optimistic).
- **Uncertainty:** day-block bootstrap (resample whole UTC days, 2,000 draws) for beta and for the holdout MSE differences.
- **Verdict DIFFUSION SUPPORTED** only if all of these hold:
  (a) the 95% interval of beta on the holdout excludes 0, with the same sign as on the fit set;
  (b) the fit-set-selected kernel beats Null on holdout MSE, and the 95% interval of the difference excludes 0;
  (c) the **placebo check passes**: the 95% bootstrap interval of the ratio `beta_pre / beta48` (holdout) lies entirely below 0.5, i.e. the pre-event
      drift is demonstrably less than half the post-event drift. "The interval includes 0" is **not** enough, because a small sample passes that for
      free; an interval that includes 0 but reaches 0.5 or more is inconclusive and counts as a fail. If the price had already moved before the headline, the headline is most likely *describing* the move ("BTC surges"), which
      is reverse causality, not diffusion.
- Otherwise the verdict is **NOT SUPPORTED** and the reason is printed: no effect, or an effect that started before the news.
- **Subgroups are reported separately, never pooled silently:** BTC-only (raw returns) and alts-only (BTC-adjusted), each with its own beta48 and interval.
  The verdict uses the pooled holdout, but if the two subgroup betas have opposite signs the verdict text says so. A sensitivity run with raw (unadjusted)
  returns for every coin is also reported, which is the alternative in owner decision c.
- Which kernel won (Instant, Exponential or Bass) is reported **descriptively** alongside the verdict. Its parameters (tau, or p and q) get
  bootstrap intervals but carry no pass/fail threshold.

## Known weaknesses, stated in advance
- **Reverse causality is the main risk.** Many crypto headlines are written *after* a price move. Criterion (c) exists for that reason, and a
  failing placebo is an expected and informative outcome, not a bug.
- **Mixed labellers.** The deepseek and rule-based labels have different scales. The holdout result is also reported for deepseek-only events.
  A difference between the two is noted, not tuned away.
- **About 2.5 months of dense data**, in one market regime. Any result is about this period only.
- Publication timestamps come from the sources. An hour-level error moves t0 by one bar. The 1 h bar is the finest resolution claimed.

## Slices (each on its own task branch and PR; nothing merged by the loop)
1. **Synthetic harness:** `core/diffusion_sentiment.py` (event construction, the kernels, the fit, the bootstrap) plus tests that inject known
   beta/tau and beta/p/q with noise and check the fit recovers them within tolerance (AC deliverable 4). No real data is read.
2. **Real-data run.** It runs only after the owner approves this pre-registration and its slice-1 PR is merged. One run, written to
   `training_ground/results/phase_10_0_*.json`, verdict as defined above.
3. **Teaching material:** `docs/diffusion_sentiment_explained.md` and a matplotlib plot of the non-parametric kernel with the fitted curves and
   bootstrap band (AC deliverables 2 and 3).

## What this is not
It is not a trading signal, a forecast or a profit claim. No order path, no change to the paper runner, and no change to the sentiment pipeline.
A DIFFUSION SUPPORTED verdict would justify a separate, pre-registered test of a strategy. It would not justify trading.

## Owner decisions needed before slice 2
a. Is the 2026-09-01 start (with the eligibility rule and the 500-event gate) and the 60/40 time split acceptable?
b. Is a 48 h horizon with a 6 h de-clustering window acceptable?
c. Is a BTC-beta-adjusted abnormal return acceptable for alts? The alternative is raw returns.

## Revision history
- 2026-11-17: first draft (overnight work-loop).
- 2026-11-22: revised after the owner-requested review of PR #56. Nothing was run on prices or returns; the profile used counts of the news table only. Changes:
  eligibility rule and ISO8601 parsing; expected sample and the 500-eligible-event gate; normalisation K(48)=1 and a kernel-free beta48 for criterion (a);
  placebo criterion rewritten as a ratio (an underpowered interval can no longer pass it); kernel chosen by fit-set cross-validation instead of on the holdout;
  BTC / alts subgroups and a raw-return sensitivity run reported separately.
