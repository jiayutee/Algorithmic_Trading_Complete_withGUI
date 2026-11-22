# Phase 10.0 pre-registration (APPROVED 2026-11-22): how news sentiment gets priced in over time

Written 2026-11-17 by the overnight work-loop, **before any model code exists and before any real-data number has been computed.**
Status: **APPROVED by the owner on 2026-11-22 (all three decisions below accepted at the defaults written here; no protocol change).** Slice 1 may start; no real-data number exists yet. This file is a plan, not a result. Phase 10.0 is a LEARNING exercise: the bar is that the owner
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
- **Headlines:** `news_store.sqlite3`, table `news`. Profiled read-only on 2026-11-17 (counts only, no returns computed): 4,680 rows; dense
  coverage starts 2026-09 (Sep 1,205 / Oct 2,033 / Nov to date 922); 3,078 rows since 2026-09-01 carry a `*USDT` ticker, 2,114 of them with a
  non-neutral label. Labeller mix in that subset: deepseek-chat 2,836, rule-based-headline-v1 242.
- **Universe:** the crypto tickers that appear in the headlines and have Binance spot hourly klines (BTC, ETH, XRP, SOL, ADA, DOGE, ...).
- **Prices:** Binance 1h klines via the existing `core.data_loader` path. Completed bars only.
- **Event window:** headlines published from 2026-09-01 to the run date. Earlier rows are too sparse to use.

## Event construction (fixed)
1. Keep a headline only if it has exactly one `*USDT` ticker, a non-neutral label, and a timestamp that is not exactly midnight (00:00:00 is
   taken to mean date-only precision; 2 such rows exist since 09-01).
2. Event time t0 = the first hourly bar that **opens after** the publication time. Nothing at or before the publication time counts as reaction.
3. **De-clustering:** for the same ticker, a headline within 6 h of an earlier kept headline is merged into that event, and the event's `s` is
   the mean of the merged shocks. This stops one story told by five outlets from counting five times.
4. **Abnormal return:** for BTC, the raw log return. For every other coin, the log return minus beta_mkt × BTC's log return, where beta_mkt is
   estimated on the 30 days of hourly bars ending at t0 (past data only).
5. **Horizons:** h = 1 … 48 hours. **Pre-event placebo window:** h = -24 … -1.

## Evaluation (fixed now, before any data is looked at)
- **Split by time, not at random:** events in the first 60% of the event-time range are used to fit; the last 40% are the holdout. All verdict
  numbers come from the holdout only.
- **Minimum sample:** at least 200 de-clustered holdout events, otherwise the verdict is **INSUFFICIENT DATA** and no shape is reported as a
  finding.
- **Model comparison:** each kernel above is fitted on the fit set (least squares on CAR(h) / s_i over all events and horizons), then scored by
  holdout mean squared error.
- **Uncertainty:** day-block bootstrap (resample whole UTC days, 2,000 draws) for beta and for the holdout MSE differences.
- **Verdict DIFFUSION SUPPORTED** only if all of these hold:
  (a) the 95% interval of beta on the holdout excludes 0, with the same sign as on the fit set;
  (b) the best non-null kernel beats Null on holdout MSE, and the 95% interval of the difference excludes 0;
  (c) the **placebo check passes**: the pre-event window shows no drift in the sentiment direction (the 95% interval of the pre-event beta
      includes 0). If the price had already moved before the headline, the headline is most likely *describing* the move ("BTC surges"), which
      is reverse causality, not diffusion.
- Otherwise the verdict is **NOT SUPPORTED** and the reason is printed: no effect, or an effect that started before the news.
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

## Owner decisions (resolved 2026-11-22)
Owner said "approved, go ahead with the defaults" in chat on 2026-11-22. The protocol above is frozen as of that date; any later change must be
a dated amendment below, made before the real-data run, never after seeing its numbers.

a. **2026-09-01 start and 60/40 time split: approved.** At approval time the window was about 82 days, so the holdout is roughly 33 days.
b. **48 h horizon with 6 h de-clustering: approved.**
c. **BTC raw log return, other coins BTC-beta-adjusted (30-day hourly beta, past data only): approved.**

Sample-size risk accepted in advance: the 200-event holdout minimum has not been checked against the de-clustered event count (headline yield of
the free collectors is low). If it is not met the verdict is INSUFFICIENT DATA and is reported as such; the bar is not lowered, and the run may
instead wait for more data.

## Amendments
**2026-11-22 (slice 1, before any real-data run): operationalisations the text left open. No criterion, threshold or data rule changed.**
- CAR(h) is the sum of abnormal hourly log returns of bars t0 .. t0+h-1 (CAR(1) = the return of the first bar that opens after publication).
- "Fit on the first 60% of the event-time range" means the first 60% of the TIME SPAN between the first and last event, not 60% of the events.
- The kernel is fitted by least squares of CAR(h) on beta*s*K(h) over all events and horizons 1..48. For a fixed shape the best beta is closed form;
  shape parameters are found on a fixed grid (tau 0.25..500 h; Bass p 0.001..3.2, q 0.001..10). Nothing is tuned afterwards.
- Criterion (a) beta holdout = the best kernel's shape (fixed from the fit set) re-scaled on the holdout; criterion (b) = holdout MSE of Null minus
  holdout MSE of the best non-null kernel (parameters from the fit set), both with day-block bootstrap on the holdout.
- Criterion (c) placebo = least-squares slope of the abnormal return over the 24 h BEFORE t0 on s, holdout only; it passes if its 95% interval includes 0.
- Events whose 48 h window is incomplete, or whose coin has fewer than 360 hourly bars of history in the 720 before t0, are dropped and counted.
- Slice 1 implements `diffusion_verdict` exactly as above; tests on synthetic data with a planted signal (recovery, pure-noise false-positive rate,
  a pre-headline move, too few events) are in `tests/test_diffusion_sentiment.py`.
