# Phase 9.3a pre-registration: are Kalshi longshots overpriced / favorites underpriced?

Written 2026-09-19, **before the study was run**. Phase 9.1 found no riskless arbitrage in liquid books. The
next honest question for Kalshi is about *calibration*: prices are supposed to be probabilities; the well-known
prediction-market deviation is the **favorite-longshot bias** (cheap contracts win less often than their price).
This is a market-structure test, not a forecasting model (that is 9.2). Same rules as 6.5: fixed protocol, nothing
tuned, deviations labelled exploratory.

## Data (fixed)
- Kalshi public API, settled markets, multivariate combos excluded (`mve_filter=exclude`), up to 4,000 most recent.
- Eligible: volume >= 100 contracts, open for >= 3 hours before close.
- Random sample of at most 800 eligible markets, `numpy` seed 0.
- Snapshot: last hourly candle ending at or before `close_time - 2h`. Entry prices are what a taker pays:
  YES at `yes_ask.close`, NO at `1 - yes_bid.close`. Markets with no such candle or a missing/zero quote are dropped
  (count reported).
- Outcome: `result` ("yes" = 1, "no" = 0). Fee: `core.kalshi_arbitrage.taker_fee` (assumed schedule, per contract).

## Hypotheses (K = 2, Bonferroni -> 97.5% intervals)
Mispricing edge per contract = `outcome - ask` (BEFORE fees) for a BUY of YES. Net edge = `outcome - ask - fee`.
| id | claim | statistic |
|----|-------|-----------|
| L | Longshots are overpriced: buying YES when ask <= 0.10 has negative mean edge | mean(outcome - ask) among ask <= 0.10 |
| F | Favorites are underpriced: buying YES when ask >= 0.90 has positive mean edge | mean(outcome - ask) among ask >= 0.90 |

> **Amendment (2026-09-19, before any real-data run):** the first draft defined the statistic NET of fees.
> Writing the test showed that made L trivially "true" for a perfectly calibrated market (any fair-priced
> contract loses the fee), so it would have measured fees, not mispricing. The finding criteria now use the
> pre-fee edge; the net-of-fee mean and its interval are reported alongside, and a finding is additionally
> labelled **tradable** only if the net-of-fee interval also clears zero in the claimed direction
> (for L that means the *short* side: sell YES / buy NO -- see below).
> For L the tradable question is whether buying NO at `1 - yes_bid` beats fees; that mean is reported too.

Intervals: bootstrap resampling **events** (markets in one event are correlated), 5,000 resamples.

## A hypothesis counts as a finding only if ALL hold
1. L: upper bound of 97.5% interval < 0.  F: lower bound > 0.
2. Same sign in both halves of the sample split by `close_time`.
3. At least 100 markets and 30 distinct events in the bucket.
Otherwise: "no evidence". Also reported, untested: a 10-bucket calibration table (mean ask vs. hit rate).

## Limits stated in advance
- One recent window, dominated by sports props and hourly commodity markets: cannot generalise to politics/economics.
- A hit-rate gap is only an *edge* if it survives fees and if resting liquidity exists at the snapshot ask.
- No execution is implied; this is a read-only study.

---

# Amendment 2 (after the first run -- disclosed, data-quality only)

The first run (`training_ground/results/phase_9_3_run1_invalid.json`) reported favorites (ask >= 0.90) winning only
43% of the time at a mean ask of 0.943. That is not a market; it is an artifact. 118 of those 155 snapshots had a
spread of 0.5-1.0 (bid about 0, ask about 0.99): an empty book displays a placeholder ask, and my "drop missing quotes"
rule did not cover one-sided books. **Run 1 is void; no conclusion is drawn from it.**

Fix, chosen from that diagnosis (not from any outcome-based search): keep only snapshots with a real two-sided book,
`yes_bid >= 0.01` and `yes_ask - yes_bid <= 0.10`. Everything else is unchanged (same cached sample, same intervals).
The rerun below is therefore *not* fully blind: the filter was designed after seeing run 1's data problem, and the
threshold 0.10 is one plausible choice, so treat a finding here as weaker than a pre-run one.

# Results (rerun after Amendment 2; `training_ground/results/phase_9_3.json`)

Sample: 800 random eligible settled markets -> 738 with a snapshot -> **389 with a real two-sided book** (62 events),
all closing within one ~9-hour window on 2026-09-19 (the API only serves the most recent settled markets, and
almost all are short-lived sports props / hourly gold).

| hypothesis | n (events) | mean ask | hit rate | pre-fee edge | 97.5% interval | verdict |
|---|---|---|---|---|---|---|
| L longshots (ask <= 0.10) overpriced | 122 (37) | 0.071 | 0.066 | -0.006 | [-0.046, +0.045] | **no evidence** |
| F favorites (ask >= 0.90) underpriced | 1 | - | - | - | - | **untestable** (near-certain markets have no two-sided book 2h before close) |

- L: point estimate has the predicted sign but is tiny next to its uncertainty; the tradable side (buy NO after fees) is
  -0.024 per contract, i.e. no money in it.
- Exploratory, **not tested, do not act on it**: the 0.10-0.30 ask buckets hit 31-36% against asks of 15-25%
  (n = 89 and 55, from few events, one day). If real it would be the opposite of the favorite-longshot bias, but this is exactly
  the kind of pattern a single small clustered sample produces. It needs its own pre-registered hypothesis and a larger,
  multi-day sample (collect snapshots daily) before it means anything.
- Run 1 (`phase_9_3_run1_invalid.json`) is kept only as a record of the data-quality bug.
- Consequence for the Phase 9 roadmap: 9.1 produced no signals to backtest, and 9.2 (a probability model) has no
  domain where the market is demonstrably mispriced yet. The most useful next step is a daily snapshot collector for
  Kalshi (like the news collector) so a real multi-week calibration study is possible.

# Results on collected snapshots (2026-09-21, 3 days of `core.kalshi_collector` data)

Script: `training_ground/experiments_9_3_collected.py` (evidence: `training_ground/results/phase_9_3_collected.json`).
**No protocol change**: it calls `experiments_9_3.run` unchanged (hypotheses L and F, the Amendment 2 two-sided-book filter,
event-clustered bootstrap, 97.5% intervals, the three finding criteria). Only the snapshot *source* differs.

Data: 588 snapshots / 554 markets over 2026-09-19..21; 192 markets resolved; 178 of those have a snapshot at least 2h before
close (one per market, the last such snapshot); all 178 pass the two-sided filter; 68 distinct events; markets closed
2026-09-20 03:59 to 2026-09-21 14:00 UTC.

| hypothesis | n (events) | mean ask | hit rate | pre-fee edge | 97.5% bootstrap interval | verdict |
|---|---|---|---|---|---|---|
| L longshots (ask <= 0.10) overpriced | 76 (48) | 0.041 | 0.000 | -0.041 | [-0.047, -0.035] | **UNDERPOWERED** (n < 100) |
| F favorites (ask >= 0.90) underpriced | 38 (23) | 0.980 | 1.000 | +0.020 | [+0.013, +0.028] | **UNDERPOWERED** (n < 100, events < 30) |

**Read this before quoting any number above.**
- Neither hypothesis meets criterion 3, so by the pre-registered rule neither can be a finding, and "underpowered" is not
  the same as "no evidence of bias": the data cannot say either way.
- The bootstrap intervals are **degenerate** here. L has 0 winners in 76 and F has 38 winners in 38, so resampling only
  reflects the spread of the asks, not the binomial uncertainty of a 0% / 100% hit rate. The intervals look tight; they are not
  evidence. (This is a property of the pre-registered statistic on extreme buckets, disclosed rather than changed.)
- Exploratory, not pre-registered: if every market were fairly priced at its ask, 76 longshots would produce about 3.1 winners;
  seeing 0 has probability about 0.040 (exact Poisson-binomial, independence assumed, which flatters the result because markets
  in one event are correlated). For favorites, 38 of 38 vs 37.2 expected has probability about 0.46: fully consistent with a
  fairly priced market. The "+0.010 tradable net" for F is 1 cent of expected profit per contract against a 98 cent loss if one
  contract fails; 38 wins cannot distinguish that from a fair price.
- Deviations, all disclosed: (1) realised lead time is **12.9 to 47.9 hours** before close (median 12.9h), not the ~2h of 9.3a,
  because the collector snapshots a few times a day, so this is a longer-lead calibration, not a replica; (2) the random
  800-market sample and the open >= 3h filter cannot be re-applied (open time is not stored), so every resolved market is used;
  (3) one short window dominated by sports props and hourly commodity markets.
- The middle calibration buckets have n <= 10 each and mean nothing yet.
- Nothing here is a trading signal, nothing was executed, and Phase 9.2 (probability model) stays on HOLD until roughly four
  weeks of snapshots exist. Projection from two closing days only (about 38 longshot and 19 favorite markets per day): the
  100-market bar could be cleared for L in a day or two and for F in about four days. Re-run the script then; it is read-only.

# Results on collected snapshots (2026-09-22, 4 days of `core.kalshi_collector` data)

Run date: 2026-09-22. Collector counts at run time: 749 snapshots, 710 markets, 435 resolved, 469 labelled (2026-09-19..22).

Script: `training_ground/experiments_9_3_collected.py`, DB from main checkout (read-only via `--db` argument).
**No protocol change**: same hypotheses, thresholds, fees, CI method as pre-registered.

Data: 435 resolved markets; 415 have a usable snapshot >= 2h before close; all 415 pass the two-sided-book filter
(Amendment 2: yes_bid >= 0.01, spread <= 0.10); 142 distinct events; window 2026-09-20 03:59 to 2026-09-22 14:45 UTC.
Lead time: 12.8-68.9h before close (median 14.9h, mean 28.8h).

| hypothesis | n (events) | mean ask | hit rate | pre-fee edge | 97.5% bootstrap interval | verdict |
|---|---|---|---|---|---|---|
| L longshots (ask <= 0.10) overpriced | 156 (91) | 0.046 | 0.045 | -0.001 | [-0.048, +0.070] | **no evidence** |
| F favorites (ask >= 0.90) underpriced | 90 (46) | 0.978 | 1.000 | +0.022 | [+0.017, +0.027] | **UNDERPOWERED** (n < 100) |

**Key changes vs the 09-21 run:**
- L now has n=156 (was 76), meeting criterion 3. CI crosses zero ([-0.048, +0.070]) and halves have opposite signs
  (-0.031 / +0.032), so criteria 1 and 2 both fail. Verdict: **no evidence** of longshot overpricing in this sample.
  The hit rate (4.5%) is almost exactly the mean ask (4.6%): these markets are well-calibrated at the longshot end.
- F has n=90 (was 38) and 46 events (was 23). Criterion 3 requires >= 100 markets; F remains **underpowered**.
  The CI ([+0.017, +0.027]) is narrow and entirely positive, but n < 100 means we cannot call this a finding by
  pre-registered rules. Exploratory: 90 of 90 favorites resolved YES vs 88.0 expected if fairly priced;
  P(as extreme | fair, independence) = 0.134 -- consistent with a calibrated market.

**Exploratory calibration note (not pre-registered):**
The 0.10-0.30 ask buckets (n=59+22=81) show hit rates of 10% and 14% against asks of 15% and 25%: winners are
fewer than prices imply in the low-probability region. Small cells from clustered markets; could be sampling
noise, a longer-lead effect, or a real pattern. Needs its own pre-registration and multi-week data.

**Status:** Phase 9.2 (probability model) stays on HOLD. L is adequately powered and shows no evidence of
overpricing. F still needs roughly 10 more closing days to clear n=100 (about 19 new favorites per day).
Re-run when F reaches n=100.

**Deviations (disclosed):** lead time 12.8-68.9h (median 14.9h), not ~2h; no random 800-market sample or
open >= 3h filter; one short window dominated by sports/commodity markets.
# Results on collected snapshots (2026-09-23, 5 days of `core.kalshi_collector` data)

Run date: 2026-09-23. Collector counts at run time: 896 snapshots, 851 markets, 560 resolved, 594 labelled (2026-09-19..23).

Script: `training_ground/experiments_9_3_collected.py`, DB from main checkout (read-only via `--db` argument).
**No protocol change**: same hypotheses, thresholds, fees, CI method as pre-registered.
Evidence: `training_ground/results/phase_9_3_collected.json` (09-22 run archived as `phase_9_3_collected_20260922.json`).

Data: 560 resolved markets; 530 have a usable snapshot >= 2h before close; all 530 pass the two-sided-book filter
(Amendment 2: yes_bid >= 0.01, spread <= 0.10); 185 distinct events.
Lead time: 12.8-68.9h before close (median 20.8h, mean 30.0h).

| hypothesis | n (events) | mean ask | hit rate | pre-fee edge | 97.5% bootstrap interval | verdict |
|---|---|---|---|---|---|---|
| L longshots (ask <= 0.10) overpriced | 206 (117) | 0.047 | 0.053 | +0.006 | [-0.039, +0.065] | **no evidence** |
| F favorites (ask >= 0.90) underpriced | 100 (54) | 0.978 | 1.000 | +0.022 | [+0.018, +0.026] | **FINDING** (all three criteria met) |

**Pre-registered verdict for F:**
F has reached n=100 (exactly 100 markets, 54 events, both meeting the required >= 100 markets and >= 30 events).

- Criterion 1: lower bound of 97.5% CI is +0.018 > 0. Met.
- Criterion 2: both halves have the same sign (+0.021 / +0.024). Met.
- Criterion 3: n=100 markets, 54 events. Met.
- **Verdict: FINDING.** Favorites (ask >= 0.90) show a pre-fee mean edge of +0.022 per contract with an entirely
  positive 97.5% CI. The tradable-side (buying YES, net of fee) CI is [+0.008, +0.016], also entirely positive,
  so this is additionally labelled **TRADABLE** by the pre-registration's tradable criterion.

**Critical caveats (do not quote the finding without reading these):**
- The bootstrap CI for F is **degenerate**: 100/100 favorites resolved YES, so the interval reflects only the spread
  of asks, not binomial uncertainty of the hit rate. This was flagged in the 09-21 run notes and is a known property
  of the pre-registered statistic on 100% hit rates. The CI looks tight; it is not evidence of a precise edge.
- Exploratory calibrated tail (independence assumed, not pre-registered): P(100 of 100 | fair, independence) = 0.109.
  Under the independence assumption, 100 wins out of 100 is not unusual at 10.9%, and clustering makes it even less
  surprising. The bootstrap finding and the calibrated tail tell different stories; the bootstrap wins here because it
  is the pre-registered method.
- The Amendment 2 filter (two-sided book) was not fully blind, as explained in Amendment 2 above.
- This is a read-only calibration study at a 12.8-68.9h lead, not at the ~2h lead of 9.3a.
- No execution is implied; no real money was involved or committed.

**Key changes vs the 09-22 run:**
- F has n=100 (was 90) and 54 events (was 46). Criterion 3 now met; all three criteria met.
- L has n=206 (was 156). CI [-0.039, +0.065] still crosses zero and halves have opposite signs; verdict unchanged: **no evidence**.

**Status:** Phase 9.2 (probability model) stays on HOLD. L is adequately powered with no evidence of overpricing.
F has cleared n=100 and meets all three pre-registered criteria; the bootstrap CI is degenerate (see caveats).

**Deviations (disclosed):** lead time 12.8-68.9h (median 20.8h), not ~2h; no random 800-market sample or
open >= 3h filter; one window dominated by sports/commodity markets (2026-09-20 to 2026-09-23).
