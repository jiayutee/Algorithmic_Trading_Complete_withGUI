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
