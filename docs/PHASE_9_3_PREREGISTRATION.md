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
