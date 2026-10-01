# Phase 9.3 re-run on collected snapshots — 2026-09-26

**Run date:** 2026-09-26
**Script:** `training_ground/experiments_9_3_collected.py`
**DB:** `training_ground/datasets/kalshi_snapshots.sqlite3` (shared checkout, read-only)
**Output:** `training_ground/results/phase_9_3_collected_2026-09-26.json`
**No protocol change.** Same hypotheses (L and F), thresholds, fees, CI method, and three finding criteria as pre-registered in `docs/PHASE_9_3_PREREGISTRATION.md`. Previous evidence file (`phase_9_3_collected.json`, 09-23 run) was not overwritten.

## Command

```
~/miniconda3/bin/python3 training_ground/experiments_9_3_collected.py \
  --db /path/to/training_ground/datasets/kalshi_snapshots.sqlite3 \
  --out training_ground/results/phase_9_3_collected_2026-09-26.json
```

## Data window

- Collector counts at run time: 1264 snapshots, 1213 distinct markets, 962 resolved, 1008 labelled snapshots
- Window: 2026-09-19 to 2026-09-26 (7 distinct closing days: 2026-09-20 to 2026-09-26)
- After >= 2h lead filter and two-sided-book filter (Amendment 2: yes_bid >= 0.01, spread <= 0.10):
  - 919 markets pass (vs 530 in the 09-23 run)
  - 309 distinct events
  - 43 resolved markets had no usable snapshot >= 2h before close
- Lead time: 5.9 to 68.9 h before close (mean 32.8 h, median 37.9 h)

## Key results

| Hypothesis | n (events) | Mean ask | Hit rate | Pre-fee edge | 97.5% bootstrap CI | Verdict |
|---|---|---|---|---|---|---|
| L longshots (ask <= 0.10) | 356 (189) | 0.044 | 0.037 | -0.0076 | [-0.035, +0.030] | **no evidence** |
| F favorites (ask >= 0.90) | 123 (71) | 0.977 | 1.000 | +0.0227 | [+0.019, +0.027] | **FINDING** |

Previous run (09-23): L n=206 (117 events), F n=100 (54 events), F FINDING with CI [+0.018, +0.026].

### F finding: criteria check

- Criterion 1: lower bound of 97.5% CI = +0.019 > 0. Met.
- Criterion 2: both halves have the same sign (+0.021 / +0.025). Met.
- Criterion 3: n=123 markets, 71 events. Met.
- **Verdict: FINDING** (persists from 09-23 run). Tradable-side (net of fee) CI: [+0.009, +0.017], entirely positive — TRADABLE by pre-registered criterion.

### F: P(all YES | fairly priced at ask)

Product of the 123 individual ask values = **0.0577** (5.77e-02). This equals P(exactly 123 YES | markets are fair and independent). Under the independence assumption, seeing 123 of 123 resolve YES has probability ~5.8%. The independence assumption is wrong (events are clustered); clustering makes the true probability higher than 5.8%.

Exploratory calibrated tail (computed but not the pre-registered test): P(>= 123 YES | fair, independence) = 0.058. The pre-registered test uses the bootstrap, not this tail.

### L: verdict unchanged

L hit rate 3.7% vs mean ask 4.4%. Point estimate is negative (matching the predicted direction) but the CI [-0.035, +0.030] crosses zero and the two halves (+0.004 / -0.021) have opposite signs. Criteria 1 and 2 both fail. Verdict: **no evidence** of longshot overpricing.

## Category mix (all 919 two-sided markets)

| Category | Markets | Events | Notes |
|---|---|---|---|
| Commodity | 334 | 98 | Diesel/gas price (KXDIESELD), AAA gas (KXAAAGASD) |
| City high temperature | 307 | 141 | KXHIGH* (daily high temp in OKC, PHX, LAX, etc.) |
| Sports | 169 | 60 | NBA/WNBA/EPL/La Liga/NCAAF point spreads and totals |
| Weather (rainfall) | 86 | 6 | KXRAIN* |
| Politics | 23 | 4 | Trump approval (KXTRUMPAPPROVE) |

F favorites (n=123) are almost entirely commodity markets: 115 of 123 are commodity (65 events), 4 rainfall/weather, 3 city-high, 1 sports. The F FINDING is a commodity-specific result, not a general market finding.

## Effective sample size note

Seven distinct closing days (2026-09-20 to 2026-09-26) but markets are tightly clustered: each day has many related markets from a few event families (e.g., 14 KXDIESELD markets on a single day, 23 KXRAIN markets on two days). The 309 events and 71 F-events represent far fewer independent observations than the raw counts suggest. The bootstrap resamples events (not markets) to account for within-event correlation, but cross-event and cross-day correlation among related commodity families is not captured.

## Limitations (explicit)

1. **Degenerate CI for F.** F has 123 of 123 markets resolve YES. The bootstrap CI reflects only the spread of asks (all near 0.977), not the binomial uncertainty of the 100% hit rate. The interval looks tight but is not evidence of a precise edge.
2. **Amendment 2 filter is not fully blind.** The two-sided-book filter was designed after seeing a data-quality problem in run 1 (see PHASE_9_3_PREREGISTRATION.md Amendment 2). Treat this as weaker than a fully pre-registered finding.
3. **Short window, clustered events.** Seven closing days dominated by hourly commodity and city-temperature markets. Cannot generalise to politics, economics, or multi-day markets.
4. **Longer lead time than 9.3a.** Median lead is 37.9 h, not the ~2 h of the original study. Favorites at 38 h before close are already near certain; the study may capture that structural certainty rather than a mispricing at the margin.
5. **No execution implied.** This is a read-only calibration study. No real money was involved or committed. Phase 9.2 stays on HOLD. Interpretation of F is the owner's call.
6. **Assumed fees.** Fee schedule from `core.kalshi_arbitrage.taker_fee` is assumed, not verified against Kalshi's current published schedule.
7. **Stale text in the JSON.** The third `protocol_deviations` string in the output JSON is fixed text in the script and still says "markets closing 2026-09-20..21". The actual closing window of this run is 2026-09-20 to 2026-09-26 (see Data window above). The string was left as is, because changing the script is outside this analysis-only run.
