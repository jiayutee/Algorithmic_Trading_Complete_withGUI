# News quality baseline — 2026-09-30 (report only)

CONTINUATION_PLAN item 2, first slice. **Status: implemented + tested on a task branch; not merged, not deployed.**
Plan/notebook entry is **pending** (not edited while PRs #31/#32/#33 are open).

## How it was produced
- Canonical `news_store.sqlite3` SHA-256 before and after: `b84effcbb9a85284056014058c5fba840100ea3987cc686ae451d8bb1b0df473` (unchanged).
- Report ran against a scratch copy, opened `mode=ro&immutable=1`: `python scripts/news_quality_baseline.py --db <copy>`.
- 1664 items, fetched 2026-05-25 .. 2026-09-30. Offline test: `tests/test_news_quality_baseline.py` (4 tests).

## Headline findings (descriptive, from this one local store)
1. **Publication time is missing for web-search sources.** duckduckgo (98.1%) and brave (96.2%) items carry a
   publication time within 60 s of the fetch time, i.e. the fetch time was substituted. Freshness/lag cannot be measured
   for 43.6% of the store, and any point-in-time use of those items is only as good as the fetch time.
2. **Duplicates are concentrated in duckduckgo:** 58.2% exact / 63.4% near-duplicate headlines (URL is UNIQUE, so the
   same story arrives under different URLs or query variants). openbb 0.4%, brave 0.4-1.5%, gdelt 0-8.3%, rss 0%.
3. **Most items have no symbol:** 963 of 1664 (58%) have an empty `tickers` list; the rest is dominated by BTCUSDT (232)
   and AAPL (189). Tagging quality (relevance) was NOT measured — it needs labels.
4. **Lag (publication -> fetch), where measurable:** openbb median 11.6 h, p90 81.6 h; rss median 0.8 h; gdelt median
   603 h. These mostly reflect back-fill of older articles in batch runs, not feed latency.
5. **Coverage is batchy:** ingest on 27 of 129 days; longest gap 26 days overall. The daily collector has run more
   regularly since 2026-09-17, but the older history is sparse.

## Overall
| group | items | exact_duplicate_rate | near_duplicate_rate | fetch_time_substituted_share | date_only_midnight_share | negative_lag_count | lag_hours_n | lag_hours_median | lag_hours_p90 |
|---|---|---|---|---|---|---|---|---|---|
| all | 1664 | 17.2% | 19.5% | 43.6% | 0.2% | 3 | 936 | 12.7 | 533.42 |

Coverage: {"first_ingest_day": "2026-05-25", "last_ingest_day": "2026-09-30", "span_days": 129, "ingest_days": 27, "longest_gap_days": 26}

## By source family
| group | items | exact_duplicate_rate | near_duplicate_rate | fetch_time_substituted_share | date_only_midnight_share | negative_lag_count | lag_hours_n | lag_hours_median | lag_hours_p90 |
|---|---|---|---|---|---|---|---|---|---|
| openbb | 790 | 0.4% | 0.4% | 0.0% | 0.0% | 0 | 790 | 11.64 | 81.64 |
| duckduckgo | 481 | 58.2% | 63.4% | 98.1% | 0.0% | 0 | 9 | 0.02 | 0.52 |
| brave | 263 | 0.4% | 1.5% | 96.2% | 1.5% | 0 | 10 | 0.1 | 12488.4 |
| gdelt | 84 | 0.0% | 8.3% | 0.0% | 0.0% | 0 | 84 | 603.16 | 1356.27 |
| rss | 40 | 0.0% | 0.0% | 0.0% | 0.0% | 0 | 40 | 0.76 | 14.41 |
| unknown | 6 | 0.0% | 0.0% | 0.0% | 0.0% | 3 | 3 | 705.52 | 705.54 |

Coverage per family:

- openbb: {"first_ingest_day": "2026-07-30", "last_ingest_day": "2026-09-28", "span_days": 61, "ingest_days": 13, "longest_gap_days": 19}
- duckduckgo: {"first_ingest_day": "2026-06-23", "last_ingest_day": "2026-09-30", "span_days": 100, "ingest_days": 21, "longest_gap_days": 25}
- brave: {"first_ingest_day": "2026-05-26", "last_ingest_day": "2026-09-19", "span_days": 117, "ingest_days": 13, "longest_gap_days": 27}
- gdelt: {"first_ingest_day": "2026-05-25", "last_ingest_day": "2026-09-18", "span_days": 117, "ingest_days": 8, "longest_gap_days": 38}
- rss: {"first_ingest_day": "2026-09-19", "last_ingest_day": "2026-09-25", "span_days": 7, "ingest_days": 3, "longest_gap_days": 2}
- unknown: {"first_ingest_day": "2026-06-22", "last_ingest_day": "2026-07-11", "span_days": 20, "ingest_days": 3, "longest_gap_days": 9}

## By symbol (tickers field)
| group | items | exact_duplicate_rate | near_duplicate_rate | fetch_time_substituted_share | date_only_midnight_share | negative_lag_count | lag_hours_n | lag_hours_median | lag_hours_p90 |
|---|---|---|---|---|---|---|---|---|---|
| (none) | 963 | 27.4% | 30.5% | 39.1% | 0.0% | 0 | 586 | 8.99 | 210.13 |
| BTCUSDT | 232 | 7.3% | 8.6% | 59.1% | 1.7% | 1 | 94 | 8.94 | 41.78 |
| AAPL | 189 | 1.1% | 2.6% | 64.0% | 0.0% | 1 | 67 | 161.4 | 849.92 |
| XRPUSDT | 79 | 0.0% | 0.0% | 10.1% | 0.0% | 0 | 71 | 17.47 | 38.26 |
| ETHUSDT | 77 | 0.0% | 0.0% | 39.0% | 0.0% | 1 | 46 | 16.39 | 40.88 |
| ADAUSDT | 28 | 0.0% | 0.0% | 28.6% | 0.0% | 0 | 20 | 229.52 | 954.65 |
| TSLA | 27 | 0.0% | 0.0% | 92.6% | 0.0% | 0 | 2 | 705.53 | 705.54 |
| SOLUSDT | 24 | 0.0% | 0.0% | 29.2% | 0.0% | 0 | 17 | 17.14 | 43.55 |
| DOGEUSDT | 21 | 0.0% | 0.0% | 19.1% | 0.0% | 0 | 17 | 190.15 | 397.26 |
| LTCUSDT | 14 | 0.0% | 7.1% | 42.9% | 0.0% | 0 | 8 | 1228.49 | 2230.23 |
| BNBUSDT | 9 | 0.0% | 0.0% | 11.1% | 0.0% | 0 | 8 | 91.74 | 204.09 |
| LATEST STOCK MARKET NEWS | 1 | 0.0% | 0.0% | 100.0% | 0.0% | 0 | 0 | None | None |


## Limitations (explicit)
- Near-duplicate = identical after normalisation (lowercase, trailing " - Publisher" removed, punctuation stripped);
  paraphrased duplicates are not caught, so near-duplicate rates are a lower bound.
- "negative_lag_count": publication time after fetch time (clock/parse issue); 3 items, all from the `unknown` family.
- brave's p90 lag (12488 h) comes from only 10 measurable items and is not meaningful.
- Not measured: symbol relevance, sentiment, fetch/pipeline latency (not stored), paid feeds. No store writes.
