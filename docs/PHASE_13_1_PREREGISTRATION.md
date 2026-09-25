# Phase 13.1 pre-registration: do the Market Context event readings predict later price moves?

Written 2026-09-25, **before the test was run on any outcome data**. Market Context labels each news event
"bullish case", "bearish case" or unclear using fixed rules (`core/news_interpretation.py`, method
`deterministic-rules-v1`). The scenario lines on the chart do not use those labels because nothing has shown
they carry information. This test asks the question so that a later change to the chart is earned, not assumed.
Same rules as 6.5 and 9.3: fixed protocol, nothing tuned, deviations labelled.

## Data (fixed)
- Events: rows of `news_store.sqlite3` whose `tickers` contain a collected crypto symbol (`core.news_collector.SYMBOLS`).
  Each is filtered exactly as Market Context filters it (`core.news_context.keep_for_context`: no undated web-search
  results, no evergreen explainer pages, relevance `direct` or `macro`) and read with `interpret_news` (rules only,
  no model, so nothing in the reading can know later prices).
- Prices: Binance daily klines via `core.order_flow_data.fetch_klines`.
- Unit: one **symbol-day**. Its read is `bullish` if the day has more bullish- than bearish-case events, `bearish` if
  more bearish than bullish, otherwise the day is skipped (no read, or a tie).
- Return: entry is the **close of the UTC day of publication** (the reader cannot trade inside that day's move),
  exit is the close `h` days later, log return. Horizons `h` in {1, 3, 7} days; **primary horizon is 3**.
- Excess return: the symbol's mean `h`-day log return over all its days in the sample is subtracted, so a rising
  market does not make bullish reads look skilful.
- Directional excess return = `+excess` for a bullish read, `-excess` for a bearish read.

## Hypothesis (one claim, three horizons -> Bonferroni, 98.33% intervals)
H: the mean directional excess return of the readings is greater than 0.
Interval: moving-block bootstrap over dates (block 7 days, 5,000 resamples, seed 0), pooled across symbols with
whole dates resampled together, using `core.ml_validation._block_resample_dates`.

## Support requires ALL of
1. At the primary horizon (3 days) the lower bound of the 98.33% interval is above 0.
2. The mean is positive in both halves of the sample split by date.
3. At least **30 symbol-days with a read** and at least **10 bullish and 10 bearish** reads.
4. A placebo passes: reading labels shuffled across the symbol-days 2,000 times (seed 0) give a mean directional excess
   at least as large as the observed one in fewer than 5% of shuffles.

Verdicts: `INCONCLUSIVE` if condition 3 fails (the test cannot tell, and says so); `NOT SUPPORTED` if condition 3
holds but any other fails; `SUPPORTED (one sample)` if all hold. A supported verdict is one look at a short window of
survivorship-biased coins: replicate on a later, disjoint period before trusting it, and it would still not be an
edge until costs and a trading rule are tested.

## Limits stated in advance
- The stored news history is thin (the collector started recently; BTC has ~30 days with any items). Expect
  `INCONCLUSIVE`. The script is built to be re-run as the collector accumulates history; the protocol above is frozen.
- The rules mark a direction for only a few narrow headline patterns (completed hacks, earnings-type wording), so
  bullish/bearish reads are rare for crypto.
- AI research notes are **not** tested here. They can only be scored on events that happened after the model's training
  data ends (otherwise the model may already know the outcome), and none are logged yet. Forward logging of AI notes
  is a separate step.
- Deviations from this protocol are labelled exploratory and appended below, not edited in.

# Results (run 2026-09-25 on the stored news; `training_ground/results/phase_13_1.json`)
**INCONCLUSIVE -- zero directional reads, so the readings cannot be tested at all yet.**

Of 427 stored events tagged with the collected symbols, Market Context's filter kept 320 (96 undated web-search results,
5 explainer pages and 6 off-topic items removed). All 320 read `unknown`: no event met the narrow completed-event wording
the rules require for a bullish or bearish case (213 are `unclassified`, 51 `market_flows`, 18 `regulation`, 13
`network_activity`, 12 `monetary_policy`, 8 `earnings`, 3 `security`, 2 `macro_data`). With 0 symbol-days carrying a
read against the required 30 (10 per side), no interval or verdict on predictive value can be computed, and none is claimed.

What this does say: the current rules essentially never fire on crypto news, so the chart's bullish/bearish labels are
almost always "unclear" and carry no tested information either way. The protocol was not changed after seeing this.
Ways to get a testable signal are separate, pre-registered steps: (1) test the sentiment-pipeline headline tone (the
existing H3 route, needs the collector's 300 qualifying days), (2) log AI research notes going forward and score them later.
