# Continuation plan — 2026-09-19

Source of intent: [AlgoTrader notebook](algotrader_notebook.html), confirmed by the owner. Historical notebook assessments predate merged PRs #3–#7; use its latest update and the current code together.

## Delivered in this continuation
- [x] Reuse the application's configured sources, ticker/text routing, deadline and circuit breaker in `scripts/smoke_news.py`.
- [x] Expose per-source delivery timestamps, item counts, elapsed time, failures, cooldown and missing configuration.
- [x] Record a [live BTC/AAPL probe](artifacts/news-source-smoke-2026-09-19.json) without sentiment scoring or production database writes.
- [x] Add offline regression coverage and a durable [handoff](CODEX_HANDOFF.md).

## Evidence and limits
The live probe delivered five raw items per symbol, all from OpenBB. Brave and DuckDuckGo returned no items; RSS and GDELT timed out or returned slow empty results. NewsAPI and EventRegistry were unconfigured. Each symbol's fetch took about six seconds. Exit zero means some raw delivery for every symbol, not that every provider, article relevance or sentiment is healthy. Initialization time is outside the fetch budget; existing daemon workers can finish after it.

## Remaining work, in recommended order
1. **Provider failure visibility** — *implemented and tested on branch `orchestrator/day85-news-failure-visibility` (PR pending review, not merged/deployed, 2026-09-20):* adapters now classify `ok_empty` / `rate_limited` / `auth_failed` / `parse_error` / `timeout` / `error`, surfaced via the health registry, `source_status()` and the smoke report; offline fixtures added; probe repeated ([evidence](artifacts/news-failure-classification-2026-09-20.md)). **Still open:** the 6 s shared deadline hides the adapter's own cause for the slow sources (brave/rss/gdelt), OpenBB classification is best-effort, and DuckDuckGo CAPTCHA pages read as empty.
2. **News quality baseline** — *first slice merged 2026-10-01 (PR #34, report only; see the 2026-09-30 entry below); fix slice proposed, not implemented:* measure symbol relevance, duplicate rate, publication freshness, coverage and latency on representative equities/crypto. Compare an explicitly selected paid feed against that baseline before buying or integrating it. No paid feed evaluation has been completed.
3. **Sentiment and news research:** build labeled evaluation and sufficient point-in-time news history; test incremental out-of-sample value after costs. Current diagnostics neither validate sentiment nor establish a tradable news signal.
4. **Paper operations:** inspect the existing execution status UI before adding persistent alerts, protective paper stop rules and multi-strategy allocation. Preserve one runner, idempotency, entry blocks that permit exits, and paper-only defaults.
5. **Research automation:** finish scheduling and reviewed promotion-to-paper linkage from Phase 12; retain holdout/cost gates and full experiment provenance. Existing quant methods and paper execution already exist; profitability remains unproven.
6. **Notebook research backlog:** Kalshi probability modeling (9.2), remaining 9.3 work and the user's Hawkes lesson comprehension checkpoint remain open. Real-money execution is a separate future scope, not enabled by this continuation.

## Update contract
For every implementation slice, update this plan, notebook, evidence/handoff, the Notion sprint task and GitHub commit/PR. Record exactly what was tested and what remains. Never mark an entire phase complete for a diagnostic-only change.

## 2026-09-20 — unified agent workflow
The owner requested that overnight and interactive agents use the same workflow. The orchestrator and its release specialist now require isolated task branches/worktrees, full local tests, a PR against main, latest-head CI reporting and owner review before merge. Unmerged tasks remain In progress. Retries reuse a recorded, exclusively owned task worktree; pending-review PRs are not duplicated. Runtime checkout updates remain separate from merging.

See [workflow evidence and activation notes](artifacts/orchestrator-pr-workflow-2026-09-20.md). The news and trading backlog above is unchanged.

## 2026-09-21 — chart-linked news context
Implemented a Dash-first Market Context tab from the owner's MEXC reference: event markers and selection, conditional-case filters, observed price context, expandable evidence-linked explanations and source coverage. Shared interpretation code can serve desktop later. See [evidence and limitations](artifacts/news-context-2026-09-21.md). Merged as PR #21.

## 2026-09-22 — optional AI research on top of Market Context
Owner asked to add AI research to the analysis, using a free model if the configured sentiment pipeline is good enough to build on. `core/sentiment.py` was reviewed: FinBERT (optional) -> DeepSeek LLM (opt-in) -> keyword rule-based fallback, with chunking and per-row fallback on partial LLM failures -- judged solid enough to extend rather than replace.

Added `core/ai_research.py`: an on-demand, per-event research note from Groq (free tier, `GROQ_API_KEY`), reasoning only from the same supplied headline/summary text plus the existing sentiment label, never fetched automatically (a "Get AI research on selected event" button, so cost/latency stays bounded to what the reader actually opens). Any missing key, network error, timeout or malformed JSON returns `None` and the deterministic `core/news_interpretation.py` reading is shown unaffected -- this is additive, not a replacement path.

Not done: no evaluation of the AI research note's own accuracy or calibration (it is presented as a hypothesis to verify, consistently with the deterministic path's disclaimers, not tested against outcomes). The owner has since added `GROQ_API_KEY` to `.env` (2026-09-22); the live success path has not yet been exercised against the real API. See [evidence artifact](artifacts/ai-research-2026-09-22.md).

## 2026-09-23 — research loop: trend-filtered candidates
The loop had four candidates (MACD/RSI, EMA Crossover, Stochastic, GBM) because those are the only standalone signal strategies that run in the current environment; TD3/DDPG/FinRL need `stable_baselines3`/`finrl` plus trained model files, the LSTM is deprecated, and the trend filter is an overlay rather than a signal source. Added a `trend_overlay` flag to `Candidate` and four wrapped variants ("... + Trend", 8 candidates total) so the overlay is tested on the same promotion rules as everything else. The Bonferroni level widens automatically with the candidate count. Expectation, from Phases 6.7-6.9: drawdown improves, Sharpe versus buy-and-hold does not clear the bar. No result has been produced yet -- the loop has not been re-run with the new candidates, and no promotion is implied. RL candidates were not added: they would need installed dependencies, trained artifacts and a pre-registered training protocol first.

## 2026-09-24 — research loop: first 8-candidate evaluation (scratch copy)

Ran `python -m core.research_loop run` against a scratch copy of `training_ground/results/experiments.sqlite3` (canonical file SHA-256 verified unchanged before and after: `4a86ff64...f26d34`). Git commit `d00c66b`, 8 symbols, 700 eval bars, data through 2026-09-23, Bonferroni 99.375% CI per candidate (0.05/8).

**Result: 0 promoted, 0 retired. All 8 remain on trial.**

| Candidate | Sharpe | Diff vs B&H | 99.4% CI | Trades | maxDD% | PASS? |
|---|---|---|---|---|---|---|
| MACD/RSI | 0.38 | -0.10 | [-3.15, +2.95] | 9 | -5% | fail (3 reasons) |
| EMA Crossover | 0.88 | +0.41 | [-0.93, +1.74] | 128 | -39% | fail (CI lower only) |
| Stochastic | 0.36 | -0.12 | [-3.48, +3.24] | 97 | -261% | fail (3 reasons) |
| GBM (LightGBM) | -0.51 | -0.98 | [-3.50, +1.62] | 1192 | -56% | fail (2 reasons) |
| MACD/RSI + Trend | 0.59 | +0.11 | [-3.19, +3.20] | 5 | -3% | fail (3 reasons) |
| EMA Crossover + Trend | 0.57 | +0.10 | [-2.48, +2.99] | 100 | -15% | fail (2 reasons) |
| Stochastic + Trend | 0.75 | +0.28 | [-2.88, +3.17] | 83 | -107% | fail (3 reasons) |
| GBM + Trend | -0.73 | -1.20 | [-4.09, +2.06] | 622 | -40% | fail (2 reasons) |

Prior expectation (Phases 6.7-6.9): trend overlay reduces drawdown, does not produce a Sharpe edge. This run agrees: drawdown reduced in all four trend-filtered variants; no trend-filtered variant clears the promotion bar. EMA Crossover remains the closest to promotion (all non-CI tests pass; CI lower -0.93 still below zero). Canonical state untouched; applying any promotion/retirement decision is an owner decision. Evidence: [research-loop-8cand-2026-09-24.md](artifacts/research-loop-8cand-2026-09-24.md) and sibling JSON. Status: evidence only (not merged to main as a code change; docs committed to branch `worktree-agent-a4738bd4b58b22a7b`, PR pending).

## 2026-09-23 — Phase 9.3 re-run (5 days of collected data)

Ran `training_ground/experiments_9_3_collected.py` with the main checkout's Kalshi DB (896 snapshots, 560 resolved,
594 labelled, 2026-09-19..23). No protocol change.

- L (longshots, ask <= 0.10): n=206 (117 events), CI97.5 [-0.039, +0.065]. Verdict: **no evidence** of overpricing.
- F (favorites, ask >= 0.90): n=100 (54 events), CI97.5 [+0.018, +0.026]. **Pre-registered FINDING** (all three
  criteria met for the first time). Bootstrap CI is degenerate (100/100 hit rate); exploratory calibrated tail gives
  P=0.109 under independence -- not statistically surprising. See caveats in PHASE_9_3_PREREGISTRATION.md.
- Phase 9.2 stays on HOLD per pre-registered rules.
- Evidence: `training_ground/results/phase_9_3_collected.json`; archive: `phase_9_3_collected_20260922.json`.

## 2026-09-23 — AI research live activation smoke
One real Groq call through the Market Context path failed with HTTP 404: Groq has retired the default model `llama-3.3-70b-versatile`, so the keyed feature still showed no AI block. The key is valid. One verification call on `openai/gpt-oss-120b` succeeded (1.9 s), and the note rendered beside the unchanged deterministic reading. The orchestrator PR changes only the default model. Single-sample caveat: the note read a past price surge as "bullish" at 0.80 confidence. Accuracy, calibration and rate limits are still untested. Not merged or deployed. See [smoke evidence](artifacts/ai-research-smoke-2026-09-23.md).

Follow-up (review of this PR, 2026-09-25): the replacement `openai/gpt-oss-120b` is a reasoning model whose hidden reasoning shares the completion budget, so the original `max_tokens=400` failed intermittently with HTTP 400 (`max completion tokens reached before generating a valid document`): 3 of 8 live calls succeeded. At `max_tokens=1500` 6 of 6 succeeded (a second setting with `reasoning_effort=low` also 6 of 6). The limit is now 1500 with a regression test. Small samples; the note's accuracy is still unevaluated.

## 2026-09-25 — scenario fans on the Market Context chart, and the test of the event readings
Owner asked to extrapolate price action for bullish, bearish and unclear/mixed contexts, then said a straight line is too simple. The chart now draws scenario **fans**: 500 paths per scenario, block-bootstrapped from the loaded candles' own daily log returns (mean removed, 5-day blocks so volatility clustering survives), with a constant per-bar tilt of +-1 sigma/sqrt(horizon) for bullish/bearish and none for range. Each scenario shows the middle-50% band, the median (dotted) and one jagged example path; horizon 7/14/30 bars; deterministic for the same candles. These are simulations, not forecasts. The legend counts reported events per reading, but the counts do not tilt the fans.

Before letting the news lean the fans, the readings had to be tested: [Phase 13.1](PHASE_13_1_PREREGISTRATION.md) was pre-registered (committed before any outcome data was touched) and run through `training_ground/experiments_13_1.py`. **Result: INCONCLUSIVE.** Of 427 stored events for the collected symbols, 320 passed the Market Context filter and all 320 read `unknown`; there are zero bullish/bearish symbol-days against the 30 required, so nothing can be said about predictive value. The rules essentially never fire on crypto news. Next candidates, each its own pre-registration: the sentiment pipeline's headline tone (H3, needs the collector's 300 qualifying days) and AI research notes logged going forward (scoring them on past events would let the model see outcomes it was trained on).

## 2026-09-25 — model readings colour the Market Context events
Owner asked why every event reads unclear. The rule-based label can only be directional for security and earnings headlines that name the asset in completed-event wording (0 of 320 stored crypto events qualified), so most of the timeline was grey by design. Added an optional, best-effort **model reading** (Groq, same key): on refresh, the 24 most recent events the rules left non-directional are sent in chunks of 10 for one label each (bullish / bearish / mixed / unclear + confidence + a short reason). Rules win when they fire; a model reading colours an event only at confidence >= 0.6 (fixed in advance, not tuned) and is drawn as a hollow circle, labelled "model, untested", with the reason on the card. Cached per headline for 6 hours; a rate limit or error leaves the rule labels and says so in the status line; off without `GROQ_API_KEY`. The counts feed the legend but still do not tilt the scenario fans.

Live check (10 real BTC headlines, two runs): 10 of 10 labelled in about 3 s. Known weakness: the model still reads some price-move headlines ("Bitcoin Surges Past $87,000 ...") as bullish although the prompt tells it not to, i.e. it partly echoes the move that already happened. The labels have not been tested against later price moves; a pre-registered test would need the labels logged going forward (scoring past events with a model trained later can leak outcomes).

## 2026-09-28 — DNS-outage follow-ups and overnight-schedule proposal (Day 93, report + proposal only)
The paper runner's 09-28 00:07→03:33 heartbeat stall was **Mac sleep, not a defect**. The lid was closed from 09-27 16:54 to 09-28 11:25, and the runner's 300 s poll wait counts only awake seconds, which the 14 short DarkWakes supplied after about 3.4 h of wall time. The heartbeat (wall clock) correctly read about 12 256 s. No missing timeout was found. The 14 Kalshi markets that hit DNS errors on 09-27 all resolved on 09-28 07:13, so no data was lost; `resolve` exiting rc=0 despite errors is recorded as a visibility gap. 18 git worktrees were reported (nothing deleted). Evidence: [dns-outage-followups-2026-09-28.md](artifacts/dns-outage-followups-2026-09-28.md). The overnight misses have the same root cause. [Proposal](proposals/overnight_schedule_resilience.md) recommends a caffeinate window (on AC) + agenda polling in the work-loop + a "running late" Telegram. The owner decides; no schedule, launchd or pmset setting was changed.

## 2026-09-25 — post-merge validation of main 19f8841 (report only)

Throwaway worktree on main after the owner merged #16, #17, #18, #20, #25 and #26. Full suite: base env 1484 passed, 3.11 env 1486
passed (1 skipped each); main CI green. The Market Context scenario-fan figure is deterministic (same sha256 across two processes
and commits). `training_ground/experiments_13_1.py` reproduces the committed `phase_13_1.json` exactly (only `run_at` differs), but
that result has n=0 directional reads, so its INCONCLUSIVE verdict means "no data yet", not "tested and failed". No bugs found.

## 2026-09-26 — post-merge validation of main 2f9337f (PR #30 model readings, PR #19 options chain; report only)

- **Suite:** base env 1489 passed, 4 skipped; 3.11 env 1491 passed, 4 skipped (was 1484/1486 with 1 skipped on 19f8841).
- **Groq key unset:** model readings report status `off`, rule labels unchanged, no model-reading field added, no network call.
- **Mocked Groq:** a reading with confidence 0.59 does not colour an event; a firing rule beats a 0.99 model reading; a 429 or
  a connection error leaves the rule labels intact (existing tests in `tests/test_news_context.py` / `tests/test_ai_research.py`
  plus scratch checks).
- **Live Groq (one call each):** on-demand research note parsed fine in 2.3 s with all expected keys, no truncation, so the
  max_tokens fix holds. A 5-headline model-readings batch returned 5 of 5 readings in 1.4 s.
- No bugs found. This checks plumbing only; the model readings are still untested against later price moves.

## 2026-09-26 — Phase 9.3 re-run (7 days of collected data)

Ran `training_ground/experiments_9_3_collected.py` with the main checkout's Kalshi DB (1264 snapshots, 962 resolved,
2026-09-19..26). No protocol change. Previous result file archived as `phase_9_3_collected.json` (09-23 run);
new output: `training_ground/results/phase_9_3_collected_2026-09-26.json`.

- **Data:** 919 markets pass the two-sided-book filter (was 530), 309 distinct events, 7 distinct closing days.
  Lead time: 5.9–68.9 h (median 37.9 h). Category mix: commodity 334, city-high-temp 307, sports 169, rainfall 86, politics 23.
- **L (longshots, ask <= 0.10):** n=356 (189 events), mean ask 0.044, hit rate 0.037, CI97.5 [-0.035, +0.030].
  Verdict: **no evidence** of overpricing (CI crosses zero, halves have opposite signs).
- **F (favorites, ask >= 0.90):** n=123 (71 events), mean ask 0.977, hit rate 1.000, CI97.5 [+0.019, +0.027].
  Verdict: **FINDING** (all three criteria met, second consecutive run). F is 115/123 commodity markets.
  P(all 123 YES | fairly priced, independence) = 0.058. Bootstrap CI is degenerate (100% hit rate reflects only ask spread).
- Phase 9.2 stays on HOLD. This is analysis only; no execution path exists or is implied. Interpretation of F is the owner's call.
- Evidence: `training_ground/results/phase_9_3_collected_2026-09-26.json` and `docs/artifacts/kalshi-9-3-rerun-2026-09-26.md`.
- **Status (2026-10-01):** merged by the owner (PR #31). Analysis only; Phase 9.4 (any execution) not started.

## 2026-09-29 — Kalshi resolve reports lookup errors (Day 94)
`python -m core.kalshi_collector resolve` now prints `WARN kalshi resolve: N lookup error(s)` and exits 3 when any market
lookup failed (closed-but-unsettled markets alone stay rc=0). No change to resolve logic, schema or labels. 3 new tests.
**Implemented, tested, merged 2026-10-01 (PR #33).** The running checkout is on main, so the 07:05/12:05/17:05 collector
runs use it from 2026-10-02; a live `END rc=3` will only appear on a run that has errors. Evidence:
[kalshi-resolve-exit-code-2026-09-29.md](artifacts/kalshi-resolve-exit-code-2026-09-29.md).

## 2026-09-30 — news quality baseline (Day 95, report only)
`scripts/news_quality_baseline.py` reads a scratch copy of the news store (read-only) and reports duplicates, publication
freshness, coverage and lag per source and symbol. Findings: web-search sources (duckduckgo 98%, brave 96%) store the
fetch time as publication time; duckduckgo is 58% exact duplicates; 58% of items have no symbol; ingest is batchy
(27 of 129 days). Symbol relevance was not measured (needs labels). **Implemented, tested, merged 2026-10-01 (PR #34).**
Nothing to deploy (report tool). Evidence: [news-quality-baseline-2026-09-30.md](artifacts/news-quality-baseline-2026-09-30.md).
Next: the fix-slice proposal in `docs/proposals/news_quality_fix_slice.md` (proposal only).

## 2026-10-01 — post-merge validation of main 79eaaa9 (Day 96, report only)
After the owner merged #31, #32, #33 and #34: base env 1499 passed, 1 skipped; 3.11 env 1501 passed, 1 skipped; main CI 36921635059 success.
Kalshi resolve on a scratch copy with forced lookup failures printed the WARN line and exited 3 (canonical DB hash
unchanged). The news baseline re-ran on a scratch copy (1849 items; numbers within ~2 points of 09-30; canonical hash
unchanged by the run). No bugs found. Evidence: [post-merge-validation-79eaaa9-2026-10-01.md](artifacts/post-merge-validation-79eaaa9-2026-10-01.md).
