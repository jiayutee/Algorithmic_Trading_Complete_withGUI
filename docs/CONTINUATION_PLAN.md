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
1. **Provider failure visibility:** distinguish HTTP rate limits/auth failures/parser failures from legitimate empty results at source adapters; add offline fixtures, then repeat the same probe. Current adapters can swallow errors, so an empty status cannot establish the cause.
2. **News quality baseline:** measure symbol relevance, duplicate rate, publication freshness, coverage and latency on representative equities/crypto. Compare an explicitly selected paid feed against that baseline before buying or integrating it. No paid feed evaluation has been completed.
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

## 2026-09-25 — scenario projections on the Market Context chart
Owner asked to extrapolate price action for bullish, bearish and unclear/mixed contexts. Implemented as labelled scenarios, not forecasts: from the last close, a bullish and a bearish straight line end one standard deviation (loaded candles' daily log-return sigma times the square root of the horizon) up and down, and the unclear/mixed scenario is flat with the +-1 sigma cone shaded. Horizon 7/14/30 bars, toggle on by default. The legend shows how many reported events read each way, but those counts do not tilt the lines: no model here has shown predictive skill (Phases 6.5-6.9), so the lines show what a typical-size move would look like, not which is likely. Needs 30 loaded candles. Not evaluated: any claim that the scenarios anticipate price.
