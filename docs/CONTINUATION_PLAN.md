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
