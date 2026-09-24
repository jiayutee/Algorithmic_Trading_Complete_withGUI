# Claude / Codex handoff — updated 2026-09-24

## Latest: first 8-candidate research loop evaluation (2026-09-24)
Ran the research loop (`python -m core.research_loop run`) against a **scratch copy** of the canonical experiment log (SHA-256 verified unchanged). All 8 candidates remain on trial — 0 promoted, 0 retired. Trend overlay reduces drawdown in all four variants but produces no Sharpe edge, consistent with the Phases 6.7-6.9 pre-registration. EMA Crossover is closest to promotion (all non-CI tests pass; CI lower –0.93 under the 99.375% Bonferroni level). Evidence in [artifacts/research-loop-8cand-2026-09-24.md](artifacts/research-loop-8cand-2026-09-24.md) and its JSON sibling. Canonical promote/retire state untouched; applying decisions is an owner decision.

## Previous: optional AI research on Market Context
Branch `feat/ai-research-groq`, base `codex/news-event-timeline` (PR #21). Adds `core/ai_research.py`: an on-demand,
per-event Groq call (free tier, opt-in `GROQ_API_KEY`) that reasons only from the supplied headline/summary text
plus the existing `core/sentiment.py` label, returning `None` on any failure so the deterministic
`core/news_interpretation.py` reading is always shown regardless. Dash-only, button-triggered (not automatic).
See [feature evidence](artifacts/ai-research-2026-09-22.md). `GROQ_API_KEY` not yet in `.env` -- inert until the
owner adds one.

## Previous: chart-linked Market Context
Branch `codex/news-event-timeline`, base `215e62b`. Dash Market Context snapshots the loaded candles and fetches news on demand; core/news_interpretation.py supplies conditional explanations without model calls. Source/pipeline files intentionally untouched while provider PR16 is open. See [feature evidence](artifacts/news-context-2026-09-21.md) and Notion task https://app.notion.com/p/3e2d2ab050d98108bc0ddaa22221c02d . Implemented for review, not deployed. Full validation and PR links recorded on the task/PR.

Desktop, macro calendar and grounded model interpretation remain future slices; no predictive price curve or trading changes.

## Latest: orchestrator PR workflow
Owner requested removal of the overnight direct-main exception. Updated AGENTS.md, CLAUDE.md, the canonical orchestrator runbook, its Claude wrapper and both release-specialist definitions. Includes Claude PR #13 helper (7ab47ce), wired into the launcher/runbook, plus a regression fix preserving unpushed local commits when recreating worktrees. All implementation runs prepare task-branch PRs; no automatic merging. Pending review stays In progress. See [workflow evidence](artifacts/orchestrator-pr-workflow-2026-09-20.md) for validation and activation limits.

Notion: https://app.notion.com/p/3e0d2ab050d981218af5e78bd6a3fb90

## Previous delivery (historical evidence)

## Start here
Read [the owner-confirmed notebook](algotrader_notebook.html) and [remaining plan](CONTINUATION_PLAN.md). This slice starts at merged main `ad69e318cd7cdedbb3a06d63e900f054213564f4`, on `codex/news-health-handoff`, in `/private/tmp/algotrader-news-health`.

The current project is `/Users/jiayutee/Dev/Projects/Algorithmic_Trading_Complete_withGUI`; the desktop task originally pointed at an older backup. The running paper agent was not restarted or modified. Claude also had active worktrees: reconcile concurrent changes before merging.

## Delivered
`NewsPipeline.source_status()` and the existing health registry expose safe structured diagnostics. `scripts/smoke_news.py` probes actual configured sources, uses actual ticker/text routing, honors both Brave key aliases, and emits JSON. It does not score sentiment or open the news store. Credentials, article bodies, request URLs and raw exception text are excluded from the report.

[Evidence artifact](artifacts/news-source-smoke-2026-09-19.json): OpenBB delivered five raw items each for BTCUSDT and AAPL; other enabled sources returned empty/slow-empty/timed-out outcomes. This is degraded delivery, not a repaired news system. The artifact records the precommit base and dirty state at execution time.

[Notion task](https://app.notion.com/p/3e0d2ab050d981d5829def6cde9a4e9c).
GitHub: [PR #8](https://github.com/jiayutee/Algorithmic_Trading_Complete_withGUI/pull/8), implementation commit `603ac89`. Merged by Claude as `b3c8d51`; current main includes PRs #8–#11.

## Validation
- Focused: `python -m pytest tests/test_news_diagnostics.py tests/test_news_hardening.py tests/test_news_pipeline.py -q --tb=short` — 48 passed.
- Full: `python -m pytest --ignore=test_gui.py -q --tb=short` — 1,216 passed, 1 skipped, 14 warnings in 212.33 seconds.
- Live: `python scripts/smoke_news.py --symbols BTCUSDT AAPL --env-file /path/to/project/.env --output docs/artifacts/news-source-smoke-2026-09-19.json` — exit 0, both probes degraded, approximately 12 seconds total fetch time.
- Python: local Miniconda base 3.9. Full tests require OpenBB access to its user log directory; the sandbox-only attempt failed on that permission and was rerun with approval.

At original publication GitHub had not yet reported checks. Subsequent main CI passed after Claude merged PRs #8–#11. The counts above describe the original local run. Documentation copies were placed in the current project folder for discovery before merge; source code remains on the PR branch. If Git blocks checkout because those files are untracked, preserve or compare them against the PR before removing the duplicates.

## Next action
Follow item 1 in CONTINUATION_PLAN.md: typed provider failures and replayable fixtures. Do not infer provider causes from empty output. No paid feed was purchased, no news edge was proved, and no live trading was enabled. Historical plan items are not all completed.
