# Claude / Codex handoff — updated 2026-09-20

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
