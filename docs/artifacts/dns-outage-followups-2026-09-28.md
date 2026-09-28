# DNS-outage follow-ups — evidence (Day 93, work-loop 2026-09-28 23:20)

Report only. No code, launchd, pmset or scheduled-task settings were changed. Sources: `pmset -g log`,
`logs/execution.log`, `logs/collectors.log`, read-only queries of `training_ground/datasets/kalshi_snapshots.sqlite3`.

## 1. Paper-execution heartbeat stall (09-28 00:07 → 03:33): Mac sleep, not a code defect

**What happened to the machine.** The lid was closed at 2026-09-27 16:54:32 (`Entering Sleep state due to 'Clamshell Sleep'`)
and the next full wake was 2026-09-28 11:25:53 (`DarkWake to FullWake ... due to UserActivity`). The Mac spent the whole
night in Deep Idle, with only short DarkWakes (Power Nap / dasd maintenance) roughly every 10–17 minutes. Most lasted 2–30 s.

**What the runner did.** `logs/execution.log` has no lines between 00:07:34 and 03:33:02. Every burst of log lines
that night starts within a second of a DarkWake (for example 00:01:19 → 00:01:31, 00:07:24 → 00:07:24, 05:55:47 → 05:55:47).
The Yahoo/Binance DNS failures at 20:29–00:07 are fetches made during those few seconds of dark wake, often before the
network was back, so they failed.

**Why the gap is 3.4 h and not 5 min.** The runner waits `POLL_SECONDS=300` between ticks with `threading.Event.wait()`
in 60 s chunks (`core/execution/service.py:_wait_renewing_lease`). On macOS that wait is measured on a clock that stops
while the machine sleeps, so it counts only awake seconds. Between 00:07:42 and 03:33:02 there were 14 DarkWakes.
Together they add up to about 250–300 s of awake time, which is the 300 s poll. The heartbeat is `time.time()`
(wall clock, set in `_publish_status`), so its age at 03:33 was about 3 h 25 min of wall time (the reported 12256 s).
That is correct: the age is measured in wall-clock time, and no UTC/local-time mixing is involved.

**Timeouts.** No missing timeout was found. The Yahoo chart fallback uses `timeout=15` with 2 retries (`core/yahoo_chart.py`),
the requests path in `core/data_loader.py` uses `timeout=15`, and CCXT keeps its default 10 s. The one apparently long
call (05:55:47 CCXT fetch → 06:17:02 error) also spans a sleep (DarkWakes at 05:55, 06:11, 06:17).

**Decision.** Record the finding and close the row. No code change. The runner is healthy (daily bars, so a few hours
of lag cannot skip a decision). Keeping the Mac awake is item 3's proposal, not a runner fix.

## 2. Kalshi collector after the 09-27 DNS outage: no data lost

- The 09-27 17:16 resolve logged DNS errors for 14 markets (`KXAAAGASD{NJ,NY,OH,OR}-26SEP27-*`, 18:26–18:40), rc=0.
- All 14 are now in `outcomes`, all resolved by the 09-28 07:13 run (`resolved_at 2026-09-28T05:13:42Z`): 12 NO, 2 YES
  (OR 5.0450, OR 5.0500). Each has exactly one snapshot, and none is missing or labelled twice.
- The 09-28 07:13 run itself had 1 error (`KXAAAGASDCO-26SEP28-4.1750`, `{'resolved': 28, 'pending': 89, 'errors': 1}`).
  That market resolved at 15:05Z (YES).
- 09-28 12:05: `{'resolved': 0, 'pending': 148, 'errors': 0}`. 09-28 17:05: `{'resolved': 152, 'pending': 3, 'errors': 0}`.
- The 3 markets still pending are unrelated to the outage. They are sports props that closed 09-22 to 09-25
  (`KXNFLTD-26SEP20CARATL-CARJSANDERS0-2`, two `KXWNBAPTS-*`), which Kalshi has not finalised yet.
- Store totals now: 1611 snapshots, 1546 markets, 9 days, 1256 resolved (871 NO / 385 YES).

**Visibility finding (no fix tonight):** `python -m core.kalshi_collector resolve` exits `rc=0` even when some lookups
failed, so `logs/collectors.log`'s `END rc=0` line hides partial failures. They are visible only in the `errors`
count and the WARNING lines. A possible follow-up is a non-zero exit, or a `PARTIAL` tag in `run_collectors.sh`, when `errors > 0`.

## 4. Git worktrees (report only, nothing deleted)

18 worktrees in `git worktree list` besides the main checkout. "merged" means the branch tip is an ancestor of `main` 2f9337f.

| Worktree | Branch | Head | Uncommitted files | Status |
|---|---|---|---|---|
| /private/tmp/algotrader-news-timeline | codex/news-event-timeline | 083bbad | n/a (git marks it *prunable*; the folder is no longer a checkout) | merged |
| agent-a1d1df438b2e9f5c2 | worktree-agent-… | 215e62b | 2 (tests/test_ibkr_manager.py, tests/test_options_chain.py) | merged |
| agent-a2467634f85b232c1 | worktree-agent-… | 4afb6ee | 0 | **unmerged, local only** (1 commit: "inspect(paper-ops): audit execution status surfaces and surface hb_age_s") |
| agent-a27f9ae75346fb859 | worktree-agent-… | 2f9337f | 3 (plan, notebook, kalshi-9-3-rerun artifact: probably copies of PR #31's docs) | merged |
| agent-a4681bbc29a3a69d0 | worktree-agent-… | 215e62b | 3 | merged |
| agent-a47c0330bb3a7503f | worktree-agent-… | 9da0c92 | 0 | merged |
| agent-a4b907d2405c6aa7f | worktree-agent-… | 215e62b | 4 | merged |
| agent-a61455e4d78b70d42 | worktree-agent-… | 8b1b67c | 3 (core/ta_engine.py, ui/main_window.py, test_indicators.py) | **unmerged, local only**: old Day-14 merge commits |
| agent-a667ca007f2fecf73 | worktree-agent-… | 215e62b | 6 | merged |
| agent-afae5b3ad6d72f3fa | worktree-agent-… | 215e62b | 1 | merged |
| orchestrator-day85-dash-options-chain | orchestrator/day85-dash-options-chain | d376d6b | 0 | merged |
| orchestrator-day85-ibkr-mock-gapcheck | orchestrator/day85-ibkr-mock-gapcheck | 8653aec | 0 | merged |
| orchestrator-day85-kalshi-collector-eval | orchestrator/day85-kalshi-collector-eval | 004f6c0 | 0 | merged |
| orchestrator-day85-news-failure-visibility | orchestrator/day85-news-failure-visibility | 7e8c4f8 | 0 | merged |
| orchestrator-day85-paper-ops-inspection | orchestrator/day85-paper-ops-inspection | 86020b0 | 0 | merged |
| orchestrator-day88-ai-research-smoke | orchestrator/day88-ai-research-smoke | 4b042cf | 0 | merged |
| orchestrator-day89-research-loop-8cand | orchestrator/day89-research-loop-8cand | da5614a | 0 | merged |
| orchestrator-day91-kalshi-9-3-rerun | orchestrator/day91-kalshi-9-3-rerun | 6cf6de0 | 0 | unmerged, pushed (PR #31, with the owner) |

Plus tonight's own `orchestrator-day93-overnight-resilience` (this PR).

**Safe to remove (owner's call):** the 7 clean, merged `orchestrator-day85…day89` worktrees (`scripts/orchestrator_git.sh cleanup <name>`),
`agent-a47c0330bb3a7503f` (clean, merged), and the prunable `/tmp` entry (`git worktree prune`).
**Look before removing:** the 6 agent worktrees with uncommitted files, and the 2 local-only unmerged branches
(a2467634's paper-ops audit commit may be worth keeping; a61455e4 looks like stale Day-14 work).
