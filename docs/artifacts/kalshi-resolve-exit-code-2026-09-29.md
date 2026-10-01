# Kalshi resolve: visible errors (2026-09-29, Day 94)

Status: implemented + tested on branch `orchestrator/day94-kalshi-resolve-rc`; PR open; **not merged, not deployed**.
CONTINUATION_PLAN.md / notebook entry: **pending** (held back while PRs #31/#32 are open, to avoid doc conflicts).

## Problem (Day 93 finding)
`python -m core.kalshi_collector resolve` exited 0 even when market lookups failed (the 09-27 DNS outage: 14 lookup
errors, rc=0 in `logs/collectors.log`). Failures were only visible by reading the printed dict.

## Change
- `core/kalshi_collector.py` `main()`: when `resolve` has `errors > 0`, print
  `WARN kalshi resolve: N lookup error(s); ...` and return `RESOLVE_ERRORS_RC` (3).
- `pending > 0` alone (closed but not settled) stays rc=0: it is normal, not an error.
- No change to `resolve()` logic, the SQLite schema or outcome labels; `collect`/`status` exit codes unchanged.

## Wrapper
`scripts/run_collectors.sh kalshi` runs `collect` then `resolve` with no `set -e`, so both still run independently;
the `END rc=3` line now appears in `logs/collectors.log` when lookups fail. The script's own exit code is resolve's rc
(launchd only logs it; it does not change scheduling).

## Tests
`tests/test_kalshi_collector.py`: 3 new tests through `main()` with a mocked HTTP session (404 -> rc 3 + WARN;
pending-only -> rc 0; clean resolve -> rc 0).
Full suite: base env 1495 passed, 1 skipped; 3.11 env 1497 passed, 1 skipped.
