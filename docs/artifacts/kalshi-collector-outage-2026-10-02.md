# Kalshi collector outage, 2026-10-02: diagnosis and resolve fail-fast

Day 97 work-loop (2026-10-02 23:20 firing). Sources: `logs/collectors.log`, `pmset -g log`,
`python -m core.kalshi_collector status`, read-only `sqlite3` counts on the canonical snapshot DB (no writes).

## What happened

| Slot (local) | Actually ran | Result |
|---|---|---|
| 07:05 collect | 07:20:36 | **rc=1**: `KalshiError: request to /markets failed ... NameResolutionError` (DNS could not resolve `api.elections.kalshi.com`) |
| 07:05 resolve | 07:20:40 → 08:38:48 (78 min) | rc=3 + WARN line (lookup errors) |
| 12:05 collect | 12:18:29 | rc=0: scanned 3000, stored 53 snapshots at 10:18:30Z |
| 12:05 resolve | 12:18:31 → 12:18:59 | rc=0 |
| 17:05 collect | 17:13:05 | **rc=1**: same DNS `NameResolutionError` traceback |
| 17:05 resolve | 17:13:08 → 20:31:33 (3 h 18 min) | rc=3: `{'resolved': 6, 'pending': 2, 'errors': 67}` + WARN line |

Snapshot slots on 2026-10-02: **1 of 3** stored (12:05 only, 53 snapshots). The 07:05 and 17:05 slots are lost; they
cannot be back-filled (a snapshot is "what the book looked like then"). For comparison 2026-10-01 also stored only two
slots (10:19Z = 23 rows, 15:08Z = 109 rows).

## Cause: the Mac was asleep with the lid closed, not a Kalshi/API problem

`pmset -g log`: **06:41:20 `Entering Sleep state due to 'Clamshell Sleep'` (on battery)**, then only short DarkWakes
(87 on 2026-10-02, typically 2-30 s each, ~15-17 min apart) until **21:09:28 DarkWake to FullWake** (on AC) and
21:57:28 FullWake (user activity).

- launchd runs a missed `StartCalendarInterval` job at the next wake, so 07:05 ran at the 07:20:36 DarkWake and
  17:05 at the 17:13 DarkWake. In most DarkWakes the network is not usable yet, so DNS fails at once (`[Errno 8]
  nodename nor servname provided`). The 12:18 DarkWake happened to have network, so that slot succeeded.
- The long resolve durations are **sleep time, not retry time**. The log shows bursts of 3-4 failed lookups ~4 s apart
  (3 HTTP attempts each, back-off 1 s + 2 s) and then a ~10-17 min gap, matching the DarkWake rhythm: the process is
  frozen while the Mac sleeps and makes a little progress on each wake. 6 markets resolved in wakes that had network.
- Nothing is wrong with the data: failed lookups write nothing, so those markets stay unresolved and are retried
  next run (the Day 94 rc=3 + WARN behaviour worked as designed — this is also the first live post-merge rc=3 observation).

## Fix in this PR (small, behaviour-preserving)

`core/kalshi_collector.py` `resolve`: if **3 lookups in a row cannot connect at all** (the underlying exception is a
`requests.ConnectionError`: DNS failure, refused, unreachable), stop early, log
`Kalshi resolve: 3 lookups in a row could not connect; stopping early, N market(s) not tried this run`, and return a
new `skipped` count. HTTP-level errors (404, 5xx after retries) and any successful lookup reset the streak, so a
single bad ticker can never stop the run.

Unchanged: exit code (still rc=3 with the WARN line, which now adds `stopped early (N not tried: no connection)`),
database schema, outcome labels, the `collect` command, launchd schedule. Untried markets stay unresolved exactly like
failed ones and are picked up next run.

What it does **not** fix: lost `collect` slots. With the lid closed on battery the Mac is simply not online at 07:05 /
17:05. That is the owner's overnight/daytime schedule decision (see the "Overnight schedule resilience proposal" row:
keep the lid open on AC, `pmset` wake schedule, or a non-laptop host). No schedule, launchd or pmset change here.

## Tests

`tests/test_kalshi_collector.py` (offline, mocked HTTP session):
- network down for every lookup → stops after 3, `skipped` = rest, the remaining markets are never requested, no outcome written;
- five HTTP 404s → no early stop (`skipped` 0);
- connection failures interrupted by a successful lookup → streak resets, no early stop;
- `main(["resolve"])` with the network down → rc=3 and the WARN line mentions the early stop.

## Phase 9.2 HOLD counts (2026-10-02 23:30, `python -m core.kalshi_collector status`)

snapshots 2042, markets 1977, days 13 (first 2026-09-19T10:17:46Z, last 2026-10-02T10:18:30Z),
resolved markets 1736, labelled snapshots 1801.
