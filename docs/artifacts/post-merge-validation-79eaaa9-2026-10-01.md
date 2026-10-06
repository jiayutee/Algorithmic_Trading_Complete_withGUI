# Post-merge validation of main 79eaaa9 (2026-10-01, Day 96, report only)

The owner merged PRs #31 (Phase 9.3 re-run), #33 (Kalshi resolve rc=3), #34 (news quality baseline) and #32
(overnight-schedule proposal, docs only) on 2026-10-01 ~20:23-20:27. This checks main `79eaaa9` after the merge.
No code was changed. The running checkout is on `79eaaa9`, so the collectors now run the rc=3 code; the first
live run with it is the 2026-10-02 07:05 Kalshi job (not observed here).

## Results
| check | result |
|---|---|
| Full suite, base env (3.9), task worktree on `79eaaa9` | 1499 passed, 1 skipped |
| Full suite, 3.11 env (`~/.venvs/algotrader311`) | 1501 passed, 1 skipped |
| main CI run 36921635059 (`CI – pytest`, push, head `79eaaa9`) | completed / **success** (2026-10-01 20:30 UTC). Earlier merge commits 79063ca and 6baa7c6 also green. No open PRs. |
| Kalshi resolve WARN / rc=3 path | **works** on a scratch copy (see below) |
| News quality baseline on a scratch copy | **runs**, rc=0; canonical store SHA-256 unchanged by the run (see below) |

## Kalshi resolve rc=3 (scratch copy)
- Canonical DB `training_ground/datasets/kalshi_snapshots.sqlite3` copied with the SQLite backup API (read-only source).
  SHA-256 before and after: `1a223122da4781beea1ebda19d4ec6ad313e0cc50eb665bd555b4fe454fd40a2` (unchanged).
- Lookups were forced to fail without touching DNS by pointing `HTTPS_PROXY`/`HTTP_PROXY` at a closed local port:
  `KALSHI_DB_PATH=<copy> HTTPS_PROXY=http://127.0.0.1:9 python -m core.kalshi_collector resolve`
- Output: `{'resolved': 0, 'pending': 0, 'errors': 22}`, then
  `WARN kalshi resolve: 22 lookup error(s); those markets stay unresolved until the next run`, **exit code 3**.
- The 2026-10-01 18:08 live resolve (3 DNS errors, `END rc=0`) ran before the merge, so rc=0 there is expected.
- Not exercised: the wrapper `scripts/run_collectors.sh kalshi` writing `END rc=3` to `logs/collectors.log` (running it would
  write to the live log and DB). The wrapper passes resolve's rc through unchanged; the first live confirmation will
  only appear on a run that actually has lookup errors.
- Collector status at check time: 12 days, 1989 snapshots, 1924 markets, 1670 resolved, 1735 labelled snapshots.

## News quality baseline (scratch copy)
- Canonical `news_store.sqlite3` SHA-256 before and after the run: `c1393857bedaeb06286f7dd1ef368a00abda5dfc0f1edcfeea947a76ef33ee61`
  (unchanged by this run; it differs from the 09-30 baseline hash because the daily collector has added items since).
- `python scripts/news_quality_baseline.py --db <copy>` -> rc=0. 1849 items (was 1664 on 09-30), 2026-05-25..2026-10-01,
  28 ingest days, longest gap 26 days.
- Headline numbers moved only slightly from the 09-30 baseline: exact duplicates 15.6% (17.2%), near 17.8% (19.5%),
  fetch-time substituted 45.4% (43.6%). duckduckgo still 58.1% exact duplicates and 97.7% substituted times; brave 92.2%
  substituted. The findings in `news-quality-baseline-2026-09-30.md` still hold.

## Limitations
- Report only. No bug was found, so nothing was fixed. Validation of a merged state is not deployment evidence for the
  07:05 collector run.
- Phase 9.3 (#31) is merged with its F FINDING recorded; Phase 9.4 (any execution) is not started and needs the owner.
