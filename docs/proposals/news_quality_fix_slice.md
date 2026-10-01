# News Quality Fix — Slice 1 Proposal

**Status: proposal only — not implemented, not merged.**
Date: 2026-10-01

---

## 1. Goal and non-goals

### Goal

Address the three most tractable quality problems in the news store revealed by the baseline report (docs/artifacts/news-quality-baseline-2026-09-30.md):

1. Mark items whose `datetime_utc` is the fetch time, not a real publication date, so downstream consumers can filter or weight them appropriately.
2. Reduce near-identical duplicate headlines from the same search-engine source arriving under different URLs before they enter the store.
3. Surface ingest gaps in `python -m core.news_collector status` so the owner can see when collection missed days.

Scope is additive and read-compatible. No structural changes to existing rows, no schema breakage for old readers.

### Non-goals

- No rewriting of historical rows (no UPDATE or DELETE on existing data).
- No paid news feed integration.
- No changes to sentiment scoring, sentiment caching, or the merge/aggregation logic in `core/news_pipeline.py`.
- No scheduling changes (the collector schedule is already a launchd concern and out of scope here).
- No symbol relevance tagging (needs human labels).

---

## 2. Smallest first slice: publication-time reliability flag

### Problem

`DuckDuckGoSource.fetch_classified` (core/news_sources.py line 742) constructs `NewsItem.datetime_utc` as `coerce_datetime(pd.Timestamp.utcnow())` because DuckDuckGo HTML search results carry no publication date. `BraveSearchSource.fetch_classified` (lines 563-592) does the same when the result has no `published`/`date`/`age` field. `coerce_datetime` (core/news_sources.py line 92-96) returns `datetime.now(timezone.utc)` when it receives `None` or a value that pandas cannot parse. The effect: 98% of DuckDuckGo rows and 96% of Brave rows have `datetime_utc` within 60 seconds of `created_at` (the SQLite row-insertion timestamp).

### Proposed flag: `pub_time_is_fetch_time`

Add a boolean column `pub_time_is_fetch_time INTEGER DEFAULT 0` to the `news` table. The value is set **at item-construction time inside each source adapter** when the provider gave no date, not inferred after the fact by the 60-second heuristic.

Where to set it:

- `DuckDuckGoSource.fetch_classified` (core/news_sources.py ~line 741): always `True` — this source never provides a publication date.
- `BraveSearchSource.fetch_classified` (core/news_sources.py ~line 572-578): `True` when `published_at` resolves to `None` before `coerce_datetime` is called (i.e., none of `published`, `date`, `age`, `page_age`, `timestamp` are present in the result dict).

`NewsItem` (core/news_sources.py lines 65-80) gains a new field `pub_time_is_fetch_time: bool = False`. `NewsStore.add_items` (core/news_store.py lines 53-97) writes the value alongside the other fields.

### Existing rows

Existing rows keep `pub_time_is_fetch_time = 0` (the column default). They are not touched. Callers that want a conservative signal can derive a read-time flag using the same 60-second rule already implemented in `scripts/news_quality_baseline.py` (line 29, `SUBSTITUTED_SECONDS = 60`): `abs(created_at - datetime_utc) < 60s`. Both signals can coexist — the stored flag is authoritative for new rows; the read-time heuristic covers old rows at query time without needing a migration UPDATE.

### Schema change

One additive column in the migration, backward compatible. Old readers that do not select the column are unaffected. The UNIQUE constraint on `url` is unchanged.

---

## 3. Dedupe rule

### Problem

The URL UNIQUE constraint prevents exact URL duplicates from entering the store. It does not prevent the same story from arriving under different URLs (e.g., the same article from different query variants, or a canonical vs. a tracking-parameter-stripped URL that differs from a redirected URL). The baseline found 58% exact and 63% near-duplicate headlines for DuckDuckGo (1664-item store). The pipeline's `_prefilter` method (core/news_pipeline.py lines 608-632) already drops same-session duplicates in memory using a normalised headline key (`_normalized_headline_key`) and canonicalised URL, but this only runs over items fetched in one pipeline call. Items arriving in separate collector runs (different days) bypass it.

### Proposed dedupe key and location

**Key:** `(headline_hash, source_family, day_bucket)`

- `headline_hash` is the existing `sha256(headline.strip().lower())` already computed by `_headline_hash` (core/news_store.py line 22-25) and stored in the `headline_hash` column.
- `source_family` is the `metadata.source_api` value (e.g., `"duckduckgo"`, `"brave"`, `"openbb"`). The same headline from two different families is a valid independent confirmation and should not be deduped.
- `day_bucket` is `date(datetime_utc)` — a 24-hour UTC day window. Items with the same normalised headline from the same source family on the same day are treated as the same story regardless of URL.

**Where it applies: insert time, inside `NewsStore.add_items`.**

Before the existing `INSERT OR IGNORE`, add a check: `SELECT id FROM news WHERE headline_hash = ? AND json_extract(metadata, '$.source_api') = ? AND date(datetime_utc) = date(?)`. If a row is found, skip the insert. This runs at O(1) per item using the existing `idx_news_headline_hash` index plus a full-expression scan on the narrow result set.

An alternative is a partial unique index on `(headline_hash, date(datetime_utc))` filtered to rows sharing a source family, but SQLite partial indexes do not support expressions on JSON columns. A composite index on `(headline_hash, date(datetime_utc))` (two text columns) is cheap and would accelerate the lookup without enforcing a constraint (the constraint logic stays in Python so source-family scope is preserved).

**What is NOT caught:**

- Paraphrased duplicates (same story, different words) — the near-duplicate rate in the baseline is already a lower bound of what a full semantic dedupe would remove.
- Cross-family duplicates from genuinely independent sources are intentionally kept.
- Items in already-stored history are not retroactively deduped (no-rewrite-of-history rule).

---

## 4. Ingest-gap reporting

### Problem

`python -m core.news_collector status` (core/news_collector.py lines 87-97) currently shows per-symbol qualifying-day counts and ETA toward the H3 threshold. It does not surface the overall ingest calendar: longest gap, last ingest date, or whether the daily collector has actually run recently.

### Proposed change

Extend `format_status` (core/news_collector.py line 87) to include a single summary line at the top showing:

- Last ingest date (max `date(created_at)` across all rows in the store).
- Consecutive days since last ingest (`today - max_ingest_date`).
- Longest gap in the collection window.

These numbers come from the existing `coverage()` function (core/news_collector.py lines 32-59), which already queries the store grouped by day. The only addition is reading `max(date(created_at))` and computing the delta to today to form a "stale for N days" indicator.

No schema change is needed. No scheduling changes.

---

## 5. Schema/migration impact

### New column

```sql
ALTER TABLE news ADD COLUMN pub_time_is_fetch_time INTEGER DEFAULT 0;
```

This is a single `ALTER TABLE … ADD COLUMN` with a literal default, which SQLite executes O(1) (no table rewrite). Old readers that do not name this column in their SELECT are unaffected. The existing `DEFAULT_DB` path (`news_store.sqlite3`) is git-ignored and local-only.

### New index (optional, for dedupe lookup performance)

```sql
CREATE INDEX IF NOT EXISTS idx_news_headline_date ON news(headline_hash, date(datetime_utc));
```

This is additive. The index is not used by any existing query and does not change any UNIQUE constraint. Creating it on an existing store takes a few seconds on ~2000 rows.

### Migration file

A new file `migrations/0002_pub_time_flag.sql` containing both statements. `NewsStore._ensure_tables` (core/news_store.py lines 42-51) is extended to also run `migrations/0002_pub_time_flag.sql` on startup, wrapped in the existing `try/except FileNotFoundError` pattern.

### Backward compatibility

- Old readers (code that predates this change) see `pub_time_is_fetch_time = 0` for all existing rows. This is conservative: it says "we don't know" rather than "this is a real date", which is slightly pessimistic for DuckDuckGo rows (which are always substituted). The read-time 60-second heuristic covers old rows at query time.
- `NewsItem.pub_time_is_fetch_time` defaults to `False`, so source adapters not yet updated continue to produce the correct conservative default.
- If `ALTER TABLE … ADD COLUMN` is run against a store that already has the column, SQLite raises "duplicate column name". The existing `try/except Exception` block in `_ensure_tables` (core/news_store.py line 48) will log a warning and continue cleanly.

---

## 6. Tests to add

All tests are offline and fixture-based. No network calls, no writes to `news_store.sqlite3`.

### tests/test_news_store_pub_flag.py

1. Insert a `NewsItem` with `pub_time_is_fetch_time=True`; assert the stored row has `pub_time_is_fetch_time = 1`.
2. Insert a `NewsItem` with `pub_time_is_fetch_time=False` (default); assert the stored row has `pub_time_is_fetch_time = 0`.
3. Open a store created from the old schema (simulate via in-memory SQLite with `CREATE TABLE news … -- no pub_time_is_fetch_time`); assert that after `_ensure_tables` the column is present and all pre-existing rows have value `0`.

### tests/test_news_store_dedupe.py

4. Insert two items with identical `headline_hash`, same `source_api` in metadata, same UTC day, different URLs; assert only the first is stored (dedupe fires).
5. Insert two items with identical `headline_hash`, different `source_api` in metadata; assert both are stored (cross-family kept).
6. Insert two items with identical `headline_hash`, same `source_api`, different UTC days (one day apart); assert both are stored (day-window boundary).

### tests/test_duckduckgo_pub_flag.py

7. Construct a `DuckDuckGoSource` with a fixture HTML response (no network); assert that all returned `NewsItem` objects have `pub_time_is_fetch_time=True`.

### Validating with the baseline script

Run on a scratch copy of the canonical store, comparing SHA-256 before and after:

```
cp news_store.sqlite3 /tmp/scratch_news.sqlite3
sha256sum news_store.sqlite3
python scripts/news_quality_baseline.py --db /tmp/scratch_news.sqlite3
sha256sum news_store.sqlite3   # must be identical to the first line
```

The baseline script opens files with `mode=ro&immutable=1` (scripts/news_quality_baseline.py line 115), so it cannot write. The canonical-store SHA must not change. Migration testing goes against a separate in-memory or temp-file store.

---

## 7. Rollback

The changes are additive. To roll back:

1. Revert the source code commits.
2. The `pub_time_is_fetch_time` column and the `idx_news_headline_date` index remain in existing local stores (SQLite does not auto-remove them) but are ignored by the reverted code. They do not affect any existing query or constraint.
3. The `migrations/0002_pub_time_flag.sql` file can be removed. If the column already exists in a local store, the `ALTER TABLE … ADD COLUMN` will fail on next startup; the `try/except` in `_ensure_tables` will log a warning and continue.

No destructive rollback step needed. No data is lost.

---

## 8. Open questions for the owner

1. **Source-family scope for dedupe.** The current proposal dedupes within `(source_family, day)`. Should it also fire across families (same headline from DuckDuckGo and Brave)? That would further reduce noise but loses cross-source confirmation as a signal.

2. **Backfill the flag for existing DuckDuckGo rows.** All ~481 existing DuckDuckGo rows will have `pub_time_is_fetch_time = 0` after migration (the column default), which is misleading — those rows were always substituted. A one-time `UPDATE news SET pub_time_is_fetch_time = 1 WHERE json_extract(metadata, '$.source_api') = 'duckduckgo'` would fix them, but violates the no-rewrite-of-history rule. Decision needed: apply the one-time UPDATE, or accept that the read-time 60-second heuristic is the proxy for old rows?

3. **Dedupe window size.** One calendar UTC day is proposed. A tighter window (e.g., 6 hours) reduces false positives when a story evolves intra-day; a wider window (e.g., 3 days) catches duplicates across missed collector runs. Given the batchy collection history, 1 day is a reasonable starting point.

4. **Near-duplicate normalisation strength.** The current `_normalized_headline_key` (core/news_pipeline.py line 181) strips trailing `" - Publisher"` and punctuation. A heavier normalisation (stop-word removal, stemming) would catch more paraphrases but risks false positives and requires an additional dependency. Is the simple rule sufficient for now?

5. **Priority relative to other CONTINUATION_PLAN items.** This slice addresses data quality, not strategy edge. If the H3 news-sentiment hypothesis test (target: 300 qualifying days) is the higher-priority gate, this slice can be deferred without blocking that test.
