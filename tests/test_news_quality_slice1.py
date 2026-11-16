"""News quality slice 1 (docs/proposals/news_quality_fix_slice.md): fetch-time flag, same-story dedupe, ingest gaps."""
import sqlite3
from datetime import date, datetime, timezone
from unittest.mock import MagicMock

import requests

from core.news_collector import format_ingest, ingest_summary
from core.news_sources import BraveSearchSource, DuckDuckGoSource, NewsItem, has_usable_datetime
from core.news_store import NewsStore
from test_news_failure_classification import _BRAVE_NEWS_BODY, _DDG_HTML_OK, _make_response, _session_returns


def _item(headline, url, *, day=24, hour=12, api=None, fetch_time=False):
    return NewsItem(datetime_utc=datetime(2026, 5, day, hour, tzinfo=timezone.utc), source="unittest", headline=headline,
                    url=url, metadata={"source_api": api} if api else {}, pub_time_is_fetch_time=fetch_time)


def _rows(path, sql="SELECT headline, pub_time_is_fetch_time FROM news ORDER BY id"):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


# ------------------------------------------------------------- 1. publication-time flag
def test_flag_is_stored_true_and_false(tmp_path):
    db = str(tmp_path / "n.sqlite3")
    NewsStore(db).add_items([_item("A story", "https://x/a", fetch_time=True), _item("B story", "https://x/b")])
    assert _rows(db) == [("A story", 1), ("B story", 0)]


def test_old_schema_gains_the_column_and_existing_rows_read_zero(tmp_path):
    db = str(tmp_path / "old.sqlite3")
    conn = sqlite3.connect(db)
    conn.executescript("""CREATE TABLE news (id INTEGER PRIMARY KEY AUTOINCREMENT, datetime_utc TEXT NOT NULL, source TEXT,
        headline TEXT, headline_hash TEXT, url TEXT UNIQUE, summary TEXT, content TEXT, language TEXT, tickers TEXT, entities TEXT,
        event_type TEXT, sentiment TEXT, impact_score REAL, source_reliability REAL, metadata TEXT,
        created_at TEXT DEFAULT (datetime('now')));
        INSERT INTO news (datetime_utc, headline, url) VALUES ('2026-05-01T00:00:00+00:00', 'old row', 'https://x/old');""")
    conn.commit(); conn.close()
    ns = NewsStore(db)
    ns.add_items([_item("new row", "https://x/new", fetch_time=True)])
    assert _rows(db) == [("old row", 0), ("new row", 1)]            # history is not rewritten
    NewsStore(db)                                                     # re-opening an already-migrated store is harmless


def test_has_usable_datetime():
    assert not has_usable_datetime(None) and not has_usable_datetime("") and not has_usable_datetime("not a date")
    assert has_usable_datetime("2026-05-24T12:00:00Z") and has_usable_datetime(datetime(2026, 5, 24, tzinfo=timezone.utc))


def test_duckduckgo_items_are_always_marked_fetch_time():
    resp = MagicMock(spec=requests.Response)
    resp.status_code, resp.text = 200, _DDG_HTML_OK
    resp.raise_for_status.return_value = None
    items, _ = DuckDuckGoSource(session=_session_returns(resp)).fetch_classified("Bitcoin")
    assert items and all(i.pub_time_is_fetch_time for i in items)


def test_brave_flag_follows_whether_the_provider_gave_a_date():
    undated = BraveSearchSource(api_key="k", session=_session_returns(_make_response(200, _BRAVE_NEWS_BODY)))
    items, _ = undated.fetch_classified("Bitcoin")
    has_date = bool(_BRAVE_NEWS_BODY["results"][0].get("age") or _BRAVE_NEWS_BODY["results"][0].get("published")
                    or _BRAVE_NEWS_BODY["results"][0].get("date") or _BRAVE_NEWS_BODY["results"][0].get("page_age"))
    assert items[0].pub_time_is_fetch_time is (not has_date)
    body = {"results": [{**_BRAVE_NEWS_BODY["results"][0], "published": "2026-05-24T09:00:00Z"}]}
    dated, _ = BraveSearchSource(api_key="k", session=_session_returns(_make_response(200, body))).fetch_classified("Bitcoin")
    assert dated[0].pub_time_is_fetch_time is False


# ------------------------------------------------------------- 2. same-story dedupe
def test_same_headline_same_family_same_day_different_url_is_skipped(tmp_path):
    ns = NewsStore(str(tmp_path / "d.sqlite3"))
    assert ns.add_items([_item("Story", "https://x/1", api="duckduckgo"), _item("Story", "https://y/2", api="duckduckgo")]) == 1


def test_same_headline_other_family_is_kept(tmp_path):
    ns = NewsStore(str(tmp_path / "d.sqlite3"))
    assert ns.add_items([_item("Story", "https://x/1", api="duckduckgo"), _item("Story", "https://y/2", api="brave")]) == 2


def test_same_headline_next_day_is_kept(tmp_path):
    ns = NewsStore(str(tmp_path / "d.sqlite3"))
    assert ns.add_items([_item("Story", "https://x/1", day=24, api="brave"), _item("Story", "https://y/2", day=25, api="brave")]) == 2


def test_dedupe_works_across_separate_collector_runs(tmp_path):
    db = str(tmp_path / "d.sqlite3")
    assert NewsStore(db).add_items([_item("Story", "https://x/1", api="brave")]) == 1
    assert NewsStore(db).add_items([_item("Story", "https://z/9", api="brave", hour=18)]) == 0


# ------------------------------------------------------------- 3. ingest gaps
def _db_with_created(tmp_path, days):
    db = str(tmp_path / "g.sqlite3")
    NewsStore(db)
    conn = sqlite3.connect(db)
    for n, d in enumerate(days):
        conn.execute("INSERT INTO news (datetime_utc, headline, url, created_at) VALUES (?, ?, ?, ?)",
                     ("2026-05-01T00:00:00+00:00", f"h{n}", f"https://x/{n}", f"{d} 07:30:00"))
    conn.commit(); conn.close()
    return db


def test_ingest_summary_reports_last_day_staleness_and_longest_gap(tmp_path):
    db = _db_with_created(tmp_path, ["2026-09-01", "2026-09-02", "2026-09-06", "2026-09-07"])
    s = ingest_summary(db, today=date(2026, 9, 10))
    assert s["last_ingest"] == "2026-09-07" and s["days_since"] == 3
    assert s["longest_gap_days"] == 3 and s["gap_between"] == ("2026-09-02", "2026-09-06")
    text = format_ingest(s)
    assert "STALE" in text and "longest gap 3 day(s)" in text


def test_ingest_summary_is_quiet_when_current_and_safe_when_empty(tmp_path):
    db = _db_with_created(tmp_path, ["2026-09-09", "2026-09-10"])
    assert "STALE" not in format_ingest(ingest_summary(db, today=date(2026, 9, 10)))
    assert ingest_summary(str(tmp_path / "missing.sqlite3"))["last_ingest"] is None
    assert format_ingest(ingest_summary(str(tmp_path / "missing.sqlite3"))) == "Ingest: nothing stored yet."
    assert not (tmp_path / "missing.sqlite3").exists()                # read-only: never creates a database
