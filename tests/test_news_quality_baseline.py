"""Offline tests for scripts/news_quality_baseline.py (temp SQLite file, no network)."""
import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "news_quality_baseline", Path(__file__).resolve().parents[1] / "scripts" / "news_quality_baseline.py")
nqb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(nqb)

ROWS = [
    # (datetime_utc, created_at, source, headline, tickers, metadata)
    ("2026-09-01T10:00:00+00:00", "2026-09-01 12:00:00", "openbb:CCN", "Bitcoin rises - CCN", '["BTCUSDT"]', None),
    ("2026-09-01T11:00:00+00:00", "2026-09-01 12:00:00", "openbb:TheStreet", "Bitcoin Rises!", '["BTCUSDT"]', None),
    ("2026-09-01T11:00:00+00:00", "2026-09-01 12:00:00", "openbb:TheStreet", "Bitcoin Rises!", '["BTCUSDT"]', None),
    ("2026-09-04T09:00:10+00:00", "2026-09-04 09:00:12", "duckduckgo", "Apple news", '["AAPL"]',
     '{"source_api": "duckduckgo"}'),
    ("2026-09-04T00:00:00+00:00", "2026-09-04 06:00:00", "gdelt.com", "Market wrap", "[]", '{"source_api": "gdelt"}'),
    ("2026-09-04T08:00:00+00:00", "2026-09-04 07:00:00", "rss feed", "Future dated", "[]", '{"source_api": "rss"}'),
]


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "news.sqlite3"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE news (id INTEGER PRIMARY KEY, datetime_utc TEXT NOT NULL, source TEXT, headline TEXT,"
                " url TEXT UNIQUE, tickers TEXT, metadata TEXT, created_at TEXT)")
    for i, (p_, c, s, h, t, m) in enumerate(ROWS):
        con.execute("INSERT INTO news (datetime_utc, created_at, source, headline, url, tickers, metadata)"
                    " VALUES (?,?,?,?,?,?,?)", (p_, c, s, h, f"https://x/{i}", t, m))
    con.commit()
    con.close()
    return p


def test_normalise_title_strips_publisher_and_punctuation():
    assert nqb.normalise_title("Bitcoin rises - CCN") == nqb.normalise_title("Bitcoin Rises!") == "bitcoin rises"


def test_source_family_and_symbols():
    assert nqb.source_family("openbb:CCN", None) == "openbb"
    assert nqb.source_family("x", '{"source_api": "brave"}') == "brave"
    assert nqb.source_family("x", "not json") == "unknown"
    assert nqb.symbols("[]") == ["(none)"] and nqb.symbols('["AAPL"]') == ["AAPL"]


def test_report_metrics(db):
    rep = nqb.build_report(nqb.load_rows(str(db)))
    o = rep["overall"]
    assert o["items"] == 6
    assert o["exact_duplicate_rate"] == round(1 / 6, 4)       # one identical headline
    assert o["near_duplicate_rate"] == round(2 / 6, 4)        # plus the " - CCN" variant
    assert o["fetch_time_substituted_share"] == round(1 / 6, 4)  # duckduckgo row, 2 s apart
    assert o["date_only_midnight_share"] == round(1 / 6, 4)
    assert o["negative_lag_count"] == 1
    assert o["lag_hours_n"] == 4 and o["lag_hours_median"] == 1.5  # lags 2,1,1,6 h
    assert rep["coverage"] == {"first_ingest_day": "2026-09-01", "last_ingest_day": "2026-09-04",
                               "span_days": 4, "ingest_days": 2, "longest_gap_days": 2}
    assert list(rep["by_source_family"])[0] == "openbb"
    assert rep["by_symbol"]["BTCUSDT"]["items"] == 3 and rep["by_symbol"]["(none)"]["items"] == 2


def test_read_only_and_cli(db, capsys):
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    assert nqb.main(["--db", str(db), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["overall"]["items"] == 6
    assert nqb.main(["--db", str(db)]) == 0
    assert "## By source family" in capsys.readouterr().out
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before
    assert sorted(x.name for x in db.parent.iterdir()) == ["news.sqlite3"]  # no journal/WAL created
