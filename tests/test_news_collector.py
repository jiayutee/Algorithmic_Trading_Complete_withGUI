"""News collector: coverage math against the H3 threshold, and fault-tolerant collection."""
import os
from datetime import date, datetime, timedelta, timezone

import pytest

import core.news_collector as nc
from core.news_sources import NewsItem
from core.news_store import NewsStore


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    monkeypatch.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))          # NewsStore reads migrations/ relative to CWD


def _store(tmp_path):
    return NewsStore(str(tmp_path / "news.sqlite3")), str(tmp_path / "news.sqlite3")


def _add(store, symbol, day, n, tag=None):
    items = [NewsItem(datetime_utc=datetime(day.year, day.month, day.day, 9, i, tzinfo=timezone.utc), source="t",
                      headline=f"{symbol} headline {day} #{i}", url=f"https://x.com/{symbol}/{day}/{i}",
                      tickers=[tag or symbol]) for i in range(n)]
    store.add_items(items)


def test_no_database_means_zero_coverage_not_a_crash(tmp_path):
    cov = nc.coverage(["BTCUSDT"], db_path=str(tmp_path / "missing.sqlite3"))
    assert cov["BTCUSDT"]["items"] == 0 and cov["BTCUSDT"]["pct_of_target"] == 0.0 and cov["BTCUSDT"]["eta_days"] is None


def test_days_qualify_only_with_enough_headlines_and_only_for_that_symbol(tmp_path):
    store, path = _store(tmp_path)
    d0 = date(2026, 1, 1)
    for i in range(10):                       # 10 days: alternating 3 headlines (qualifies) and 2 (does not)
        _add(store, "BTCUSDT", d0 + timedelta(days=i), 3 if i % 2 == 0 else 2)
    _add(store, "ETHUSDT", d0, 9)
    store.close()
    c = nc.coverage(["BTCUSDT", "ETHUSDT", "SOLUSDT"], db_path=path)
    assert c["BTCUSDT"]["days"] == 10 and c["BTCUSDT"]["qualifying_days"] == 5 and c["BTCUSDT"]["items"] == 25
    assert c["BTCUSDT"]["first"] == "2026-01-01" and c["BTCUSDT"]["last"] == "2026-01-10"
    assert c["ETHUSDT"]["qualifying_days"] == 1 and c["SOLUSDT"]["items"] == 0
    assert c["BTCUSDT"]["pct_of_target"] == pytest.approx(100 * 5 / 300, abs=0.05)


def test_eta_extrapolates_the_observed_rate_and_reads_reached(tmp_path):
    store, path = _store(tmp_path)
    d0 = date(2026, 1, 1)
    for i in range(0, 20, 2):                                 # qualifying on every other day over a 19-day span
        _add(store, "BTCUSDT", d0 + timedelta(days=i), 3)
    store.close()
    c = nc.coverage(["BTCUSDT"], db_path=path)["BTCUSDT"]
    assert c["qualifying_days"] == 10 and c["days_per_week_rate"] == pytest.approx(10 / 19 * 7, abs=0.02)
    assert c["eta_days"] == round((300 - 10) / (10 / 19))
    assert nc.coverage(["BTCUSDT"], db_path=path, target_days=10)["BTCUSDT"]["eta_days"] == 0
    assert "reached" in nc.format_status(nc.coverage(["BTCUSDT"], db_path=path, target_days=10), target_days=10)


def test_collect_calls_the_pipeline_per_symbol_and_survives_a_failure():
    class Pipe:
        def fetch_news_items(self, sym, limit=25):
            if sym == "ETHUSDT":
                raise RuntimeError("all sources down")
            return [1, 2, 3]
    lines = []
    res = nc.collect(["BTCUSDT", "ETHUSDT", "SOLUSDT"], pipeline=Pipe(), progress=lines.append)
    assert res["BTCUSDT"]["items"] == 3 and res["SOLUSDT"]["items"] == 3
    assert res["ETHUSDT"]["error"] == "all sources down" and any("ERROR" in l for l in lines)


def test_status_lists_ready_symbols_and_the_pre_registered_threshold(tmp_path):
    txt = nc.format_status(nc.coverage(["BTCUSDT"], db_path=str(tmp_path / "none.sqlite3")))
    assert "300 days with >= 3 headlines" in txt and "none yet" in txt
    assert (nc.TARGET_DAYS, nc.MIN_ITEMS_PER_DAY) == (300, 3)                # matches docs/PHASE_6_5_PREREGISTRATION.md


def test_cli_status_runs_against_the_default_store(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(nc, "DEFAULT_DB", str(tmp_path / "nothing.sqlite3"))
    assert nc._cli(["status", "--symbols", "BTCUSDT,ETHUSDT"]) == 0
    assert "BTCUSDT" in capsys.readouterr().out


def test_collect_pauses_between_symbols_but_not_before_the_first(monkeypatch):
    sleeps = []
    monkeypatch.setattr(nc.time, "sleep", sleeps.append)
    class Pipe:
        def fetch_news_items(self, sym, limit=25):
            return []
    nc.collect(["A", "B", "C"], pipeline=Pipe(), progress=lambda m: None, pause_seconds=1.5)
    assert sleeps == [1.5, 1.5]
    sleeps.clear()
    nc.collect(["A", "B"], pipeline=Pipe(), progress=lambda m: None, pause_seconds=0)
    assert sleeps == []
