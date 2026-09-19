"""Source diagnostics must distinguish delivery from fallback and never score/store news."""
import json
import sys
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from core.news_health import SourceHealthRegistry
from core.news_pipeline import NewsPipeline
from core.news_sources import BaseNewsSource, NewsItem
from scripts.smoke_news import main, positive_seconds, run_report


class Source(BaseNewsSource):
    def __init__(self, name, items=(), error=None, wait=None, query_style="text"):
        self.name, self.items, self.error, self.wait = name, list(items), error, wait
        self.query_style, self.queries = query_style, []

    def fetch(self, query, limit=5):
        self.queries.append(query)
        if self.wait:
            self.wait.wait(2)
        if self.error:
            raise RuntimeError(self.error)
        return self.items


def story():
    return NewsItem(datetime_utc=datetime.now(timezone.utc), source="publisher", headline="Bitcoin news")


def test_probe_uses_app_routing_and_reports_partial_delivery_without_scoring_or_storing(monkeypatch):
    good = Source("brave", [story()])
    ticker = Source("openbb_news", query_style="ticker")
    broken = Source("broken", error="https://example.com/?api_key=DO_NOT_EXPORT")
    pipe = NewsPipeline(sources=[good, ticker, broken], health=SourceHealthRegistry())
    monkeypatch.setattr(pipe, "_open_store", lambda: pytest.fail("must not write news store"))
    monkeypatch.setattr(pipe.sentiment_analyzer, "analyze_many", lambda _: pytest.fail("must not score"))
    report = run_report(pipe, ["BTCUSDT"])
    probe = report["probes"][0]
    rows = {r["name"]: r for r in probe["sources"]}
    assert report["usable"] and probe["status"] == "degraded"
    assert good.queries == ["Bitcoin"] and ticker.queries == ["BTCUSDT"]
    assert rows["brave"]["status"] == "ok" and rows["brave"]["last_success_at"]
    assert rows["openbb_news"]["status"] == "empty" and not rows["openbb_news"]["last_success_at"]
    assert rows["broken"]["status"] == "error"
    assert "DO_NOT_EXPORT" not in json.dumps(report)


def test_timeout_and_cooldown_are_not_reported_as_success():
    release = threading.Event()
    src = Source("hung", [story()], wait=release)
    pipe = NewsPipeline(sources=[src], deadline_seconds=0.05,
                        health=SourceHealthRegistry(threshold=1))
    try:
        first = run_report(pipe, ["BTCUSDT"])
        row = first["probes"][0]["sources"][0]
        assert not first["usable"] and row["last_result"] == "timeout"
        assert row["status"] == "cooldown" and row["retry_in_seconds"] > 0
        assert first["elapsed_seconds"] < 1.0
        second = run_report(pipe, ["BTCUSDT"])
        assert not second["usable"] and len(src.queries) == 1
    finally:
        release.set()


def test_recovery_clears_error_but_preserves_last_delivery_when_response_is_empty():
    health = SourceHealthRegistry()
    health.record("a", 0, 0.2, error="failure")
    health.record("a", 3, 0.1)
    success = health.snapshot()["a"]
    assert success["last_reason"] == "" and success["last_status"] == "ok"
    health.record("a", 0, 0.1)
    empty = health.snapshot()["a"]
    assert empty["last_status"] == "empty" and empty["last_success_at"] == success["last_success_at"]
    health.record("a", 0, 5.0)
    assert health.snapshot()["a"]["last_status"] == "slow_empty"


@pytest.mark.parametrize("key", ["BRAVE_SEARCH_API_KEY", "BRAVE_API_KEY"])
def test_configuration_matches_app_key_aliases_and_reports_missing_optional_sources(monkeypatch, key):
    for name in ("BRAVE_SEARCH_API_KEY", "BRAVE_API_KEY", "NEWSAPI_API_KEY", "RSS_FEEDS", "RSS_FEED", "EVENTREGISTRY_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(key, "secret-never-exported")
    monkeypatch.setitem(sys.modules, "openbb", None)
    pipe = NewsPipeline.from_env()
    rows = {r["name"]: r for r in pipe.source_status()}
    assert rows["brave"]["enabled"] and rows["brave"]["status"] == "not_checked"
    assert not rows["newsapi"]["enabled"] and not rows["openbb_news"]["enabled"]
    assert all("mcp" not in r["name"] for r in rows.values())
    assert "secret-never-exported" not in json.dumps(rows)


def test_cli_writes_artifact_and_reports_unavailable_without_network(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(NewsPipeline, "from_env", lambda: NewsPipeline(sources=[]))
    output = tmp_path / "report.json"
    assert main(["--symbols", "BTCUSDT", "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report == json.loads(capsys.readouterr().out)
    assert report["probes"][0]["status"] == "unavailable"


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1"])
def test_invalid_deadlines_cannot_disable_the_fetch_budget(value):
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        positive_seconds(value)
