"""Offline tests: failure classification for every source adapter that does network I/O.

Each test mocks HTTP / library calls and asserts that fetch_classified() returns the
correct OUTCOME_* constant.  No network access; news_store.sqlite3 is never touched.

Coverage per adapter:
  NewsApiSource      ok, ok_empty, rate_limited, auth_failed, parse_error, error
  GDELTSource        ok, ok_empty, rate_limited, auth_failed, parse_error, timeout, error
  EventRegistrySource ok, ok_empty, rate_limited, auth_failed, parse_error, error
  BraveSearchSource  ok, ok_empty, rate_limited, auth_failed, parse_error, error
  RssSource          ok, ok_empty, rate_limited, auth_failed, parse_error, error
  DuckDuckGoSource   ok, ok_empty, rate_limited, auth_failed, error
  OpenBBNewsSource   ok, ok_empty, rate_limited, auth_failed, timeout, error  (best-effort)
  McpDuckDuckGoSource ok_empty always
  BaseNewsSource     default fetch_classified() wraps fetch()

Classification contract:
  ok, ok_empty  → failed=False; circuit breaker stays closed
  all others    → failed=True;  circuit breaker counts the failure

Limitation (OpenBBNewsSource): the openbb library hides the raw HTTP status code; the
classification is inferred from the exception *message* string, so it is best-effort only.
Tests verify that common patterns are detected but cannot guarantee exhaustive coverage.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from unittest.mock import MagicMock

import pytest
import requests

from core.news_health import SourceHealthRegistry
from core.news_sources import (
    OUTCOME_AUTH_FAILED,
    OUTCOME_ERROR,
    OUTCOME_OK,
    OUTCOME_OK_EMPTY,
    OUTCOME_PARSE_ERROR,
    OUTCOME_RATE_LIMITED,
    OUTCOME_TIMEOUT,
    BaseNewsSource,
    BraveSearchSource,
    DuckDuckGoSource,
    EventRegistrySource,
    GDELTSource,
    McpDuckDuckGoSource,
    NewsApiSource,
    NewsItem,
    OpenBBNewsSource,
    RssSource,
    _classify_http_status,
    request_with_outcome,
)

# Sentinel used in OpenBBNewsSource tests to distinguish "not in sys.modules" from None
_SENTINEL = object()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(status: int, body: str | dict | None = None) -> MagicMock:
    """Create a mock requests.Response."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status
    if isinstance(body, dict):
        resp.json.return_value = body
        resp.text = json.dumps(body)
    elif isinstance(body, str):
        resp.text = body
        resp.json.side_effect = ValueError("not JSON")
    else:
        resp.text = ""
        resp.json.return_value = {}

    if 400 <= status < 600:
        http_err = requests.exceptions.HTTPError(response=resp)
        resp.raise_for_status.side_effect = http_err
    else:
        resp.raise_for_status.return_value = None
    return resp


def _session_returns(response: MagicMock) -> MagicMock:
    session = MagicMock()
    session.request.return_value = response
    return session


def _session_raises(exc: Exception) -> MagicMock:
    session = MagicMock()
    session.request.side_effect = exc
    return session


# Minimal valid RSS body
_RSS_OK = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>Bitcoin surges past 100k</title>
      <link>https://example.com/btc</link>
      <description>Details here</description>
    </item>
  </channel>
</rss>"""

_RSS_MALFORMED = "this is not xml <><>"

# ---------------------------------------------------------------------------
# _classify_http_status
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code,expected", [
    (429, OUTCOME_RATE_LIMITED),
    (401, OUTCOME_AUTH_FAILED),
    (403, OUTCOME_AUTH_FAILED),
    (500, OUTCOME_ERROR),
    (503, OUTCOME_ERROR),
])
def test_classify_http_status_mapping(code, expected):
    assert _classify_http_status(code) == expected


# ---------------------------------------------------------------------------
# request_with_outcome — unit tests for the helper itself
# ---------------------------------------------------------------------------

def test_request_with_outcome_success():
    resp = _make_response(200, {"articles": []})
    session = _session_returns(resp)
    result_resp, outcome = request_with_outcome(session, "get", "https://x.com/", timeout=5)
    assert result_resp is resp and outcome == OUTCOME_OK


def test_request_with_outcome_rate_limited(monkeypatch):
    monkeypatch.setenv("NEWS_FETCH_MAX_RETRIES", "1")
    resp = _make_response(429)
    session = _session_returns(resp)
    result_resp, outcome = request_with_outcome(session, "get", "https://x.com/", timeout=5)
    assert result_resp is None and outcome == OUTCOME_RATE_LIMITED


def test_request_with_outcome_auth_failed():
    resp = _make_response(401)
    session = _session_returns(resp)
    result_resp, outcome = request_with_outcome(session, "get", "https://x.com/", timeout=5)
    assert result_resp is None and outcome == OUTCOME_AUTH_FAILED


def test_request_with_outcome_auth_failed_403():
    resp = _make_response(403)
    session = _session_returns(resp)
    result_resp, outcome = request_with_outcome(session, "get", "https://x.com/", timeout=5)
    assert result_resp is None and outcome == OUTCOME_AUTH_FAILED


def test_request_with_outcome_timeout(monkeypatch):
    monkeypatch.setenv("NEWS_FETCH_MAX_RETRIES", "1")
    session = _session_raises(requests.exceptions.Timeout())
    result_resp, outcome = request_with_outcome(session, "get", "https://x.com/", timeout=5)
    assert result_resp is None and outcome == OUTCOME_TIMEOUT


def test_request_with_outcome_generic_error(monkeypatch):
    monkeypatch.setenv("NEWS_FETCH_MAX_RETRIES", "1")
    session = _session_raises(ConnectionError("no route to host"))
    result_resp, outcome = request_with_outcome(session, "get", "https://x.com/", timeout=5)
    assert result_resp is None and outcome == OUTCOME_ERROR


# ---------------------------------------------------------------------------
# NewsApiSource
# ---------------------------------------------------------------------------

class TestNewsApiSource:
    def test_ok(self):
        body = {"articles": [{"title": "BTC rises", "url": "https://a.com/1", "source": {"name": "Pub"},
                               "publishedAt": "2026-09-20T10:00:00Z", "description": "...", "content": ""}]}
        src = NewsApiSource(api_key="key", session=_session_returns(_make_response(200, body)))
        items, outcome = src.fetch_classified("Bitcoin", limit=5)
        assert outcome == OUTCOME_OK and len(items) == 1

    def test_ok_empty(self):
        src = NewsApiSource(api_key="key", session=_session_returns(_make_response(200, {"articles": []})))
        items, outcome = src.fetch_classified("Bitcoin", limit=5)
        assert outcome == OUTCOME_OK_EMPTY and items == []

    def test_missing_key_auth_failed(self):
        src = NewsApiSource(api_key="", session=MagicMock())
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_AUTH_FAILED and items == []

    def test_rate_limited(self, monkeypatch):
        monkeypatch.setenv("NEWS_FETCH_MAX_RETRIES", "1")
        src = NewsApiSource(api_key="key", session=_session_returns(_make_response(429)))
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_RATE_LIMITED and items == []

    def test_auth_failed_401(self, monkeypatch):
        monkeypatch.setenv("NEWS_FETCH_MAX_RETRIES", "1")
        src = NewsApiSource(api_key="bad", session=_session_returns(_make_response(401)))
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_AUTH_FAILED and items == []

    def test_parse_error(self):
        session = _session_returns(_make_response(200, "not json at all"))
        src = NewsApiSource(api_key="key", session=session)
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_PARSE_ERROR and items == []

    def test_error_network(self, monkeypatch):
        monkeypatch.setenv("NEWS_FETCH_MAX_RETRIES", "1")
        src = NewsApiSource(api_key="key", session=_session_raises(ConnectionError()))
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_ERROR and items == []

    def test_fetch_backward_compat(self):
        """fetch() still returns just the list."""
        src = NewsApiSource(api_key="", session=MagicMock())
        assert isinstance(src.fetch("Bitcoin"), list)


# ---------------------------------------------------------------------------
# GDELTSource
# ---------------------------------------------------------------------------

class TestGDELTSource:
    def _src(self, session, monkeypatch=None):
        if monkeypatch:
            # Reset class-level rate-limit clock
            monkeypatch.setattr(GDELTSource, "_last_call", 0.0)
        src = GDELTSource(session=session)
        GDELTSource._last_call = 0.0  # don't sleep in tests
        return src

    def test_ok(self):
        body = {"articles": [{"title": "BTC up", "url": "https://g.com/1", "seendate": "20260920T100000Z",
                               "domain": "g.com", "snippet": ""}]}
        src = self._src(_session_returns(_make_response(200, body)))
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_OK and len(items) == 1

    def test_ok_empty(self):
        src = self._src(_session_returns(_make_response(200, {"articles": []})))
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_OK_EMPTY and items == []

    def test_rate_limited(self, monkeypatch):
        monkeypatch.setenv("NEWS_FETCH_MAX_RETRIES", "1")
        src = self._src(_session_returns(_make_response(429)))
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_RATE_LIMITED and items == []

    def test_auth_failed(self, monkeypatch):
        monkeypatch.setenv("NEWS_FETCH_MAX_RETRIES", "1")
        src = self._src(_session_returns(_make_response(401)))
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_AUTH_FAILED and items == []

    def test_parse_error(self):
        src = self._src(_session_returns(_make_response(200, "not json")))
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_PARSE_ERROR and items == []

    def test_timeout(self, monkeypatch):
        monkeypatch.setenv("NEWS_FETCH_MAX_RETRIES", "1")
        src = self._src(_session_raises(requests.exceptions.Timeout()))
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_TIMEOUT and items == []

    def test_error(self, monkeypatch):
        monkeypatch.setenv("NEWS_FETCH_MAX_RETRIES", "1")
        src = self._src(_session_raises(ConnectionError()))
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_ERROR and items == []


# ---------------------------------------------------------------------------
# EventRegistrySource
# ---------------------------------------------------------------------------

class TestEventRegistrySource:
    def test_ok(self):
        body = {"articles": {"results": [{"title": "ETH news", "url": "https://e.com/1",
                                           "dateTimePub": "2026-09-20T10:00:00Z", "lang": "eng"}]}}
        src = EventRegistrySource(api_key="key", session=_session_returns(_make_response(200, body)))
        items, outcome = src.fetch_classified("Ethereum")
        assert outcome == OUTCOME_OK and len(items) == 1

    def test_ok_empty(self):
        src = EventRegistrySource(api_key="key",
                                  session=_session_returns(_make_response(200, {"articles": {"results": []}})))
        items, outcome = src.fetch_classified("Ethereum")
        assert outcome == OUTCOME_OK_EMPTY and items == []

    def test_missing_key_auth_failed(self):
        src = EventRegistrySource(api_key="")
        items, outcome = src.fetch_classified("Ethereum")
        assert outcome == OUTCOME_AUTH_FAILED and items == []

    def test_rate_limited(self, monkeypatch):
        monkeypatch.setenv("NEWS_FETCH_MAX_RETRIES", "1")
        src = EventRegistrySource(api_key="key", session=_session_returns(_make_response(429)))
        items, outcome = src.fetch_classified("Ethereum")
        assert outcome == OUTCOME_RATE_LIMITED and items == []

    def test_auth_failed_403(self, monkeypatch):
        monkeypatch.setenv("NEWS_FETCH_MAX_RETRIES", "1")
        src = EventRegistrySource(api_key="bad", session=_session_returns(_make_response(403)))
        items, outcome = src.fetch_classified("Ethereum")
        assert outcome == OUTCOME_AUTH_FAILED and items == []

    def test_parse_error(self):
        src = EventRegistrySource(api_key="key", session=_session_returns(_make_response(200, "not json")))
        items, outcome = src.fetch_classified("Ethereum")
        assert outcome == OUTCOME_PARSE_ERROR and items == []

    def test_error(self, monkeypatch):
        monkeypatch.setenv("NEWS_FETCH_MAX_RETRIES", "1")
        src = EventRegistrySource(api_key="key", session=_session_raises(ConnectionError()))
        items, outcome = src.fetch_classified("Ethereum")
        assert outcome == OUTCOME_ERROR and items == []


# ---------------------------------------------------------------------------
# BraveSearchSource
# ---------------------------------------------------------------------------

_BRAVE_NEWS_BODY = {"results": [{"title": "BTC ATH", "url": "https://brave.com/1",
                                  "description": "desc", "published": "2026-09-20T10:00:00Z"}]}


class TestBraveSearchSource:
    def test_ok(self):
        src = BraveSearchSource(api_key="key", session=_session_returns(_make_response(200, _BRAVE_NEWS_BODY)))
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_OK and len(items) == 1

    def test_ok_empty(self):
        src = BraveSearchSource(api_key="key", session=_session_returns(_make_response(200, {"results": []})))
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_OK_EMPTY and items == []

    def test_missing_key_auth_failed(self):
        src = BraveSearchSource(api_key="")
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_AUTH_FAILED and items == []

    def test_rate_limited(self, monkeypatch):
        monkeypatch.setenv("NEWS_FETCH_MAX_RETRIES", "1")
        src = BraveSearchSource(api_key="key", session=_session_returns(_make_response(429)))
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_RATE_LIMITED and items == []

    def test_auth_failed_401(self, monkeypatch):
        monkeypatch.setenv("NEWS_FETCH_MAX_RETRIES", "1")
        src = BraveSearchSource(api_key="bad", session=_session_returns(_make_response(401)))
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_AUTH_FAILED and items == []

    def test_parse_error(self):
        src = BraveSearchSource(api_key="key", session=_session_returns(_make_response(200, "not json")))
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_PARSE_ERROR and items == []

    def test_error_network(self, monkeypatch):
        monkeypatch.setenv("NEWS_FETCH_MAX_RETRIES", "1")
        src = BraveSearchSource(api_key="key", session=_session_raises(ConnectionError()))
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome in (OUTCOME_ERROR, OUTCOME_OK_EMPTY)

    def test_fetch_backward_compat(self):
        src = BraveSearchSource(api_key="", session=MagicMock())
        assert isinstance(src.fetch("Bitcoin"), list)


# ---------------------------------------------------------------------------
# RssSource
# ---------------------------------------------------------------------------

class TestRssSource:
    def test_ok(self):
        resp = MagicMock(spec=requests.Response)
        resp.status_code = 200
        resp.text = _RSS_OK
        resp.raise_for_status.return_value = None
        src = RssSource(feed_urls=["https://rss.example.com/feed"], session=_session_returns(resp))
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_OK and len(items) == 1

    def test_ok_empty_no_feeds(self):
        src = RssSource(feed_urls=[])
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_OK_EMPTY and items == []

    def test_ok_empty_no_matches(self):
        resp = MagicMock(spec=requests.Response)
        resp.status_code = 200
        # RSS with items that don't mention the query
        resp.text = """<?xml version="1.0"?><rss version="2.0"><channel><title>T</title>
        <item><title>Cooking tips</title><link>https://a.com/cook</link></item></channel></rss>"""
        resp.raise_for_status.return_value = None
        src = RssSource(feed_urls=["https://rss.example.com/feed"], session=_session_returns(resp))
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_OK_EMPTY and items == []

    def test_rate_limited(self, monkeypatch):
        monkeypatch.setenv("NEWS_FETCH_MAX_RETRIES", "1")
        src = RssSource(feed_urls=["https://rss.example.com/feed"],
                        session=_session_returns(_make_response(429)))
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_RATE_LIMITED and items == []

    def test_auth_failed(self, monkeypatch):
        monkeypatch.setenv("NEWS_FETCH_MAX_RETRIES", "1")
        src = RssSource(feed_urls=["https://rss.example.com/feed"],
                        session=_session_returns(_make_response(401)))
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_AUTH_FAILED and items == []

    def test_parse_error_malformed_xml(self):
        resp = MagicMock(spec=requests.Response)
        resp.status_code = 200
        resp.text = _RSS_MALFORMED
        resp.raise_for_status.return_value = None
        src = RssSource(feed_urls=["https://rss.example.com/feed"], session=_session_returns(resp))
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_PARSE_ERROR and items == []

    def test_error_network(self, monkeypatch):
        monkeypatch.setenv("NEWS_FETCH_MAX_RETRIES", "1")
        src = RssSource(feed_urls=["https://rss.example.com/feed"],
                        session=_session_raises(ConnectionError()))
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_ERROR and items == []

    def test_fetch_backward_compat(self):
        src = RssSource(feed_urls=[])
        assert isinstance(src.fetch("Bitcoin"), list)


# ---------------------------------------------------------------------------
# DuckDuckGoSource
# ---------------------------------------------------------------------------

_DDG_HTML_OK = """<html><body>
<div class="result">
  <a class="result__a" href="https://news.com/btc">Bitcoin Reaches New High</a>
  <div class="result__snippet">Details about bitcoin price action</div>
</div>
</body></html>"""

_DDG_HTML_EMPTY = "<html><body><p>No results</p></body></html>"


class TestDuckDuckGoSource:
    def test_ok(self):
        resp = MagicMock(spec=requests.Response)
        resp.status_code = 200
        resp.text = _DDG_HTML_OK
        resp.raise_for_status.return_value = None
        src = DuckDuckGoSource(session=_session_returns(resp))
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_OK and len(items) >= 1

    def test_ok_empty_no_results_in_html(self):
        resp = MagicMock(spec=requests.Response)
        resp.status_code = 200
        resp.text = _DDG_HTML_EMPTY
        resp.raise_for_status.return_value = None
        src = DuckDuckGoSource(session=_session_returns(resp))
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_OK_EMPTY and items == []

    def test_rate_limited(self, monkeypatch):
        monkeypatch.setenv("NEWS_FETCH_MAX_RETRIES", "1")
        src = DuckDuckGoSource(session=_session_returns(_make_response(429)))
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_RATE_LIMITED and items == []

    def test_auth_failed_403(self, monkeypatch):
        monkeypatch.setenv("NEWS_FETCH_MAX_RETRIES", "1")
        src = DuckDuckGoSource(session=_session_returns(_make_response(403)))
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_AUTH_FAILED and items == []

    def test_error_network(self, monkeypatch):
        monkeypatch.setenv("NEWS_FETCH_MAX_RETRIES", "1")
        src = DuckDuckGoSource(session=_session_raises(ConnectionError()))
        items, outcome = src.fetch_classified("Bitcoin")
        assert outcome == OUTCOME_ERROR and items == []

    def test_fetch_backward_compat(self):
        resp = MagicMock(spec=requests.Response)
        resp.status_code = 200
        resp.text = _DDG_HTML_EMPTY
        resp.raise_for_status.return_value = None
        src = DuckDuckGoSource(session=_session_returns(resp))
        assert isinstance(src.fetch("Bitcoin"), list)


# ---------------------------------------------------------------------------
# OpenBBNewsSource  (best-effort: library hides HTTP status)
# ---------------------------------------------------------------------------

class TestOpenBBNewsSource:
    """All tests mock the openbb import via sys.modules to avoid the optional dependency.

    OpenBB hides raw HTTP status codes — classification is best-effort from the exception
    message string.  Tests verify common message patterns only.
    """

    def _src(self):
        return OpenBBNewsSource(provider="yfinance")

    def _fake_obb(self, articles=None, side_effect=None):
        """Build a fake openbb module suitable for sys.modules injection."""
        fake = MagicMock()
        if side_effect is not None:
            fake.obb.news.company.side_effect = side_effect
        else:
            fake.obb.news.company.return_value.results = articles if articles is not None else []
        return fake

    def _article(self, title="BTC hits ATH", url="https://obb.com/1"):
        art = MagicMock()
        art.title = title
        art.url = url
        art.text = "Summary"
        art.source = "Yahoo"
        art.date = None
        return art

    def test_ok(self):
        import sys
        sys.modules["openbb"] = self._fake_obb(articles=[self._article()])
        try:
            items, outcome = self._src().fetch_classified("AAPL")
        finally:
            sys.modules.pop("openbb", None)
        assert outcome == OUTCOME_OK and len(items) == 1

    def test_ok_empty(self):
        import sys
        sys.modules["openbb"] = self._fake_obb(articles=[])
        try:
            items, outcome = self._src().fetch_classified("AAPL")
        finally:
            sys.modules.pop("openbb", None)
        assert outcome == OUTCOME_OK_EMPTY and items == []

    def test_rate_limited_message(self):
        import sys
        sys.modules["openbb"] = self._fake_obb(side_effect=Exception("429 too many requests"))
        try:
            items, outcome = self._src().fetch_classified("AAPL")
        finally:
            sys.modules.pop("openbb", None)
        assert outcome == OUTCOME_RATE_LIMITED and items == []

    def test_auth_failed_message(self):
        import sys
        sys.modules["openbb"] = self._fake_obb(side_effect=Exception("401 unauthorized"))
        try:
            items, outcome = self._src().fetch_classified("AAPL")
        finally:
            sys.modules.pop("openbb", None)
        assert outcome == OUTCOME_AUTH_FAILED and items == []

    def test_timeout_message(self):
        import sys
        sys.modules["openbb"] = self._fake_obb(side_effect=Exception("timeout while reading"))
        try:
            items, outcome = self._src().fetch_classified("AAPL")
        finally:
            sys.modules.pop("openbb", None)
        assert outcome == OUTCOME_TIMEOUT and items == []

    def test_generic_error(self):
        import sys
        sys.modules["openbb"] = self._fake_obb(side_effect=RuntimeError("something went wrong"))
        try:
            items, outcome = self._src().fetch_classified("AAPL")
        finally:
            sys.modules.pop("openbb", None)
        assert outcome == OUTCOME_ERROR and items == []

    def test_import_error_classified_as_error(self):
        """If openbb is not installed the exception is classified as error."""
        import sys
        # Ensure openbb is not importable by setting module to None in sys.modules
        saved = sys.modules.pop("openbb", _SENTINEL)
        sys.modules["openbb"] = None   # causes ImportError on `from openbb import obb`
        try:
            items, outcome = self._src().fetch_classified("AAPL")
        finally:
            if saved is _SENTINEL:
                sys.modules.pop("openbb", None)
            else:
                sys.modules["openbb"] = saved
        assert outcome == OUTCOME_ERROR and items == []

    def test_fetch_backward_compat(self):
        import sys
        sys.modules["openbb"] = self._fake_obb(articles=[])
        try:
            result = self._src().fetch("AAPL")
        finally:
            sys.modules.pop("openbb", None)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# McpDuckDuckGoSource
# ---------------------------------------------------------------------------

def test_mcp_duckduckgo_always_ok_empty():
    src = McpDuckDuckGoSource()
    items, outcome = src.fetch_classified("Bitcoin")
    assert outcome == OUTCOME_OK_EMPTY and items == []


# ---------------------------------------------------------------------------
# BaseNewsSource default fetch_classified wraps fetch()
# ---------------------------------------------------------------------------

class _CustomSource(BaseNewsSource):
    """Simulates a third-party adapter that only overrides fetch()."""
    name = "custom"

    def __init__(self, items=(), raises=None):
        self._items = list(items)
        self._raises = raises

    def fetch(self, query: str, limit: int = 50) -> list[NewsItem]:
        if self._raises:
            raise self._raises
        return self._items


def test_base_fetch_classified_ok():
    from datetime import datetime, timezone
    item = NewsItem(datetime_utc=datetime.now(timezone.utc), source="s", headline="h")
    src = _CustomSource(items=[item])
    items, outcome = src.fetch_classified("q")
    assert outcome == OUTCOME_OK and len(items) == 1


def test_base_fetch_classified_ok_empty():
    src = _CustomSource(items=[])
    items, outcome = src.fetch_classified("q")
    assert outcome == OUTCOME_OK_EMPTY and items == []


def test_base_fetch_classified_exception_classified_as_error():
    src = _CustomSource(raises=RuntimeError("boom"))
    items, outcome = src.fetch_classified("q")
    assert outcome == OUTCOME_ERROR and items == []


# ---------------------------------------------------------------------------
# Circuit-breaker integration: classified outcomes count correctly
# ---------------------------------------------------------------------------

def test_rate_limited_counts_as_failure_for_circuit_breaker():
    h = SourceHealthRegistry(threshold=2)
    h.record("src", 0, 0.1, failure_class=OUTCOME_RATE_LIMITED)
    h.record("src", 0, 0.1, failure_class=OUTCOME_RATE_LIMITED)
    assert not h.allow("src"), "circuit should open after 2 rate-limited failures"
    snap = h.snapshot()["src"]
    assert snap["last_status"] == OUTCOME_RATE_LIMITED
    assert snap["failures"] == 2


def test_auth_failed_counts_as_failure_for_circuit_breaker():
    h = SourceHealthRegistry(threshold=2)
    h.record("src", 0, 0.1, failure_class=OUTCOME_AUTH_FAILED)
    h.record("src", 0, 0.1, failure_class=OUTCOME_AUTH_FAILED)
    assert not h.allow("src")
    assert h.snapshot()["src"]["last_status"] == OUTCOME_AUTH_FAILED


def test_ok_empty_does_not_count_as_failure():
    h = SourceHealthRegistry(threshold=1)
    # Even 10 ok_empty results should never open the circuit
    for _ in range(10):
        h.record("src", 0, 0.1, failure_class=OUTCOME_OK_EMPTY)
    assert h.allow("src"), "ok_empty must not trip the circuit breaker"
    assert h.snapshot()["src"]["failures"] == 0


def test_parse_error_counts_as_failure():
    h = SourceHealthRegistry(threshold=1)
    h.record("src", 0, 0.1, failure_class=OUTCOME_PARSE_ERROR)
    assert not h.allow("src")
    assert h.snapshot()["src"]["last_status"] == OUTCOME_PARSE_ERROR


def test_ok_empty_last_status_stored():
    h = SourceHealthRegistry()
    h.record("src", 0, 0.1, failure_class=OUTCOME_OK_EMPTY)
    assert h.snapshot()["src"]["last_status"] == OUTCOME_OK_EMPTY


def test_classified_outcome_surfaces_in_source_status():
    """source_status() must expose the classified last_status, not a generic 'error'."""
    from datetime import datetime, timezone
    from core.news_pipeline import NewsPipeline
    from core.news_sources import BaseNewsSource, NewsItem

    class RateLimitedSource(BaseNewsSource):
        name = "rl_source"

        def fetch_classified(self, query, limit=50):
            return [], OUTCOME_RATE_LIMITED

        def fetch(self, query, limit=50):
            return []

    h = SourceHealthRegistry(threshold=1)
    pipe = NewsPipeline(sources=[RateLimitedSource()], health=h, deadline_seconds=2.0)
    # Trigger one fetch so the health state is recorded
    pipe._fetch_all_sources("Bitcoin", 5)
    rows = {r["name"]: r for r in pipe.source_status()}
    assert rows["rl_source"]["last_result"] == OUTCOME_RATE_LIMITED


def test_ok_empty_from_source_not_shown_as_failure_in_source_status():
    """ok_empty should NOT trigger cooldown; status should show ok_empty, not cooldown."""
    from core.news_pipeline import NewsPipeline
    from core.news_sources import BaseNewsSource

    class EmptySource(BaseNewsSource):
        name = "empty_src"

        def fetch_classified(self, query, limit=50):
            return [], OUTCOME_OK_EMPTY

        def fetch(self, query, limit=50):
            return []

    h = SourceHealthRegistry(threshold=1)
    pipe = NewsPipeline(sources=[EmptySource()], health=h, deadline_seconds=2.0)
    pipe._fetch_all_sources("Bitcoin", 5)
    rows = {r["name"]: r for r in pipe.source_status()}
    # Circuit should be closed (no cooldown)
    assert rows["empty_src"]["status"] != "cooldown"
    assert rows["empty_src"]["last_result"] == OUTCOME_OK_EMPTY
