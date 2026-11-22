"""OpenBB news source: Yahoo-search fallback when yfinance's own news call is broken (2026-11-22 outage).

yfinance 0.2.55 raised "YAHOO! FINANCE IS CURRENTLY DOWN" while Yahoo itself answered, so openbb_news returned nothing
for weeks and its circuit stayed open. The fallback reads Yahoo's public search endpoint, which carries real publish times.
No network: ``requests.get`` and ``openbb`` are mocked.
"""
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from core.news_sources import OUTCOME_ERROR, OUTCOME_OK, OpenBBNewsSource

SEARCH = {"news": [
    {"title": "Bitcoin slides as ETF outflows grow", "link": "https://finance.yahoo.com/a1", "publisher": "Reuters",
     "providerPublishTime": 1795321020},
    {"title": "No time given", "link": "https://finance.yahoo.com/a2", "publisher": "X"},
    {"title": "", "link": "https://finance.yahoo.com/a3", "providerPublishTime": 1795321021},
]}


@pytest.fixture
def broken_yfinance(monkeypatch):
    obb = MagicMock()
    obb.news.company.side_effect = RuntimeError("[Unexpected Error] -> RuntimeError -> *** YAHOO! FINANCE IS CURRENTLY DOWN! ***")
    mod = types.ModuleType("openbb")
    mod.obb = obb
    monkeypatch.setitem(sys.modules, "openbb", mod)


def _resp(status=200, payload=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload if payload is not None else SEARCH
    return r


def test_fallback_returns_items_with_real_publish_time(broken_yfinance, monkeypatch):
    monkeypatch.setenv("OPENBB_NEWS_YAHOO_FALLBACK", "true")
    with patch("core.news_sources.requests.get", return_value=_resp()) as get:
        items, outcome = OpenBBNewsSource().fetch_classified("BTCUSDT")
    assert outcome == OUTCOME_OK
    assert len(items) == 1                              # undated and headline-less rows are dropped
    it = items[0]
    assert it.datetime_utc.timestamp() == 1795321020 and it.pub_time_is_fetch_time is False
    assert it.source == "openbb:Reuters"
    assert get.call_args.kwargs["params"]["q"] == "Bitcoin"   # pair -> BTC-USD -> coin name (Yahoo finds crypto by name)


def test_crypto_name_search_keeps_only_items_tied_to_the_coin(broken_yfinance, monkeypatch):
    monkeypatch.setenv("OPENBB_NEWS_YAHOO_FALLBACK", "true")
    payload = {"news": [
        {"title": "Fed holds rates", "link": "u1", "providerPublishTime": 1795321020, "relatedTickers": ["BTC-USD"]},
        {"title": "Bitcoin dips", "link": "u2", "providerPublishTime": 1795321021, "relatedTickers": []},
        {"title": "Unrelated stock news", "link": "u3", "providerPublishTime": 1795321022, "relatedTickers": ["AAPL"]},
    ]}
    with patch("core.news_sources.requests.get", return_value=_resp(payload=payload)):
        items, _ = OpenBBNewsSource().fetch_classified("BTCUSDT")
    assert [i.headline for i in items] == ["Fed holds rates", "Bitcoin dips"]
    with patch("core.news_sources.requests.get") as get:                  # a coin we cannot name: no call at all
        assert OpenBBNewsSource().fetch_classified("ZZZUSDT") == ([], OUTCOME_ERROR)
        get.assert_not_called()


def test_fallback_failure_still_reports_error(broken_yfinance, monkeypatch):
    monkeypatch.setenv("OPENBB_NEWS_YAHOO_FALLBACK", "true")
    with patch("core.news_sources.requests.get", return_value=_resp(status=429)):
        assert OpenBBNewsSource().fetch_classified("AAPL") == ([], OUTCOME_ERROR)
    with patch("core.news_sources.requests.get", side_effect=RuntimeError("dns")):
        assert OpenBBNewsSource().fetch_classified("AAPL") == ([], OUTCOME_ERROR)


def test_fallback_is_off_by_env_and_for_other_providers(broken_yfinance, monkeypatch):
    with patch("core.news_sources.requests.get") as get:                       # conftest default: switched off
        assert OpenBBNewsSource().fetch_classified("AAPL") == ([], OUTCOME_ERROR)
        monkeypatch.setenv("OPENBB_NEWS_YAHOO_FALLBACK", "true")
        assert OpenBBNewsSource(provider="benzinga").fetch_classified("AAPL") == ([], OUTCOME_ERROR)
        get.assert_not_called()


def test_undated_openbb_article_is_flagged_as_fetch_time(monkeypatch):
    art = types.SimpleNamespace(title="Headline", url="https://x/y", text="t", source="S", date=None)
    obb = MagicMock()
    obb.news.company.return_value = types.SimpleNamespace(results=[art])
    mod = types.ModuleType("openbb")
    mod.obb = obb
    monkeypatch.setitem(sys.modules, "openbb", mod)
    items, _ = OpenBBNewsSource().fetch_classified("AAPL")
    assert items[0].pub_time_is_fetch_time is True
