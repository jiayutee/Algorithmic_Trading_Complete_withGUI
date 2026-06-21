from datetime import datetime, timezone
import json
from core.news_store import NewsStore
from core.news_sources import NewsItem


def _make_item(headline, url, minutes=0, ticker=None):
    return NewsItem(
        datetime_utc=datetime(2026,5,24,12,minutes,tzinfo=timezone.utc),
        source="unittest",
        headline=headline,
        url=url,
        summary="",
        content="",
        tickers=[ticker] if ticker else [],
    )


def test_news_store_insert_and_dedupe(tmp_path):
    db_path = tmp_path / "test_news_store.sqlite3"
    ns = NewsStore(str(db_path))
    items = [
        _make_item("ACME posts profit", "https://example.com/a?utm=1", ticker="ACME"),
        _make_item("ACME posts profit", "https://example.com/a?utm=2", ticker="ACME"),
        _make_item("ACME launches product", "https://example.com/b", ticker="ACME"),
    ]
    inserted = ns.add_items(items)
    assert inserted >= 2
    results = ns.query_by_ticker("ACME", limit=10)
    assert len(results) >= 2
    ns.close()
