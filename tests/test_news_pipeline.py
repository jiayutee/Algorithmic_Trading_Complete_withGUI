from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from core.news_pipeline import NewsPipeline, _classify_event_type, canonicalize_url
from core.news_sources import BaseNewsSource, NewsItem, RssSource
from core.sentiment import SentimentAnalyzer


@dataclass
class DummySource(BaseNewsSource):
    name: str = "dummy"
    reliability: float = 0.8
    items: list[NewsItem] | None = None

    def fetch(self, query: str, limit: int = 50) -> list[NewsItem]:
        return list(self.items or [])[:limit]


def _make_item(headline: str, url: str, minutes: int = 0, source: str = "newsapi") -> NewsItem:
    return NewsItem(
        datetime_utc=datetime(2026, 5, 24, 12, minutes, tzinfo=timezone.utc),
        source=source,
        headline=headline,
        url=url,
        summary="",
        content="",
        source_reliability=0.9,
    )


def test_canonicalize_url_removes_tracking():
    url = "https://example.com/story?utm_source=x&fbclid=y&id=42"
    assert canonicalize_url(url) == "https://example.com/story?id=42"


def test_event_classifier_picks_up_earnings_and_mna():
    assert _classify_event_type("Company beats earnings and raises guidance") == "earnings"
    assert _classify_event_type("Firm announces acquisition deal") == "mna"


def test_pipeline_deduplicates_and_enriches():
    source_items = [
        _make_item("Tesla beats earnings estimates", "https://example.com/a?utm_source=x"),
        _make_item("Tesla beats earnings estimates", "https://example.com/a?utm_source=y", minutes=1),
        _make_item("Tesla announces new product launch", "https://example.com/b", minutes=2),
    ]
    pipeline = NewsPipeline(
        sources=[DummySource(items=source_items)],
        sentiment_analyzer=SentimentAnalyzer(force_rule_based=True),
    )

    frame = pipeline.fetch_news_dataframe("TSLA", company_name="Tesla")
    assert not frame.empty
    assert len(frame) == 2
    assert "positive" in frame.columns
    assert "impact_score" in frame.columns
    assert frame.iloc[0]["headline"] in {"Tesla beats earnings estimates", "Tesla announces new product launch"}


def test_aggregate_news_features_builds_time_buckets():
    pipeline = NewsPipeline(sentiment_analyzer=SentimentAnalyzer(force_rule_based=True))
    frame = pd.DataFrame(
        [
            {
                "datetime": datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc),
                "headline": "Tesla beats earnings estimates",
                "source": "newsapi",
                "positive": 0.8,
                "negative": 0.1,
                "neutral": 0.1,
                "sentiment_confidence": 0.8,
                "sentiment_balance": 0.7,
                "sentiment_magnitude": 0.7,
                "impact_score": 0.9,
                "source_reliability": 0.9,
                "news_count": 1,
                "event_type": "earnings",
                "event_earnings": 1,
                "event_guidance": 0,
                "event_mna": 0,
                "event_analyst": 0,
                "event_macro": 0,
                "event_regulatory": 0,
                "event_product": 0,
                "event_litigation": 0,
                "event_dividend": 0,
                "event_general": 0,
            },
            {
                "datetime": datetime(2026, 5, 24, 12, 5, tzinfo=timezone.utc),
                "headline": "Tesla launches new battery product",
                "source": "rss",
                "positive": 0.7,
                "negative": 0.2,
                "neutral": 0.1,
                "sentiment_confidence": 0.7,
                "sentiment_balance": 0.5,
                "sentiment_magnitude": 0.5,
                "impact_score": 0.6,
                "source_reliability": 0.7,
                "news_count": 1,
                "event_type": "product",
                "event_earnings": 0,
                "event_guidance": 0,
                "event_mna": 0,
                "event_analyst": 0,
                "event_macro": 0,
                "event_regulatory": 0,
                "event_product": 1,
                "event_litigation": 0,
                "event_dividend": 0,
                "event_general": 0,
            },
        ]
    )

    aggregated = pipeline.aggregate_news_features(frame, freq="15min")
    assert not aggregated.empty
    assert "news_count" in aggregated.columns
    assert aggregated.iloc[0]["news_count"] == 2
    assert aggregated.iloc[0]["event_earnings"] == 1
    assert aggregated.iloc[0]["event_product"] == 1


def test_merge_features_into_prices_handles_mixed_datetime_precision():
    pipeline = NewsPipeline(sentiment_analyzer=SentimentAnalyzer(force_rule_based=True))

    price_index = pd.DatetimeIndex(
        [
            datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc),
            datetime(2026, 5, 24, 12, 5, tzinfo=timezone.utc),
        ]
    )
    price_df = pd.DataFrame({"close": [100.0, 101.5]}, index=price_index)

    news_df = pd.DataFrame(
        {
            "datetime": pd.Series(
                ["2026-05-24 12:00:00.123", "2026-05-24 12:04:00.456"]
            ).astype("datetime64[ms]"),
            "headline": ["Tesla beats earnings", "Tesla launches product"],
            "source": ["newsapi", "rss"],
            "positive": [0.8, 0.7],
            "negative": [0.1, 0.2],
            "neutral": [0.1, 0.1],
            "sentiment_confidence": [0.8, 0.7],
            "sentiment_balance": [0.7, 0.5],
            "sentiment_magnitude": [0.7, 0.5],
            "impact_score": [0.9, 0.6],
            "source_reliability": [0.9, 0.7],
            "news_count": [1, 1],
            "event_type": ["earnings", "product"],
            "event_earnings": [1, 0],
            "event_guidance": [0, 0],
            "event_mna": [0, 0],
            "event_analyst": [0, 0],
            "event_macro": [0, 0],
            "event_regulatory": [0, 0],
            "event_product": [0, 1],
            "event_litigation": [0, 0],
            "event_dividend": [0, 0],
            "event_general": [0, 0],
        }
    )

    merged = pipeline.merge_features_into_prices(price_df, news_df, interval="1D")
    assert not merged.empty
    assert "news_count" in merged.columns
    assert merged["news_count"].sum() > 0


def test_rss_source_parses_items(monkeypatch):
    xml_payload = """
    <rss version="2.0">
      <channel>
        <title>Example Feed</title>
        <item>
          <title>Bitcoin rallies after ETF approval</title>
          <link>https://example.com/bitcoin</link>
          <description>ETF approval boosts risk appetite</description>
          <pubDate>Sat, 24 May 2026 12:00:00 GMT</pubDate>
        </item>
      </channel>
    </rss>
    """.strip()

    class Response:
        status_code = 200
        text = xml_payload

        def raise_for_status(self):
            return None

    def fake_request(self, method, url, **kwargs):
        return Response()

    # RssSource calls session.request() via request_with_retries, not session.get()
    monkeypatch.setattr("requests.Session.request", fake_request)
    source = RssSource(feed_urls=["https://example.com/rss"])
    items = source.fetch("bitcoin", limit=10)
    assert len(items) == 1
    assert items[0].headline == "Bitcoin rallies after ETF approval"
