from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any
import os
import re

import pandas as pd

from core.logger import logger
from core.news_sources import (
    BaseNewsSource,
    BraveSearchSource,
    EventRegistrySource,
    GDELTSource,
    NewsApiSource,
    NewsItem,
    RssSource,
    canonicalize_url,
    fuzzy_title_match,
    DuckDuckGoSource,
    McpDuckDuckGoSource,
)
from core.sentiment import SentimentAnalyzer
from core.news_store import NewsStore


EVENT_TYPES = [
    "earnings",
    "guidance",
    "mna",
    "analyst",
    "macro",
    "regulatory",
    "product",
    "litigation",
    "dividend",
    "general",
]

EVENT_KEYWORDS = {
    "earnings": ["earnings", "eps", "revenue", "quarter", "q1", "q2", "q3", "q4", "results", "profit", "loss", "beat", "miss"],
    "guidance": ["guidance", "outlook", "forecast", "raises guidance", "cuts guidance", "revises guidance"],
    "mna": ["acquire", "acquisition", "merger", "merges", "takeover", "buyout", "deal", "stake"],
    "analyst": ["upgrade", "downgrade", "initiates", "price target", "target price", "reiterates", "cuts target", "raises target"],
    "macro": ["cpi", "inflation", "fed", "fomc", "rates", "rate hike", "rate cut", "jobs", "payrolls", "gdp", "pmi", "ppi", "unemployment"],
    "regulatory": ["sec", "doj", "lawsuit", "investigation", "probe", "fine", "settlement", "antitrust", "sanction"],
    "product": ["launch", "launches", "announces", "unveils", "release", "product", "chip", "software", "ai"],
    "litigation": ["lawsuit", "sued", "court", "appeal", "litigation"],
    "dividend": ["dividend", "buyback", "repurchase", "shareholder return"],
}

SOURCE_WEIGHTS = {
    "newsapi": 0.9,
    "brave": 0.86,
    "eventregistry": 0.82,
    "gdelt": 0.75,
    "rss": 0.7,
}

DEFAULT_NUMERIC_COLUMNS = [
    "positive",
    "negative",
    "neutral",
    "sentiment_confidence",
    "sentiment_balance",
    "sentiment_magnitude",
    "impact_score",
    "source_reliability",
    "news_count",
]


def interval_to_pandas_freq(interval: str) -> str:
    mapping = {
        "1m": "1min",
        "2m": "2min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "60m": "60min",
        "1h": "1h",
        "1d": "1D",
        "1wk": "1W",
        "1mo": "1MS",
    }
    return mapping.get(interval, "1D")


def _query_variants(symbol: str, company_name: str | None = None) -> list[str]:
    variants = []
    if company_name:
        variants.append(company_name)
    if symbol:
        variants.append(symbol)
        upper = symbol.upper()
        variants.append(upper)
        variants.append(upper.replace("-", ""))
        variants.append(upper.replace("USDT", ""))
    seen: set[str] = set()
    deduped: list[str] = []
    for variant in variants:
        variant = variant.strip()
        if variant and variant.lower() not in seen:
            deduped.append(variant)
            seen.add(variant.lower())
    return deduped or [symbol]


def _classify_event_type(text: str) -> str:
    text_lower = (text or "").lower()
    for event_type, keywords in EVENT_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            return event_type
    return "general"


def _normalize_index_to_utc_naive(idx):
    """Normalize a datetime-like index/array to UTC-naive nanosecond precision.

    Uses `pd.to_datetime(..., utc=True)` then converts to UTC and localizes to None.
    Returns a `pd.DatetimeIndex`.
    """
    # pd.to_datetime will return tz-aware datetimes when utc=True
    dt = pd.to_datetime(idx, utc=True, errors="coerce")
    # convert to UTC then drop tz info (make naive)
    dt = dt.tz_convert("UTC").tz_localize(None)
    # ensure numpy dtype is nanosecond for compatibility with other data
    try:
        dt = dt.astype("datetime64[ns]")
    except Exception:
        # fallback: reconstruct via to_datetime
        dt = pd.to_datetime(dt).astype("datetime64[ns]")
    return pd.DatetimeIndex(dt)


def _extract_entities(text: str, symbol: str, company_name: str | None = None) -> tuple[list[str], list[dict[str, Any]]]:
    text_lower = (text or "").lower()
    tickers: list[str] = []
    entities: list[dict[str, Any]] = []

    symbol_variants = {symbol.upper(), symbol.upper().replace("-", ""), symbol.upper().replace("USDT", "")}
    if any(variant.lower() in text_lower for variant in symbol_variants if variant):
        tickers.append(symbol.upper())
        entities.append({"text": symbol.upper(), "type": "TICKER", "confidence": 1.0})

    if company_name:
        company_tokens = [token for token in re.findall(r"[A-Za-z0-9]+", company_name.lower()) if token]
        if company_tokens and all(token in text_lower for token in company_tokens):
            if symbol.upper() not in tickers:
                tickers.append(symbol.upper())
            entities.append({"text": company_name, "type": "ORG", "confidence": 0.95})

    return tickers, entities


def _score_impact(sentiment: dict[str, float], event_type: str, source: str, tickers: list[str]) -> float:
    positive = float(sentiment.get("positive", 0.0))
    negative = float(sentiment.get("negative", 0.0))
    neutral = float(sentiment.get("neutral", 0.0))
    sentiment_strength = min(1.0, abs(positive - negative) + max(positive, negative) * 0.5 + (1.0 - neutral) * 0.2)
    event_weight = {
        "earnings": 1.0,
        "guidance": 0.95,
        "mna": 1.0,
        "regulatory": 0.85,
        "analyst": 0.8,
        "macro": 0.9,
        "product": 0.7,
        "litigation": 0.75,
        "dividend": 0.65,
        "general": 0.45,
    }.get(event_type, 0.45)
    source_weight = SOURCE_WEIGHTS.get((source or "").lower(), 0.5)
    mention_bonus = min(len(tickers), 3) * 0.05
    score = (0.55 * sentiment_strength + 0.35 * event_weight + mention_bonus) * source_weight
    return round(max(0.0, min(score, 1.0)), 4)


@lru_cache(maxsize=1)
def get_default_news_pipeline() -> "NewsPipeline":
    return NewsPipeline.from_env()


class NewsPipeline:
    def __init__(
        self,
        sources: list[BaseNewsSource] | None = None,
        sentiment_analyzer: SentimentAnalyzer | None = None,
        max_workers: int = 4,
    ):
        self.sources = sources or []
        self.sentiment_analyzer = sentiment_analyzer or SentimentAnalyzer()
        self.max_workers = max_workers

    @classmethod
    def from_env(cls) -> "NewsPipeline":
        sources: list[BaseNewsSource] = []

        # Optionally enable the MCP-managed DuckDuckGo source as the first source.
        if os.getenv("USE_MCP_DDG", "1") == "1":
            try:
                sources.append(McpDuckDuckGoSource())
            except Exception:
                # If for some reason the class isn't available, log and continue.
                logger.info("MCP DuckDuckGo source requested but unavailable; skipping.")

        brave_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip() or os.getenv("BRAVE_API_KEY", "").strip()
        if brave_key:
            sources.append(BraveSearchSource(api_key=brave_key))

        # Prefer DuckDuckGo HTML search first for broader coverage/enrichment
        sources.append(DuckDuckGoSource())

        newsapi_key = os.getenv("NEWSAPI_API_KEY", "").strip()
        if newsapi_key:
            sources.append(NewsApiSource(api_key=newsapi_key))

        sources.append(GDELTSource())

        eventregistry_key = os.getenv("EVENTREGISTRY_API_KEY", "").strip()
        if eventregistry_key:
            sources.append(EventRegistrySource(api_key=eventregistry_key))

        rss_feeds_env = os.getenv("RSS_FEEDS", "").strip()
        rss_feeds = [feed.strip() for feed in re.split(r"[\n,]", rss_feeds_env) if feed.strip()]
        if rss_feeds:
            sources.append(RssSource(feed_urls=rss_feeds))

        # Add DuckDuckGo HTML search as a fallback / enrichment source
        sources.append(DuckDuckGoSource())

        return cls(sources=sources)

    def fetch_news_items(self, symbol: str, company_name: str | None = None, limit: int = 50) -> list[NewsItem]:
        query_variants = _query_variants(symbol, company_name)
        gathered: list[NewsItem] = []

        if not self.sources:
            logger.warning("No news sources configured. Returning an empty result set.")
            return []

        with ThreadPoolExecutor(max_workers=min(self.max_workers, max(1, len(self.sources)))) as executor:
            futures = []
            for source in self.sources:
                query = query_variants[0]
                futures.append(executor.submit(source.fetch, query, limit))

            for future in as_completed(futures):
                try:
                    gathered.extend(future.result() or [])
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("News source task failed: %s", exc)

        items = self._enrich_and_deduplicate(gathered, symbol=symbol, company_name=company_name)

        # Persist deduplicated/enriched items to local news store (non-fatal)
        try:
            store = NewsStore()
            inserted = store.add_items(items)
            if inserted:
                logger.info("Persisted %s new news items for %s", inserted, symbol)
            store.close()
        except Exception as exc:  # pragma: no cover - do not fail fetch on persistence errors
            logger.warning("Failed to persist news items: %s", exc)

        return items

    def fetch_news_dataframe(self, symbol: str, company_name: str | None = None, limit: int = 50) -> pd.DataFrame:
        items = self.fetch_news_items(symbol=symbol, company_name=company_name, limit=limit)
        if not items:
            return pd.DataFrame()
        return pd.DataFrame([self._item_to_row(item) for item in items])

    def aggregate_news_features(self, news_df: pd.DataFrame, freq: str = "1D") -> pd.DataFrame:
        if news_df is None or news_df.empty:
            return pd.DataFrame()

        frame = news_df.copy()
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["datetime"]).sort_values("datetime")
        if frame.empty:
            return pd.DataFrame()

        frame = frame.set_index("datetime")
        # normalize index to UTC-naive nanosecond precision
        try:
            frame.index = _normalize_index_to_utc_naive(frame.index)
        except Exception:
            # fallback: ensure it's a DatetimeIndex
            frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True, errors="coerce")).tz_convert("UTC").tz_localize(None)
        for column in ["positive", "negative", "neutral", "sentiment_confidence", "sentiment_balance", "sentiment_magnitude", "impact_score", "source_reliability"]:
            if column not in frame.columns:
                frame[column] = 0.0

        for event_type in EVENT_TYPES:
            event_column = f"event_{event_type}"
            if event_column not in frame.columns:
                frame[event_column] = 0

        frame["news_count"] = 1

        aggregated = frame.resample(freq).agg(
            {
                "positive": "mean",
                "negative": "mean",
                "neutral": "mean",
                "sentiment_confidence": "mean",
                "sentiment_balance": "mean",
                "sentiment_magnitude": "mean",
                "impact_score": "mean",
                "source_reliability": "mean",
                "news_count": "sum",
                "headline": "count",
                "source": "nunique",
                "event_earnings": "sum",
                "event_guidance": "sum",
                "event_mna": "sum",
                "event_analyst": "sum",
                "event_macro": "sum",
                "event_regulatory": "sum",
                "event_product": "sum",
                "event_litigation": "sum",
                "event_dividend": "sum",
                "event_general": "sum",
            }
        )
        aggregated = aggregated.rename(columns={"headline": "headline_count", "source": "source_count"})
        aggregated["news_flow_ratio"] = aggregated["sentiment_balance"].fillna(0.0) / aggregated["news_count"].replace(0, 1)
        return aggregated.fillna(0.0)

    def merge_features_into_prices(self, price_df: pd.DataFrame, news_df: pd.DataFrame, interval: str = "1D") -> pd.DataFrame:
        if price_df is None or price_df.empty:
            return price_df

        frame = price_df.copy()
        if not isinstance(frame.index, pd.DatetimeIndex):
            if "Datetime" in frame.columns:
                frame["Datetime"] = pd.to_datetime(frame["Datetime"], utc=True, errors="coerce")
                frame = frame.set_index("Datetime")
            else:
                raise ValueError("price_df must be indexed by Datetime or include a Datetime column")

        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        else:
            frame.index = frame.index.tz_convert("UTC")

        # normalize price index to UTC-naive nanosecond precision
        frame.index = _normalize_index_to_utc_naive(frame.index)

        aggregated = self.aggregate_news_features(news_df, freq=interval_to_pandas_freq(interval))
        if aggregated.empty:
            for column in DEFAULT_NUMERIC_COLUMNS:
                if column not in frame.columns:
                    frame[column] = 0.0
            return frame

        # ensure aggregated index uses same UTC-naive nanosecond precision
        try:
            aggregated.index = _normalize_index_to_utc_naive(aggregated.index)
        except Exception:
            aggregated.index = pd.DatetimeIndex(pd.to_datetime(aggregated.index, utc=True, errors="coerce")).tz_convert("UTC").tz_localize(None)

        merged = pd.merge_asof(
            frame.sort_index(),
            aggregated.sort_index(),
            left_index=True,
            right_index=True,
            direction="backward",
            allow_exact_matches=True,
        )

        for column in merged.columns:
            if pd.api.types.is_numeric_dtype(merged[column]):
                merged[column] = merged[column].fillna(0.0)

        return merged

    def _enrich_and_deduplicate(self, items: list[NewsItem], symbol: str, company_name: str | None = None) -> list[NewsItem]:
        if not items:
            return []

        scored_items = []
        for item in items:
            text = " ".join([item.headline or "", item.summary or "", item.content or ""]).strip()
            tickers, entities = _extract_entities(text, symbol=symbol, company_name=company_name)
            event_type = _classify_event_type(text)
            scored_items.append(
                NewsItem(
                    datetime_utc=item.datetime_utc,
                    source=item.source,
                    headline=item.headline,
                    url=canonicalize_url(item.url),
                    summary=item.summary,
                    content=item.content,
                    language=item.language,
                    tickers=tickers or item.tickers,
                    entities=entities or item.entities,
                    event_type=event_type,
                    sentiment=item.sentiment,
                    impact_score=item.impact_score,
                    source_reliability=item.source_reliability,
                    metadata=item.metadata,
                )
            )

        sentiment_texts = [" ".join([item.headline, item.summary]).strip() for item in scored_items]
        sentiment_results = self.sentiment_analyzer.analyze_many(sentiment_texts)

        enriched: list[NewsItem] = []
        for item, sentiment_result in zip(scored_items, sentiment_results):
            sentiment = {
                "positive": sentiment_result.positive,
                "negative": sentiment_result.negative,
                "neutral": sentiment_result.neutral,
                "label": sentiment_result.label,
                "confidence": sentiment_result.confidence,
                "model_name": sentiment_result.model_name,
            }
            impact_score = _score_impact(sentiment, item.event_type, item.source, item.tickers)
            enriched.append(
                NewsItem(
                    datetime_utc=item.datetime_utc,
                    source=item.source,
                    headline=item.headline,
                    url=item.url,
                    summary=item.summary,
                    content=item.content,
                    language=item.language,
                    tickers=item.tickers,
                    entities=item.entities,
                    event_type=item.event_type,
                    sentiment=sentiment,
                    impact_score=impact_score,
                    source_reliability=item.source_reliability,
                    metadata=item.metadata,
                )
            )

        return self._deduplicate(enriched)

    def _deduplicate(self, items: list[NewsItem]) -> list[NewsItem]:
        ranked = sorted(items, key=lambda item: (item.datetime_utc, item.source_reliability, item.impact_score), reverse=True)
        unique: list[NewsItem] = []
        seen_urls: set[str] = set()
        domain_counts: dict[str, int] = {}
        max_per_domain = 3

        for item in ranked:
            canonical_url = canonicalize_url(item.url)
            if canonical_url and canonical_url in seen_urls:
                continue

            duplicate = False
            for existing in unique:
                if canonical_url and canonical_url == canonicalize_url(existing.url):
                    duplicate = True
                    break
                if item.headline and existing.headline and fuzzy_title_match(item.headline, existing.headline) >= 0.94:
                    duplicate = True
                    break

            if duplicate:
                continue

            # domain diversity: avoid too many items from same host
            domain = ""
            try:
                from urllib.parse import urlsplit

                domain = urlsplit(canonical_url or item.url or "").netloc.lower()
            except Exception:
                domain = ""

            cnt = domain_counts.get(domain, 0)
            if domain and cnt >= max_per_domain:
                continue

            if canonical_url:
                seen_urls.add(canonical_url)
            unique.append(item)
            if domain:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

        return sorted(unique, key=lambda item: item.datetime_utc, reverse=True)

    @staticmethod
    def _item_to_row(item: NewsItem) -> dict[str, Any]:
        sentiment = item.sentiment or {}
        row = {
            "datetime": item.datetime_utc,
            "source": item.source,
            "headline": item.headline,
            "link": item.url,
            "summary": item.summary,
            "content": item.content,
            "language": item.language,
            "tickers": item.tickers,
            "entities": item.entities,
            "event_type": item.event_type,
            "positive": float(sentiment.get("positive", 0.0)),
            "negative": float(sentiment.get("negative", 0.0)),
            "neutral": float(sentiment.get("neutral", 0.0)),
            "sentiment_label": sentiment.get("label", "neutral"),
            "sentiment_confidence": float(sentiment.get("confidence", 0.0)),
            "sentiment_model": sentiment.get("model_name", ""),
            "sentiment_balance": float(sentiment.get("positive", 0.0)) - float(sentiment.get("negative", 0.0)),
            "sentiment_magnitude": abs(float(sentiment.get("positive", 0.0)) - float(sentiment.get("negative", 0.0))),
            "impact_score": item.impact_score,
            "source_reliability": item.source_reliability,
            "news_count": 1,
        }
        for event_type in EVENT_TYPES:
            row[f"event_{event_type}"] = int(item.event_type == event_type)
        return row
