from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import re
import xml.etree.ElementTree as ET
import os
import time
import random

import pandas as pd
import requests
from bs4 import BeautifulSoup

from core.logger import logger


TRACKING_PARAMS = {
    "cmpid",
    "fbclid",
    "gclid",
    "icid",
    "mc_cid",
    "mc_eid",
    "ref",
    "utm_campaign",
    "utm_content",
    "utm_medium",
    "utm_source",
    "utm_term",
}


@dataclass
class NewsItem:
    datetime_utc: datetime
    source: str
    headline: str
    url: str = ""
    summary: str = ""
    content: str = ""
    language: str = "en"
    tickers: list[str] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    event_type: str = "general"
    sentiment: dict[str, float] = field(default_factory=dict)
    impact_score: float = 0.0
    source_reliability: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


def canonicalize_url(url: str) -> str:
    if not url:
        return ""

    parts = urlsplit(url)
    filtered_query = [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True) if key.lower() not in TRACKING_PARAMS]
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, urlencode(filtered_query), ""))


def coerce_datetime(value: Any) -> datetime:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return datetime.now(timezone.utc)
    return timestamp.to_pydatetime()


def request_with_retries(session: requests.Session, method: str, url: str, **kwargs):
    """Make an HTTP request with exponential backoff retries.

    Returns a `requests.Response` on success or `None` on final failure.
    Environment variables:
    - NEWS_FETCH_MAX_RETRIES (default 3)
    - NEWS_FETCH_BACKOFF_FACTOR (default 0.5)
    """
    try:
        max_retries = int(os.getenv("NEWS_FETCH_MAX_RETRIES", "3"))
    except Exception:
        max_retries = 3
    try:
        backoff_factor = float(os.getenv("NEWS_FETCH_BACKOFF_FACTOR", "0.5"))
    except Exception:
        backoff_factor = 0.5

    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            resp = session.request(method, url, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            if attempt >= max_retries:
                logger.warning("HTTP %s %s failed after %s attempts: %s", method.upper(), url, attempt, exc)
                return None
            # Exponential backoff with jitter
            sleep_time = backoff_factor * (2 ** (attempt - 1))
            jitter = random.uniform(0, backoff_factor)
            time.sleep(sleep_time + jitter)
            continue


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query or "").strip()


class BaseNewsSource:
    name = "base"
    reliability = 0.5

    def fetch(self, query: str, limit: int = 50) -> list[NewsItem]:
        raise NotImplementedError


class NewsApiSource(BaseNewsSource):
    name = "newsapi"
    reliability = 0.9

    def __init__(self, api_key: str | None = None, session: requests.Session | None = None):
        self.api_key = api_key or ""
        self.session = session or requests.Session()

    def fetch(self, query: str, limit: int = 50) -> list[NewsItem]:
        if not self.api_key:
            return []

        params = {
            "q": normalize_query(query),
            "language": "en",
            "pageSize": min(limit, 100),
            "sortBy": "publishedAt",
        }
        # Prefer the newer X-Api-Key header, but fall back to Authorization for
        # older setups. Use the request_with_retries helper for both attempts.
        headers = {"X-Api-Key": self.api_key}
        try:
            response = request_with_retries(self.session, "get", "https://newsapi.org/v2/everything", params=params, headers=headers, timeout=20)
            # If the helper returned None (final failure), attempt one fallback
            # using the older Authorization header.
            if response is None:
                logger.info("NewsAPI fetch: initial X-Api-Key attempt failed, trying Authorization fallback")
                response = request_with_retries(self.session, "get", "https://newsapi.org/v2/everything", params=params, headers={"Authorization": self.api_key}, timeout=20)
                if response is None:
                    return []
            payload = response.json()
        except Exception as exc:
            logger.warning("NewsAPI fetch failed for %s: %s", query, exc)
            return []

        articles = payload.get("articles", []) or []
        items: list[NewsItem] = []
        for article in articles[:limit]:
            items.append(
                NewsItem(
                    datetime_utc=coerce_datetime(article.get("publishedAt")),
                    source=(article.get("source") or {}).get("name") or self.name,
                    headline=article.get("title") or "",
                    url=article.get("url") or "",
                    summary=article.get("description") or "",
                    content=article.get("content") or "",
                    language=article.get("language") or "en",
                    metadata={"source_api": self.name},
                    source_reliability=self.reliability,
                )
            )
        return items


class GDELTSource(BaseNewsSource):
    name = "gdelt"
    reliability = 0.75

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()

    def fetch(self, query: str, limit: int = 50) -> list[NewsItem]:
        params = {
            "query": normalize_query(query),
            "mode": "ArtList",
            "format": "json",
            "sort": "HybridRel",
            "maxrecords": min(limit, 250),
        }
        try:
            response = request_with_retries(self.session, "get", "https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=20)
            if response is None:
                return []
            payload = response.json()
        except Exception as exc:
            logger.warning("GDELT fetch failed for %s: %s", query, exc)
            return []

        article_candidates = payload.get("articles") or payload.get("result", {}).get("articles") or payload.get("records") or []
        items: list[NewsItem] = []
        for article in article_candidates[:limit]:
            items.append(
                NewsItem(
                    datetime_utc=coerce_datetime(article.get("seendate") or article.get("datetime") or article.get("date") or article.get("pubDate")),
                    source=article.get("sourceCountry") or article.get("domain") or self.name,
                    headline=article.get("title") or article.get("headline") or "",
                    url=article.get("url") or article.get("link") or "",
                    summary=article.get("snippet") or article.get("description") or "",
                    content=article.get("content") or "",
                    language=article.get("language") or "en",
                    metadata={"source_api": self.name, "domain": article.get("domain")},
                    source_reliability=self.reliability,
                )
            )
        return items


class EventRegistrySource(BaseNewsSource):
    name = "eventregistry"
    reliability = 0.82

    def __init__(self, api_key: str | None = None, session: requests.Session | None = None):
        self.api_key = api_key or ""
        self.session = session or requests.Session()

    def fetch(self, query: str, limit: int = 50) -> list[NewsItem]:
        if not self.api_key:
            return []

        payload = {
            "apiKey": self.api_key,
            "keyword": normalize_query(query),
            "articlesSortBy": "date",
            "articlesCount": min(limit, 100),
            "includeArticleBody": False,
            "lang": ["eng"],
        }
        try:
            response = request_with_retries(self.session, "post", "https://eventregistry.org/api/v1/article/getArticles", json=payload, timeout=25)
            if response is None:
                return []
            payload = response.json()
        except Exception as exc:
            logger.warning("EventRegistry fetch failed for %s: %s", query, exc)
            return []

        article_candidates = (
            payload.get("articles", {}).get("results")
            or payload.get("articles", {}).get("results", {}).get("results")
            or payload.get("results")
            or []
        )
        items: list[NewsItem] = []
        for article in article_candidates[:limit]:
            source = article.get("source") or {}
            items.append(
                NewsItem(
                    datetime_utc=coerce_datetime(article.get("dateTimePub") or article.get("date") or article.get("publishedAt")),
                    source=source.get("title") or source.get("name") or self.name,
                    headline=article.get("title") or article.get("headline") or "",
                    url=article.get("url") or "",
                    summary=article.get("body") or article.get("summary") or article.get("description") or "",
                    content=article.get("body") or "",
                    language=article.get("lang") or article.get("language") or "en",
                    metadata={"source_api": self.name, "uri": article.get("uri")},
                    source_reliability=self.reliability,
                )
            )
        return items


class BraveSearchSource(BaseNewsSource):
    name = "brave"
    reliability = 0.86

    def __init__(self, api_key: str | None = None, session: requests.Session | None = None):
        self.api_key = api_key or ""
        self.session = session or requests.Session()

    @staticmethod
    def _result_text(result: dict[str, Any]) -> str:
        parts = [
            result.get("title") or "",
            result.get("description") or result.get("snippet") or "",
        ]
        extra_snippets = result.get("extra_snippets") or result.get("extraSnippets") or []
        if isinstance(extra_snippets, list):
            parts.extend([snippet for snippet in extra_snippets if isinstance(snippet, str)])
        return " ".join(part.strip() for part in parts if part).strip()

    def _fetch_endpoint(self, endpoint: str, query: str, limit: int) -> list[dict[str, Any]]:
        params = {
            "q": normalize_query(query),
            "count": min(limit, 20),
            "search_lang": "en",
            "country": "us",
            "safesearch": "off",
        }
        headers = {
            "X-Subscription-Token": self.api_key,
            "Accept": "application/json",
        }
        response = request_with_retries(self.session, "get", endpoint, params=params, headers=headers, timeout=20)
        if response is None:
            return []

        payload = response.json()
        results = payload.get("results") or payload.get("web", {}).get("results") or payload.get("news", {}).get("results") or []
        if isinstance(results, list):
            return [result for result in results if isinstance(result, dict)]
        return []

    def fetch(self, query: str, limit: int = 50) -> list[NewsItem]:
        if not self.api_key:
            return []

        endpoints = [
            ("https://api.search.brave.com/res/v1/news/search", "news"),
            ("https://api.search.brave.com/res/v1/web/search", "web"),
        ]
        results: list[dict[str, Any]] = []
        endpoint_name = "news"
        for endpoint, label in endpoints:
            try:
                results = self._fetch_endpoint(endpoint, query, limit)
            except Exception as exc:
                logger.warning("Brave Search fetch failed for %s via %s: %s", query, endpoint, exc)
                results = []
            if results:
                endpoint_name = label
                break

        items: list[NewsItem] = []
        for result in results[:limit]:
            profile = result.get("profile") or {}
            source_name = result.get("source") or result.get("publisher") or (profile.get("name") if isinstance(profile, dict) else None) or self.name
            published_at = (
                result.get("published")
                or result.get("date")
                or result.get("age")
                or result.get("page_age")
                or result.get("timestamp")
            )
            summary = result.get("description") or result.get("snippet") or ""
            content = summary or self._result_text(result)
            items.append(
                NewsItem(
                    datetime_utc=coerce_datetime(published_at),
                    source=source_name,
                    headline=result.get("title") or "",
                    url=result.get("url") or result.get("link") or "",
                    summary=summary,
                    content=content,
                    language=result.get("language") or result.get("lang") or "en",
                    metadata={"source_api": self.name, "endpoint": endpoint_name, "raw_source": source_name},
                    source_reliability=self.reliability,
                )
            )

        return items


class RssSource(BaseNewsSource):
    name = "rss"
    reliability = 0.7

    def __init__(self, feed_urls: list[str] | None = None, session: requests.Session | None = None):
        self.feed_urls = [feed.strip() for feed in (feed_urls or []) if feed.strip()]
        self.session = session or requests.Session()

    @staticmethod
    def _matches_query(text: str, query: str) -> bool:
        if not query:
            return True
        query_terms = [term.lower() for term in re.findall(r"[A-Za-z0-9$.-]+", query)]
        text_lower = text.lower()
        return any(term in text_lower for term in query_terms if term)

    def fetch(self, query: str, limit: int = 50) -> list[NewsItem]:
        items: list[NewsItem] = []
        for feed_url in self.feed_urls:
            try:
                response = request_with_retries(self.session, "get", feed_url, timeout=20)
                if response is None:
                    raise RuntimeError("max retries exceeded")
            except Exception as exc:
                logger.warning("RSS fetch failed for %s: %s", feed_url, exc)
                continue

            try:
                root = ET.fromstring(response.text)
            except ET.ParseError as exc:
                logger.warning("RSS parse failed for %s: %s", feed_url, exc)
                continue

            channel = root.find("channel")
            channel_title = "rss"
            if channel is not None and channel.findtext("title"):
                channel_title = channel.findtext("title") or "rss"

            for entry in root.findall(".//item"):
                title = (entry.findtext("title") or "").strip()
                summary = (entry.findtext("description") or entry.findtext("summary") or "").strip()
                content = (entry.findtext("content:encoded") or "").strip()
                text = f"{title} {summary} {content}".strip()
                if not self._matches_query(text, query):
                    continue

                url = (entry.findtext("link") or "").strip()
                pub_date = entry.findtext("pubDate") or entry.findtext("updated") or entry.findtext("published")
                items.append(
                    NewsItem(
                        datetime_utc=coerce_datetime(pub_date),
                        source=channel_title,
                        headline=title,
                        url=url,
                        summary=summary,
                        content=content,
                        metadata={"source_api": self.name, "feed_url": feed_url},
                        source_reliability=self.reliability,
                    )
                )
                if len(items) >= limit:
                    return items
        return items


def fuzzy_title_match(left: str, right: str) -> float:
    return SequenceMatcher(None, (left or "").lower(), (right or "").lower()).ratio()


class DuckDuckGoSource(BaseNewsSource):
    """Lightweight DuckDuckGo HTML search adapter.

    Uses the HTML endpoint to fetch organic search results and returns
    NewsItem objects with the extracted headline, snippet and link.
    """
    name = "duckduckgo"
    reliability = 0.65

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()

    def fetch(self, query: str, limit: int = 25) -> list[NewsItem]:
        q = normalize_query(query)
        url = "https://html.duckduckgo.com/html/"
        params = {"q": q}
        items: list[NewsItem] = []
        try:
            resp = request_with_retries(self.session, "get", url, params=params, timeout=15)
            if resp is None:
                return []
            html = resp.text
        except Exception as exc:
            logger.warning("DuckDuckGo fetch failed for %s: %s", query, exc)
            return []

        soup = BeautifulSoup(html, "html.parser")
        results = soup.find_all("div", class_=re.compile(r"result|result__body"))
        for r in results:
            if len(items) >= limit:
                break
            a = r.find("a", href=True)
            if not a:
                continue
            href = a.get("href") or ""
            title = (a.get_text() or "").strip()
            snippet_tag = r.find("a") or r.find("div", class_=re.compile(r"snippet|result__snippet"))
            snippet = (snippet_tag.get_text() or "").strip() if snippet_tag is not None else ""
            items.append(
                NewsItem(
                    datetime_utc=coerce_datetime(pd.Timestamp.utcnow()),
                    source="duckduckgo",
                    headline=title,
                    url=canonicalize_url(href),
                    summary=snippet,
                    content=snippet,
                    language="en",
                    metadata={"source_api": "duckduckgo"},
                    source_reliability=self.reliability,
                )
            )

        return items


class McpDuckDuckGoSource(BaseNewsSource):
    """MCP DuckDuckGo source placeholder.

    Note: The actual MCP DuckDuckGo integration is performed by the MCP agent
    at runtime. This placeholder exists so the pipeline can include the
    source when configured, but it intentionally returns an empty list and
    logs an informational message explaining that the agent handles the
    integration.
    """
    name = "mcp_duckduckgo"
    reliability = 0.85

    def fetch(self, query: str, limit: int = 50) -> list[NewsItem]:
        logger.info(
            "MCP DuckDuckGo fetch requested for query=%s limit=%s: MCP integration is handled by the agent at runtime; returning placeholder empty list.",
            query,
            limit,
        )
        return []
