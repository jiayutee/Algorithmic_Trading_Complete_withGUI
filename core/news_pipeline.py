from __future__ import annotations

from functools import lru_cache
from typing import Any
import os
import re
import threading
import time

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
    OpenBBNewsSource,
)
from core.sentiment import SentimentAnalyzer
from core.news_health import HEALTH, SourceHealthRegistry
from core.news_store import NewsStore, _headline_hash


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
    "news_sentiment",  # mean(positive - negative) for news in bar window; 0.0 = no news / neutral
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


# Search engines and news sites say "Bitcoin", not "BTCUSDT". Searching the exchange pair
# returned mostly irrelevant exchange-contract pages (5 of 25 useful vs 20 of 25 for "Bitcoin").
SYMBOL_ALIASES: dict[str, list[str]] = {
    "BTC": ["Bitcoin"],
    "ETH": ["Ethereum", "Ether"],
    "SOL": ["Solana"],
    "BNB": ["Binance Coin", "BNB Chain"],
    "XRP": ["XRP", "Ripple"],
    "ADA": ["Cardano"],
    "DOGE": ["Dogecoin"],
    "LTC": ["Litecoin"],
    "AVAX": ["Avalanche"],
    "DOT": ["Polkadot"],
    "LINK": ["Chainlink"],
}


def _base_ticker(symbol: str) -> str:
    upper = (symbol or "").upper().replace("-", "").replace("/", "")
    for quote in ("USDT", "USDC", "USD", "BUSD"):
        if upper.endswith(quote) and len(upper) > len(quote):
            return upper[: -len(quote)]
    return upper


def _symbol_aliases(symbol: str) -> list[str]:
    return SYMBOL_ALIASES.get(_base_ticker(symbol), [])


def _query_variants(symbol: str, company_name: str | None = None) -> list[str]:
    variants = []
    if company_name:
        variants.append(company_name)
    variants.extend(_symbol_aliases(symbol))          # "Bitcoin" before "BTCUSDT"
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


def _mentions(text_lower: str, term: str) -> bool:
    """Whole-word match, so "eth" no longer matches "method" and "ada" no longer matches "canada"."""
    term = (term or "").lower().strip()
    return bool(term) and re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text_lower) is not None


# --- content filters ------------------------------------------------------------------
_NON_NEWS_PATTERNS = re.compile(
    r"(stock price|share price|stock quote|price,? (quote|news|chart)|quote (&|and) (news|chart)|"
    r"official (web)?site|homepage|home page|overview\b|stock chart|live price|price (today|live))",
    re.I,
)


def _is_mostly_latin(text: str, threshold: float = 0.85) -> bool:
    """Cheap language screen: keep text whose letters are mostly ASCII/Latin (drops CJK, Devanagari, Cyrillic...)."""
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return True
    return sum(ord(c) < 0x250 for c in letters) / len(letters) >= threshold


def _is_non_news_page(headline: str, symbol: str, company_name: str | None) -> bool:
    """Search engines return quote pages and company home pages alongside news; they carry no
    sentiment and dilute the features (they were being scored as confident 'neutral' news)."""
    h = (headline or "").strip()
    if _NON_NEWS_PATTERNS.search(h):
        return True
    bare = re.sub(r"[^a-z0-9 ]", "", h.lower()).strip()
    names = {bare_ for bare_ in [re.sub(r"[^a-z0-9 ]", "", n.lower()).strip()
                                 for n in [symbol, company_name or "", _base_ticker(symbol), *_symbol_aliases(symbol)]] if bare_}
    return bare in names                                  # a title that is only the name: a landing page


def _normalized_headline_key(headline: str) -> str:
    h = re.sub(r"\s+[-|\u2013\u2014]\s+[^-|\u2013\u2014]{2,40}$", "", headline or "")   # strip trailing " - Source"
    return re.sub(r"[^a-z0-9 ]", "", h.lower()).strip()


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

    symbol_variants = {symbol.upper(), symbol.upper().replace("-", ""), _base_ticker(symbol), *_symbol_aliases(symbol)}
    if any(_mentions(text_lower, variant) for variant in symbol_variants if variant):
        tickers.append(symbol.upper())
        entities.append({"text": symbol.upper(), "type": "TICKER", "confidence": 1.0})

    if company_name:
        company_tokens = [token for token in re.findall(r"[A-Za-z0-9]+", company_name.lower()) if token]
        if company_tokens and all(_mentions(text_lower, token) for token in company_tokens):
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
        deadline_seconds: float | None = None,
        health: SourceHealthRegistry | None = None,
        store_path: str | None = None,
    ):
        self.sources = sources or []
        self.sentiment_analyzer = sentiment_analyzer or SentimentAnalyzer()
        self.max_workers = max_workers
        # One slow or rate-limited source must not hold up the rest: sources still running when
        # the budget expires are abandoned (their thread is a daemon) and counted as failures.
        self.deadline_seconds = float(
            deadline_seconds if deadline_seconds is not None else os.getenv("NEWS_FETCH_DEADLINE_SECONDS", "6")
        )
        self.health = health or HEALTH
        self.store_path = store_path

    def _open_store(self) -> NewsStore | None:
        try:
            return NewsStore(self.store_path) if self.store_path else NewsStore()
        except Exception as exc:  # noqa: BLE001 -- the store is an optimisation, never a hard dependency
            logger.warning("News store unavailable: %s", exc)
            return None

    @classmethod
    def from_env(cls) -> "NewsPipeline":
        sources: list[BaseNewsSource] = []

        # Brave Search (best quality, needs API key)
        brave_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip() or os.getenv("BRAVE_API_KEY", "").strip()
        if brave_key:
            sources.append(BraveSearchSource(api_key=brave_key))
            logger.info("News source: Brave Search enabled")

        # NewsAPI (needs API key from newsapi.org)
        newsapi_key = os.getenv("NEWSAPI_API_KEY", "").strip()
        if newsapi_key:
            sources.append(NewsApiSource(api_key=newsapi_key))
            logger.info("News source: NewsAPI enabled")

        # RSS feeds (comma or newline separated URLs in RSS_FEEDS env var)
        rss_feeds_env = (
            os.getenv("RSS_FEEDS", "").strip()
            or os.getenv("RSS_FEED", "").strip()  # legacy single-feed key
        )
        rss_feeds = [feed.strip() for feed in re.split(r"[\n,]", rss_feeds_env) if feed.strip()]
        if rss_feeds:
            sources.append(RssSource(feed_urls=rss_feeds))
            logger.info("News source: RSS enabled (%d feed(s))", len(rss_feeds))

        # DuckDuckGo HTML scrape (no key needed, always added as fallback)
        sources.append(DuckDuckGoSource())
        logger.info("News source: DuckDuckGo HTML enabled")

        # OpenBB news (no key needed for yfinance provider; set OPENBB_NEWS_PROVIDER
        # in .env for paid providers like "benzinga" or "biztoc")
        # BACKUP NOTE: original sources above remain active — OpenBB is additive.
        openbb_provider = os.getenv("OPENBB_NEWS_PROVIDER", "yfinance").strip()
        try:
            from openbb import obb  # noqa: F401 — just check it's importable
            sources.append(OpenBBNewsSource(provider=openbb_provider))
            logger.info("News source: OpenBB enabled (provider=%s)", openbb_provider)
        except ImportError:
            logger.warning("OpenBB not installed — skipping (pip install openbb openbb-yfinance)")

        # GDELT (no key needed, rate-limited to 1 req/6s)
        sources.append(GDELTSource())
        logger.info("News source: GDELT enabled")

        # EventRegistry (optional paid API)
        eventregistry_key = os.getenv("EVENTREGISTRY_API_KEY", "").strip()
        if eventregistry_key:
            sources.append(EventRegistrySource(api_key=eventregistry_key))
            logger.info("News source: EventRegistry enabled")

        if not sources:
            logger.warning("No news sources configured — set BRAVE_SEARCH_API_KEY or NEWSAPI_API_KEY in .env")

        return cls(sources=sources)

    def _fetch_all_sources(self, query: str, limit: int) -> list[NewsItem]:
        """Run every healthy source in parallel under one time budget."""
        active = []
        for source in self.sources:
            if self.health.allow(source.name):
                active.append(source)
            else:
                logger.info("News source %s skipped for %.0fs more (circuit open after repeated failures)",
                            source.name, self.health.seconds_until_retry(source.name))
        if not active:
            return []

        lock = threading.Lock()
        results: dict[str, tuple[list[NewsItem], float, str]] = {}
        closed = False

        def worker(source: BaseNewsSource) -> None:
            t0 = time.monotonic()
            try:
                items, err = list(source.fetch(query, limit) or []), ""
            except Exception as exc:  # noqa: BLE001
                items, err = [], f"{type(exc).__name__}: {exc}"[:120]
            with lock:
                if not closed:                       # ignore stragglers that finish after the deadline
                    results[source.name] = (items, time.monotonic() - t0, err)

        threads = [threading.Thread(target=worker, args=(src,), daemon=True, name=f"news-{src.name}") for src in active]
        started = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join(max(0.0, self.deadline_seconds - (time.monotonic() - started)))
        with lock:
            closed = True
            done = dict(results)

        gathered: list[NewsItem] = []
        summary = []
        for source in active:
            if source.name in done:
                items, seconds, err = done[source.name]
                failed = self.health.record(source.name, len(items), seconds, err)
                gathered.extend(items)
                summary.append(f"{source.name}={len(items)}{'!' if failed else ''}/{seconds:.1f}s")
            else:
                self.health.record_timeout(source.name, self.deadline_seconds)
                summary.append(f"{source.name}=TIMEOUT")
        logger.info("News fetch %r: %s (budget %.0fs, %.1fs used)", query, ", ".join(summary),
                    self.deadline_seconds, time.monotonic() - started)
        return gathered

    def fetch_news_items(self, symbol: str, company_name: str | None = None, limit: int = 50) -> list[NewsItem]:
        query_variants = _query_variants(symbol, company_name)

        if not self.sources:
            logger.warning("No news sources configured. Returning an empty result set.")
            return []

        gathered = self._fetch_all_sources(query_variants[0], limit)
        store = self._open_store()
        try:
            items = self._enrich_and_deduplicate(gathered, symbol=symbol, company_name=company_name, store=store)

            # Persist deduplicated/enriched items to local news store (non-fatal)
            if store is not None:
                try:
                    inserted = store.add_items(items)
                    store.upgrade_sentiments({_headline_hash(i.headline): i.sentiment for i in items if i.sentiment})
                    if inserted:
                        logger.info("Persisted %s new news items for %s", inserted, symbol)
                except Exception as exc:  # pragma: no cover - do not fail fetch on persistence errors
                    logger.warning("Failed to persist news items: %s", exc)
        finally:
            if store is not None:
                store.close()

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
        # news_sentiment: mean sentiment balance (positive − negative) for all news items
        # published within the resampled bar window.  Range [−1, 1].  Matches sentiment_balance
        # by construction; named separately so strategy code has a single canonical column to read.
        aggregated["news_sentiment"] = aggregated["sentiment_balance"]
        return aggregated.fillna(0.0)

    def merge_features_into_prices(self, price_df: pd.DataFrame, news_df: pd.DataFrame, interval: str = "1D") -> pd.DataFrame:
        """Merge time-aligned news sentiment features into a price OHLCV DataFrame.

        Join strategy (``merge_asof`` backward carry-forward):
          1. News items are resampled into time buckets matching ``interval``
             (e.g. "1D" → daily, "1h" → hourly).  Sentiment scores are aggregated
             as the **mean** over all news items in each bucket.
          2. For each price bar, ``pd.merge_asof`` with ``direction="backward"``
             attaches the most-recent news bucket whose timestamp is ≤ the bar's
             timestamp.  This preserves causal order — no future news leaks in.
          3. Price bars that precede *any* news bucket receive 0.0 for all numeric
             sentiment columns (interpreted as "no news / neutral sentiment").

        Added column ``news_sentiment``:
            Mean sentiment balance (positive − negative) for all news items
            published within the resampled bar window.  Range [−1, 1].
            0.0 indicates either no news or a perfectly balanced sentiment mix.
            When ``news_df`` is empty the column is set to 0.0 for every bar.
        """
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

    def _prefilter(self, items: list[NewsItem], symbol: str, company_name: str | None) -> list[NewsItem]:
        """Drop what should never be scored: non-Latin-script text, quote/landing pages, and
        near-identical headlines (same story, different URL) -- before paying to score them."""
        kept, seen_keys, seen_urls = [], set(), set()
        dropped = {"language": 0, "not_news": 0, "duplicate": 0}
        for item in items:
            headline = item.headline or ""
            if not _is_mostly_latin(headline + " " + (item.summary or "")):
                dropped["language"] += 1
                continue
            if _is_non_news_page(headline, symbol, company_name):
                dropped["not_news"] += 1
                continue
            url, key = canonicalize_url(item.url), _normalized_headline_key(headline)
            if (url and url in seen_urls) or (key and key in seen_keys):
                dropped["duplicate"] += 1
                continue
            if url:
                seen_urls.add(url)
            if key:
                seen_keys.add(key)
            kept.append(item)
        if any(dropped.values()):
            logger.info("News prefilter for %s: kept %d/%d, dropped %s", symbol, len(kept), len(items), dropped)
        return kept

    def _score_with_cache(self, texts: list[str], headlines: list[str], store: NewsStore | None) -> list:
        """Sentiment for each text, re-using an earlier good score for the same headline.

        A refresh re-fetches mostly the same headlines; scoring them again costs an LLM call and
        seconds. Cached scores from the weak rule-based fallback are NOT trusted -- those get re-scored.
        """
        cached: dict[str, dict] = {}
        if store is not None:
            try:
                cached = store.get_cached_sentiments([_headline_hash(h) for h in headlines])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Sentiment cache lookup failed: %s", exc)
        from core.sentiment import SentimentResult
        out: list = [None] * len(texts)
        todo = []
        for i, h in enumerate(headlines):
            c = cached.get(_headline_hash(h))
            if c:
                out[i] = SentimentResult(positive=float(c["positive"]), negative=float(c["negative"]),
                                         neutral=float(c["neutral"]), label=str(c["label"]),
                                         confidence=float(c["confidence"]), model_name=str(c["model_name"]))
            else:
                todo.append(i)
        if todo:
            fresh = self.sentiment_analyzer.analyze_many([texts[i] for i in todo])
            for i, r in zip(todo, fresh):
                out[i] = r
        if len(todo) != len(texts):
            logger.info("Sentiment: %d/%d headlines served from cache, %d scored", len(texts) - len(todo), len(texts), len(todo))
        return out

    def _enrich_and_deduplicate(self, items: list[NewsItem], symbol: str, company_name: str | None = None,
                                store: NewsStore | None = None) -> list[NewsItem]:
        if not items:
            return []
        items = self._prefilter(items, symbol, company_name)
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
        sentiment_results = self._score_with_cache(sentiment_texts, [item.headline for item in scored_items], store)

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
