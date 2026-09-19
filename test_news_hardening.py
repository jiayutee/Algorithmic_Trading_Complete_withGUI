"""Phase 11.0: news pipeline failure modes found by benchmarking the real sources.

  * one rate-limited source (GDELT ~30 s, RSS 429 storms) held every refresh hostage
  * "BTCUSDT" is a poor search query; "Bitcoin" finds far more relevant news
  * "eth" matched "method" (substring tagging)
  * non-English pages and stock-quote / landing pages were scored as confident 'neutral' news
  * every refresh re-scored headlines it had already scored (LLM call + seconds)
"""
import time
from datetime import datetime, timezone

import pytest

from core.news_health import SourceHealthRegistry
from core.news_pipeline import (
    NewsPipeline, _extract_entities, _is_mostly_latin, _is_non_news_page, _mentions,
    _normalized_headline_key, _query_variants,
)
from core.news_sources import BaseNewsSource, NewsItem
from core.news_store import NewsStore, _headline_hash
from core.sentiment import SentimentResult

NOW = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


def item(headline, url="", source="fake", summary=""):
    return NewsItem(datetime_utc=NOW, source=source, headline=headline, url=url, summary=summary)


class FakeSource(BaseNewsSource):
    def __init__(self, name, items=(), delay=0.0, error=None):
        self.name, self._items, self.delay, self.error, self.calls, self.queries = name, list(items), delay, error, 0, []

    def fetch(self, query, limit=50):
        self.calls += 1
        self.queries.append(query)
        time.sleep(self.delay)
        if self.error:
            raise RuntimeError(self.error)
        return list(self._items)


class FakeAnalyzer:
    """Counts how many texts it was asked to score."""
    def __init__(self, model="deepseek-chat"):
        self.model, self.scored = model, []

    def analyze_many(self, texts):
        texts = list(texts)
        self.scored += texts
        return [SentimentResult(0.7, 0.1, 0.2, "positive", 0.7, self.model) for _ in texts]


def pipeline(sources, tmp_path, analyzer=None, deadline=1.0, health=None):
    return NewsPipeline(sources=sources, sentiment_analyzer=analyzer or FakeAnalyzer(), deadline_seconds=deadline,
                        health=health or SourceHealthRegistry(), store_path=str(tmp_path / "news.sqlite3"))


@pytest.fixture(autouse=True)
def _cwd_for_migrations(monkeypatch):
    import os
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))     # NewsStore reads migrations/ relative to CWD


# ------------------------------------------------------------- circuit breaker

class Clock:
    t = 0.0
    def __call__(self):
        return self.t


def test_circuit_opens_after_repeated_failures_and_recovers_after_the_cooldown():
    clk = Clock()
    h = SourceHealthRegistry(threshold=2, base_cooldown=100, clock=clk)
    assert h.allow("gdelt")
    h.record("gdelt", 0, 0.1, error="boom")
    assert h.allow("gdelt")                                   # one failure is tolerated
    h.record("gdelt", 0, 0.1, error="boom")
    assert not h.allow("gdelt") and h.seconds_until_retry("gdelt") == pytest.approx(100)
    clk.t = 101
    assert h.allow("gdelt")                                   # cool-down over: one trial
    h.record("gdelt", 0, 0.1, error="boom")                   # trial fails -> longer cool-down
    assert not h.allow("gdelt") and h.seconds_until_retry("gdelt") == pytest.approx(200)
    clk.t = 400
    h.record("gdelt", 5, 0.5)                                 # success closes the circuit and resets the backoff
    assert h.allow("gdelt") and h.snapshot()["gdelt"]["consecutive_failures"] == 0


def test_a_fast_empty_answer_is_not_a_failure_but_a_slow_empty_one_is():
    h = SourceHealthRegistry(slow_empty_seconds=3.0)
    assert h.record("a", 0, 0.4) is False                     # "no news for this query"
    assert h.record("a", 0, 6.0) is True                      # swallowed 429/retry storm
    assert h.record("a", 3, 6.0) is False                     # slow but delivered


def test_cooldown_is_capped():
    clk = Clock()
    h = SourceHealthRegistry(threshold=1, base_cooldown=100, max_cooldown=250, clock=clk)
    for _ in range(6):
        h.record("x", 0, 0, error="e")
    assert h.seconds_until_retry("x") == pytest.approx(250)


# ------------------------------------------------------------- deadline budget

def test_a_hanging_source_cannot_hold_up_the_refresh(tmp_path):
    fast = FakeSource("fast", [item("Bitcoin rallies past resistance", "https://a.com/1")])
    slow = FakeSource("gdelt", [item("Late story about Bitcoin", "https://b.com/2")], delay=5.0)
    pipe = pipeline([fast, slow], tmp_path, deadline=0.4)
    t = time.monotonic()
    items = pipe.fetch_news_items("BTCUSDT")
    elapsed = time.monotonic() - t
    assert elapsed < 2.0, f"refresh took {elapsed:.1f}s despite a 0.4s budget"
    assert [i.headline for i in items] == ["Bitcoin rallies past resistance"]
    assert pipe.health.snapshot()["gdelt"]["consecutive_failures"] == 1
    assert "timed out" in pipe.health.snapshot()["gdelt"]["last_reason"]


def test_a_source_that_keeps_failing_is_skipped_on_later_refreshes(tmp_path):
    bad = FakeSource("gdelt", delay=5.0)
    good = FakeSource("brave", [item("Bitcoin ETF inflows hit record", "https://c.com/3")])
    pipe = pipeline([bad, good], tmp_path, deadline=0.3, health=SourceHealthRegistry(threshold=2))
    for _ in range(2):
        pipe.fetch_news_items("BTCUSDT")
    assert bad.calls == 2
    t = time.monotonic()
    pipe.fetch_news_items("BTCUSDT")                           # circuit is open: not called, no waiting
    assert bad.calls == 2 and time.monotonic() - t < 0.2


def test_a_source_that_raises_is_recorded_and_others_still_deliver(tmp_path):
    broken = FakeSource("rss", error="HTTP 429")
    good = FakeSource("brave", [item("Ethereum upgrade goes live", "https://d.com/4")])
    pipe = pipeline([broken, good], tmp_path)
    assert len(pipe.fetch_news_items("ETHUSDT")) == 1
    snap = pipe.health.snapshot()
    assert snap["rss"]["failures"] == 1 and "429" in snap["rss"]["last_reason"]


def test_stragglers_finishing_after_the_deadline_do_not_leak_into_results(tmp_path):
    slow = FakeSource("slow", [item("Stale Bitcoin item", "https://e.com/5")], delay=0.6)
    pipe = pipeline([slow], tmp_path, deadline=0.2)
    assert pipe.fetch_news_items("BTCUSDT") == []
    time.sleep(0.8)                                            # the straggler completes; nothing crashes


# ------------------------------------------------------------ query and tagging

def test_crypto_pairs_are_searched_by_name():
    assert _query_variants("BTCUSDT")[0] == "Bitcoin"
    assert _query_variants("ETHUSDT")[:2] == ["Ethereum", "Ether"]
    assert _query_variants("AAPL")[0] == "AAPL"                # unknown symbols unchanged
    assert _query_variants("AAPL", "Apple Inc")[0] == "Apple Inc"


def test_the_source_actually_receives_the_better_query(tmp_path):
    src = FakeSource("brave", [item("Bitcoin news", "https://f.com/6")])
    pipeline([src], tmp_path).fetch_news_items("BTCUSDT")
    assert src.queries == ["Bitcoin"]


def test_tagging_uses_whole_words_and_aliases():
    assert _extract_entities("Bitcoin rallies 5% today", "BTCUSDT")[0] == ["BTCUSDT"]      # by name
    assert _extract_entities("BTC breaks 100k", "BTCUSDT")[0] == ["BTCUSDT"]
    assert _extract_entities("A new method for ethics review", "ETHUSDT")[0] == []          # was a false positive
    assert _extract_entities("Ethereum gas fees drop", "ETHUSDT")[0] == ["ETHUSDT"]
    assert _extract_entities("Visit Canada for the holidays", "ADAUSDT")[0] == []
    assert _mentions("eth surges", "eth") and not _mentions("method", "eth")


# ------------------------------------------------------------------ content filters

@pytest.mark.parametrize("text,ok", [
    ("Apple beats earnings estimates as iPhone demand rises", True),
    ("एसएलआरको गोलीसहित जडिबुटीबाट चोरीमा संलग्न व्यक्ति पक्राउ", False),     # Nepali
    ("OpenAI和苹果的联盟即将破裂", False),                                        # Chinese
    ("Bitcoin at $64,000: what traders watch next", True),
    ("", True),
])
def test_language_screen(text, ok):
    assert _is_mostly_latin(text) is ok


@pytest.mark.parametrize("headline,is_page", [
    ("Apple (AAPL) Stock Price & Overview", True),
    ("Apple Inc. (AAPL) Stock Price, News, Quote & History - Yahoo Finance", True),
    ("Apple", True),                                            # bare landing page title
    ("AAPL Stock Quote Price and Forecast | CNN", True),
    ("Apple stock rises as AI servers ramp", False),
    ("Tech stocks today: Apple's iPhone 18 Pro goes on sale", False),
])
def test_non_news_pages_are_recognised(headline, is_page):
    assert _is_non_news_page(headline, "AAPL", "Apple") is is_page


def test_only_real_english_news_reaches_the_scorer(tmp_path):
    analyzer = FakeAnalyzer()
    src = FakeSource("ddg", [
        item("Apple (AAPL) Stock Price & Overview", "https://x.com/quote"),
        item("OpenAI和苹果的联盟即将破裂", "https://x.com/cn"),
        item("Apple raises dividend after record quarter", "https://x.com/news1"),
    ])
    out = pipeline([src], tmp_path, analyzer=analyzer).fetch_news_items("AAPL", company_name="Apple")
    assert [i.headline for i in out] == ["Apple raises dividend after record quarter"]
    assert len(analyzer.scored) == 1


# ------------------------------------------------------------------- dedup + cache

def test_same_story_with_a_source_suffix_or_a_new_url_is_scored_once(tmp_path):
    analyzer = FakeAnalyzer()
    src = FakeSource("brave", [
        item("Bitcoin surges past $70,000 - Reuters", "https://a.com/x?utm_source=tw"),
        item("Bitcoin surges past $70,000 | CoinDesk", "https://b.com/y"),
        item("Bitcoin surges past $70,000", "https://a.com/x"),                 # same URL after canonicalisation
        item("Ethereum devs schedule the next upgrade", "https://c.com/z"),
    ])
    out = pipeline([src], tmp_path, analyzer=analyzer).fetch_news_items("BTCUSDT")
    assert len(out) == 2 and len(analyzer.scored) == 2
    assert _normalized_headline_key("Bitcoin surges past $70,000 - Reuters") == _normalized_headline_key("Bitcoin surges past $70,000")


def test_a_refresh_does_not_rescore_headlines_it_already_scored(tmp_path):
    analyzer = FakeAnalyzer()
    src = FakeSource("brave", [item("Bitcoin ETF sees record inflows", "https://a.com/1"),
                               item("Bitcoin miners sell holdings", "https://a.com/2")])
    pipe = pipeline([src], tmp_path, analyzer=analyzer)
    pipe.fetch_news_items("BTCUSDT")
    assert len(analyzer.scored) == 2
    out = pipe.fetch_news_items("BTCUSDT")                      # same headlines again
    assert len(analyzer.scored) == 2                            # nothing new was scored
    assert all(i.sentiment["model_name"] == "deepseek-chat" for i in out)


def test_only_new_headlines_are_scored_on_a_partial_overlap(tmp_path):
    analyzer = FakeAnalyzer()
    src = FakeSource("brave", [item("Bitcoin ETF sees record inflows", "https://a.com/1")])
    pipe = pipeline([src], tmp_path, analyzer=analyzer)
    pipe.fetch_news_items("BTCUSDT")
    src._items.append(item("Bitcoin hashrate hits a new high", "https://a.com/3"))
    pipe.fetch_news_items("BTCUSDT")
    assert len(analyzer.scored) == 2 and "hashrate" in analyzer.scored[-1]


def test_weak_rule_based_scores_are_rescored_and_upgraded_in_the_store(tmp_path):
    weak = FakeAnalyzer(model="rule-based-headline-v1")
    src = FakeSource("brave", [item("Bitcoin ETF sees record inflows", "https://a.com/1")])
    pipe = pipeline([src], tmp_path, analyzer=weak)
    pipe.fetch_news_items("BTCUSDT")
    store = NewsStore(str(tmp_path / "news.sqlite3"))
    assert store.get_cached_sentiments([_headline_hash("Bitcoin ETF sees record inflows")]) == {}   # not trusted
    store.close()

    good = FakeAnalyzer(model="deepseek-chat")
    pipe.sentiment_analyzer = good
    pipe.fetch_news_items("BTCUSDT")
    assert len(good.scored) == 1                               # re-scored by the real analyzer
    store = NewsStore(str(tmp_path / "news.sqlite3"))
    cached = store.get_cached_sentiments([_headline_hash("Bitcoin ETF sees record inflows")])
    store.close()
    assert next(iter(cached.values()))["model_name"] == "deepseek-chat"      # upgraded in place
    pipe.fetch_news_items("BTCUSDT")
    assert len(good.scored) == 1                               # and now served from cache


def test_cache_failure_falls_back_to_scoring(tmp_path, monkeypatch):
    analyzer = FakeAnalyzer()
    src = FakeSource("brave", [item("Bitcoin ETF sees record inflows", "https://a.com/1")])
    pipe = pipeline([src], tmp_path, analyzer=analyzer)
    monkeypatch.setattr(NewsStore, "get_cached_sentiments", lambda self, h: (_ for _ in ()).throw(RuntimeError("db locked")))
    assert len(pipe.fetch_news_items("BTCUSDT")) == 1 and len(analyzer.scored) == 1


def test_pipeline_works_without_a_store(tmp_path, monkeypatch):
    analyzer = FakeAnalyzer()
    src = FakeSource("brave", [item("Bitcoin ETF sees record inflows", "https://a.com/1")])
    pipe = pipeline([src], tmp_path, analyzer=analyzer)
    monkeypatch.setattr(pipe, "_open_store", lambda: None)
    assert len(pipe.fetch_news_items("BTCUSDT")) == 1


def test_no_sources_returns_empty_without_error(tmp_path):
    assert pipeline([], tmp_path).fetch_news_items("BTCUSDT") == []


def test_default_budget_is_six_seconds_and_can_be_overridden(monkeypatch):
    monkeypatch.delenv("NEWS_FETCH_DEADLINE_SECONDS", raising=False)
    assert NewsPipeline(sources=[]).deadline_seconds == 6.0
    monkeypatch.setenv("NEWS_FETCH_DEADLINE_SECONDS", "3")
    assert NewsPipeline(sources=[]).deadline_seconds == 3.0
    assert NewsPipeline(sources=[], deadline_seconds=1.5).deadline_seconds == 1.5
