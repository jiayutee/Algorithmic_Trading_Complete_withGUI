#!/usr/bin/env python3
"""
Diagnostic script — tests Brave Search and DuckDuckGo news sources.

Run with:
    conda activate myenv
    python scripts/test_news_sources.py

Optional env vars (load from .env automatically):
    BRAVE_SEARCH_API_KEY  or  BRAVE_API_KEY
"""
import sys
import os

# Allow running from the scripts/ directory or from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

QUERY = "Apple stock earnings"
LIMIT = 3

# ── colour helpers ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):  print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg):print(f"  {RED}✗{RESET} {msg}")
def info(msg):print(f"  {CYAN}→{RESET} {msg}")
def warn(msg):print(f"  {YELLOW}⚠{RESET} {msg}")
def header(t): print(f"\n{BOLD}{t}{RESET}\n" + "─" * len(t))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Brave Search
# ─────────────────────────────────────────────────────────────────────────────
def test_brave():
    header("Brave Search API")

    api_key = (
        os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
        or os.getenv("BRAVE_API_KEY", "").strip()
    )

    if not api_key:
        fail("No BRAVE_SEARCH_API_KEY / BRAVE_API_KEY set in environment or .env")
        warn("Add it to your .env:  BRAVE_SEARCH_API_KEY=your_key_here")
        warn("Free tier: https://api.search.brave.com  (2 000 req/month)")
        return False

    info(f"Key found: {api_key[:6]}…{api_key[-4:]}")

    try:
        from core.news_sources import BraveSearchSource
        source = BraveSearchSource(api_key=api_key)

        info(f"Fetching up to {LIMIT} results for: \"{QUERY}\"")
        items = source.fetch(QUERY, limit=LIMIT)
    except Exception as e:
        fail(f"Exception during fetch: {e}")
        return False

    if not items:
        fail("Fetch returned 0 items — key may be invalid or rate-limited")
        return False

    ok(f"Got {len(items)} result(s)")
    for i, item in enumerate(items, 1):
        print(f"\n  [{i}]  {CYAN}{item.headline[:80]}{RESET}")
        print(f"       URL:     {item.url[:80]}")
        print(f"       Source:  {item.source}")
        print(f"       Time:    {item.datetime_utc}")
        print(f"       Sentiment label will be filled by pipeline")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 2. DuckDuckGo (no API key needed)
# ─────────────────────────────────────────────────────────────────────────────
def test_duckduckgo():
    header("DuckDuckGo (HTML scrape — no key needed)")

    try:
        from core.news_sources import DuckDuckGoSource
        source = DuckDuckGoSource()

        info(f"Fetching up to {LIMIT} results for: \"{QUERY}\"")
        items = source.fetch(QUERY, limit=LIMIT)
    except Exception as e:
        fail(f"Exception during fetch: {e}")
        return False

    if not items:
        fail("Fetch returned 0 items — DuckDuckGo may have changed its HTML structure or rate-limited this IP")
        warn("DDG HTML scraping is fragile; consider Brave or NewsAPI for production use")
        return False

    ok(f"Got {len(items)} result(s)")
    for i, item in enumerate(items, 1):
        print(f"\n  [{i}]  {CYAN}{item.headline[:80]}{RESET}")
        print(f"       URL:     {item.url[:80]}")
        print(f"       Snippet: {item.summary[:80]}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 3. End-to-end through NewsPipeline (uses whichever sources are configured)
# ─────────────────────────────────────────────────────────────────────────────
def test_pipeline():
    header("NewsPipeline end-to-end (AAPL)")

    try:
        from core.news_pipeline import NewsPipeline
        from core.sentiment import SentimentAnalyzer

        pipeline = NewsPipeline.from_env()
        # Override analyzer to avoid requiring FinBERT model download
        pipeline.sentiment_analyzer = SentimentAnalyzer(force_rule_based=True)
        info(f"Pipeline has {len(pipeline.sources)} source(s): {[s.name for s in pipeline.sources]}")
        info("Running pipeline.fetch_news_dataframe(\"AAPL\") …")
        df = pipeline.fetch_news_dataframe("AAPL", company_name="Apple")
    except Exception as e:
        fail(f"Pipeline raised: {e}")
        return False

    if df.empty:
        warn("Pipeline returned empty DataFrame — all sources may be unconfigured or offline")
        return False

    ok(f"Pipeline returned {len(df)} article(s)")
    print()
    for _, row in df.head(LIMIT).iterrows():
        label = row.get("sentiment_label", "?")
        score = row.get("sentiment_confidence", 0.0)
        event = row.get("event_type", "?")
        print(f"  • {CYAN}{str(row.get('headline',''))[:70]}{RESET}")
        print(f"    sentiment={label} ({score:.0%})  event={event}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{BOLD}News Source Diagnostics{RESET}")
    print("=" * 40)

    results = {
        "Brave Search": test_brave(),
        "DuckDuckGo":   test_duckduckgo(),
        "NewsPipeline": test_pipeline(),
    }

    header("Summary")
    for name, passed in results.items():
        if passed:
            ok(f"{name}: working")
        else:
            fail(f"{name}: failed or not configured")

    print()
    if all(results.values()):
        print(f"{GREEN}{BOLD}All sources operational.{RESET}")
    elif any(results.values()):
        print(f"{YELLOW}{BOLD}Partial — at least one source is working.{RESET}")
    else:
        print(f"{RED}{BOLD}No sources returned results. Check your .env and network.{RESET}")
    print()


if __name__ == "__main__":
    main()
