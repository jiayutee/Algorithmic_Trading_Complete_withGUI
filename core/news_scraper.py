from __future__ import annotations

import pandas as pd

from core.news_pipeline import get_default_news_pipeline


def scrape_and_analyze_finviz_news(ticker: str, company_name: str | None = None, limit: int = 50) -> pd.DataFrame:
    """Backward-compatible facade that now fetches and enriches news from multiple sources."""
    pipeline = get_default_news_pipeline()
    return pipeline.fetch_news_dataframe(symbol=ticker, company_name=company_name, limit=limit)


if __name__ == "__main__":
    ticker_symbol = "TSLA"
    news_with_sentiment = scrape_and_analyze_finviz_news(ticker_symbol)

    if not news_with_sentiment.empty:
        print(news_with_sentiment.head())
    else:
        print(f"No news or analysis found for {ticker_symbol}")
