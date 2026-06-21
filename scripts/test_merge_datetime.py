#!/usr/bin/env python3
"""Test datetime normalization in NewsPipeline.merge_features_into_prices

Creates synthetic price and news data with different datetime precisions
and checks that merge_features_into_prices completes without dtype errors.
"""
import sys
from datetime import datetime, timedelta
import pandas as pd

from core.news_pipeline import NewsPipeline


def make_price_df(start: datetime, periods: int = 5):
    rng = [start + timedelta(days=i) for i in range(periods)]
    # price index with microsecond precision
    idx = pd.to_datetime([int(dt.timestamp() * 1_000_000) for dt in rng], unit="us", utc=True)
    df = pd.DataFrame({"Open": [100 + i for i in range(periods)],
                       "High": [101 + i for i in range(periods)],
                       "Low": [99 + i for i in range(periods)],
                       "Close": [100 + i for i in range(periods)],
                       "Volume": [1000 + i for i in range(periods)]}, index=idx)
    return df


def make_news_df(start: datetime, periods: int = 5):
    rng = [start + timedelta(days=i, hours=1) for i in range(periods)]
    # news datetimes with millisecond precision in a column named 'datetime'
    news_ts = pd.to_datetime([int(dt.timestamp() * 1_000) for dt in rng], unit="ms", utc=True)
    df = pd.DataFrame({"datetime": news_ts,
                       "headline": [f"News {i}" for i in range(periods)],
                       "source": ["test"] * periods})
    return df


def main():
    start = datetime.utcnow() - timedelta(days=10)
    price = make_price_df(start, periods=10)
    news = make_news_df(start + timedelta(days=1), periods=6)

    print(f"price.index.dtype: {price.index.dtype}")
    print(f"news.datetime.dtype: {news['datetime'].dtype}")

    pipeline = NewsPipeline(sources=[])
    try:
        merged = pipeline.merge_features_into_prices(price, news, interval="1d")
        print("merge successful. Merged columns:", list(merged.columns))
        print(merged.head())
    except Exception as e:
        print("merge failed:", e)
        sys.exit(2)


if __name__ == '__main__':
    main()
