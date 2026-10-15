"""Cache-backed Binance history, scroll-back loading and per-interval stock limits in DataLoader (offline)."""
import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import core.kline_cache as kc
from core.data_loader import DataLoader

MIN = 60_000


def fake_exchange(calls):
    def fetch(symbol, interval, start_ms, end_ms):
        calls.append((symbol, interval, start_ms, end_ms))
        step = {"1m": MIN, "5m": 5 * MIN, "1h": 60 * MIN, "1d": 1440 * MIN}[interval]
        first = start_ms + (-start_ms) % step
        return [[t, 100.0, 101.0, 99.0, 100.5, 1.0] for t in range(first, end_ms, step)]
    return fetch


@pytest.fixture
def loader():
    return DataLoader.__new__(DataLoader)


def test_binance_history_comes_through_the_cache_and_is_not_refetched(loader, monkeypatch):
    calls = []
    monkeypatch.setattr(kc, "fetch_range_http", fake_exchange(calls))
    a = loader._get_binance_historical("BTCUSDT", 1, "1m")
    assert 1400 <= len(a) <= 1442 and a.index.name == "Datetime" and list(a.columns) == kc.COLUMNS
    n = len(calls)
    b = loader._get_binance_historical("BTCUSDT", 1, "1m")
    assert len(calls) <= n + 1 and abs(len(b) - len(a)) <= 2           # only the newest closed bar / the forming one is asked for again
    assert all(e - s <= 5 * MIN for _, _, s, e in calls[n:])


def test_falls_back_to_ccxt_when_the_cached_path_fails(loader):
    loader._get_binance_historical_ccxt = MagicMock(return_value="from-ccxt")
    # conftest makes the HTTP fetcher raise, i.e. the cached path fails
    assert loader._get_binance_historical("BTCUSDT", 5, "1d") == "from-ccxt"


def test_intervals_binance_rest_does_not_offer_use_ccxt(loader):
    loader._get_binance_historical_ccxt = MagicMock(return_value="from-ccxt")
    assert loader._get_binance_historical("BTCUSDT", 5, "45m") == "from-ccxt"


def test_older_candles_are_the_bars_just_before_the_given_time(loader, monkeypatch):
    calls = []
    monkeypatch.setattr(kc, "fetch_range_http", fake_exchange(calls))
    before = pd.Timestamp("2026-10-10 12:00:00")
    df = loader.get_older_candles("BTCUSDT", "1m", before, bars=1000)
    assert len(df) == 1000 and df.index[-1] == before - pd.Timedelta(minutes=1) and df.index[0] == before - pd.Timedelta(minutes=1000)


def test_older_candles_respect_the_oldest_allowed_floor(loader, monkeypatch):
    monkeypatch.setattr(kc, "fetch_range_http", fake_exchange([]))
    before = pd.Timestamp("2026-10-10 12:00:00")
    df = loader.get_older_candles("BTCUSDT", "1m", before, bars=1000, oldest_allowed=before - pd.Timedelta(minutes=300))
    assert len(df) == 300 and df.index[0] == before - pd.Timedelta(minutes=300)
    assert loader.get_older_candles("BTCUSDT", "1m", before, 1000, oldest_allowed=before).empty        # already at the floor


def test_older_candles_are_crypto_only_and_binance_intervals_only(loader):
    assert loader.get_older_candles("AAPL", "1m", pd.Timestamp("2026-10-10")).empty
    assert loader.get_older_candles("BTCUSDT", "45m", pd.Timestamp("2026-10-10")).empty


# ---------------------------------------------------------------- stocks: ask the source only for history it keeps

@pytest.mark.parametrize("interval,days,expected", [("1m", 365, 7), ("5m", 365, 59), ("15m", 365, 59), ("1h", 365, 365), ("1d", 365, 365)])
def test_stock_requests_are_clamped_to_the_source_limit_before_openbb_is_asked(loader, interval, days, expected):
    loader._get_openbb_historical = MagicMock(return_value="frame")
    assert loader._get_historical_data("AAPL", days, interval) == "frame"
    assert loader._get_openbb_historical.call_args.args[1] == expected


@pytest.mark.parametrize("interval,expected_period", [("5m", "59d"), ("15m", "59d"), ("1h", "365d"), ("1m", "7d")])
def test_yahoo_cap_is_per_interval_not_a_flat_7_days(loader, interval, expected_period):
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    with patch("core.data_loader.download_with_retry", return_value=empty) as dl, \
         patch("core.yahoo_chart.fetch_chart", return_value=empty):
        loader._get_yahoo_historical("AAPL", 365, interval)
    assert dl.call_args.kwargs["period"] == expected_period


def test_the_optional_env_ceiling_still_applies(loader, monkeypatch):
    monkeypatch.setenv("YAHOO_INTRADAY_MAX_DAYS", "3")
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    with patch("core.data_loader.download_with_retry", return_value=empty) as dl, \
         patch("core.yahoo_chart.fetch_chart", return_value=empty):
        loader._get_yahoo_historical("AAPL", 365, "5m")
    assert dl.call_args.kwargs["period"] == "3d"
