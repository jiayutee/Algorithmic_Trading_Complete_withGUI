import pytest
import pandas as pd
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock
from core.data_loader import DataLoader


def _make_ccxt_ohlcv(days: int, interval: str) -> list:
    """Return synthetic CCXT-format ohlcv: [[timestamp_ms, o, h, l, c, v], ...]"""
    freq_map = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600, '1d': 86400}
    step_s = freq_map.get(interval, 60)
    end_ts = int(time.time()) * 1000
    start_ts = end_ts - days * 86400 * 1000
    rows = []
    ts = start_ts
    i = 0
    while ts <= end_ts:
        rows.append([ts, 100 + i, 105 + i, 95 + i, 102 + i, 1000 + i])
        ts += step_s * 1000
        i += 1
    return rows


def _make_ccxt_ohlcv_since(since_ms: int, interval: str) -> list:
    """Return synthetic CCXT-format ohlcv starting from since_ms up to now.

    Unlike _make_ccxt_ohlcv, this function is anchored to an explicit start
    timestamp so that paginated calls with an advancing `since` value correctly
    return fewer and fewer rows until the loop terminates.
    """
    freq_map = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600, '1d': 86400}
    step_ms = freq_map.get(interval, 60) * 1000
    end_ts = int(time.time()) * 1000
    rows = []
    ts = since_ms
    i = 0
    while ts <= end_ts:
        rows.append([ts, 100 + i, 105 + i, 95 + i, 102 + i, 1000 + i])
        ts += step_ms
        i += 1
    return rows


@pytest.fixture
def data_loader(monkeypatch):
    loader = DataLoader(live_api_key="test_key", live_secret_key="test_secret",
                        kucoin_key="test_kucoin_key", kucoin_secret="test_kucoin_secret",
                        binance_key="test_binance_key", binance_secret="test_binance_secret")

    # Patch the public ccxt.binance exchange used by _get_binance_historical.
    # The mock must correctly terminate pagination: when `since` advances near
    # "now", only a small slice (possibly 0 rows) should be returned so the
    # while-True loop in _get_binance_historical breaks on `len(ohlcv) < limit`.
    mock_exchange = MagicMock()
    mock_exchange.milliseconds.return_value = int(time.time() * 1000)
    mock_exchange.rateLimit = 0

    def _fake_fetch_ohlcv(symbol, timeframe="1d", since=None, limit=1000):
        start = since if since is not None else (int(time.time() * 1000) - 86400 * 1000)
        rows = _make_ccxt_ohlcv_since(start, timeframe)
        return rows[:limit]

    mock_exchange.fetch_ohlcv.side_effect = _fake_fetch_ohlcv
    mock_exchange.fetch_ticker.return_value = {"last": 50000.0}
    loader.binance_public = mock_exchange
    return loader


def test_get_historical_data_yahoo_1m(data_loader):
    # This test might fail if yfinance doesn't have 1m data for the symbol/period
    # It's more of an integration test with yfinance
    try:
        df = data_loader._get_historical_data("AAPL", days=7, interval="1m")
        assert not df.empty
        assert "Datetime" == df.index.name
        assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
    except ValueError as e:
        pytest.skip(f"Skipping Yahoo 1m data test due to yfinance limitation: {e}")

def test_get_historical_data_yahoo_5m(data_loader):
    try:
        df = data_loader._get_historical_data("AAPL", days=60, interval="5m")
        assert not df.empty
        assert "Datetime" == df.index.name
        assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
    except ValueError as e:
        pytest.skip(f"Skipping Yahoo 5m data test due to yfinance limitation: {e}")

def test_get_historical_data_yahoo_15m(data_loader):
    try:
        df = data_loader._get_historical_data("AAPL", days=60, interval="15m")
        assert not df.empty
        assert "Datetime" == df.index.name
        assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
    except ValueError as e:
        pytest.skip(f"Skipping Yahoo 15m data test due to yfinance limitation: {e}")

def test_get_binance_data_1m(data_loader):
    df = data_loader._get_binance_historical("BTCUSDT", days=1, interval="1m")
    assert not df.empty
    assert "Datetime" == df.index.name
    assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])

def test_get_binance_data_5m(data_loader):
    df = data_loader._get_binance_historical("BTCUSDT", days=5, interval="5m")
    assert not df.empty
    assert "Datetime" == df.index.name
    assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])

def test_get_binance_data_15m(data_loader):
    df = data_loader._get_binance_historical("BTCUSDT", days=15, interval="15m")
    assert not df.empty
    assert "Datetime" == df.index.name
    assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
