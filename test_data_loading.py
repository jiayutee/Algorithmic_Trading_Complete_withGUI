import pytest
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from core.data_loader import DataLoader
from core.news_pipeline import NewsPipeline
from core.news_sources import BaseNewsSource, NewsItem


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


# ---------------------------------------------------------------------------
# Task 2: OpenBB equity + Yahoo Finance fallback validation
# ---------------------------------------------------------------------------

def _make_yahoo_df(symbol: str, rows: int = 10) -> pd.DataFrame:
    """Return a minimal synthetic Yahoo-Finance-style DataFrame."""
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=rows, freq="1D", name="Datetime")
    idx = idx.tz_localize(None)  # yfinance returns tz-naive DatetimeIndex
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(rows)],
            "High": [105.0 + i for i in range(rows)],
            "Low":  [95.0  + i for i in range(rows)],
            "Close":[102.0 + i for i in range(rows)],
            "Volume":[1000  + i for i in range(rows)],
        },
        index=idx,
    )


# BLOCKER DOCUMENTATION: OpenBB not installed in this environment.
# When `openbb` is not importable, _get_openbb_historical raises ImportError
# and _get_historical_data automatically falls back to _get_yahoo_historical.
# Install with: pip install openbb openbb-yfinance
def test_openbb_not_installed_falls_back_to_yahoo(data_loader, monkeypatch):
    """Confirm AAPL fetch falls through to Yahoo Finance when OpenBB is absent.

    BLOCKER: openbb package is not installed in this environment.
    The fallback path is tested here by forcing ImportError from _get_openbb_historical
    and confirming _get_historical_data returns a valid Yahoo Finance DataFrame.
    """
    synthetic = _make_yahoo_df("AAPL")

    # Force _get_openbb_historical to raise ImportError (simulates missing package)
    def _openbb_unavailable(symbol, days, interval="1d"):
        raise ImportError("No module named 'openbb'")

    monkeypatch.setattr(data_loader, "_get_openbb_historical", _openbb_unavailable)

    # Mock _get_yahoo_historical so test is offline
    monkeypatch.setattr(data_loader, "_get_yahoo_historical", lambda *args, **kwargs: synthetic)

    df = data_loader._get_historical_data("AAPL", days=30, interval="1d")
    assert not df.empty
    assert df.index.name == "Datetime"
    assert all(col in df.columns for col in ["Open", "High", "Low", "Close", "Volume"])


def test_yahoo_finance_fallback_aapl(data_loader, monkeypatch):
    """Yahoo Finance fallback returns valid OHLCV for AAPL (offline mock)."""
    synthetic = _make_yahoo_df("AAPL")

    # Both OpenBB and yahoo are mocked so no network calls are made
    def _openbb_fail(*args, **kwargs):
        raise Exception("forced OpenBB failure")

    monkeypatch.setattr(data_loader, "_get_openbb_historical", _openbb_fail)
    monkeypatch.setattr(data_loader, "_get_yahoo_historical", lambda *args, **kwargs: synthetic)

    df = data_loader._get_historical_data("AAPL", days=30, interval="1d")
    assert not df.empty
    assert all(col in df.columns for col in ["Open", "High", "Low", "Close", "Volume"])


def test_historical_data_btcusdt_uses_binance_not_openbb(data_loader):
    """BTCUSDT is routed to Binance CCXT, not OpenBB — OpenBB is never called for crypto."""
    # binance_public is already mocked in the data_loader fixture.
    # This test asserts that the Binance path produces valid results without
    # needing OpenBB at all (OpenBB is only for non-crypto symbols).
    df = data_loader._get_historical_data("BTCUSDT", days=5, interval="1d")
    assert not df.empty
    assert df.index.name == "Datetime"
    assert all(col in df.columns for col in ["Open", "High", "Low", "Close", "Volume"])
    # Verify it came from the CCXT mock (values are deterministic)
    assert df["Open"].iloc[0] == pytest.approx(100.0, abs=1)


def test_historical_data_ethusdt_uses_binance_not_openbb(data_loader):
    """ETHUSDT (crypto) is routed to Binance CCXT, confirming OpenBB is not used for crypto."""
    df = data_loader._get_historical_data("ETHUSDT", days=5, interval="1d")
    assert not df.empty
    assert df.index.name == "Datetime"
    assert all(col in df.columns for col in ["Open", "High", "Low", "Close", "Volume"])


def test_news_pipeline_returns_results_for_aapl(monkeypatch):
    """News pipeline returns ≥1 article for AAPL via a DummySource (offline).

    OpenBB news source requires the openbb package; the DummySource here confirms
    the pipeline wiring is correct independent of OpenBB availability.
    """
    dummy_items = [
        NewsItem(
            datetime_utc=datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc),
            source="dummy",
            headline="Apple beats earnings estimates for Q3",
            url="https://example.com/aapl-earnings",
            summary="AAPL posted strong results.",
            source_reliability=0.9,
        )
    ]

    class DummyNewsSource(BaseNewsSource):
        name = "dummy"
        reliability = 0.9

        def fetch(self, query: str, limit: int = 50) -> list[NewsItem]:
            return dummy_items[:limit]

    pipeline = NewsPipeline(sources=[DummyNewsSource()])
    items = pipeline.fetch_news_items("AAPL")
    assert len(items) >= 1
    assert any("apple" in item.headline.lower() or "aapl" in item.headline.lower() for item in items)


def test_news_pipeline_returns_results_for_btcusdt(monkeypatch):
    """News pipeline returns ≥1 article for BTCUSDT (crypto) via a DummySource."""
    dummy_items = [
        NewsItem(
            datetime_utc=datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc),
            source="dummy",
            headline="Bitcoin surges above $70k on ETF inflows",
            url="https://example.com/btc-rally",
            summary="BTCUSDT breaks resistance.",
            source_reliability=0.85,
        )
    ]

    class DummyNewsSource(BaseNewsSource):
        name = "dummy"
        reliability = 0.85

        def fetch(self, query: str, limit: int = 50) -> list[NewsItem]:
            return dummy_items[:limit]

    pipeline = NewsPipeline(sources=[DummyNewsSource()])
    items = pipeline.fetch_news_items("BTCUSDT")
    assert len(items) >= 1


def test_news_pipeline_returns_results_for_eth(monkeypatch):
    """News pipeline returns ≥1 article for ETH (ETHUSDT) via a DummySource."""
    dummy_items = [
        NewsItem(
            datetime_utc=datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc),
            source="dummy",
            headline="Ethereum gas fees drop ahead of network upgrade",
            url="https://example.com/eth-upgrade",
            summary="ETHUSDT network improvement.",
            source_reliability=0.85,
        )
    ]

    class DummyNewsSource(BaseNewsSource):
        name = "dummy"
        reliability = 0.85

        def fetch(self, query: str, limit: int = 50) -> list[NewsItem]:
            return dummy_items[:limit]

    pipeline = NewsPipeline(sources=[DummyNewsSource()])
    items = pipeline.fetch_news_items("ETHUSDT")
    assert len(items) >= 1


def test_openbb_equity_blocker_documented():
    """Explicit blocker test: openbb package is NOT installed in this environment.

    Impact:
    - obb.equity.price.historical() is unavailable for AAPL, BTCUSDT, ETH.
    - _get_historical_data() for stock symbols (AAPL) falls back to Yahoo Finance.
    - Crypto symbols (BTCUSDT, ETHUSDT) are unaffected — they use Binance CCXT.
    - obb.news.company() for news is unavailable (OpenBBNewsSource returns empty).

    Resolution: pip install openbb openbb-yfinance (requires Python >=3.9, ~500MB).
    No API key required for the default yfinance provider.
    """
    try:
        import openbb  # noqa: F401
        pytest.skip("openbb IS installed — blocker resolved, remove this test")
    except ImportError:
        pass  # Expected: openbb not installed — blocker confirmed
