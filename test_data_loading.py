import pytest
import pandas as pd
from datetime import datetime, timedelta
from core.data_loader import DataLoader
from brokers.binance_connector import BinanceConnector

# Mocking BinanceConnector for isolated testing
class MockBinanceConnector:
    def get_historical_klines(self, symbol, interval, start_str, end_str=None):
        # Simulate data for testing
        if interval == '1m':
            freq = 'T'
        elif interval == '5m':
            freq = '5T'
        elif interval == '15m':
            freq = '15T'
        elif interval == '1h':
            freq = 'H'
        elif interval == '1d':
            freq = 'D'
        else:
            raise ValueError("Unsupported interval for mock data")

        start_date = datetime.strptime(start_str, '%Y-%m-%d')
        end_date = datetime.now() if end_str is None else datetime.strptime(end_str, '%Y-%m-%d')

        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        df = pd.DataFrame({
            'Open': [100 + i for i in range(len(date_range))],
            'High': [105 + i for i in range(len(date_range))],
            'Low': [95 + i for i in range(len(date_range))],
            'Close': [102 + i for i in range(len(date_range))],
            'Volume': [1000 + i for i in range(len(date_range))]
        }, index=date_range)
        df.index.name = 'Datetime'
        return df

@pytest.fixture
def data_loader():
    # Initialize DataLoader with dummy keys for testing purposes
    # In a real scenario, you might mock the external API calls
    return DataLoader(live_api_key="test_key", live_secret_key="test_secret",
                      kucoin_key="test_kucoin_key", kucoin_secret="test_kucoin_secret",
                      binance_key="test_binance_key", binance_secret="test_binance_secret")

@pytest.fixture(autouse=True)
def mock_binance_connector(monkeypatch):
    # This fixture will replace the actual BinanceConnector with our mock version
    def mock_init(self, api_key, secret_key, paper=True):
        self.client = None # No actual client needed for mock
    monkeypatch.setattr(BinanceConnector, '__init__', mock_init)
    monkeypatch.setattr(BinanceConnector, 'get_historical_klines', MockBinanceConnector().get_historical_klines)


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
    df = data_loader._get_binance_data("BTCUSDT", days=1, interval="1m")
    assert not df.empty
    assert "Datetime" == df.index.name
    assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])

def test_get_binance_data_5m(data_loader):
    df = data_loader._get_binance_data("BTCUSDT", days=5, interval="5m")
    assert not df.empty
    assert "Datetime" == df.index.name
    assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])

def test_get_binance_data_15m(data_loader):
    df = data_loader._get_binance_data("BTCUSDT", days=15, interval="15m")
    assert not df.empty
    assert "Datetime" == df.index.name
    assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
