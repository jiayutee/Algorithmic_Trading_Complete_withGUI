"""Yahoo fallback symbol mapping and Binance failure logging (paper-execution audit 2026-10-03, findings 2 and 3)."""
import logging
from unittest.mock import MagicMock, patch

import pytest

from core.data_loader import DataLoader


@pytest.mark.parametrize("symbol,expected", [
    ("BTCUSDT", "BTC-USD"), ("ETHUSDT", "ETH-USD"), ("SOLUSDT", "SOL-USD"), ("ADAUSDT", "ADA-USD"),     # the old map
    ("BNBUSDT", "BNB-USD"), ("XRPUSDT", "XRP-USD"), ("DOGEUSDT", "DOGE-USD"), ("LTCUSDT", "LTC-USD"),    # the gap
    ("btcusdt", "BTC-USD"),
])
def test_every_usdt_pair_maps_to_its_yahoo_usd_pair(symbol, expected):
    assert DataLoader._yahoo_crypto_symbol(symbol) == expected


@pytest.mark.parametrize("symbol", ["AAPL", "USDT", "", None, "BTC-USD"])
def test_non_usdt_symbols_pass_through_unchanged(symbol):
    assert DataLoader._yahoo_crypto_symbol(symbol) == (symbol if symbol is not None else "")


def test_fallback_asks_yahoo_for_bnb_usd():
    dl = DataLoader.__new__(DataLoader)
    with patch.object(DataLoader, "_get_yahoo_historical", return_value="frame") as yahoo:
        assert dl._get_yahoo_crypto_historical("BNBUSDT", 30, "1d") == "frame"
    yahoo.assert_called_once_with("BNB-USD", 30, "1d")


def test_binance_batch_failure_logs_the_exception_class_and_symbol(caplog):
    dl = DataLoader.__new__(DataLoader)
    dl.binance_public = MagicMock()
    dl.binance_public.milliseconds.return_value = 1_700_000_000_000
    dl.binance_public.fetch_ohlcv.side_effect = TimeoutError("request timed out after 10s")
    with caplog.at_level(logging.WARNING, logger="AlgoTrading"):
        with pytest.raises(ValueError):
            dl._get_binance_historical_ccxt("BTCUSDT", 5, "1d")
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "BTCUSDT" in text and "TimeoutError" in text and "timed out" in text
