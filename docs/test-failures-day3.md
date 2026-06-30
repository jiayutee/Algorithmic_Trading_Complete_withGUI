# Test Run Results — Day 3 (2026-06-30)

## Environment

- Python: `/Users/jiayutee/.pyenv/versions/3.11.9/bin/python3` (3.11.9)
- pytest: 9.1.1
- Run command: `python3 -m pytest test_data_loading.py test_brokers.py test_strategies.py test_news_pipeline.py test_news_store.py test_gui.py --tb=short -q --timeout=30`

## Summary

| Result  | Count |
|---------|-------|
| PASSED  | 63    |
| FAILED  | 3     |
| ERRORS  | 0     |
| Total   | 66    |

**All failures are in `test_data_loading.py`.**

---

## Failures

### 1. `test_data_loading.py::test_get_binance_data_1m`

**Error:**
```
AttributeError: 'DataLoader' object has no attribute '_get_binance_data'
```

**Root cause guess:**
The test calls `data_loader._get_binance_data("BTCUSDT", days=1, interval="1m")`, but the method was renamed to `_get_binance_historical` in `core/data_loader.py`. The test was written against an older API. The method `_get_binance_historical` still exists and accepts the same arguments. This is a test/code name mismatch — the tests need updating (or an alias added to `DataLoader`).

**File:** `test_data_loading.py` lines 84–88
**Method expected:** `DataLoader._get_binance_data`
**Method that exists:** `DataLoader._get_binance_historical` (same signature: `symbol`, `days`, `interval`)

---

### 2. `test_data_loading.py::test_get_binance_data_5m`

**Error:**
```
AttributeError: 'DataLoader' object has no attribute '_get_binance_data'
```

**Root cause guess:** Same as #1 — method renamed from `_get_binance_data` to `_get_binance_historical`.

**File:** `test_data_loading.py` lines 90–94

---

### 3. `test_data_loading.py::test_get_binance_data_15m`

**Error:**
```
AttributeError: 'DataLoader' object has no attribute '_get_binance_data'
```

**Root cause guess:** Same as #1 — method renamed from `_get_binance_data` to `_get_binance_historical`.

**File:** `test_data_loading.py` lines 96–100

---

## Fix Required

**Assigned to:** Data loading specialist

**Options:**
1. **Update the 3 failing tests** to call `_get_binance_historical` instead of `_get_binance_data` (simplest fix, no production code change).
2. **Add an alias** `_get_binance_data = _get_binance_historical` in `DataLoader` (keeps old test API without changing tests).

Note: The `_get_binance_historical` method makes **live Binance API calls** via CCXT. The mock `MockBinanceConnector` fixture in the test file patches `BinanceConnector.get_historical_klines` (old python-binance style), not the CCXT-backed `binance_public` connector used in `_get_binance_historical`. After the rename fix, these tests will likely become live integration tests unless additional mocking is added for `self.binance_public.fetch_ohlcv`.

---

## Environment Setup Notes (for future runs)

The system Python `/usr/bin/python3` (3.9.6) has no packages. The following were installed to `/Users/jiayutee/.pyenv/versions/3.11.9` to make tests runnable:

```
pip install pytest pytest-timeout pandas numpy backtrader yfinance ccxt python-binance websocket-client beautifulsoup4 requests PyQt5 PyQtChart
```

Consider adding a `setup-test-env.sh` script or pinning a `.python-version` file pointing to `3.11.9`.
