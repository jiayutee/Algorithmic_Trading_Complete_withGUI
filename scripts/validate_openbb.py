"""OpenBB data-pipeline validation (Day 5 T6f — Launch Roadmap: Data Pipeline).

Standalone script (not part of the pytest suite) that hits OpenBB's real
providers over the network to confirm we get usable OHLCV data for the three
symbols called out on the roadmap: AAPL (equity), BTCUSDT (crypto), ETH
(crypto).

Usage:
    ~/miniconda3/bin/python3 scripts/validate_openbb.py

Notes on crypto coverage:
- core/data_loader.py intentionally does NOT route crypto through OpenBB —
  BTCUSDT/ETHUSDT are fetched via Binance CCXT (see _get_binance_historical),
  with OpenBB reserved for equities only. This script separately validates
  OpenBB's own crypto.price.historical endpoint so we have a documented
  answer for "does OpenBB's crypto coverage work, and in what symbol format"
  in case we ever want to use it as an additional crypto fallback.
- OpenBB's yfinance crypto provider expects Yahoo-style tickers, i.e.
  "BTC-USD" / "ETH-USD" — NOT the Binance-style "BTCUSDT"/"ETHUSDT" used
  elsewhere in this codebase. Passing "BTCUSDT" gets mangled by the
  yfinance/openbb-yfinance symbol-normalisation logic into garbage like
  "BTCU-SDT" (404s). Only the dash-qualified "<COIN>-USD" form works.
- Unlike obb.equity.price.historical(), obb.crypto.price.historical()'s
  to_df() returns a plain object-dtype Index of date strings (a "date"
  column that never got promoted to a DatetimeIndex) — callers must
  explicitly `pd.to_datetime(df.index)` before treating it as a
  DatetimeIndex. This script does that coercion and _check_df() verifies
  the result.
- Known upstream flakiness: yfinance's HTTP client (yfinance/data.py)
  picks a `random.choice(USER_AGENTS)` per session; some randomly-picked
  UAs get an immediate 429 from Yahoo while others succeed on the very
  next call with identical parameters. This is unrelated to real request
  volume/throttling — retrying a few times resolves it. The retry helper
  below accounts for this.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta

import pandas as pd


def _retry(fn, attempts: int = 8, delay: float = 1.5):
    """Call fn() up to `attempts` times, treating empty-result exceptions as
    retryable (yfinance's random User-Agent selection makes single calls
    flaky independent of real rate limiting — see module docstring)."""
    last_exc = None
    for i in range(attempts):
        try:
            df = fn()
            if df is not None and not df.empty:
                return df, None
            last_exc = RuntimeError("empty result")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
        time.sleep(delay)
    return None, last_exc


def _check_df(df: pd.DataFrame, label: str) -> bool:
    if df is None or df.empty:
        print(f"  [FAIL] {label}: empty DataFrame")
        return False

    # OpenBB's crypto provider (yfinance) returns a plain object-dtype index
    # of date strings rather than a DatetimeIndex — coerce it here, same as
    # core/data_loader.py._get_openbb_historical does for the equity path.
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        except Exception:
            print(f"  [FAIL] {label}: index is not a DatetimeIndex and could not be coerced ({type(df.index)})")
            return False

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        print(f"  [FAIL] {label}: no numeric OHLCV-like columns found ({list(df.columns)})")
        return False

    start, end = df.index.min(), df.index.max()
    print(f"  [OK]   {label}: {len(df)} rows, {start} -> {end}, columns={list(df.columns)}")
    print(df.tail(2).to_string())
    return True


def validate_equity(symbol: str, days: int = 30) -> bool:
    from openbb import obb

    end = datetime.now()
    start = end - timedelta(days=days)
    print(f"\n=== Equity: {symbol} (obb.equity.price.historical, provider=yfinance) ===")

    def _fetch():
        result = obb.equity.price.historical(
            symbol,
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            interval="1d",
            provider="yfinance",
        )
        return result.to_df()

    df, exc = _retry(_fetch)
    if df is None:
        print(f"  [FAIL] {symbol}: {exc!r}")
        return False
    return _check_df(df, symbol)


def validate_crypto(candidates: list[str], days: int = 30) -> tuple[bool, str | None]:
    from openbb import obb

    end = datetime.now()
    start = end - timedelta(days=days)
    print(f"\n=== Crypto: trying {candidates} (obb.crypto.price.historical, provider=yfinance) ===")
    for sym in candidates:

        def _fetch(sym=sym):
            result = obb.crypto.price.historical(
                sym,
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
                interval="1d",
                provider="yfinance",
            )
            return result.to_df()

        df, exc = _retry(_fetch)
        if df is None:
            print(f"  [FAIL] {sym}: {exc!r}")
            continue
        if _check_df(df, sym):
            return True, sym
    return False, None


def main() -> int:
    try:
        import openbb  # noqa: F401
    except ImportError:
        print("openbb is not installed. Run: pip install openbb openbb-yfinance")
        return 1

    results = {}
    results["AAPL"] = validate_equity("AAPL")
    ok, working_symbol = validate_crypto(["BTCUSDT", "BTC-USD", "BTCUSD"])
    results["BTCUSDT"] = ok
    if working_symbol:
        print(f"\n  -> BTCUSDT resolves via OpenBB/yfinance as symbol: {working_symbol}")

    ok, working_symbol = validate_crypto(["ETH", "ETHUSDT", "ETH-USD", "ETHUSD"])
    results["ETH"] = ok
    if working_symbol:
        print(f"\n  -> ETH resolves via OpenBB/yfinance as symbol: {working_symbol}")

    print("\n=== Summary ===")
    for symbol, ok in results.items():
        print(f"  {symbol}: {'PASS' if ok else 'FAIL'}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
