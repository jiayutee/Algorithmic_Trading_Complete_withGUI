"""
conftest.py — repo-wide pytest fixtures.

Keeps retry-with-backoff code (core/yf_session.py's download_with_retry /
fetch_earnings_dates_with_retry) from introducing real wall-clock sleeps into
the test suite. Tests that exercise the empty/error paths of yfinance calls
(mocked) would otherwise trigger genuine exponential-backoff delays between
retry attempts. Forcing a single attempt means "no second attempt" -> no
sleep, while leaving the retry loop's logic itself under test (it still runs
its empty/exception-handling branch once).

This does not affect production behavior: outside pytest, YF_FETCH_MAX_RETRIES
defaults to 3 as documented in core/yf_session.py.
"""

import pytest


@pytest.fixture(autouse=True)
def _fast_yf_retries(monkeypatch):
    monkeypatch.setenv("YF_FETCH_MAX_RETRIES", "1")
