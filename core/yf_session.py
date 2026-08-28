"""
core/yf_session.py — Shared browser-like requests.Session for yfinance calls,
plus retry-with-backoff wrappers around the two yfinance entry points this
codebase uses (yf.download, Ticker.earnings_dates).

Yahoo Finance's endpoints increasingly rate-limit/block requests that carry
the default python-requests User-Agent (``YFRateLimitError: Too Many
Requests``). The standard workaround is to attach a realistic browser
User-Agent to the HTTP session yfinance uses.

Note: we deliberately do NOT use ``yf.utils.get_default_session()`` — that
helper doesn't exist in the yfinance version pinned in requirements.txt
(<0.2.60, kept there because newer yfinance releases require a curl_cffi
session internally that breaks openbb-yfinance==1.4.2's plain
requests.Session — see the note in requirements.txt). Instead we build a
plain ``requests.Session`` with a spoofed header and pass it explicitly via
the ``session=`` kwarg that ``yf.download()`` / ``yf.Ticker()`` already
accept in this version.

This session is only used by our own direct yfinance calls
(core/data_loader.py, core/backtester.py). It has no effect on OpenBB's
internal HTTP client, which manages its own session for the
openbb-yfinance provider path.

Retry behaviour
----------------
Unlike a plain HTTP 429, yfinance's ``yf.download()`` does NOT raise on
Yahoo's rate limit — it catches ``YFRateLimitError`` internally per-ticker,
prints "N Failed download", and returns an *empty* DataFrame. So the retry
condition here is "result came back empty", not "an exception was raised".
``Ticker.earnings_dates`` can raise directly, so that path retries on
exceptions too. Backoff mirrors the style already used for HTTP fetches in
core/news_sources.py (exponential + jitter), with its own env-var knobs:

- YF_FETCH_MAX_RETRIES (default 3)
- YF_FETCH_BACKOFF_FACTOR (default 1.0, seconds)
"""

import os
import random
import time

import requests
import yfinance as yf

from core.logger import logger

_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_session: "requests.Session | None" = None


def get_yf_session() -> requests.Session:
    """Return a shared requests.Session with a browser-like User-Agent.

    Lazily created and cached at module level so every yfinance call in the
    process reuses one session (and its connection pool) rather than
    spoofing headers on a fresh session per call.
    """
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": _BROWSER_USER_AGENT})
    return _session


def _retry_config():
    try:
        max_retries = int(os.getenv("YF_FETCH_MAX_RETRIES", "3"))
    except Exception:
        max_retries = 3
    try:
        backoff_factor = float(os.getenv("YF_FETCH_BACKOFF_FACTOR", "1.0"))
    except Exception:
        backoff_factor = 1.0
    return max_retries, backoff_factor


def _sleep_for_attempt(attempt: int, backoff_factor: float) -> float:
    sleep_time = backoff_factor * (2 ** (attempt - 1))
    jitter = random.uniform(0, backoff_factor)
    return sleep_time + jitter


def download_with_retry(*args, **kwargs):
    """``yf.download()`` with exponential-backoff retries on rate limiting.

    Retries while the returned DataFrame is empty (yfinance's own signal
    that the last attempt was rate-limited or otherwise failed), up to
    YF_FETCH_MAX_RETRIES attempts. Returns whatever the final attempt
    returned — including an empty DataFrame if every attempt failed, so
    callers keep their existing "empty means no data" handling unchanged.

    ``session`` defaults to :func:`get_yf_session` unless the caller passes
    one explicitly.
    """
    kwargs.setdefault("session", get_yf_session())
    max_retries, backoff_factor = _retry_config()
    label = args[0] if args else kwargs.get("tickers", "?")

    attempt = 0
    df = None
    while attempt < max_retries:
        attempt += 1
        df = yf.download(*args, **kwargs)
        if df is not None and not df.empty:
            return df
        if attempt >= max_retries:
            logger.warning(
                "yf.download(%s) returned empty after %s attempt(s) — "
                "likely still rate-limited; giving up.", label, attempt
            )
            return df
        sleep_time = _sleep_for_attempt(attempt, backoff_factor)
        logger.warning(
            "yf.download(%s) returned empty (attempt %s/%s), possible rate "
            "limit; retrying in %.1fs", label, attempt, max_retries, sleep_time
        )
        time.sleep(sleep_time)
    return df


def fetch_earnings_dates_with_retry(symbol: str):
    """``yf.Ticker(symbol).earnings_dates`` with exponential-backoff retries.

    Retries on both a raised exception and an empty/None result. Returns the
    DataFrame from ``Ticker.earnings_dates`` (may be None or empty if every
    attempt failed), matching the caller's existing empty-check handling.
    """
    max_retries, backoff_factor = _retry_config()

    attempt = 0
    last_exc = None
    while attempt < max_retries:
        attempt += 1
        try:
            ticker = yf.Ticker(symbol, session=get_yf_session())
            df = ticker.earnings_dates
            if df is not None and not df.empty:
                return df
            last_exc = None
        except Exception as exc:  # noqa: BLE001 — yfinance raises assorted types
            last_exc = exc
            df = None

        if attempt >= max_retries:
            if last_exc is not None:
                logger.warning(
                    "Ticker(%s).earnings_dates failed after %s attempt(s): %s",
                    symbol, attempt, last_exc,
                )
            return df

        sleep_time = _sleep_for_attempt(attempt, backoff_factor)
        logger.warning(
            "Ticker(%s).earnings_dates empty/failed (attempt %s/%s); "
            "retrying in %.1fs", symbol, attempt, max_retries, sleep_time
        )
        time.sleep(sleep_time)
    return None
