import pytest

from core import kline_cache as kc
from core.kline_cache import fetch_range_http as real_fetch_range_http   # the repo-wide offline fixture swaps the module attribute

MIN = 60_000


class FakeExchange:
    """Serves 1-minute candles for every minute since t=0; counts what was asked of it."""
    def __init__(self, gaps=()):
        self.calls, self.gaps = [], set(gaps)

    def __call__(self, symbol, interval, start_ms, end_ms):
        self.calls.append((interval, start_ms, end_ms))
        return [[t, 100 + t / MIN, 101 + t / MIN, 99 + t / MIN, 100.5 + t / MIN, 1.0]
                for t in range(start_ms - start_ms % MIN, end_ms, MIN) if t >= start_ms and t not in self.gaps]


@pytest.fixture
def cache(tmp_path):
    return kc.KlineCache(str(tmp_path / "k.sqlite3"))


NOW = 10_000 * MIN + 30_000                        # 30 s into the minute whose candle is still forming


def test_first_request_fetches_and_second_reads_the_file(cache):
    ex = FakeExchange()
    a = kc.get_klines("BTCUSDT", "1m", 9_000 * MIN, 9_500 * MIN, cache=cache, fetch=ex, now_ms=NOW)
    assert len(a) == 500 and a.index.name == "Datetime" and list(a.columns) == kc.COLUMNS
    assert len(ex.calls) == 1
    b = kc.get_klines("BTCUSDT", "1m", 9_000 * MIN, 9_500 * MIN, cache=cache, fetch=ex, now_ms=NOW)
    assert len(ex.calls) == 1 and b.equals(a)                              # nothing re-downloaded


def test_a_wider_request_fetches_only_the_missing_older_and_newer_parts(cache):
    ex = FakeExchange()
    kc.get_klines("BTCUSDT", "1m", 9_000 * MIN, 9_500 * MIN, cache=cache, fetch=ex, now_ms=NOW)
    ex.calls.clear()
    df = kc.get_klines("BTCUSDT", "1m", 8_800 * MIN, 9_700 * MIN, cache=cache, fetch=ex, now_ms=NOW)
    assert sorted((s, e) for _, s, e in ex.calls) == [(8_800 * MIN, 9_000 * MIN), (9_500 * MIN, 9_700 * MIN)]
    assert len(df) == 900 and cache.coverage("BTCUSDT", "1m") == (8_800 * MIN, 9_700 * MIN)


def test_the_forming_candle_is_returned_but_never_stored(cache):
    ex = FakeExchange()
    df = kc.get_klines("BTCUSDT", "1m", 9_990 * MIN, 10_001 * MIN, cache=cache, fetch=ex, now_ms=NOW)
    assert df.index[-1] == __import__("pandas").to_datetime(10_000 * MIN, unit="ms")        # the forming bar is on the chart
    assert cache.coverage("BTCUSDT", "1m") == (9_990 * MIN, 10_000 * MIN)                    # ... but coverage stops before it
    assert len(cache.read("BTCUSDT", "1m", 10_000 * MIN, 10_001 * MIN)) == 0                 # and it is not in the file


def test_a_refresh_after_time_passes_fetches_only_the_newly_closed_candles(cache):
    ex = FakeExchange()
    kc.get_klines("BTCUSDT", "1m", 9_900 * MIN, 10_001 * MIN, cache=cache, fetch=ex, now_ms=NOW)
    ex.calls.clear()
    later = NOW + 5 * MIN
    df = kc.get_klines("BTCUSDT", "1m", 9_900 * MIN, 10_006 * MIN, cache=cache, fetch=ex, now_ms=later)
    asked = sorted((s, e) for _, s, e in ex.calls)
    assert (10_000 * MIN, 10_005 * MIN) in asked                                  # the 5 candles that closed meanwhile
    assert all(e - s <= 5 * MIN for s, e in asked)                                # never the whole history again
    assert len(df) == 106


def test_exchange_gaps_do_not_cause_refetching(cache):
    ex = FakeExchange(gaps={9_100 * MIN})
    a = kc.get_klines("BTCUSDT", "1m", 9_000 * MIN, 9_200 * MIN, cache=cache, fetch=ex, now_ms=NOW)
    assert len(a) == 199
    kc.get_klines("BTCUSDT", "1m", 9_000 * MIN, 9_200 * MIN, cache=cache, fetch=ex, now_ms=NOW)
    assert len(ex.calls) == 1                                                     # the hole is covered, not 'missing'


def test_a_request_wholly_outside_the_cache_replaces_coverage_so_a_hole_is_never_trusted(cache):
    ex = FakeExchange()
    kc.get_klines("BTCUSDT", "1m", 100 * MIN, 200 * MIN, cache=cache, fetch=ex, now_ms=NOW)
    df = kc.get_klines("BTCUSDT", "1m", 9_000 * MIN, 9_100 * MIN, cache=cache, fetch=ex, now_ms=NOW)
    assert cache.coverage("BTCUSDT", "1m") == (9_000 * MIN, 9_100 * MIN) and len(df) == 100


def test_symbols_and_intervals_do_not_mix(cache):
    ex = FakeExchange()
    kc.get_klines("BTCUSDT", "1m", 9_000 * MIN, 9_100 * MIN, cache=cache, fetch=ex, now_ms=NOW)
    assert cache.coverage("ETHUSDT", "1m") is None and cache.coverage("BTCUSDT", "5m") is None
    assert len(cache.read("ETHUSDT", "1m", 0, 10**15)) == 0


# ---------------------------------------------------------------- the HTTP fetcher (no network: `get` is injected)

class Resp:
    def __init__(self, rows=None, status=200, headers=None):
        self._rows, self.status_code, self.headers = rows or [], status, headers or {}

    def json(self):
        return self._rows

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_http_fetch_pages_by_1000_bars_and_returns_everything_in_order():
    seen = []

    def get(url, params, timeout):
        seen.append(params)
        s, e = params["startTime"], params["endTime"]
        return Resp([[t, "1", "2", "0.5", "1.5", "3"] for t in range(s, e + 1, MIN)])

    rows = real_fetch_range_http("BTCUSDT", "1m", 0, 2_500 * MIN, get=get, sleep=lambda s: None)
    assert [r[0] for r in rows] == list(range(0, 2_500 * MIN, MIN))
    assert len(seen) == 3 and all(p["limit"] == 1000 and p["symbol"] == "BTCUSDT" for p in seen)


def test_http_fetch_waits_out_a_rate_limit_and_retries():
    waits, calls = [], []

    def get(url, params, timeout):
        calls.append(1)
        return Resp(status=429, headers={"Retry-After": "3"}) if len(calls) == 1 else Resp([[0, "1", "2", "0.5", "1.5", "3"]])

    rows = real_fetch_range_http("BTCUSDT", "1m", 0, MIN, get=get, sleep=waits.append)
    assert len(rows) == 1 and 3.0 in waits


def test_http_fetch_gives_a_clear_error_after_repeated_failures():
    def get(url, params, timeout):
        raise TimeoutError("read timed out")

    with pytest.raises(RuntimeError, match="BTCUSDT 1m.*read timed out"):
        real_fetch_range_http("BTCUSDT", "1m", 0, MIN, get=get, sleep=lambda s: None)
