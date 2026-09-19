import numpy as np
import pandas as pd
import pytest

from core import yahoo_chart as yc


def payload(n=30, adj=None, tsstart=1_700_000_000, gmtoffset=-18000, nulls=()):
    ts = [tsstart + i * 86400 for i in range(n)]
    close = [100.0 + i for i in range(n)]
    q = {"open": [c - 1 for c in close], "high": [c + 2 for c in close], "low": [c - 2 for c in close], "close": list(close),
         "volume": [1000 + i for i in range(n)]}
    for i in nulls:
        for k in q:
            q[k][i] = None
    ind = {"quote": [q]}
    if adj is not None:
        ind["adjclose"] = [{"adjclose": adj}]
    return {"chart": {"result": [{"meta": {"gmtoffset": gmtoffset}, "timestamp": ts, "indicators": ind}], "error": None}}


class R:
    def __init__(self, status=200, body=None):
        self.status_code, self._b = status, body

    def json(self):
        return self._b


class Sess:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_parses_ohlcv_normalises_daily_bars_to_midnight_and_sorts(monkeypatch):
    monkeypatch.setattr(yc.time, "sleep", lambda s: None)
    df = yc.fetch_chart("AAPL", 60, "1d", session=Sess([R(200, payload())]), now=1_700_000_000 + 40 * 86400)
    assert list(df.columns) == yc.EMPTY_COLUMNS and len(df) == 30 and df.index.is_monotonic_increasing
    assert (df.index == df.index.normalize()).all() and df.index.name == "Datetime" and df.index.tz is None
    assert df["Close"].iloc[0] == 100.0 and df["Volume"].iloc[-1] == 1029


def test_intraday_bars_keep_their_time_of_day():
    df = yc.fetch_chart("AAPL", 5, "5m", session=Sess([R(200, payload(n=10, tsstart=1_700_000_000 + 3600))]))
    assert (df.index != df.index.normalize()).any()


def test_prices_are_split_and_dividend_adjusted_like_yfinance_default():
    adj = [50.0 + i / 2 for i in range(30)]                          # adjclose = half of close at the start, converging
    df = yc.fetch_chart("AAPL", 60, "1d", session=Sess([R(200, payload(adj=adj))]))
    assert df["Close"].iloc[0] == pytest.approx(50.0) and df["Open"].iloc[0] == pytest.approx(99.0 * 50.0 / 100.0)


def test_null_rows_are_dropped_and_missing_volume_becomes_zero():
    body = payload(nulls=(3, 7))
    body["chart"]["result"][0]["indicators"]["quote"][0]["volume"][5] = None
    df = yc.fetch_chart("AAPL", 60, "1d", session=Sess([R(200, body)]))
    assert len(df) == 28 and df["Volume"].isna().sum() == 0


def test_never_raises_and_returns_an_empty_frame_on_any_failure(monkeypatch):
    monkeypatch.setattr(yc.time, "sleep", lambda s: None)
    for resp in ([R(429), R(429), R(429)], [ConnectionError("down")] * 3, [R(200, {"chart": {"result": None}})],
                 [R(200, {"chart": {"result": [{"timestamp": []}]}})], [R(200, {"garbage": 1})]):
        df = yc.fetch_chart("AAPL", 60, "1d", session=Sess(resp))
        assert df.empty and list(df.columns) == yc.EMPTY_COLUMNS


def test_retries_a_transient_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(yc.time, "sleep", lambda s: None)
    s = Sess([R(429), R(200, payload())])
    assert len(yc.fetch_chart("AAPL", 60, "1d", session=s)) == 30 and len(s.calls) == 2


def test_the_loader_falls_back_to_the_chart_endpoint_when_the_library_is_rate_limited(monkeypatch):
    import core.data_loader as dl
    sample = yc.fetch_chart("AAPL", 60, "1d", session=Sess([R(200, payload())]))
    monkeypatch.setattr(dl, "download_with_retry", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr("core.yahoo_chart.fetch_chart", lambda *a, **k: sample)
    out = dl.DataLoader()._get_yahoo_historical("AAPL", 60, "1d")
    assert len(out) == 30
    monkeypatch.setattr("core.yahoo_chart.fetch_chart", lambda *a, **k: yc._empty())
    assert dl.DataLoader()._get_yahoo_historical("AAPL", 60, "1d").empty                    # both failed: empty, as before
