"""Dash: crypto minute charts start small and load older candles as the user scrolls left (offline)."""
import pandas as pd
import pytest

import dash_app.callbacks as cb
from core import chart_window as cw


def bars(end, n, step="1min", base=100.0):
    idx = pd.date_range(end=end, periods=n, freq=step)
    return pd.DataFrame({"Open": base, "High": base + 1, "Low": base - 1, "Close": base + 0.5, "Volume": 1.0}, index=idx)


class FakeLoader:
    def __init__(self):
        self.calls = []

    def get_older_candles(self, symbol, interval, before, bars=1000, oldest_allowed=None):
        self.calls.append((symbol, interval, pd.Timestamp(before), bars, oldest_allowed))
        step = pd.Timedelta(minutes=1)
        n = bars if oldest_allowed is None else min(bars, int((pd.Timestamp(before) - oldest_allowed) / step))
        return globals()["bars"](pd.Timestamp(before) - step, max(0, n)) if n > 0 else pd.DataFrame(columns=cb._OHLCV)


@pytest.fixture
def chart(monkeypatch):
    df = bars("2026-10-10 12:00", 1500)
    cb._remember_chart("BTCUSDT", "1m", df, days_requested=365)
    yield df
    cb._chart_hist.update(key=None, df=None, oldest_allowed=None, exhausted=False)


def relayout(start, end):
    return {"xaxis.range[0]": str(start), "xaxis.range[1]": str(end)}


def ops(patch):
    return {tuple(op["location"]): op for op in patch._operations}


def test_the_view_reaching_the_left_edge_prepends_the_next_older_chunk(chart):
    loader = FakeLoader()
    start = chart.index[0] + pd.Timedelta(minutes=30)               # well inside the left 15% of the loaded range
    patch, status = cb.older_candles_patch(relayout(start, chart.index[-1]), "BTCUSDT", "1m", loader=loader)
    assert patch is not None and "2,500" in status
    assert loader.calls[0][2] == chart.index[0] and loader.calls[0][3] == cw.CHUNK_BARS
    o = ops(patch)
    assert ("data", 0, "x") in o and len(o[("data", 0, "x")]["params"]["value"]) == 2500
    assert len(o[("data", 0, "close")]["params"]["value"]) == 2500
    assert o[("data", 0, "x")]["params"]["value"][0] < o[("data", 0, "x")]["params"]["value"][-1]       # oldest first, in order
    assert len(cb._chart_hist["df"]) == 2500                                  # remembered for the next scroll


def test_nothing_is_fetched_while_there_is_plenty_of_data_off_to_the_left(chart):
    loader = FakeLoader()
    mid = chart.index[1000]
    assert cb.older_candles_patch(relayout(mid, chart.index[-1]), "BTCUSDT", "1m", loader=loader) == (None, None)
    assert loader.calls == []


def test_the_layout_is_never_touched_so_the_users_zoom_stays_put(chart):
    patch, _ = cb.older_candles_patch(relayout(chart.index[0], chart.index[-1]), "BTCUSDT", "1m", loader=FakeLoader())
    assert not any(loc[0] == "layout" for loc in ops(patch))


@pytest.mark.parametrize("relayout_data", [None, {}, {"xaxis.autorange": True}, {"yaxis.range[0]": 1, "yaxis.range[1]": 2},
                                           {"xaxis.range[0]": "not a date", "xaxis.range[1]": "nope"}])
def test_other_relayout_events_are_ignored(chart, relayout_data):
    loader = FakeLoader()
    assert cb.older_candles_patch(relayout_data, "BTCUSDT", "1m", loader=loader) == (None, None)
    assert loader.calls == []


def test_a_different_symbol_or_interval_or_a_stock_is_ignored(chart):
    r = relayout(chart.index[0], chart.index[-1])
    loader = FakeLoader()
    assert cb.older_candles_patch(r, "ETHUSDT", "1m", loader=loader) == (None, None)     # chart holds BTC
    assert cb.older_candles_patch(r, "BTCUSDT", "5m", loader=loader) == (None, None)
    assert cb.older_candles_patch(r, "AAPL", "1m", loader=loader) == (None, None)
    assert loader.calls == []


def test_it_stops_at_the_days_limit_and_says_so(monkeypatch):
    df = bars("2026-10-10 12:00", 1500)
    cb._remember_chart("BTCUSDT", "1m", df, days_requested=1)
    cb._chart_hist["oldest_allowed"] = df.index[0] - pd.Timedelta(minutes=200)         # only 200 more minutes allowed
    loader = FakeLoader()
    patch, status = cb.older_candles_patch(relayout(df.index[0], df.index[-1]), "BTCUSDT", "1m", loader=loader)
    assert patch is not None and len(cb._chart_hist["df"]) == 1700                     # got the 200 that were allowed
    patch2, status2 = cb.older_candles_patch(relayout(cb._chart_hist["df"].index[0], df.index[-1]), "BTCUSDT", "1m", loader=loader)
    assert patch2 is None and "Days" in status2                                        # next scroll: told why it stops
    assert cb.older_candles_patch(relayout(cb._chart_hist["df"].index[0], df.index[-1]), "BTCUSDT", "1m", loader=loader) == (None, None)
    cb._chart_hist.update(key=None, df=None, exhausted=False)


def test_it_stops_at_the_memory_cap(monkeypatch):
    monkeypatch.setattr(cw, "MAX_BARS", 1500)
    cb._remember_chart("BTCUSDT", "1m", bars("2026-10-10 12:00", 1500), days_requested=365)
    loader = FakeLoader()
    patch, status = cb.older_candles_patch(relayout(cb._chart_hist["df"].index[0], cb._chart_hist["df"].index[-1]),
                                           "BTCUSDT", "1m", loader=loader)
    assert patch is None and "limit" in status and loader.calls == []
    cb._chart_hist.update(key=None, df=None, exhausted=False)


def test_a_failing_fetch_never_breaks_the_chart(chart):
    class Boom:
        def get_older_candles(self, *a, **k):
            raise RuntimeError("network down")
    assert cb.older_candles_patch(relayout(chart.index[0], chart.index[-1]), "BTCUSDT", "1m", loader=Boom()) == (None, None)
    assert len(cb._chart_hist["df"]) == 1500 and not cb._chart_hist["exhausted"]       # and it will try again on the next scroll


def test_the_callback_is_registered_on_relayout_and_writes_the_figure_and_status():
    import dash_app.app as m
    for key, c in m.app.callback_map.items():
        if [i["property"] for i in c["inputs"]] == ["relayoutData"]:
            assert "main-chart.figure" in key and "chart-status.children" in key
            return
    pytest.fail("scroll-back callback not registered")


def test_a_drag_far_beyond_the_loaded_data_fetches_enough_to_cover_the_view(chart):
    loader = FakeLoader()
    far = chart.index[0] - pd.Timedelta(minutes=3000)                       # view starts 3,000 minutes before the data
    patch, status = cb.older_candles_patch(relayout(far, far + pd.Timedelta(minutes=500)), "BTCUSDT", "1m", loader=loader)
    assert patch is not None and loader.calls[0][3] == 3150                  # 3,000 + 30% of the 500-minute view
    assert len(cb._chart_hist["df"]) == 1500 + 3150
