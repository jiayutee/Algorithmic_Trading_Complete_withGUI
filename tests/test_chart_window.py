import pytest

from core import chart_window as cw


def test_a_year_of_minutes_is_cut_to_a_first_screen_for_crypto_and_the_user_is_told():
    p = cw.plan_window("1m", 365, crypto=True)
    assert p.capped_by == "initial_window" and p.days == 2 and p.bars == 2880      # 1,500 bars rounds up to whole days
    assert "scroll left" in p.note and "365 days" in p.note and "1m" in p.note
    # in particular it is nowhere near the 525,600 candles that took ~8 minutes to fetch
    assert p.bars < 3000


@pytest.mark.parametrize("interval,days,expect_days", [("5m", 365, 6), ("15m", 365, 16), ("1h", 365, 63), ("4h", 365, 250)])
def test_other_intervals_get_a_first_screen_of_about_1500_to_3000_bars(interval, days, expect_days):
    p = cw.plan_window(interval, days, crypto=True)
    assert p.days == expect_days and p.capped_by == "initial_window"
    assert cw.INITIAL_BARS <= p.bars <= 2 * cw.INITIAL_BARS       # whole days, so it rounds up (never down to a thin chart)


def test_daily_and_short_requests_are_left_alone():
    for interval, days in (("1d", 365), ("1d", 1500), ("1m", 1), ("5m", 3), ("1h", 30)):
        p = cw.plan_window(interval, days, crypto=True)
        assert (p.days, p.capped_by, p.note) == (days, None, "")


@pytest.mark.parametrize("interval,days,limit", [("1m", 365, 7), ("5m", 365, 59), ("15m", 365, 59), ("1h", 1000, 729)])
def test_stocks_are_cut_to_what_the_source_keeps_with_a_message(interval, days, limit):
    p = cw.plan_window(interval, days, crypto=False)
    assert p.days == limit and p.capped_by == "provider" and str(limit) in p.note and str(days) in p.note


def test_hourly_stock_history_is_not_cut_to_the_one_minute_limit():
    """The old flat 7-day cap applied to 1h/5m/15m too, although Yahoo keeps ~2 years of hourly and ~60 days of 5-15 minute bars."""
    assert cw.plan_window("1h", 365, crypto=False).days == 365
    assert cw.plan_window("5m", 30, crypto=False).days == 30


def test_nonsense_input_is_safe():
    assert cw.plan_window("weird", 0, crypto=True).days == 1
    assert cw.interval_seconds(None) == 86400 and cw.interval_seconds("1H") == 3600


def test_when_to_load_older_candles():
    start, end = 1_000_000, 2_000_000
    assert cw.should_load_older(start - 5, start, end)                       # panned past the left edge
    assert cw.should_load_older(start + 100_000, start, end)                 # inside the left 15%
    assert not cw.should_load_older(start + 400_000, start, end)             # plenty of candles still off to the left


def test_older_chunk_is_exactly_the_bars_before_the_loaded_start():
    s, e = cw.older_chunk_range(1_700_000_000_000, "1m")
    assert e == 1_700_000_000_000 and e - s == cw.CHUNK_BARS * 60_000


def test_loading_stops_at_the_memory_cap_and_at_the_users_day_limit():
    assert cw.can_load_more(1000, None, 5_000)
    assert not cw.can_load_more(cw.MAX_BARS, None, 5_000)
    assert not cw.can_load_more(1000, 6_000, 5_000)                          # already older than the oldest allowed time
    assert cw.can_load_more(1000, 4_000, 5_000)


def test_span_text_reads_naturally():
    assert cw.span_text(90 * 60) == "90 minutes"
    assert cw.span_text(25 * 3600) == "25 hours"
    assert cw.span_text(5.2 * 86400) == "5.2 days"
    assert cw.span_text(800 * 86400) == "2.2 years"


def test_a_fast_drag_far_left_fetches_enough_to_cover_the_view_in_one_go():
    min_ms = 60_000
    loaded_start = 1_000_000 * min_ms
    # view starts 5,000 minutes before the loaded data and is 600 minutes wide: needs 5,000 + 30% margin of 600 = 5,180 bars
    assert cw.bars_needed(loaded_start - 5_000 * min_ms, loaded_start - 4_400 * min_ms, loaded_start, "1m") == 5_180
    # just touching the left edge: one normal chunk
    assert cw.bars_needed(loaded_start + 10 * min_ms, loaded_start + 600 * min_ms, loaded_start, "1m") == cw.CHUNK_BARS
    # a drag a month away is bounded, the next scroll event continues
    assert cw.bars_needed(loaded_start - 50_000 * min_ms, loaded_start - 49_400 * min_ms, loaded_start, "1m") == cw.MAX_BARS_PER_LOAD


# ---------------------------------------------------------------- zoomed out: coarser candles (like an exchange chart)

import pandas as pd


@pytest.mark.parametrize("visible,expected", [(0, 1), (1500, 1), (3000, 1), (3001, 2), (6000, 2), (9000, 3), (15000, 5),
                                              (30000, 10), (525600, 240), (5_000_000, 1440)])
def test_lod_factor_uses_nice_multiples_and_keeps_the_picture_under_the_target(visible, expected):
    assert cw.lod_factor(visible) == expected
    if visible and visible < 5_000_000:
        assert visible / cw.lod_factor(visible) <= cw.TARGET_VISIBLE_BARS * 1.0001 or cw.lod_factor(visible) == 1440


def _bars(n, start="2026-10-10 00:00", step="1min"):
    idx = pd.date_range(start, periods=n, freq=step)
    base = pd.Series(range(n), index=idx, dtype=float)
    return pd.DataFrame({"Open": base, "High": base + 2, "Low": base - 2, "Close": base + 1, "Volume": 1.0}, index=idx)


def test_resample_aggregates_ohlcv_correctly_and_aligns_to_the_clock():
    df = _bars(120)                                              # two hours of 1m candles starting on the hour
    out = cw.resample_ohlcv(df, 60, "1m")
    assert len(out) == 2 and out.index[0] == pd.Timestamp("2026-10-10 00:00") and out.index[1] == pd.Timestamp("2026-10-10 01:00")
    first = out.iloc[0]
    assert first["Open"] == 0 and first["Close"] == 60 and first["High"] == 61 and first["Low"] == -2 and first["Volume"] == 60


def test_resample_buckets_are_epoch_aligned_even_when_the_data_starts_mid_bucket():
    df = _bars(100, start="2026-10-10 00:17")
    out = cw.resample_ohlcv(df, 15, "1m")
    assert all(ts.minute % 15 == 0 for ts in out.index)
    assert out["Volume"].sum() == 100                            # nothing lost or double counted


def test_resample_with_factor_one_or_empty_input_is_a_no_op():
    df = _bars(10)
    assert cw.resample_ohlcv(df, 1, "1m") is df
    assert cw.resample_ohlcv(df.iloc[0:0], 5, "1m").empty
