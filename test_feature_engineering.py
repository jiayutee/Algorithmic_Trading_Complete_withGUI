"""Phase 6.0: the feature matrix must be free of lookahead.

The strongest check is *truncation invariance*: features computed on the first k
rows must equal the first k rows of features computed on the full history. If any
feature at row t used data after t, cutting the future off would change row t.
"""
import numpy as np
import pandas as pd
import pytest

from core.feature_engineering import (
    align_macro, build_feature_matrix, build_features, clean_xy, make_target,
    news_features, technical_features, time_features,
)


def _ohlcv(n=300, seed=1, start="2024-01-01", freq="D"):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    return pd.DataFrame(
        {"Open": close * (1 + rng.normal(0, 0.002, n)), "High": high, "Low": low, "Close": close,
         "Volume": rng.uniform(500, 1500, n)},
        index=pd.date_range(start, periods=n, freq=freq),
    )


def _with_news(df, seed=2):
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["sentiment_balance"] = rng.normal(0, 0.4, len(df))
    out["sentiment_magnitude"] = rng.uniform(0, 1, len(df))
    out["sentiment_confidence"] = rng.uniform(0.5, 1, len(df))
    out["news_count"] = rng.integers(0, 6, len(df)).astype(float)
    out["impact_score"] = rng.uniform(0, 1, len(df))
    out["news_flow_ratio"] = rng.normal(0, 0.2, len(df))
    return out


def _macro(n=300, seed=3, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame({"vix": 20 + np.cumsum(rng.normal(0, 0.5, n)),
                         "us10y": 4 + np.cumsum(rng.normal(0, 0.02, n))}, index=idx)


# ---------------------------------------------------------------- no-lookahead

@pytest.mark.parametrize("cut", [60, 101, 150, 233, 299])
def test_truncating_the_future_does_not_change_past_features(cut):
    df, macro = _with_news(_ohlcv()), _macro()
    kwargs = dict(include_news=True, macro=macro, macro_lag="0D")
    full = build_features(df, **kwargs)
    truncated = build_features(df.iloc[:cut], macro=macro.iloc[:cut], include_news=True, macro_lag="0D")
    pd.testing.assert_frame_equal(full.iloc[:cut], truncated, check_freq=False, rtol=1e-9, atol=1e-12)


def test_rewriting_the_future_does_not_change_past_features():
    df = _with_news(_ohlcv())
    t = 180
    base = build_features(df, include_news=True)
    mutated = df.copy()
    rng = np.random.default_rng(99)
    mutated.iloc[t + 1:, :5] = rng.uniform(1, 1000, size=mutated.iloc[t + 1:, :5].shape)   # garbage prices
    mutated.iloc[t + 1:, 5:] = 123.0                                                       # garbage news
    after = build_features(mutated, include_news=True)
    pd.testing.assert_frame_equal(base.iloc[: t + 1], after.iloc[: t + 1], check_freq=False, rtol=1e-9, atol=1e-12)


def test_the_check_would_actually_catch_a_leak():
    """Sanity check on the test itself: a feature using shift(-1) must fail truncation invariance."""
    df = _ohlcv()
    leaky = lambda d: d["Close"].pct_change().shift(-1).to_frame("leak")   # peeks one bar ahead
    assert not leaky(df).iloc[:150].equals(leaky(df.iloc[:150]))


# ---------------------------------------------------------------------- macro

def test_macro_value_is_invisible_until_its_publication_lag_has_passed():
    idx = pd.date_range("2024-03-01", periods=30, freq="D")
    macro = pd.DataFrame({"cpi": [1.0, 1.0, 999.0]}, index=pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-10"]))
    out = align_macro(idx, macro, publication_lag="15D")
    assert (out.loc["2024-03-10":"2024-03-24", "cpi"] == 1.0).all()      # released 03-25
    assert out.loc["2024-03-25", "cpi"] == 999.0
    assert (out.loc[: "2024-03-24", "cpi"] < 999).all()


def test_macro_lag_zero_uses_the_latest_known_observation_only():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    macro = pd.DataFrame({"vix": [10, 20, 30]}, index=pd.to_datetime(["2023-12-31", "2024-01-03", "2024-01-05"]))
    out = align_macro(idx, macro)
    assert list(out["vix"]) == [10, 10, 20, 20, 30]


def test_no_macro_gives_no_columns():
    idx = pd.date_range("2024-01-01", periods=3)
    assert align_macro(idx, pd.DataFrame()).shape == (3, 0)


# ------------------------------------------------------------ shape / contents

def test_multiindex_output_and_scale_free_columns():
    df = _ohlcv()
    fm = build_feature_matrix({"AAA": df, "BBB": df * 50}, include_news=False)
    assert fm.index.names == ["symbol", "timestamp"]
    assert set(fm.index.get_level_values("symbol")) == {"AAA", "BBB"}
    a, b = fm.loc["AAA"], fm.loc["BBB"]
    # prices differ 50x but scale-free features must not (volume also scaled, so drop it)
    for col in ("ret_5", "rsi_14", "ema_ratio", "bb_pos", "atr_pct_14"):
        np.testing.assert_allclose(a[col].dropna().values, b[col].dropna().values, rtol=1e-6)


def test_accepts_lower_and_upper_case_ohlc_columns():
    df = _ohlcv()
    lower = df.rename(columns=str.lower)
    pd.testing.assert_frame_equal(build_features(df), build_features(lower))


def test_news_features_are_zero_when_no_news_columns_and_decay_when_present():
    df = _ohlcv(50)
    zero = news_features(df)
    assert (zero.values == 0).all()
    d = df.copy(); d["sentiment_balance"] = 0.0; d.iloc[10, d.columns.get_loc("sentiment_balance")] = 1.0
    nf = news_features(d, half_life_bars=2.0)
    assert nf["sentiment_ewm"].iloc[10] > nf["sentiment_ewm"].iloc[12] > nf["sentiment_ewm"].iloc[20] > 0
    assert nf["sentiment_ewm"].iloc[9] == 0                      # nothing before the shock


def test_time_features_depend_only_on_timestamp():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    a, b = time_features(idx), time_features(idx)
    pd.testing.assert_frame_equal(a, b)
    assert "hour_sin" not in a.columns
    intraday = time_features(pd.date_range("2024-01-01", periods=5, freq="h"))
    assert "hour_sin" in intraday.columns


def test_missing_ohlc_columns_raise():
    with pytest.raises(ValueError):
        technical_features(pd.DataFrame({"Close": [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3)))


# --------------------------------------------------------------------- targets

def test_target_looks_forward_and_leaves_the_tail_unlabelled():
    df = _ohlcv(30)
    y = make_target(df, horizon=1, kind="direction")
    close = df["Close"]
    assert y.iloc[3] == float(close.iloc[4] > close.iloc[3])
    assert np.isnan(y.iloc[-1])
    r = make_target(df, horizon=3, kind="return")
    assert np.isnan(r.iloc[-3:]).all() and r.iloc[0] == pytest.approx(close.iloc[3] / close.iloc[0] - 1)
    with pytest.raises(ValueError):
        make_target(df, horizon=0)


def test_clean_xy_drops_warmup_and_unlabelled_rows_and_keeps_alignment():
    df = _ohlcv(120)
    X, y = clean_xy(build_features(df), make_target(df))
    assert len(X) == len(y) and X.index.equals(y.index)
    assert not X.isna().any().any() and not y.isna().any()
    assert len(X) < 120 - 1            # warm-up (MA50 etc.) rows were dropped
