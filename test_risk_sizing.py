"""Phase 6.6 building blocks: volatility forecasts and exposure rules must only use the past."""
import numpy as np
import pandas as pd
import pytest

from core.ml_validation import paired_block_bootstrap_stat_diff, walk_forward_predict_panel
from core.risk_sizing import (
    apply_no_trade_band, log_range, naive_vol_forecast, perf_stats, portfolio_returns, sharpe, vol_target_exposure,
)


def _ohlc(n=300, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    rngs = np.abs(rng.normal(0.02, 0.01, n))
    return pd.DataFrame({"High": close * (1 + rngs / 2), "Low": close * (1 - rngs / 2), "Close": close,
                         "Open": close}, index=pd.date_range("2024-01-01", periods=n))


def test_log_range_is_positive_and_floored():
    df = _ohlc()
    assert (log_range(df) > 0).all()
    flat = df.copy(); flat["High"] = flat["Low"]
    assert (log_range(flat) == 1e-5).all()               # a zero-range bar must not give -inf/NaN in logs
    assert (np.log(log_range(flat)) > -20).all()


@pytest.mark.parametrize("cut", [40, 120, 250])
def test_naive_forecast_and_exposure_do_not_use_future_bars(cut):
    df = _ohlc()
    full = vol_target_exposure(naive_vol_forecast(log_range(df)))
    part = vol_target_exposure(naive_vol_forecast(log_range(df.iloc[:cut])))
    pd.testing.assert_series_equal(full.iloc[:cut], part, check_freq=False)


def test_rewriting_the_future_leaves_past_exposure_unchanged():
    df = _ohlc()
    base = vol_target_exposure(naive_vol_forecast(log_range(df)))
    mutated = df.copy()
    mutated.iloc[200:, mutated.columns.get_loc("High")] *= 5                     # wild future ranges
    after = vol_target_exposure(naive_vol_forecast(log_range(mutated)))
    pd.testing.assert_series_equal(base.iloc[:200], after.iloc[:200], check_freq=False)


def test_exposure_is_bounded_nan_during_warmup_and_scale_free():
    sig = pd.Series(np.abs(np.random.default_rng(1).normal(0.02, 0.008, 200)) + 0.005)
    w = vol_target_exposure(sig, k=0.7, cap=1.0, min_history=30)
    assert w.iloc[:29].isna().all() and w.iloc[29:].notna().all()
    assert (w.dropna() >= 0).all() and (w.dropna() <= 1.0).all()
    pd.testing.assert_series_equal(w, vol_target_exposure(sig * 37.0, k=0.7, cap=1.0, min_history=30))  # bias-proof


def test_high_forecast_vol_means_less_exposure():
    sig = pd.Series([0.02] * 40 + [0.06])                       # a volatility spike vs history
    w = vol_target_exposure(sig, k=0.7, cap=1.0, min_history=30)
    assert w.iloc[-1] == pytest.approx(0.7 * 0.02 / 0.06)
    calm = pd.Series([0.02] * 40 + [0.01])
    assert vol_target_exposure(calm, k=0.7, cap=1.0, min_history=30).iloc[-1] == 1.0     # capped, no leverage


def test_no_trade_band_only_moves_on_big_changes():
    target = pd.Series([0.5, 0.55, 0.45, 0.62, 0.9, 0.88, np.nan, 0.3])
    out = apply_no_trade_band(target, band=0.10)
    assert out.iloc[:4].tolist() == [0.5, 0.5, 0.5, 0.62]        # 0.62 is >0.10 from 0.5
    assert out.iloc[4] == 0.9 and out.iloc[5] == 0.9 and np.isnan(out.iloc[6]) and out.iloc[7] == 0.3


def test_portfolio_return_and_costs_by_hand():
    idx = pd.date_range("2024-01-01", periods=3)
    e = pd.DataFrame({"A": [1.0, 0.5, 0.5], "B": [0.0, 0.0, 1.0]}, index=idx)
    r = pd.DataFrame({"A": [0.02, 0.02, 0.02], "B": [0.04, 0.04, 0.04]}, index=idx)
    out = portfolio_returns(e, r, fee=0.01)
    # day0: gross (0.02+0)/2 = 0.01, turnover (1+0)/2 -> cost 0.005
    # day1: gross (0.01+0)/2 = 0.005, turnover (0.5+0)/2 -> cost 0.0025
    # day2: gross (0.01+0.04)/2 = 0.025, turnover (0+1)/2 -> cost 0.005
    np.testing.assert_allclose(out.to_numpy(), [0.01 - 0.005, 0.005 - 0.0025, 0.025 - 0.005])


def test_perf_stats_and_sharpe_basics():
    r = pd.Series([0.01, -0.02, 0.015, 0.0, 0.01] * 60)
    s = perf_stats(r)
    assert s["max_drawdown"] < 0 and s["ann_vol"] > 0 and s["sharpe"] == pytest.approx(sharpe(r.to_numpy()))
    assert perf_stats(pd.Series([], dtype=float)) == {}
    assert np.isnan(sharpe([1.0, 1.0, 1.0]))                                # zero variance -> undefined, not inf


# ----------------------------------------------------- statistical machinery

def test_paired_stat_diff_ci_detects_a_real_edge_and_not_a_tie():
    rng = np.random.default_rng(3)
    n = 900
    dates = pd.date_range("2024-01-01", periods=n)
    market = rng.normal(0.0005, 0.02, n)
    a = market + rng.normal(0.001, 0.002, n)                     # same market, small consistent edge
    b = market + rng.normal(0.0, 0.002, n)
    lo, hi = paired_block_bootstrap_stat_diff(a, b, dates, sharpe, n_boot=400)
    assert lo > 0
    lo2, hi2 = paired_block_bootstrap_stat_diff(b, b + rng.normal(0, 1e-6, n), dates, sharpe, n_boot=300)
    assert lo2 < 0 < hi2


def test_regression_mode_of_the_panel_walk_forward_learns_a_continuous_target():
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(4)
    dates = pd.date_range("2024-01-01", periods=250)
    frames, ys = [], []
    for s in ("A", "B"):
        x = rng.normal(size=(250, 2))
        y = 2 * x[:, 0] + rng.normal(0, 0.3, 250)
        idx = pd.MultiIndex.from_arrays([[s] * 250, dates], names=["symbol", "timestamp"])
        frames.append(pd.DataFrame(x, index=idx, columns=["f1", "f2"])); ys.append(pd.Series(y, index=idx))
    X, y = pd.concat(frames), pd.concat(ys)
    p = walk_forward_predict_panel(lambda: HistGradientBoostingRegressor(max_iter=60, random_state=0), X, y,
                                   train_dates=100, retrain_every=25, horizon=1, task="regress")
    both = pd.concat([y, p], axis=1).dropna()
    assert both.corr().iloc[0, 1] > 0.9 and both.index.get_level_values(1).min() > dates[100]
    with pytest.raises(ValueError):
        walk_forward_predict_panel(lambda: None, X, y, train_dates=100, retrain_every=25, task="rank")
