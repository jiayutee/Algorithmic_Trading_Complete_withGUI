"""Phase 10.0 slice 1: the measuring tool is tested on SYNTHETIC data with a planted signal. No real headline or price is read."""
import numpy as np
import pandas as pd
import pytest

from core import diffusion_sentiment as ds

H = ds.HORIZONS


# ------------------------------------------------------------------ kernels
def test_kernels_start_at_zero_and_saturate_at_one():
    assert np.all(ds.kernel_null(H) == 0) and np.all(ds.kernel_instant(H) == 1)
    e = ds.kernel_exponential(H, 6.0)
    assert 0 < e[0] < e[-1] < 1 and np.all(np.diff(e) > 0) and e[-1] > 0.99
    b = ds.kernel_bass(H, 0.02, 0.4)
    assert b[0] > 0 and np.all(np.diff(b) > 0) and abs(b[-1] - 1) < 0.01
    assert ds.kernel_bass(0.0, 0.02, 0.4) == 0.0


def test_bass_starts_slower_than_exponential_with_the_same_early_pace_then_accelerates():
    # imitation (q) makes the early rise slower than its late rise, an exponential never does that
    b = ds.kernel_bass(H, 0.01, 0.5)
    inc = np.diff(np.concatenate([[0.0], b]))
    assert inc.argmax() > 0
    e = np.diff(np.concatenate([[0.0], ds.kernel_exponential(H, 8.0)]))
    assert e.argmax() == 0


# ------------------------------------------------------------------ planted-signal recovery
def test_exponential_signal_is_recovered():
    # one noisy sample can land far from the truth (tau is weakly identified), so judge the estimator over several independent samples
    fits = [ds.fit_kernel(ds.simulate_events(600, beta=0.02, kind="exponential", params=(6.0,), seed=100 + k), "exponential") for k in range(12)]
    assert np.median([f["beta"] for f in fits]) == pytest.approx(0.02, rel=0.1)
    assert np.median([f["params"][0] for f in fits]) == pytest.approx(6.0, rel=0.2)


def test_instant_signal_is_recovered_and_preferred_over_exponential():
    ev = ds.simulate_events(600, beta=0.015, kind="instant", params=(), seed=2)
    f = ds.fit_kernel(ev, "instant")
    assert f["beta"] == pytest.approx(0.015, rel=0.15)
    hold = ds.simulate_events(600, beta=0.015, kind="instant", params=(), seed=3)
    assert ds.holdout_mse(f, hold) <= ds.holdout_mse(ds.fit_kernel(ev, "exponential"), hold) * 1.01


def test_bass_signal_curve_is_recovered():
    true = {"kind": "bass", "params": (0.01, 0.35)}
    ev = ds.simulate_events(1500, beta=0.03, kind="bass", params=true["params"], noise_sd=0.003, seed=4)
    f = ds.fit_kernel(ev, "bass")
    fitted = f["beta"] * ds.kernel_of(f)
    planted = 0.03 * ds.kernel_of(true)
    assert np.max(np.abs(fitted - planted)) < 0.004            # the whole curve, not only the parameters


def test_nonparametric_kernel_follows_the_planted_shape_without_assuming_one():
    ev = ds.simulate_events(1500, beta=0.02, kind="exponential", params=(10.0,), noise_sd=0.003, seed=5)
    est = ds.nonparametric_kernel(ev)
    assert np.max(np.abs(est - 0.02 * ds.kernel_exponential(H, 10.0))) < 0.004


def test_null_kernel_predicts_zero():
    ev = ds.simulate_events(100, beta=0.0, seed=6)
    assert ds.fit_kernel(ev, "null")["beta"] == 0.0 and np.all(ds.kernel_of(ds.fit_kernel(ev, "null")) == 0)


# ------------------------------------------------------------------ verdict
def _verdict(beta, pre_beta=0.0, n=600, seed=0):
    fit = ds.simulate_events(n, beta=beta, params=(6.0,), pre_beta=pre_beta, seed=seed)
    hold = ds.simulate_events(n, beta=beta, params=(6.0,), pre_beta=pre_beta, seed=seed + 100)
    return ds.diffusion_verdict(fit, hold, n_boot=300, seed=0)


def test_a_planted_effect_is_supported():
    out = _verdict(0.02)
    assert out["verdict"] == "DIFFUSION SUPPORTED", out
    assert out["best_kernel"] in ("exponential", "bass")


def test_pure_noise_is_not_supported_across_seeds():
    # the false-positive rate must be small: of 8 pure-noise runs at most 1 may claim support
    wins = sum(_verdict(0.0, seed=s)["verdict"] == "DIFFUSION SUPPORTED" for s in range(8))
    assert wins <= 1


def test_a_move_before_the_headline_fails_the_placebo_even_if_the_after_move_is_planted():
    out = _verdict(0.02, pre_beta=0.03)
    assert out["verdict"] == "NOT SUPPORTED" and "placebo" in out["reason"]


def test_too_few_holdout_events_is_insufficient_data_not_a_finding():
    fit = ds.simulate_events(300, beta=0.02, seed=1)
    hold = ds.simulate_events(150, beta=0.02, seed=2)
    out = ds.diffusion_verdict(fit, hold, n_boot=50)
    assert out["verdict"] == "INSUFFICIENT DATA" and out["n_holdout"] == 150


# ------------------------------------------------------------------ bootstrap and split
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_event_day_does_not_depend_on_timestamp_resolution(unit):
    # regression (PR #59 CI): pandas 3 stores timestamps in microseconds by default; the day key must not assume nanoseconds
    ev = ds.simulate_events(200, beta=0.02, days=30, seed=7)
    expected = ev.day.copy()
    assert len(np.unique(expected)) > 20
    ev.t0 = ev.t0.as_unit(unit)
    assert np.array_equal(ev.day, expected)


def test_day_block_bootstrap_keeps_whole_days_together():
    ev = ds.simulate_events(300, beta=0.02, days=30, seed=7)
    rng = np.random.default_rng(0)
    counts = dict(zip(*np.unique(ev.day, return_counts=True)))
    res = ds._resample_days(ev, rng)
    for d, c in zip(*np.unique(res.day, return_counts=True)):
        assert c % counts[d] == 0          # a drawn day brings ALL its events, k times


def test_parameter_intervals_cover_the_planted_values():
    ev = ds.simulate_events(500, beta=0.02, params=(6.0,), seed=8)
    ci = ds.bootstrap_params(ev, "exponential", n_boot=200, seed=0)
    assert ci["beta"][0] < 0.02 < ci["beta"][1] and ci["param0"][0] < 6.0 < ci["param0"][1]


def test_split_is_by_time_range_60_40_and_never_mixes():
    ev = ds.simulate_events(400, beta=0.02, days=100, seed=9)
    fit, hold = ds.split_by_time(ev)
    assert fit.n + hold.n == ev.n and fit.t0.max() < hold.t0.min()
    span = (ev.t0.max() - ev.t0.min())
    assert fit.t0.max() <= ev.t0.min() + 0.6 * span


# ------------------------------------------------------------------ event construction rules
def _headlines(rows):
    return pd.DataFrame(rows, columns=["published", "tickers", "label", "shock"])


def test_headline_selection_rules():
    df = _headlines([
        ("2026-09-10 12:30", ["ETHUSDT"], "positive", 0.8),            # kept
        ("2026-09-10 13:30", ["ETHUSDT", "BTCUSDT"], "positive", 0.5),  # two USDT tickers
        ("2026-09-10 14:30", ["ETHUSDT"], "neutral", 0.0),              # neutral label
        ("2026-09-11 00:00:00", ["ETHUSDT"], "negative", -0.5),         # date-only precision
        ("2026-09-11 08:15", '["XRPUSDT"]', "negative", -0.7),          # JSON string tickers, kept
        ("2026-09-11 09:15", ["AAPL"], "positive", 0.4),                # no USDT ticker
    ])
    out = ds.select_headlines(df)
    assert list(out["ticker"]) == ["ETHUSDT", "XRPUSDT"]


def test_declustering_merges_within_6h_of_the_anchor_and_averages():
    df = pd.DataFrame({"published": pd.to_datetime(["2026-09-10 08:00", "2026-09-10 11:00", "2026-09-10 13:30",
                                                     "2026-09-10 14:30", "2026-09-10 08:30"], utc=True),
                       "ticker": ["ETHUSDT"] * 4 + ["XRPUSDT"], "shock": [0.8, 0.4, 0.6, 0.2, 0.9]})
    out = ds.decluster(df).sort_values(["ticker", "published"]).reset_index(drop=True)
    eth = out[out.ticker == "ETHUSDT"]
    assert len(eth) == 2 and eth.iloc[0]["shock"] == pytest.approx(0.6)       # 08:00, 11:00 merged (3 h) and 13:30 (5.5 h): mean(.8,.4,.6)
    assert eth.iloc[1]["shock"] == pytest.approx(0.2)                          # 14:30 is 6.5 h after the anchor: a new event
    assert len(out[out.ticker == "XRPUSDT"]) == 1                              # other tickers never merge with it


def test_event_time_is_the_first_bar_opening_after_publication():
    assert ds.first_bar_after(pd.Timestamp("2026-09-10 12:30", tz="UTC")) == pd.Timestamp("2026-09-10 13:00", tz="UTC")
    assert ds.first_bar_after(pd.Timestamp("2026-09-10 13:00", tz="UTC")) == pd.Timestamp("2026-09-10 14:00", tz="UTC")   # on the hour: next bar


# ------------------------------------------------------------------ abnormal returns from prices (synthetic, noiseless)
def _prices(post_slope=1.5, extra=0.01, hours=24 * 60, seed=0):
    idx = pd.date_range("2026-08-01", periods=hours, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    btc_r = pd.Series(rng.normal(0, 0.005, hours), index=idx)
    t0 = pd.Timestamp("2026-09-10 13:00", tz="UTC")
    eth_r = 1.5 * btc_r
    after = idx >= t0
    eth_r[after] = post_slope * btc_r[after]
    if t0 in eth_r.index:
        eth_r[t0] += extra                                          # the planted reaction lands in the first bar after the headline
    mk = lambda r: np.exp(r.cumsum()) * 100.0
    return {"BTCUSDT": mk(btc_r), "ETHUSDT": mk(eth_r)}, btc_r, t0


def _one_headline():
    return _headlines([("2026-09-10 12:30", ["ETHUSDT"], "positive", 0.8)])


def test_abnormal_return_removes_the_market_move_and_keeps_the_planted_reaction():
    closes, _, _ = _prices()
    ev = ds.build_events(_one_headline(), closes)
    assert ev.n == 1 and ev.t0[0] == pd.Timestamp("2026-09-10 13:00", tz="UTC")
    assert np.allclose(ev.car[0], 0.01, atol=1e-9)              # exactly the planted 1%, held flat afterwards
    assert ev.pre[0] == pytest.approx(0.0, abs=1e-9)


def test_market_beta_uses_only_bars_before_the_event():
    # after t0 the coin's true beta changes to 3. If the estimate looked ahead it would not stay 1.5 and the 1.5*btc part would not cancel.
    closes, btc_r, t0 = _prices(post_slope=3.0)
    ev = ds.build_events(_one_headline(), closes)
    expected = 0.01 + np.cumsum((3.0 - 1.5) * btc_r[btc_r.index >= t0].to_numpy()[:48])
    assert np.allclose(ev.car[0], expected, atol=1e-9)


def test_btc_events_use_the_raw_return():
    closes, btc_r, t0 = _prices()
    df = _headlines([("2026-09-10 12:30", ["BTCUSDT"], "positive", 0.8)])
    ev = ds.build_events(df, closes)
    assert np.allclose(ev.car[0], np.cumsum(btc_r[btc_r.index >= t0].to_numpy()[:48]), atol=1e-9)


def test_events_without_a_full_48h_window_or_enough_history_are_dropped_and_counted():
    closes, _, _ = _prices(hours=24 * 40 + 10)                  # prices end 10 h after 2026-09-10 12:30: window incomplete
    ev = ds.build_events(_one_headline(), closes)
    assert ev.n == 0 and ev.dropped["incomplete_window"] == 1
    closes, _, _ = _prices()
    early = _headlines([("2026-08-02 12:30", ["ETHUSDT"], "positive", 0.8)])   # < 15 days of history before it
    assert ds.build_events(early, closes).dropped["short_history"] == 1
