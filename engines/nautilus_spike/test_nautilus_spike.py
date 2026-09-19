"""Spike tests. Skipped automatically where nautilus_trader is not installed (Python 3.9 base env, CI)."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("nautilus_trader")

from engines.nautilus_spike import compare as c  # noqa: E402


def wavy(n=1000, seed=3):
    """Trends that reverse every ~40 bars so the EMA cross fires repeatedly, WITH overnight gaps (open != prior close)
    so next-open and same-bar-close fills give different prices."""
    r = np.random.default_rng(seed)
    drift = np.repeat(np.tile([0.01, -0.01], n // 80 + 1), 40)[:n]
    close = 20_000 * np.cumprod(1 + drift + r.normal(0, 0.008, n))
    open_ = np.r_[close[0], close[:-1]] * (1 + r.normal(0, 0.004, n))
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.002, "Low": np.minimum(open_, close) * 0.998,
                         "Close": close, "Volume": 1e4}, index=pd.date_range("2023-01-01", periods=n))


def test_engines_agree_on_signals_after_indicator_warmup():
    df = wavy()
    bt, nt = c.run_backtrader(df), c.run_nautilus(df)
    assert bt["orders"] == nt["orders"] > 8
    bt_dates = [t.date() for t, *_ in bt["fills"]]
    nt_dates = [t.date() for t, *_ in nt["fills"]]
    # Known difference: the engines seed the EMA differently, so the FIRST cross can land a day apart. Everything after agrees.
    assert abs((bt_dates[0] - nt_dates[0]).days) <= 3
    assert bt_dates[1:] == nt_dates[1:]


def test_documented_fill_semantics_nautilus_fills_at_signal_bar_close_backtrader_at_next_open():
    df = wavy()
    bt, nt = c.run_backtrader(df), c.run_nautilus(df)
    t_bt, _, p_bt, _ = bt["fills"][1]                                # a fill after warm-up, same signal in both engines
    t_nt, _, p_nt, _ = nt["fills"][1]
    assert t_bt.date() == t_nt.date()
    assert p_bt == pytest.approx(float(df.loc[t_bt, "Open"]), rel=1e-6)                      # next bar's open
    assert p_nt == pytest.approx(float(df["Close"].shift(1).loc[t_bt]), rel=1e-6)            # the signal bar's close
    assert abs(p_bt - p_nt) > 1.0                                    # the gap makes the two semantics visibly different


def test_score_uses_one_metric_definition_for_both_engines():
    eq = pd.Series([100.0, 101.0, 99.0, 102.0], index=pd.date_range("2024-01-01", periods=4))
    s = c.score(eq)
    assert s["final_value"] == 102.0 and s["max_drawdown"] < 0
