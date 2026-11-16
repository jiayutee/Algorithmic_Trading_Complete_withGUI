import numpy as np
import pandas as pd
import pytest

from core import portfolio_rules as pr


def rets(n=300, k=6, seed=0, drift=0.0, vol=0.03):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame(rng.normal(drift, vol, (n, k)), index=idx, columns=[f"C{i}USDT" for i in range(k)])


@pytest.mark.parametrize("rule", list(pr.RULES))
def test_every_rule_is_long_only_and_never_exceeds_one(rule):
    for seed in range(5):
        w = pr.target_weights(rule, rets(seed=seed))
        assert (w >= -1e-12).all() and w.sum() <= 1.0 + 1e-9 and list(w.index) == list(rets().columns)


def test_rules_need_enough_history():
    with pytest.raises(ValueError, match="history"):
        pr.target_weights("C1", rets(n=50))


def test_ew_is_equal_and_fully_invested():
    w = pr.target_weights("EW", rets(k=5))
    assert np.allclose(w, 0.2) and w.sum() == pytest.approx(1.0)


def test_ts28_holds_only_coins_whose_28_day_return_is_positive():
    r = rets(k=3)
    r.iloc[-28:, 0] = 0.01                  # coin 0 rising
    r.iloc[-28:, 1] = -0.01                 # coin 1 falling
    w = pr.target_weights("TS28", r)
    assert w.iloc[0] == pytest.approx(1 / 3) and w.iloc[1] == 0.0


def test_vote_is_the_share_of_positive_lookbacks():
    r = rets(k=1)
    r.iloc[-112:, 0] = 0.001                # up over every window
    assert pr.vote(r).iloc[0] == 1.0
    r.iloc[-112:, 0] = -0.001
    assert pr.vote(r).iloc[0] == 0.0
    r.iloc[-112:-14, 0] = 0.01              # long windows positive, the last 14 days down
    r.iloc[-14:, 0] = -0.04
    assert 0.0 < pr.vote(r).iloc[0] < 1.0


def test_inverse_vol_gives_the_calmer_coin_more_weight_and_sums_to_one():
    r = rets(k=2, vol=0.01)
    r.iloc[:, 1] *= 4
    w = pr.inverse_vol(r)
    assert w.iloc[0] > w.iloc[1] and w.sum() == pytest.approx(1.0)


def test_c4_holds_at_most_four_and_only_trending_coins():
    r = rets(k=8, seed=3)
    for i in range(8):
        r.iloc[-56:, i] = 0.002 * (i + 1)
    r.iloc[-28:, 7] = -0.02                 # the strongest 56d coin has turned down over 28d: not eligible
    w = pr.target_weights("C4", r)
    assert (w > 0).sum() == 4 and w.iloc[7] == 0.0 and w.sum() == pytest.approx(1.0)


def test_c5_never_levers_and_cuts_exposure_when_volatility_is_high():
    calm, wild = rets(k=6, vol=0.005, drift=0.002), rets(k=6, vol=0.08, drift=0.002)
    wc, ww = pr.target_weights("C5", calm), pr.target_weights("C5", wild)
    assert wc.sum() <= 1.0 + 1e-9 and ww.sum() < wc.sum() * 0.8
    assert pr.predicted_vol(wild, ww) <= pr.VOL_TARGET * 1.01


def test_weights_use_only_the_history_they_are_given():
    r = rets()
    a = pr.target_weights("C3", r.iloc[:200])
    r2 = r.copy()
    r2.iloc[200:] = 5.0                     # absurd future data must not matter
    b = pr.target_weights("C3", r2.iloc[:200])
    assert a.equals(b)
