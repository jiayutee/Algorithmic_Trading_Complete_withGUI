import numpy as np
import pandas as pd

from training_ground import experiments_6_7 as e


def rets(n=600, k=8, seed=0, drift=0.0):
    r = np.random.default_rng(seed)
    return pd.DataFrame(r.normal(drift, 0.03, (n, k)), index=pd.date_range("2022-01-01", periods=n), columns=[f"S{i}" for i in range(k)])


def test_target_weights_shapes():
    h = rets(28)
    assert abs(e.target_weights(h, "ew").sum() - 1) < 1e-12
    x = e.target_weights(h, "xsmom")
    assert (x > 0).sum() == 3 and abs(x.sum() - 1) < 1e-12
    t = e.target_weights(h, "tsmom")
    assert t.sum() <= 1 + 1e-12 and set(t.unique()) <= {0.0, 1 / 8}


def test_xsmom_picks_the_trailing_winners():
    h = rets(28)
    h["S5"] += 0.01; h["S1"] += 0.009; h["S7"] += 0.008
    assert set(e.target_weights(h, "xsmom")[lambda w: w > 0].index) == {"S5", "S1", "S7"}


def test_no_lookahead_weights_ignore_current_and_future_rows():
    r = rets(200)
    i = 100
    base = e.target_weights(r.iloc[i - 28:i], "xsmom")
    r2 = r.copy(); r2.iloc[i:] = 0.5                                # wildly different present/future
    assert base.equals(e.target_weights(r2.iloc[i - 28:i], "xsmom"))


def planted(n=900, seed=1):
    """Persistent winners: each asset has its own slowly-changing drift, so past winners keep winning."""
    r = np.random.default_rng(seed)
    drifts = np.zeros((n, 8))
    for j in range(8):
        d = np.repeat(r.normal(0, 0.004, n // 60 + 1), 60)[:n]      # regime of ~60 days
        drifts[:, j] = d
    return pd.DataFrame(drifts + r.normal(0, 0.01, (n, 8)), index=pd.date_range("2022-01-01", periods=n),
                        columns=[f"S{i}" for i in range(8)])


def test_detects_planted_cross_sectional_momentum():
    res = e.run(planted(), n_boot=300)
    assert res["stats"]["xsmom"]["sharpe"] > res["stats"]["ew"]["sharpe"]
    assert res["comparisons"][0]["sharpe_diff"] > 0


def test_random_walk_gives_no_finding():
    res = e.run(rets(800, seed=3), n_boot=300)
    assert not any(c["criteria"]["FINDING"] for c in res["comparisons"])


def test_costs_are_charged():
    r = rets(300, seed=4)
    free, _ = e.simulate(r, "xsmom", fee=0.0)
    paid, diag = e.simulate(r, "xsmom", fee=0.01)
    assert paid.sum() < free.sum() and diag["rebalances"] > 10


def test_same_evaluation_dates_for_all_arms():
    r = rets(300)
    idx = [e.simulate(r, m)[0].index for m in ("ew", "xsmom", "tsmom")]
    assert idx[0].equals(idx[1]) and idx[1].equals(idx[2])
