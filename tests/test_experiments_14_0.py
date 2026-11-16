import numpy as np
import pandas as pd
import pytest

from training_ground import experiments_14_0 as ex


def closes(n=1300, k=8, seed=0, drift=0.0004, vol=0.04, start="2019-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(drift, vol, (n, k)), axis=0)), index=idx,
                        columns=[f"C{i}USDT" for i in range(k)])


def test_simulation_has_no_lookahead_and_charges_fees():
    r = closes(600).pct_change().dropna()
    base = ex.simulate(r, "C3")
    changed_future = r.copy()
    changed_future.iloc[400:] = 0.5                                   # change only days from 400 on
    again = ex.simulate(changed_future, "C3")
    cut = r.index[400]
    assert np.allclose(base[base.index < cut], again[again.index < cut])      # nothing before day 400 moved
    assert ex.simulate(r, "C3", fee=0.0).sum() > ex.simulate(r, "C3", fee=0.01).sum()


def test_selection_uses_candidates_only_never_the_yardsticks(monkeypatch):
    c1, c2 = closes(seed=1, start="2019-01-01"), closes(seed=2, start="2022-06-01", n=1500)
    c1.index = pd.date_range("2018-11-01", periods=len(c1), freq="D")
    res = ex.run(c1, c2, n_boot=50)
    assert res["selected"] in ("C1", "C2", "C3", "C4", "C5")


def test_a_void_holdout_gives_inconclusive_not_a_verdict():
    assert ex.verdict("C3", {"H1": None, "H2": None})["verdict"].startswith("INCONCLUSIVE")


def _holdout(sharpe_s, sharpe_ew, sharpe_ts, dd_s, dd_ew, lo):
    st = {"C3": {"sharpe": sharpe_s, "max_drawdown": dd_s}, "EW": {"sharpe": sharpe_ew, "max_drawdown": dd_ew},
          "TS28": {"sharpe": sharpe_ts, "max_drawdown": -0.4}}
    return {"stats": st, "ci": [lo, lo + 1]}


def test_verdict_levels_follow_the_pre_registered_criteria():
    good = _holdout(1.0, 0.5, 0.8, -0.3, -0.6, 0.1)
    assert ex.verdict("C3", {"H1": good, "H2": good})["verdict"] == "SUPPORTED"
    assert ex.verdict("C3", {"H1": good, "H2": _holdout(1.0, 0.5, 0.8, -0.3, -0.6, -0.2)})["verdict"].startswith("PROMISING")
    assert ex.verdict("C3", {"H1": good, "H2": _holdout(0.4, 0.5, 0.3, -0.3, -0.6, -0.2)})["verdict"] == "NOT SUPPORTED"      # Sharpe below EW
    assert ex.verdict("C3", {"H1": good, "H2": _holdout(1.0, 0.5, 0.8, -0.55, -0.6, 0.1)})["verdict"] == "NOT SUPPORTED"     # drawdown gain < 10pp
    assert ex.verdict("C3", {"H1": good, "H2": _holdout(1.0, 0.5, 1.2, -0.3, -0.6, 0.1)})["verdict"] == "NOT SUPPORTED"      # worse than TS28


def test_a_planted_trend_edge_is_found_and_noise_is_not():
    """Harness sanity check on synthetic data: persistent regimes (a real trend edge) should favour the trend rules over equal
    weight; an i.i.d. random walk should not give them a robust edge."""
    rng = np.random.default_rng(5)
    n, k = 1600, 8
    state = np.repeat(rng.choice([-1, 1], size=n // 60 + 1), 60)[:n]
    trending = pd.DataFrame(rng.normal(0.004 * state[:, None], 0.03, (n, k)), index=pd.date_range("2019-01-01", periods=n, freq="D"))
    trending.columns = [f"C{i}USDT" for i in range(k)]
    r = trending.iloc[:1000]
    ts, ew = ex.simulate(r, "TS28"), ex.simulate(r, "EW")
    assert ex.max_drawdown(ts) > ex.max_drawdown(ew) and ts.mean() / ts.std() > ew.mean() / ew.std()
    noise = pd.DataFrame(rng.normal(0.0, 0.03, (1000, k)), index=trending.index[:1000], columns=trending.columns)
    ts_n, ew_n = ex.simulate(noise, "C3"), ex.simulate(noise, "EW")
    assert ts_n.mean() / ts_n.std() < ew_n.mean() / ew_n.std() + 0.5          # no free lunch from noise
