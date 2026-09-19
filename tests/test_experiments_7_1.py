"""Phase 7.1 harness: no lookahead in the rebalance, correct cost accounting, finds planted allocation value."""
import json

import numpy as np
import pandas as pd
import pytest

import training_ground.experiments_7_1 as ex


@pytest.fixture(autouse=True)
def _isolated_experiment_log(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_LOG_PATH", str(tmp_path / "experiments.sqlite3"))


def _returns(n=800, vols=(0.01, 0.02, 0.04, 0.08), seed=0):
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.normal(0.0004, v, n) for v in vols])
    return pd.DataFrame(x, index=pd.date_range("2022-01-01", periods=n), columns=list("ABCD"))


def _universe(returns):
    return {c: {"klines": pd.DataFrame({"Close": 100 * (1 + returns[c]).cumprod()}, index=returns.index)}
            for c in returns.columns}


def test_first_day_is_the_first_after_the_lookback_and_every_method_covers_the_same_dates():
    r = _returns()
    lens = {m: ex.simulate(r, m)[0].index[0] for m in ex.METHODS}
    assert len(set(lens.values())) == 1 and list(lens.values())[0] == r.index[ex.LOOKBACK]


def test_weights_at_a_rebalance_use_only_earlier_rows():
    """Rewriting returns from day i onward must not change the return earned on day i's predecessors, nor the
    weights chosen at day i (only rows < i are visible to the optimiser)."""
    r = _returns()
    base, _ = ex.simulate(r, "risk_parity")
    mutated = r.copy()
    mutated.iloc[500:] = np.random.default_rng(9).normal(0, 0.5, mutated.iloc[500:].shape)
    other, _ = ex.simulate(mutated, "risk_parity")
    pd.testing.assert_series_equal(base.iloc[:250], other.iloc[:250])          # everything before day 500 is identical


def test_equal_weight_on_identical_assets_matches_the_hand_calculation_including_costs():
    idx = pd.date_range("2024-01-01", periods=300)
    r = pd.DataFrame({"A": 0.001, "B": 0.001}, index=idx)
    s, diag = ex.simulate(r, "equal", lookback=250, every=30, fee=0.01)
    assert s.iloc[0] == pytest.approx(0.001 - 0.01 * 1.0)                    # first allocation buys 100% of capital
    assert s.iloc[1] == pytest.approx(0.001)                                # no trade, no cost
    assert diag["rebalances"] == 2 and diag["avg_effective_assets"] == pytest.approx(2.0)


def test_risk_based_allocation_is_found_when_low_vol_assets_earn_the_same_return_per_unit_risk():
    """Planted: every asset has the same Sharpe, but vols differ 8x. Equal weight is dominated by the risky asset;
    inverse-vol / risk parity diversify across risk and should show a better Sharpe. (Length and Sharpe were chosen
    so the effect clears the pre-registered 99% level on every seed tried -- a shorter/weaker fixture simply lacks
    the statistical power, which is a test-design limit, not a code bug.)"""
    rng = np.random.default_rng(2)
    n, vols = 2600, (0.01, 0.02, 0.04, 0.08)
    x = np.column_stack([rng.normal(0.12 * v, v, n) for v in vols])
    r = pd.DataFrame(x, index=pd.date_range("2016-01-01", periods=n), columns=list("ABCD"))
    res = ex.run(_universe(r), n_boot=400)
    found = {c["method"]: c["criteria"]["FINDING"] for c in res["comparisons"]}
    assert found["inverse_vol"] and found["risk_parity"], res["comparisons"]
    assert res["stats"]["inverse_vol"]["ann_vol"] < res["stats"]["equal"]["ann_vol"]


def test_no_finding_when_all_assets_are_interchangeable():
    rng = np.random.default_rng(2)
    n = 1400
    r = pd.DataFrame(rng.normal(0.0004, 0.02, (n, 4)), index=pd.date_range("2020-01-01", periods=n), columns=list("ABCD"))
    res = ex.run(_universe(r), n_boot=300)
    assert not any(c["criteria"]["FINDING"] for c in res["comparisons"])


def test_protocol_matches_the_pre_registration():
    assert (ex.LOOKBACK, ex.REBALANCE_EVERY, ex.FEE, ex.BLOCK) == (250, 30, 0.001, 10)
    assert ex.METHODS == ["equal", "inverse_vol", "min_variance", "risk_parity", "hrp", "max_sharpe"]
    assert ex.LEVEL == pytest.approx(0.99)


def test_main_writes_results_and_logs_each_comparison(tmp_path, monkeypatch):
    from core.experiment_log import ExperimentLog
    monkeypatch.setattr(ex, "load_universe", lambda *a, **k: _universe(_returns(700, seed=5)))
    out = tmp_path / "r.json"
    assert ex.main(["--n-boot", "100", "--out", str(out)]) == 0
    assert len(json.loads(out.read_text())["comparisons"]) == 5
    runs = ExperimentLog().list_runs(tag="phase-7.1")
    assert len(runs) == 5 and all(r["model_type"] == "allocation" for r in runs)
