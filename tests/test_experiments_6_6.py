"""Phase 6.6 harness: plumbing, and proof it finds vol-targeting value when it exists and not when it doesn't."""
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

import training_ground.experiments_6_6 as ex


@pytest.fixture(autouse=True)
def _isolated_experiment_log(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_LOG_PATH", str(tmp_path / "experiments.sqlite3"))


def _universe(n=1400, symbols=("AAA", "BBB", "CCC", "DDD"), clustered=True, seed=100):
    """clustered=True: strongly persistent volatility regimes with a CONSTANT drift, so calm days pay more
    per unit of risk -- the situation where sizing by forecast volatility should help. (Regime strength and
    length were chosen so the planted effect is reliably detectable at the pre-registered 98.3% level; a
    weaker/shorter version has too little power and is a test-design limit, not a code bug.)
    clustered=False: volatility is constant, so there is nothing to time."""
    uni = {}
    for k, sym in enumerate(symbols):
        rng = np.random.default_rng(seed + k)
        if clustered:
            log_sigma = np.zeros(n)
            for t in range(1, n):
                log_sigma[t] = 0.98 * log_sigma[t - 1] + 0.18 * rng.normal()
            sigma = 0.02 * np.exp(log_sigma)
        else:
            sigma = np.full(n, 0.02)
        ret = 0.002 + sigma * rng.normal(size=n)
        close = 100 * np.exp(np.cumsum(ret))
        spread = sigma * (1.6 + 0.2 * rng.normal(size=n)).clip(0.5)
        kl = pd.DataFrame({"Open": close, "High": close * np.exp(spread / 2), "Low": close * np.exp(-spread / 2),
                           "Close": close, "Volume": rng.uniform(500, 1500, n)},
                          index=pd.date_range("2020-01-01", periods=n, freq="D"))
        uni[sym] = {"klines": kl}
    return uni


def test_vol_targeting_is_found_when_volatility_clusters_and_the_model_forecasts_it():
    res = ex.run(_universe(clustered=True), n_boot=300)
    c1, c2, c3 = res["comparisons"]
    assert c1["criteria"]["FINDING"] and c2["criteria"]["FINDING"], (c1, c2)     # timing beats matched fixed exposure
    s = res["stats"]
    assert s["GBM-VT"]["sharpe"] > s["FIXED (matched to GBM-VT)"]["sharpe"]
    assert s["GBM-VT"]["max_drawdown"] >= s["FIXED (matched to GBM-VT)"]["max_drawdown"]
    assert res["gbm_naive_forecast_log_correlation"] > 0.5                         # the two forecasts broadly agree


def test_no_finding_when_there_is_no_volatility_to_time():
    res = ex.run(_universe(clustered=False, seed=50), n_boot=300)
    assert not any(c["criteria"]["FINDING"] for c in res["comparisons"]), res["comparisons"]


def test_the_matched_fixed_benchmark_has_the_same_average_exposure_and_uses_the_same_dates():
    res = ex.run(_universe(n=900, seed=10), n_boot=100)
    s = res["stats"]
    assert s["FIXED (matched to GBM-VT)"]["avg_exposure"] == pytest.approx(s["GBM-VT"]["avg_exposure"], abs=1e-9)
    assert s["FIXED (matched to NAIVE-VT)"]["avg_exposure"] == pytest.approx(s["NAIVE-VT"]["avg_exposure"], abs=1e-9)
    assert s["BUY&HOLD (100%)"]["avg_exposure"] == 1.0
    assert 0 < s["GBM-VT"]["avg_exposure"] <= 1.0 and s["GBM-VT"]["annual_turnover"] > 0
    assert res["protocol"]["k"] == 0.7 and res["protocol"]["band"] == 0.10 and res["protocol"]["fee"] == 0.001


def test_protocol_constants_match_the_pre_registration():
    assert (ex.K_EXPOSURE, ex.CAP, ex.BAND, ex.N_COMPARISONS, ex.MIN_HISTORY) == (0.7, 1.0, 0.10, 3, 30)
    assert ex.LEVEL == pytest.approx(1 - 0.05 / 3)
    assert (ex.TRAIN, ex.RETRAIN, ex.FEE, ex.BLOCK) == (400, 20, 0.001, 10)


def test_finding_needs_all_three_criteria():
    idx = pd.date_range("2024-01-01", periods=400)
    rng = np.random.default_rng(1)
    base = pd.Series(rng.normal(0.0005, 0.02, 400), index=idx)
    better = base + 0.002                                         # a consistent edge -> passes 1 and 2
    worse_dd = {"max_drawdown": -0.5}
    ok = ex.compare("t", better, base, {"max_drawdown": -0.1}, {"max_drawdown": -0.2}, n_boot=200)
    assert ok["criteria"]["FINDING"] is True
    bad = ex.compare("t", better, base, worse_dd, {"max_drawdown": -0.2}, n_boot=200)    # deeper drawdown
    assert bad["criteria"]["3_max_drawdown_not_worse"] is False and bad["criteria"]["FINDING"] is False
    tie = ex.compare("t", base, base + rng.normal(0, 1e-6, 400), {"max_drawdown": -0.1}, {"max_drawdown": -0.1}, n_boot=200)
    assert tie["criteria"]["1_ci_lower_above_zero"] is False


def test_main_writes_results_and_logs_each_comparison(tmp_path, monkeypatch):
    from core.experiment_log import ExperimentLog
    monkeypatch.setattr(ex, "load_universe", lambda *a, **k: _universe(n=900, seed=20))
    out = tmp_path / "r.json"
    assert ex.main(["--n-boot", "100", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert len(data["comparisons"]) == 3 and "GBM-VT" in data["stats"]
    runs = ExperimentLog().list_runs(tag="phase-6.6")
    assert {r["name"] for r in runs} == {"C1 NAIVE-VT vs FIXED matched", "C2 GBM-VT vs FIXED matched", "C3 GBM-VT vs NAIVE-VT"}
    assert all("sharpe_diff" in r["metrics"] and len(r["git_commit"]) == 40 for r in runs)
