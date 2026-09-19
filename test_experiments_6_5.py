"""Phase 6.5 harness: pass/fail logic and end-to-end plumbing on synthetic universes."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

import training_ground.experiments_6_5 as ex


def _universe(n=650, symbols=("AAA", "BBB", "CCC"), planted=False, seed=0):
    """Synthetic klines. If ``planted``, tomorrow's return depends on today's taker-buy ratio."""
    uni = {}
    for k, sym in enumerate(symbols):
        rng = np.random.default_rng(seed + k)
        ratio = np.clip(rng.normal(0.5, 0.06, n), 0.2, 0.8)
        ret = rng.normal(0, 0.01, n)
        if planted:
            ret[1:] += 0.15 * (ratio[:-1] - 0.5)          # a real, learnable order-flow effect
        close = 100 * np.exp(np.cumsum(ret))
        vol = rng.uniform(1000, 2000, n)
        kl = pd.DataFrame({"Open": close, "High": close * 1.004, "Low": close * 0.996, "Close": close, "Volume": vol,
                           "Trades": rng.integers(5000, 9000, n).astype(float), "TakerBuyBase": vol * ratio},
                          index=pd.date_range("2022-01-01", periods=n, freq="D"))
        fidx = pd.date_range("2022-01-01", periods=n * 3, freq="8h")
        uni[sym] = {"klines": kl, "funding": pd.Series(rng.normal(1e-4, 2e-4, len(fidx)), index=fidx)}
    return uni


# ----------------------------------------------------------------- decision logic

def _res(**over):
    base = {"symbols": 8, "auc_ci_bonferroni": [0.52, 0.56], "symbols_above_half": 7,
            "auc_first_half": 0.54, "auc_second_half": 0.53,
            "economics": {"rule_sharpe": 1.5, "hold_sharpe": 1.0}}
    base.update(over)
    return base


def test_a_result_meeting_every_criterion_is_a_finding():
    assert ex.criteria_verdict(_res())["FINDING"] is True


@pytest.mark.parametrize("override,failing", [
    ({"auc_ci_bonferroni": [0.49, 0.56]}, "1_ci_lower_above_half"),
    ({"symbols_above_half": 5}, "2_at_least_6_of_8_symbols_above_half"),
    ({"economics": {"rule_sharpe": 0.4, "hold_sharpe": 1.0}}, "3_rule_sharpe_beats_buy_hold"),
    ({"auc_second_half": 0.49}, "4_both_halves_above_half"),
])
def test_failing_any_single_criterion_means_no_finding(override, failing):
    v = ex.criteria_verdict(_res(**override))
    assert v[failing] is False and v["FINDING"] is False


def test_bonferroni_level_matches_the_pre_registration():
    assert ex.K_DIRECTIONAL == 7 and ex.LEVEL_ADJ == pytest.approx(1 - 0.05 / 7)
    assert (ex.SYMBOLS, ex.TRAIN, ex.RETRAIN, ex.FEE, ex.BLOCK) == (
        ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "LTCUSDT", "DOGEUSDT", "SOLUSDT"], 400, 20, 0.001, 10)


def test_economic_check_charges_fees_and_uses_non_overlapping_decisions():
    dates = pd.date_range("2024-01-01", periods=6)
    idx = pd.MultiIndex.from_product([["A"], dates], names=["symbol", "timestamp"])
    p = pd.Series([0.9, 0.9, 0.9, 0.9, 0.1, 0.1], index=idx)
    r = pd.Series([0.02, 0.02, 0.02, 0.02, -0.02, -0.02], index=idx)
    e1 = ex.economic_check(p, r, horizon=1)
    assert e1["decisions"] == 6 and e1["rule_return_pct"] < ((1.02 ** 4) * (1.02 ** 2) - 1) * 100   # fees bite
    e3 = ex.economic_check(p, r, horizon=3)
    assert e3["decisions"] == 2                                   # only dates 0 and 3


# --------------------------------------------------------- end to end (synthetic)

def test_planted_order_flow_signal_is_found_only_when_the_feature_is_included():
    uni = _universe(planted=True)
    with_taker = ex.run_pooled(uni, 1, "taker", "H4a", n_boot=100)
    without = ex.run_pooled(uni, 1, "none", "H2a", n_boot=100)
    assert with_taker["auc"] > 0.60, with_taker["auc"]
    assert with_taker["auc_ci_bonferroni"][0] > 0.5
    assert without["auc"] < with_taker["auc"] - 0.05              # the feature is what made the difference


def test_pure_noise_universe_produces_no_finding():
    uni = _universe(planted=False, seed=10)
    for extra in ("none", "both"):
        r = ex.run_pooled(uni, 1, extra, f"noise-{extra}", n_boot=100)
        assert r["criteria"]["FINDING"] is False, r["auc"]
        assert 0.42 < r["auc"] < 0.58


def test_horizon_h_experiment_and_volatility_experiment_run_end_to_end():
    uni = _universe(planted=False, seed=20)
    h = ex.run_h1(uni, 5, "H1a", n_boot=50)
    assert h["symbols"] == 3 and h["horizon"] == 5 and "economics" in h
    v = ex.run_h5(uni, n_boot=50)
    assert 0.3 < v["auc_naive"] < 0.7 and "gbm_adds_value" in v


def test_main_writes_results_json(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "load_universe", lambda *a, **k: _universe(planted=True))
    out = tmp_path / "res.json"
    assert ex.main(["--only", "H4a", "--n-boot", "50", "--out", str(out)]) == 0
    import json
    data = json.loads(out.read_text())
    assert "H4a" in data["results"] and data["protocol"]["train"] == 400
