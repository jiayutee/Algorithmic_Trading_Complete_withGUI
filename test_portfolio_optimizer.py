"""Phase 7.0: allocators checked against closed-form results and structural properties."""
import numpy as np
import pandas as pd
import pytest

from core.portfolio_optimizer import (
    METHODS, hrp, inverse_volatility, min_variance, risk_contributions, risk_parity, weights,
)


def _returns(cov, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.multivariate_normal(np.zeros(len(cov)), cov, size=n)
    return pd.DataFrame(x, columns=[f"A{i}" for i in range(len(cov))])


def _cov2(s1, s2, rho):
    return np.array([[s1 ** 2, rho * s1 * s2], [rho * s1 * s2, s2 ** 2]])


@pytest.mark.parametrize("method", METHODS)
def test_every_method_gives_long_only_fully_invested_weights(method):
    r = _returns(np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]]), n=1500)
    w = weights(r, method)
    assert w.sum() == pytest.approx(1.0) and (w >= -1e-12).all() and list(w.index) == list(r.columns)


def test_identical_independent_assets_get_equal_weight_from_every_method():
    r = _returns(np.eye(4) * 0.04, n=6000)
    for m in ("equal", "inverse_vol", "risk_parity", "min_variance", "hrp"):
        np.testing.assert_allclose(weights(r, m).to_numpy(), 0.25, atol=0.04, err_msg=m)


def test_inverse_vol_matches_its_formula():
    cov = np.diag([0.01, 0.04, 0.09])                      # vols 0.1, 0.2, 0.3
    expected = np.array([1 / 0.1, 1 / 0.2, 1 / 0.3]); expected /= expected.sum()
    np.testing.assert_allclose(inverse_volatility(cov), expected)


@pytest.mark.parametrize("rho", [-0.5, 0.0, 0.3, 0.8])
def test_two_asset_risk_parity_equalises_risk_and_weights_inverse_to_volatility(rho):
    """Known result: for two assets, equal risk contribution means w1/w2 = sigma2/sigma1 for ANY correlation."""
    s1, s2 = 0.2, 0.5
    cov = _cov2(s1, s2, rho)
    w = risk_parity(cov)
    rc = risk_contributions(w, cov)
    assert rc == pytest.approx([0.5, 0.5], abs=1e-4)
    assert w[0] / w[1] == pytest.approx(s2 / s1, rel=1e-3)


def test_risk_parity_equalises_risk_across_many_correlated_assets():
    cov = np.array([[0.04, 0.012, 0.0, 0.006],
                    [0.012, 0.09, 0.02, 0.0],
                    [0.0, 0.02, 0.16, 0.03],
                    [0.006, 0.0, 0.03, 0.25]])
    rc = risk_contributions(risk_parity(cov), cov)
    assert rc == pytest.approx([0.25] * 4, abs=1e-3)


def test_min_variance_matches_the_two_asset_closed_form():
    s1, s2, rho = 0.2, 0.4, 0.3
    cov = _cov2(s1, s2, rho)
    w1 = (s2 ** 2 - rho * s1 * s2) / (s1 ** 2 + s2 ** 2 - 2 * rho * s1 * s2)
    np.testing.assert_allclose(min_variance(cov), [w1, 1 - w1], atol=1e-4)


def test_min_variance_never_has_more_variance_than_equal_weight():
    cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.36]])
    w = min_variance(cov)
    assert w @ cov @ w <= (np.full(3, 1 / 3) @ cov @ np.full(3, 1 / 3)) + 1e-9


def test_hrp_gives_an_isolated_asset_more_weight_than_each_of_a_tightly_correlated_pair():
    """A and B move together (one bet in two wrappers); C is independent. HRP should not double-count the pair."""
    rng = np.random.default_rng(3)
    common = rng.normal(0, 0.2, 5000)
    r = pd.DataFrame({"A": common + rng.normal(0, 0.02, 5000), "B": common + rng.normal(0, 0.02, 5000),
                      "C": rng.normal(0, 0.2, 5000)})
    w = weights(r, "hrp")
    assert w["C"] > w["A"] and w["C"] > w["B"] and w["C"] == pytest.approx(0.5, abs=0.06)


def test_max_sharpe_prefers_the_asset_with_the_better_risk_adjusted_return():
    rng = np.random.default_rng(4)
    r = pd.DataFrame({"good": rng.normal(0.004, 0.02, 3000), "meh": rng.normal(0.0005, 0.02, 3000)})
    w = weights(r, "max_sharpe")
    assert w["good"] > w["meh"]


def test_single_asset_and_bad_inputs():
    r = pd.DataFrame({"only": np.random.default_rng(0).normal(0, 0.01, 50)})
    assert weights(r, "hrp").iloc[0] == 1.0 and weights(r, "risk_parity").iloc[0] == 1.0
    with pytest.raises(ValueError):
        weights(r, "kelly")
    with pytest.raises(ValueError):
        weights(r.iloc[:2], "equal")


def test_nans_are_dropped_and_columns_order_is_preserved():
    r = _returns(np.diag([0.01, 0.04, 0.09]), n=500)[["A2", "A0", "A1"]]
    r.iloc[3, 0] = np.nan
    w = weights(r, "inverse_vol")
    assert list(w.index) == ["A2", "A0", "A1"] and w["A0"] > w["A1"] > w["A2"]
