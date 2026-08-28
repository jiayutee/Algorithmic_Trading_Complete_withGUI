"""Regression tests for core/volatility_lab.py — Volatility Clustering analytics.

Key correctness property under test:
  A synthetic GARCH-like series (alternating low-vol / high-vol regimes) should
  exhibit HIGHER lag-1 ACF of |returns| and LOWER Ljung-Box p-value than an
  i.i.d. Gaussian series of the same length, confirming that the volatility
  clustering measures actually detect clustering.

All tests use ~/miniconda3/bin/python3 (base env).
"""

import math
import pytest
import numpy as np

from core.volatility_lab import (
    rolling_annualized_volatility,
    excess_kurtosis,
    acf_abs_returns,
    ljung_box_test,
    same_sign_rate,
    permutation_test,
    regime_tape,
    suggest_position_size,
    compute_volatility_clustering_report,
)


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------

def _iid_returns(n: int = 500, seed: int = 0) -> list:
    """i.i.d. Gaussian returns — no volatility clustering."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.01, n).tolist()


def _clustered_returns(n: int = 500, seed: int = 0) -> list:
    """GARCH-like returns: alternate 50-bar low-vol and high-vol regimes.

    Low-vol blocks:  σ = 0.005
    High-vol blocks: σ = 0.040

    This produces strong autocorrelation in |returns| at short lags and
    a very low Ljung-Box p-value — the signature of volatility clustering.
    """
    rng = np.random.default_rng(seed)
    out: list = []
    block = 50
    while len(out) < n:
        # Low-vol block
        remaining = n - len(out)
        out.extend(rng.normal(0.0, 0.005, min(block, remaining)).tolist())
        remaining = n - len(out)
        if remaining <= 0:
            break
        # High-vol block
        out.extend(rng.normal(0.0, 0.040, min(block, remaining)).tolist())
    return out[:n]


# ---------------------------------------------------------------------------
# 1. rolling_annualized_volatility
# ---------------------------------------------------------------------------

class TestRollingAnnualizedVolatility:

    def test_empty_input_returns_empty(self):
        assert rolling_annualized_volatility([]) == []

    def test_first_window_minus_one_are_none(self):
        rets = _iid_returns(100)
        window = 21
        vols = rolling_annualized_volatility(rets, window=window)
        assert len(vols) == 100
        for i in range(window - 1):
            assert vols[i] is None, f"Position {i} should be None"

    def test_values_after_window_are_floats(self):
        rets = _iid_returns(100)
        vols = rolling_annualized_volatility(rets, window=21)
        valid = [v for v in vols if v is not None]
        assert len(valid) > 0
        for v in valid:
            assert isinstance(v, float)
            assert v >= 0.0

    def test_annualization_increases_scale(self):
        """With ann_factor=252 vol should be ~sqrt(252) times raw daily std."""
        rets = _iid_returns(100, seed=3)
        raw_std = float(np.std(rets[-21:], ddof=1))
        vols = rolling_annualized_volatility(rets, window=21, ann_factor=252)
        last_vol = vols[-1]
        assert last_vol == pytest.approx(raw_std * math.sqrt(252), rel=0.01)

    def test_high_vol_regime_higher_than_low_vol(self):
        """High-vol blocks should produce larger rolling vol estimates."""
        low_rets = [0.001] * 100    # near zero vol
        high_rets = [0.05, -0.05] * 50  # high vol
        low_vols = [v for v in rolling_annualized_volatility(low_rets, window=21) if v is not None]
        high_vols = [v for v in rolling_annualized_volatility(high_rets, window=21) if v is not None]
        assert np.mean(high_vols) > np.mean(low_vols)


# ---------------------------------------------------------------------------
# 2. excess_kurtosis
# ---------------------------------------------------------------------------

class TestExcessKurtosis:

    def test_fewer_than_4_obs_returns_nan(self):
        assert math.isnan(excess_kurtosis([0.01, -0.01, 0.02]))

    def test_normal_distribution_approx_zero(self):
        """Large sample from N(0,1) should give kurtosis near 0."""
        rng = np.random.default_rng(99)
        rets = rng.normal(0, 1, 5000).tolist()
        ek = excess_kurtosis(rets)
        assert abs(ek) < 0.5, f"Expected ~0 for normal sample, got {ek:.3f}"

    def test_fat_tail_distribution_positive_kurtosis(self):
        """A heavy-tailed distribution should yield positive excess kurtosis."""
        rng = np.random.default_rng(7)
        # Student-t with 3 df has very heavy tails (kurtosis = inf in theory, large in practice)
        rets = rng.standard_t(df=3, size=2000).tolist()
        ek = excess_kurtosis(rets)
        assert ek > 1.0, f"Expected positive excess kurtosis for t(3), got {ek:.3f}"

    def test_returns_float(self):
        rets = _iid_returns(200)
        ek = excess_kurtosis(rets)
        assert isinstance(ek, float)

    def test_empty_input_returns_nan(self):
        assert math.isnan(excess_kurtosis([]))


# ---------------------------------------------------------------------------
# 3. acf_abs_returns
# ---------------------------------------------------------------------------

class TestAcfAbsReturns:

    def test_returns_dict_with_requested_lags(self):
        rets = _iid_returns(300)
        result = acf_abs_returns(rets, lags=(1, 5, 22))
        assert set(result.keys()) == {1, 5, 22}

    def test_values_are_finite_for_sufficient_data(self):
        rets = _iid_returns(300)
        result = acf_abs_returns(rets, lags=(1, 5))
        for lag, val in result.items():
            assert not math.isnan(val), f"ACF at lag {lag} should be finite"

    def test_insufficient_data_returns_nan(self):
        result = acf_abs_returns([0.01, -0.01], lags=(1, 5))
        for lag, val in result.items():
            assert math.isnan(val), f"Expected nan for lag {lag} with short series"

    def test_clustered_series_higher_acf1_than_iid(self):
        """Clustered series must have higher lag-1 ACF(|r|) than i.i.d. series."""
        n = 500
        clustered = _clustered_returns(n, seed=0)
        iid = _iid_returns(n, seed=0)
        acf_clustered = acf_abs_returns(clustered, lags=(1,))[1]
        acf_iid = acf_abs_returns(iid, lags=(1,))[1]
        assert acf_clustered > acf_iid, (
            f"Clustered ACF(|r|) lag-1 {acf_clustered:.4f} should exceed "
            f"i.i.d. {acf_iid:.4f}"
        )

    def test_acf_values_in_valid_range(self):
        """ACF coefficients must be in [-1, 1]."""
        rets = _clustered_returns(300)
        result = acf_abs_returns(rets, lags=(1, 5, 22))
        for lag, val in result.items():
            if not math.isnan(val):
                assert -1.0 <= val <= 1.0, f"ACF at lag {lag} out of range: {val}"


# ---------------------------------------------------------------------------
# 4. ljung_box_test
# ---------------------------------------------------------------------------

class TestLjungBoxTest:

    def test_returns_dict_with_required_keys(self):
        rets = _iid_returns(200)
        lb = ljung_box_test(rets)
        assert "statistic" in lb
        assert "p_value" in lb

    def test_insufficient_data_returns_nan(self):
        lb = ljung_box_test([0.01, -0.01], lag=22)
        assert math.isnan(lb["statistic"])
        assert math.isnan(lb["p_value"])

    def test_statistic_positive_for_sufficient_data(self):
        rets = _iid_returns(300)
        lb = ljung_box_test(rets, lag=22)
        if not math.isnan(lb["statistic"]):
            assert lb["statistic"] >= 0.0

    def test_p_value_in_unit_interval(self):
        rets = _iid_returns(300)
        lb = ljung_box_test(rets, lag=22)
        if not math.isnan(lb["p_value"]):
            assert 0.0 <= lb["p_value"] <= 1.0

    def test_clustered_series_lower_pvalue_than_iid(self):
        """The key correctness property: clustered vol → lower Ljung-Box p-value."""
        n = 500
        clustered = _clustered_returns(n, seed=0)
        iid = _iid_returns(n, seed=0)
        lb_clustered = ljung_box_test(clustered, lag=22)
        lb_iid = ljung_box_test(iid, lag=22)
        # Skip if either is nan (degenerate data)
        if math.isnan(lb_clustered["p_value"]) or math.isnan(lb_iid["p_value"]):
            pytest.skip("Ljung-Box returned nan — insufficient data or numerical issue")
        assert lb_clustered["p_value"] < lb_iid["p_value"], (
            f"Clustered p-value {lb_clustered['p_value']:.4f} should be lower "
            f"than i.i.d. {lb_iid['p_value']:.4f}"
        )

    def test_strongly_clustered_series_low_pvalue(self):
        """Strongly clustered volatility should yield p < 0.05."""
        clustered = _clustered_returns(500, seed=0)
        lb = ljung_box_test(clustered, lag=22)
        assert lb["p_value"] < 0.05, (
            f"Expected p < 0.05 for clustered series, got {lb['p_value']:.4f}"
        )


# ---------------------------------------------------------------------------
# 5. same_sign_rate
# ---------------------------------------------------------------------------

class TestSameSignRate:

    def test_fewer_than_2_obs_returns_nan(self):
        assert math.isnan(same_sign_rate([0.01]))
        assert math.isnan(same_sign_rate([]))

    def test_alternating_signs_gives_zero_rate(self):
        rets = [0.01, -0.01, 0.01, -0.01, 0.01, -0.01]
        ssr = same_sign_rate(rets)
        assert ssr == pytest.approx(0.0)

    def test_same_signs_gives_one_rate(self):
        rets = [0.01, 0.02, 0.03, 0.04]
        ssr = same_sign_rate(rets)
        assert ssr == pytest.approx(1.0)

    def test_result_in_unit_interval(self):
        rets = _iid_returns(200)
        ssr = same_sign_rate(rets)
        assert 0.0 <= ssr <= 1.0


# ---------------------------------------------------------------------------
# 6. permutation_test
# ---------------------------------------------------------------------------

class TestPermutationTest:

    def test_returns_required_keys(self):
        rets = _iid_returns(200)
        pt = permutation_test(rets, n_permutations=50, seed=1)
        for key in ("observed", "shuffled_mean", "shuffled_std", "p_value", "lift_pts"):
            assert key in pt

    def test_p_value_in_unit_interval(self):
        rets = _iid_returns(200)
        pt = permutation_test(rets, n_permutations=50, seed=1)
        assert 0.0 <= pt["p_value"] <= 1.0

    def test_lift_pts_consistent_with_observed_and_mean(self):
        """lift_pts == (observed - shuffled_mean) * 100"""
        rets = _iid_returns(200)
        pt = permutation_test(rets, n_permutations=50, seed=1)
        expected_lift = (pt["observed"] - pt["shuffled_mean"]) * 100.0
        assert pt["lift_pts"] == pytest.approx(expected_lift, rel=1e-6)

    def test_seed_reproducibility(self):
        rets = _iid_returns(200)
        pt1 = permutation_test(rets, n_permutations=50, seed=42)
        pt2 = permutation_test(rets, n_permutations=50, seed=42)
        assert pt1["p_value"] == pt2["p_value"]
        assert pt1["shuffled_mean"] == pt2["shuffled_mean"]

    def test_different_seeds_may_differ(self):
        rets = _iid_returns(200)
        pt1 = permutation_test(rets, n_permutations=100, seed=1)
        pt2 = permutation_test(rets, n_permutations=100, seed=999)
        # Very unlikely to be identical with different seeds on 100 perms
        # (just check they're both finite)
        assert not math.isnan(pt1["p_value"])
        assert not math.isnan(pt2["p_value"])

    def test_clustered_series_low_pvalue(self):
        """Clustered series should yield a p-value below 0.1 for acf1_abs."""
        clustered = _clustered_returns(500, seed=0)
        pt = permutation_test(clustered, n_permutations=200, seed=42)
        assert pt["p_value"] < 0.10, (
            f"Expected p < 0.10 for clustered series, got {pt['p_value']:.4f}"
        )

    def test_insufficient_data_returns_nan(self):
        pt = permutation_test([0.01, -0.01], n_permutations=10, seed=0)
        assert math.isnan(pt["p_value"])


# ---------------------------------------------------------------------------
# 7. regime_tape
# ---------------------------------------------------------------------------

class TestRegimeTape:

    def test_returns_required_keys(self):
        rets = _iid_returns(100)
        tape = regime_tape(rets, window=21)
        assert "labels" in tape
        assert "shuffled_labels" in tape

    def test_lengths_match_input(self):
        rets = _iid_returns(100)
        tape = regime_tape(rets, window=21)
        assert len(tape["labels"]) == 100
        assert len(tape["shuffled_labels"]) == 100

    def test_first_window_minus_one_are_none(self):
        rets = _iid_returns(100)
        window = 21
        tape = regime_tape(rets, window=window)
        for i in range(window - 1):
            assert tape["labels"][i] is None, f"Position {i} should be None"

    def test_labels_are_valid_regime_names(self):
        rets = _iid_returns(150)
        tape = regime_tape(rets, window=21)
        valid = {"calm", "normal", "turbulent", None}
        for label in tape["labels"]:
            assert label in valid, f"Unexpected label: {label}"

    def test_all_three_regimes_appear_in_long_series(self):
        """A 500-bar series should have at least one bar in each regime."""
        rets = _clustered_returns(500)
        tape = regime_tape(rets, window=21)
        non_none = [l for l in tape["labels"] if l is not None]
        assert "calm" in non_none
        assert "normal" in non_none
        assert "turbulent" in non_none

    def test_empty_input_returns_all_none(self):
        tape = regime_tape([], window=21)
        assert tape["labels"] == []
        assert tape["shuffled_labels"] == []


# ---------------------------------------------------------------------------
# 8. suggest_position_size
# ---------------------------------------------------------------------------

class TestSuggestPositionSize:

    def test_returns_required_keys(self):
        rets = _iid_returns(200)
        sizing = suggest_position_size(rets, capital=100_000)
        for key in ("var_99", "cvar_99", "suggested_fraction", "suggested_notional"):
            assert key in sizing

    def test_suggested_fraction_in_unit_interval(self):
        rets = _iid_returns(200)
        sizing = suggest_position_size(rets, capital=100_000)
        assert 0.0 <= sizing["suggested_fraction"] <= 1.0

    def test_suggested_notional_equals_fraction_times_capital(self):
        rets = _iid_returns(200)
        capital = 50_000.0
        sizing = suggest_position_size(rets, capital=capital)
        expected = sizing["suggested_fraction"] * capital
        assert sizing["suggested_notional"] == pytest.approx(expected, rel=1e-6)

    def test_empty_input_returns_zero_fraction(self):
        sizing = suggest_position_size([], capital=100_000)
        assert sizing["suggested_fraction"] == pytest.approx(0.0)
        assert sizing["suggested_notional"] == pytest.approx(0.0)

    def test_single_obs_returns_zero_fraction(self):
        sizing = suggest_position_size([0.01], capital=100_000)
        assert sizing["suggested_fraction"] == pytest.approx(0.0)

    def test_var_is_negative_for_typical_returns(self):
        """99% VaR should be negative (a loss) for any realistic return series."""
        rets = _iid_returns(300)
        sizing = suggest_position_size(rets, capital=100_000, confidence=0.99)
        assert sizing["var_99"] < 0.0

    def test_cvar_leq_var(self):
        """CVaR must be <= VaR (conditional shortfall is at least as bad)."""
        rets = _iid_returns(300)
        sizing = suggest_position_size(rets, capital=100_000, confidence=0.99)
        assert sizing["cvar_99"] <= sizing["var_99"] + 1e-10

    def test_high_vol_series_lower_fraction(self):
        """Higher volatility should imply a smaller suggested fraction."""
        rng = np.random.default_rng(42)
        low_vol = rng.normal(0.0, 0.005, 300).tolist()
        high_vol = rng.normal(0.0, 0.040, 300).tolist()
        sizing_low = suggest_position_size(low_vol, capital=100_000)
        sizing_high = suggest_position_size(high_vol, capital=100_000)
        assert sizing_low["suggested_fraction"] >= sizing_high["suggested_fraction"]


# ---------------------------------------------------------------------------
# 9. compute_volatility_clustering_report (integration)
# ---------------------------------------------------------------------------

class TestComputeVolatilityClusteringReport:

    def test_returns_all_expected_keys(self):
        rets = _iid_returns(200)
        report = compute_volatility_clustering_report(rets, n_permutations=20, seed=0)
        for key in ("ann_vol_series", "excess_kurtosis", "acf_abs", "ljung_box",
                    "same_sign_rate", "permutation", "regime_tape", "dates"):
            assert key in report, f"Missing key: {key}"

    def test_dates_none_when_not_supplied(self):
        rets = _iid_returns(100)
        report = compute_volatility_clustering_report(rets, n_permutations=10, seed=0)
        assert report["dates"] is None

    def test_dates_echoed_back(self):
        rets = _iid_returns(100)
        import pandas as pd
        dates = pd.date_range("2022-01-03", periods=100, freq="B").strftime("%Y-%m-%d").tolist()
        report = compute_volatility_clustering_report(rets, dates=dates, n_permutations=10, seed=0)
        assert report["dates"] == dates

    def test_ann_vol_series_length_matches_input(self):
        rets = _iid_returns(150)
        report = compute_volatility_clustering_report(rets, n_permutations=10, seed=0)
        assert len(report["ann_vol_series"]) == 150

    def test_acf_abs_has_expected_lag_keys(self):
        rets = _iid_returns(200)
        report = compute_volatility_clustering_report(rets, n_permutations=10, seed=0)
        for lag in (1, 5, 22, 66):
            assert lag in report["acf_abs"]

    def test_regime_tape_labels_in_report(self):
        rets = _iid_returns(150)
        report = compute_volatility_clustering_report(rets, n_permutations=10, seed=0)
        assert "labels" in report["regime_tape"]
        assert "shuffled_labels" in report["regime_tape"]

    def test_permutation_keys_in_report(self):
        rets = _iid_returns(150)
        report = compute_volatility_clustering_report(rets, n_permutations=10, seed=0)
        for key in ("observed", "shuffled_mean", "shuffled_std", "p_value", "lift_pts"):
            assert key in report["permutation"]

    def test_excess_kurtosis_is_float(self):
        rets = _iid_returns(200)
        report = compute_volatility_clustering_report(rets, n_permutations=10, seed=0)
        assert isinstance(report["excess_kurtosis"], float)
