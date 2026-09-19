"""Regression tests for core/research_lab.py — Strategy Lab analytics.

Covers all eight public functions with a focus on:
- Exact numerical assertions on known inputs.
- Edge-case handling (empty, wins-only, losses-only, div-by-zero guards).
- Shape and key correctness for table-producing functions.
- Gate verdict pass/fail on clearly-good and clearly-bad synthetic reports.
"""

import math
import pytest
import numpy as np
import pandas as pd

from core.research_lab import (
    compute_drawdown_series,
    compute_rolling_sharpe,
    trade_pnl_distribution,
    monthly_returns_table,
    year_by_year_table,
    unit_economics_per_trade,
    build_strategy_book,
    evaluate_gate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _good_report(
    sharpe: float = 1.5,
    max_drawdown: float = 10.0,
    win_rate: float = 55.0,
    n_trades: int = 80,
) -> dict:
    """Minimal backtest report dict accepted by evaluate_gate."""
    return {
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "summary": {"Number of Closed Trades": n_trades},
    }


def _make_dates_returns(n: int = 60, start: str = "2023-01-03") -> tuple:
    """Return (returns, dates) as aligned lists of length n."""
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.01, n).tolist()
    dates = pd.date_range(start=start, periods=n, freq="B").strftime("%Y-%m-%d").tolist()
    return rets, dates


# ---------------------------------------------------------------------------
# 1. compute_drawdown_series
# ---------------------------------------------------------------------------

class TestComputeDrawdownSeries:

    def test_empty_input_returns_empty(self):
        assert compute_drawdown_series([]) == []

    def test_known_series_exact_values(self):
        """[100, 110, 105, 95, 115] — verify trough and recovery."""
        equity = [100.0, 110.0, 105.0, 95.0, 115.0]
        dd = compute_drawdown_series(equity)
        assert len(dd) == 5
        # First two: at or above running peak
        assert dd[0] == pytest.approx(0.0)
        assert dd[1] == pytest.approx(0.0)
        # 105 / 110 - 1 = -4.5454…%
        assert dd[2] == pytest.approx(-4.5455, abs=1e-3)
        # 95 / 110 - 1 = -13.6363…%
        assert dd[3] == pytest.approx(-13.6364, abs=1e-3)
        # 115 beats old peak of 110 → 0 drawdown
        assert dd[4] == pytest.approx(0.0)

    def test_monotone_rising_curve_all_zeros(self):
        dd = compute_drawdown_series([100.0, 110.0, 120.0, 130.0])
        assert all(v == pytest.approx(0.0) for v in dd)

    def test_monotone_falling_curve_drawdowns_negative(self):
        dd = compute_drawdown_series([100.0, 90.0, 80.0, 70.0])
        assert dd[0] == pytest.approx(0.0)
        assert all(v < 0 for v in dd[1:])

    def test_single_element(self):
        assert compute_drawdown_series([100.0]) == [pytest.approx(0.0)]

    def test_output_length_matches_input(self):
        equity = list(range(1, 21))
        assert len(compute_drawdown_series(equity)) == 20


# ---------------------------------------------------------------------------
# 2. compute_rolling_sharpe
# ---------------------------------------------------------------------------

class TestComputeRollingSharpe:

    def test_empty_input_returns_empty(self):
        assert compute_rolling_sharpe([]) == []

    def test_first_window_minus_one_entries_are_nan(self):
        rets = [0.001, -0.002, 0.003] * 30  # 90 elements
        window = 5
        rs = compute_rolling_sharpe(rets, window=window)
        assert len(rs) == 90
        # Positions 0..window-2 must be nan
        for i in range(window - 1):
            assert math.isnan(rs[i]), f"Position {i} should be nan, got {rs[i]}"

    def test_entry_at_window_is_real_float(self):
        rets = [0.001, -0.002, 0.003] * 30
        window = 5
        rs = compute_rolling_sharpe(rets, window=window)
        # rs[4] is the first valid Sharpe
        assert isinstance(rs[window - 1], float)
        assert not math.isnan(rs[window - 1])

    def test_flat_returns_produce_nan_sharpe(self):
        """All identical returns → std=0 → Sharpe should be nan."""
        rets = [0.001] * 20
        rs = compute_rolling_sharpe(rets, window=5)
        for v in rs[4:]:
            assert math.isnan(v), f"Expected nan for flat returns, got {v}"

    def test_output_length_matches_input(self):
        rets = [0.01] * 50
        assert len(compute_rolling_sharpe(rets, window=10)) == 50

    def test_positive_trending_returns_positive_sharpe(self):
        """Consistently positive returns should yield positive Sharpe values."""
        rng = np.random.default_rng(1)
        rets = (rng.normal(0.002, 0.001, 100)).tolist()  # mean >> std
        rs = compute_rolling_sharpe(rets, window=10)
        valid = [v for v in rs[9:] if not math.isnan(v)]
        assert len(valid) > 0
        assert all(v > 0 for v in valid)


# ---------------------------------------------------------------------------
# 3. trade_pnl_distribution
# ---------------------------------------------------------------------------

class TestTradePnlDistribution:

    def test_empty_input(self):
        out = trade_pnl_distribution([])
        assert out == {"bin_edges": [], "counts": [], "win_count": 0, "loss_count": 0}

    def test_win_loss_counts(self):
        trades = [150.0, -80.0, 200.0, -40.0]
        out = trade_pnl_distribution(trades, bins=5)
        assert out["win_count"] == 2
        assert out["loss_count"] == 2

    def test_bin_edge_count(self):
        trades = [10.0, -5.0, 20.0, -3.0, 15.0]
        bins = 5
        out = trade_pnl_distribution(trades, bins=bins)
        assert len(out["bin_edges"]) == bins + 1

    def test_counts_length(self):
        trades = [10.0, -5.0, 20.0, -3.0, 15.0]
        bins = 7
        out = trade_pnl_distribution(trades, bins=bins)
        assert len(out["counts"]) == bins

    def test_counts_sum_equals_num_trades(self):
        trades = [100.0, -50.0, 75.0, -25.0, 30.0]
        out = trade_pnl_distribution(trades, bins=10)
        assert sum(out["counts"]) == len(trades)

    def test_all_winners(self):
        trades = [10.0, 20.0, 30.0]
        out = trade_pnl_distribution(trades, bins=3)
        assert out["win_count"] == 3
        assert out["loss_count"] == 0

    def test_all_losers(self):
        trades = [-10.0, -20.0, -30.0]
        out = trade_pnl_distribution(trades, bins=3)
        assert out["win_count"] == 0
        assert out["loss_count"] == 3

    def test_break_even_counted_as_loss(self):
        """P&L == 0 should count towards loss_count."""
        trades = [100.0, 0.0]
        out = trade_pnl_distribution(trades, bins=5)
        assert out["win_count"] == 1
        assert out["loss_count"] == 1


# ---------------------------------------------------------------------------
# 4. monthly_returns_table
# ---------------------------------------------------------------------------

class TestMonthlyReturnsTable:

    def _two_month_data(self):
        """Return (rets, dates) spanning Jan and Feb 2023 only."""
        # 21 business days in Jan-2023, 20 in Feb-2023
        dates_jan = pd.date_range("2023-01-03", periods=21, freq="B").strftime("%Y-%m-%d").tolist()
        dates_feb = pd.date_range("2023-02-01", periods=20, freq="B").strftime("%Y-%m-%d").tolist()
        rng = np.random.default_rng(7)
        rets_jan = rng.normal(0.002, 0.01, 21).tolist()
        rets_feb = rng.normal(-0.001, 0.01, 20).tolist()
        return rets_jan + rets_feb, dates_jan + dates_feb

    def test_empty_input(self):
        out = monthly_returns_table([], [])
        assert out == {"best_month": "N/A", "worst_month": "N/A"}

    def test_mismatched_lengths(self):
        out = monthly_returns_table([0.01, 0.02], ["2023-01-03"])
        assert out == {"best_month": "N/A", "worst_month": "N/A"}

    def test_year_key_present(self):
        rets, dates = self._two_month_data()
        tbl = monthly_returns_table(rets, dates)
        assert 2023 in tbl

    def test_month_keys_present(self):
        rets, dates = self._two_month_data()
        tbl = monthly_returns_table(rets, dates)
        assert "Jan" in tbl[2023]
        assert "Feb" in tbl[2023]

    def test_best_and_worst_month_strings_present(self):
        rets, dates = self._two_month_data()
        tbl = monthly_returns_table(rets, dates)
        assert "best_month" in tbl
        assert "worst_month" in tbl
        assert tbl["best_month"] != "N/A"
        assert tbl["worst_month"] != "N/A"

    def test_monthly_return_is_float(self):
        rets, dates = self._two_month_data()
        tbl = monthly_returns_table(rets, dates)
        assert isinstance(tbl[2023]["Jan"], float)

    def test_best_month_contains_positive_sign_for_gain(self):
        """Positive best month label should contain '+'."""
        # Force Jan to be clearly the best with strong positive returns
        dates_jan = pd.date_range("2023-01-03", periods=20, freq="B").strftime("%Y-%m-%d").tolist()
        dates_feb = pd.date_range("2023-02-01", periods=20, freq="B").strftime("%Y-%m-%d").tolist()
        rets_jan = [0.005] * 20   # strongly positive
        rets_feb = [-0.003] * 20  # negative
        tbl = monthly_returns_table(rets_jan + rets_feb, dates_jan + dates_feb)
        assert "+" in tbl["best_month"]

    def test_compounding_correctness(self):
        """Single-bar month: compound return == that bar's return scaled to pct."""
        dates = ["2023-03-01"]
        rets = [0.05]   # +5 %
        tbl = monthly_returns_table(rets, dates)
        # (1.05) - 1 = 0.05 → 5.0 %
        assert tbl[2023]["Mar"] == pytest.approx(5.0, abs=1e-3)


# ---------------------------------------------------------------------------
# 5. year_by_year_table
# ---------------------------------------------------------------------------

class TestYearByYearTable:

    def test_empty_input(self):
        assert year_by_year_table([], []) == []

    def test_mismatched_lengths(self):
        assert year_by_year_table([0.01], []) == []

    def test_returns_list_of_dicts(self):
        rets, dates = _make_dates_returns(60)
        rows = year_by_year_table(rets, dates)
        assert isinstance(rows, list)
        assert len(rows) >= 1
        assert isinstance(rows[0], dict)

    def test_required_keys_present(self):
        rets, dates = _make_dates_returns(60)
        rows = year_by_year_table(rets, dates)
        expected = {"year", "return_pct", "benchmark_pct", "sharpe", "max_drawdown_pct", "num_trades_note"}
        for row in rows:
            assert expected <= set(row.keys()), f"Missing keys in row: {row.keys()}"

    def test_sorted_by_year_ascending(self):
        """Span two calendar years and confirm sort order."""
        n = 505  # roughly 2 years of business days
        rets = [0.001] * n
        dates = pd.date_range("2022-01-03", periods=n, freq="B").strftime("%Y-%m-%d").tolist()
        rows = year_by_year_table(rets, dates)
        years = [r["year"] for r in rows]
        assert years == sorted(years)

    def test_num_trades_note_value(self):
        rets, dates = _make_dates_returns(60)
        rows = year_by_year_table(rets, dates)
        for row in rows:
            assert row["num_trades_note"] == "see signals list"

    def test_max_drawdown_non_negative(self):
        rets, dates = _make_dates_returns(60)
        rows = year_by_year_table(rets, dates)
        for row in rows:
            assert row["max_drawdown_pct"] >= 0.0

    def test_benchmark_pct_none_when_not_supplied(self):
        rets, dates = _make_dates_returns(60)
        rows = year_by_year_table(rets, dates)
        for row in rows:
            assert row["benchmark_pct"] is None

    def test_benchmark_pct_float_when_supplied(self):
        rets, dates = _make_dates_returns(60)
        bench = [0.0005] * len(rets)
        rows = year_by_year_table(rets, dates, benchmark_returns=bench)
        for row in rows:
            assert row["benchmark_pct"] is not None
            assert isinstance(row["benchmark_pct"], float)


# ---------------------------------------------------------------------------
# 6. unit_economics_per_trade
# ---------------------------------------------------------------------------

class TestUnitEconomicsPerTrade:

    def test_empty_input_all_none(self):
        ue = unit_economics_per_trade([])
        assert ue["avg_pnl"] is None
        assert ue["median_pnl"] is None
        assert ue["win_rate_pct"] == pytest.approx(0.0)
        assert ue["avg_win"] is None
        assert ue["avg_loss"] is None
        assert ue["expectancy"] is None
        assert ue["profit_factor"] is None

    def test_wins_only_profit_factor_none(self):
        """No losing trades → gross losses = 0 → profit_factor must be None."""
        ue = unit_economics_per_trade([100.0, 200.0, 50.0])
        assert ue["win_rate_pct"] == pytest.approx(100.0)
        assert ue["avg_loss"] is None
        assert ue["profit_factor"] is None

    def test_losses_only_win_rate_zero(self):
        ue = unit_economics_per_trade([-50.0, -30.0, -20.0])
        assert ue["win_rate_pct"] == pytest.approx(0.0)
        assert ue["avg_win"] is None

    def test_losses_only_profit_factor_zero(self):
        """All losses → gross_wins = 0, gross_losses > 0 → profit_factor = 0.0"""
        ue = unit_economics_per_trade([-50.0, -30.0])
        # gross_losses = 80, gross_wins = 0 → 0.0 / 80.0 = 0.0
        assert ue["profit_factor"] == pytest.approx(0.0)

    def test_known_profit_factor(self):
        """[200, -50, 150, -30]: wins=350, losses=80 → PF = 350/80 = 4.375"""
        ue = unit_economics_per_trade([200.0, -50.0, 150.0, -30.0])
        assert ue["profit_factor"] == pytest.approx(4.375, rel=1e-4)

    def test_known_win_rate(self):
        ue = unit_economics_per_trade([200.0, -50.0, 150.0, -30.0])
        assert ue["win_rate_pct"] == pytest.approx(50.0)

    def test_expectancy_formula(self):
        """p(win)*avg_win + p(loss)*avg_loss for [200, -50, 150, -30]."""
        ue = unit_economics_per_trade([200.0, -50.0, 150.0, -30.0])
        # p_win = 0.5, avg_win = 175.0, p_loss = 0.5, avg_loss = -40.0
        expected_expectancy = 0.5 * 175.0 + 0.5 * (-40.0)
        assert ue["expectancy"] == pytest.approx(expected_expectancy, abs=0.01)

    def test_avg_pnl_is_arithmetic_mean(self):
        trades = [100.0, -50.0, 200.0, -50.0]
        ue = unit_economics_per_trade(trades)
        assert ue["avg_pnl"] == pytest.approx(np.mean(trades), rel=1e-5)

    def test_all_keys_present(self):
        ue = unit_economics_per_trade([10.0, -5.0])
        for key in ["avg_pnl", "median_pnl", "win_rate_pct", "avg_win",
                    "avg_loss", "expectancy", "profit_factor"]:
            assert key in ue


# ---------------------------------------------------------------------------
# 7. build_strategy_book — tested via a stub StrategyManager
# ---------------------------------------------------------------------------

class _StubWrapper:
    """Minimal strategy wrapper object."""
    pass


class _StubStrategyManager:
    """Stub for StrategyManager that allows controlled success/failure."""

    def __init__(self, strategy_map: dict):
        """strategy_map: {name: report_dict_or_Exception}"""
        self._map = strategy_map

    def get_available_strategies(self) -> list:
        return list(self._map.keys())

    def get_strategy(self, name):
        return _StubWrapper()

    def run_backtest(self, strategy_wrapper, data, cash, broker_mode):
        result = self._map[strategy_wrapper.__class__.__name__]
        # Look up by name; strategy_wrapper is always a _StubWrapper
        # Use the iteration index trick: iterate map items
        for n, v in self._map.items():
            if isinstance(v, Exception):
                raise v
            return v
        raise RuntimeError("unreachable")

    # Override to use name lookup
    def _run(self, name, data, cash):
        val = self._map[name]
        if isinstance(val, Exception):
            raise val
        return val


class _NamedStubStrategyManager(_StubStrategyManager):
    """Variant that dispatches run_backtest via stored name list."""

    def __init__(self, strategy_map):
        super().__init__(strategy_map)
        self._names = list(strategy_map.keys())
        self._call_idx = 0

    def get_strategy(self, name):
        wrapper = _StubWrapper()
        wrapper._name = name
        return wrapper

    def run_backtest(self, strategy_wrapper, data, cash, broker_mode):
        name = strategy_wrapper._name
        val = self._map[name]
        if isinstance(val, Exception):
            raise val
        return val


class TestBuildStrategyBook:

    def _make_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(5)
        n = 252
        closes = 100.0 * np.cumprod(1 + rng.normal(0.001, 0.01, n))
        idx = pd.date_range("2022-01-03", periods=n, freq="B")
        return pd.DataFrame({
            "Open": closes, "High": closes * 1.01, "Low": closes * 0.99,
            "Close": closes, "Volume": 1_000_000.0,
        }, index=idx)

    def test_successful_entries_sorted_by_sharpe_descending(self):
        sm = _NamedStubStrategyManager({
            "StratA": {"sharpe": 0.5, "max_drawdown": 10.0, "win_rate": 55.0},
            "StratB": {"sharpe": 1.5, "max_drawdown": 8.0, "win_rate": 60.0},
            "StratC": {"sharpe": 0.9, "max_drawdown": 12.0, "win_rate": 50.0},
        })
        df = self._make_df()
        book = build_strategy_book(sm, df)
        sharpes = [e["sharpe"] for e in book if "sharpe" in e]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_failed_entry_appended_after_successes(self):
        sm = _NamedStubStrategyManager({
            "GoodStrat": {"sharpe": 1.0, "max_drawdown": 10.0, "win_rate": 55.0},
            "BadStrat": RuntimeError("GPU not available"),
        })
        df = self._make_df()
        book = build_strategy_book(sm, df)
        names = [e["name"] for e in book]
        # GoodStrat should appear before BadStrat
        assert names.index("GoodStrat") < names.index("BadStrat")

    def test_error_entry_has_error_key(self):
        sm = _NamedStubStrategyManager({
            "FailStrat": RuntimeError("test error"),
        })
        df = self._make_df()
        book = build_strategy_book(sm, df)
        assert len(book) == 1
        assert "error" in book[0]
        assert book[0]["name"] == "FailStrat"

    def test_successful_entry_has_required_keys(self):
        sm = _NamedStubStrategyManager({
            "EMA": {"sharpe": 0.8, "max_drawdown": 15.0, "win_rate": 52.0},
        })
        df = self._make_df()
        book = build_strategy_book(sm, df)
        assert len(book) == 1
        entry = book[0]
        for key in ("name", "sharpe", "max_drawdown", "win_rate"):
            assert key in entry

    def test_empty_strategy_list(self):
        sm = _NamedStubStrategyManager({})
        df = self._make_df()
        assert build_strategy_book(sm, df) == []


# ---------------------------------------------------------------------------
# 8. evaluate_gate
# ---------------------------------------------------------------------------

class TestEvaluateGate:

    def test_good_report_passes(self):
        report = _good_report(sharpe=1.5, max_drawdown=10.0, win_rate=55.0, n_trades=100)
        gate = evaluate_gate(report)
        assert gate["passed"] is True

    def test_bad_report_drawdown_fails(self):
        """Drawdown far past threshold must flip passed to False."""
        report = _good_report(sharpe=1.5, max_drawdown=65.0, win_rate=55.0, n_trades=100)
        gate = evaluate_gate(report)
        assert gate["passed"] is False

    def test_bad_report_low_sharpe_fails(self):
        report = _good_report(sharpe=0.05, max_drawdown=10.0, win_rate=55.0, n_trades=100)
        gate = evaluate_gate(report)
        assert gate["passed"] is False

    def test_bad_report_too_few_trades_fails(self):
        report = _good_report(sharpe=1.5, max_drawdown=10.0, win_rate=55.0, n_trades=5)
        gate = evaluate_gate(report)
        assert gate["passed"] is False

    def test_checks_list_has_four_entries(self):
        gate = evaluate_gate(_good_report())
        assert len(gate["checks"]) == 4

    def test_check_names(self):
        gate = evaluate_gate(_good_report())
        names = [c["name"] for c in gate["checks"]]
        assert "Sharpe Ratio" in names
        assert "Max Drawdown" in names
        assert "Trade Count" in names
        assert "Win Rate Overfit Signal" in names

    def test_verdict_text_is_nonempty_string(self):
        gate = evaluate_gate(_good_report())
        assert isinstance(gate["verdict_text"], str)
        assert len(gate["verdict_text"]) > 10

    def test_custom_min_sharpe_threshold(self):
        """Sharpe 0.4 should pass with min_sharpe=0.3 but fail with min_sharpe=0.5."""
        report = _good_report(sharpe=0.4)
        assert evaluate_gate(report, thresholds={"min_sharpe": 0.3})["passed"] is True
        # Other checks still need to pass — set them to clear values
        report2 = _good_report(sharpe=0.4, max_drawdown=5.0, win_rate=50.0, n_trades=100)
        fail_result = evaluate_gate(report2, thresholds={"min_sharpe": 0.5})
        sharpe_check = next(c for c in fail_result["checks"] if c["name"] == "Sharpe Ratio")
        assert sharpe_check["passed"] is False

    def test_win_rate_overfit_flag_outside_bounds(self):
        """Win rate of 95 % is outside (20, 80) — check should have passed=False."""
        report = _good_report(sharpe=1.5, max_drawdown=10.0, win_rate=95.0, n_trades=100)
        gate = evaluate_gate(report)
        wr_check = next(c for c in gate["checks"] if c["name"] == "Win Rate Overfit Signal")
        assert wr_check["passed"] is False
        assert gate["passed"] is False

    def test_win_rate_within_bounds_check_passes(self):
        report = _good_report(sharpe=1.5, max_drawdown=10.0, win_rate=55.0, n_trades=100)
        gate = evaluate_gate(report)
        wr_check = next(c for c in gate["checks"] if c["name"] == "Win Rate Overfit Signal")
        assert wr_check["passed"] is True

    def test_checks_have_detail_string(self):
        gate = evaluate_gate(_good_report())
        for c in gate["checks"]:
            assert isinstance(c["detail"], str)
            assert len(c["detail"]) > 0

    def test_all_checks_passed_key_in_checks(self):
        gate = evaluate_gate(_good_report())
        for c in gate["checks"]:
            assert "passed" in c
            assert isinstance(c["passed"], bool)
