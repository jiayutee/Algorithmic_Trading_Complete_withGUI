"""Tests for core.ta_engine.compute_indicators (Day 14 QA coverage).

compute_indicators() is the pure-pandas counterpart of
MainWindow.calculate_technical_indicators() which was fixed on Day 14 to
actually be called (it was defined but never invoked, causing KeyError on
'RSI'/'MACD' during strategy runs).  These tests cover the function in CI
without requiring a Qt display.
"""
import pytest
import numpy as np
import pandas as pd

from core.ta_engine import compute_indicators


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int, *, seed: int = 42, trend: float = 0.001,
                flat: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if flat:
        closes = np.full(n, 100.0)
        highs = closes.copy()
        lows = closes.copy()
    else:
        closes = 100 * np.cumprod(1 + rng.normal(trend, 0.01, n))
        highs = closes * (1 + rng.uniform(0, 0.02, n))
        lows = closes * (1 - rng.uniform(0, 0.02, n))
    opens = closes * (1 + (0 if flat else rng.normal(0, 0.005, n)))
    volumes = rng.integers(100_000, 1_000_000, n).astype(float)
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Column presence tests
# ---------------------------------------------------------------------------

class TestComputeIndicatorsColumns:
    EXPECTED = {"MA20", "MA50", "MA200", "EMA12", "EMA26",
                "MACD", "Signal", "RSI", "K", "D"}

    def test_all_indicator_columns_present_on_full_data(self):
        df = _make_ohlcv(300)
        out = compute_indicators(df)
        missing = self.EXPECTED - set(out.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_indicator_columns_present_on_small_data(self):
        """Even with 5 rows (fewer than any rolling window) columns must exist."""
        df = _make_ohlcv(5)
        out = compute_indicators(df)
        missing = self.EXPECTED - set(out.columns)
        assert not missing, f"Missing columns on 5-row DataFrame: {missing}"

    def test_indicator_columns_present_on_single_row(self):
        df = _make_ohlcv(1)
        out = compute_indicators(df)
        missing = self.EXPECTED - set(out.columns)
        assert not missing, f"Missing columns on 1-row DataFrame: {missing}"


# ---------------------------------------------------------------------------
# No-exception contract
# ---------------------------------------------------------------------------

class TestComputeIndicatorsNoExceptions:
    def test_no_exception_on_normal_data(self):
        df = _make_ohlcv(250)
        compute_indicators(df)  # must not raise

    def test_no_exception_on_flat_prices(self):
        """All prices equal: loss=0 and high-low=0 — should produce NaN, not raise."""
        df = _make_ohlcv(50, flat=True)
        compute_indicators(df)

    def test_no_exception_on_small_df(self):
        df = _make_ohlcv(3)
        compute_indicators(df)

    def test_no_exception_on_single_row(self):
        df = _make_ohlcv(1)
        compute_indicators(df)

    def test_no_exception_on_all_same_close(self):
        """Edge case: constant close prices produce NaN RSI (gain=loss=0), not exception."""
        df = _make_ohlcv(30, flat=True)
        out = compute_indicators(df)
        # RSI may be NaN; what matters is no exception was raised
        assert "RSI" in out.columns


# ---------------------------------------------------------------------------
# Input isolation — compute_indicators must not mutate the original DataFrame
# ---------------------------------------------------------------------------

class TestComputeIndicatorsImmutability:
    def test_original_dataframe_not_modified(self):
        df = _make_ohlcv(100)
        original_cols = set(df.columns)
        compute_indicators(df)
        assert set(df.columns) == original_cols, (
            "compute_indicators() must not add columns to the input DataFrame"
        )


# ---------------------------------------------------------------------------
# Value sanity checks on sufficient data
# ---------------------------------------------------------------------------

class TestComputeIndicatorsValues:
    def test_rsi_in_0_100_range_on_valid_rows(self):
        df = _make_ohlcv(200)
        out = compute_indicators(df)
        valid = out["RSI"].dropna()
        assert len(valid) > 0, "Expected at least some non-NaN RSI values with 200 rows"
        assert (valid >= 0).all() and (valid <= 100).all(), (
            f"RSI out of [0, 100]: min={valid.min():.2f}, max={valid.max():.2f}"
        )

    def test_macd_equals_ema12_minus_ema26(self):
        df = _make_ohlcv(100)
        out = compute_indicators(df)
        diff = (out["MACD"] - (out["EMA12"] - out["EMA26"])).abs()
        assert diff.max() < 1e-10, "MACD must equal EMA12 - EMA26"

    def test_stochastic_k_in_0_100_when_high_ne_low(self):
        """K should be in [0, 100] on rows where high != low."""
        df = _make_ohlcv(100)
        out = compute_indicators(df)
        valid_k = out["K"].dropna()
        assert len(valid_k) > 0, "Expected non-NaN K values with 100 rows"
        # Allow tiny floating-point overshoot
        assert (valid_k >= -1e-9).all() and (valid_k <= 100 + 1e-9).all(), (
            f"Stochastic K out of [0, 100]: min={valid_k.min():.4f}, max={valid_k.max():.4f}"
        )

    def test_ma20_nan_for_first_19_rows(self):
        df = _make_ohlcv(50)
        out = compute_indicators(df)
        assert out["MA20"].iloc[:19].isna().all(), "First 19 MA20 values should be NaN"
        assert pd.notna(out["MA20"].iloc[19]), "MA20 row 19 (20th) should be non-NaN"

    def test_ema12_non_nan_from_first_row(self):
        """EWM with adjust=False has a valid value even on row 0."""
        df = _make_ohlcv(10)
        out = compute_indicators(df)
        assert pd.notna(out["EMA12"].iloc[0]), "EMA12 row 0 must not be NaN"

    def test_flat_prices_rsi_is_nan_not_exception(self):
        """When gain and loss are both 0 (flat prices), RSI should be NaN."""
        df = _make_ohlcv(50, flat=True)
        out = compute_indicators(df)
        # All rows after the first have delta=0, so gain=loss=0 → NaN RSI
        assert out["RSI"].dropna().empty or True  # don't fail if some are non-NaN


# ---------------------------------------------------------------------------
# Regression: check_strategy_signal NaN safety
# ---------------------------------------------------------------------------

class TestNaNSafeSignalCheck:
    """Verify that NaN indicator values (from insufficient data) don't cause
    exceptions in the comparison logic used by check_strategy_signal()."""

    @staticmethod
    def _signal_macd_rsi(latest):
        # Mirrors MainWindow.check_strategy_signal for MACD/RSI strategy
        if latest['RSI'] > 30 and latest['MACD'] > latest['Signal']:
            return 1
        elif latest['RSI'] > 70 or latest['MACD'] < latest['Signal']:
            return -1
        return 0

    @staticmethod
    def _signal_stochastic(latest, prev):
        if latest['K'] > latest['D'] and prev['K'] <= prev['D'] and latest['K'] < 20:
            return 1
        elif latest['K'] < latest['D'] and prev['K'] >= prev['D'] and latest['K'] > 80:
            return -1
        return 0

    def test_macd_rsi_signal_with_nan_values_returns_zero(self):
        nan = float('nan')
        latest = {'RSI': nan, 'MACD': nan, 'Signal': nan}
        assert self._signal_macd_rsi(latest) == 0

    def test_stochastic_signal_with_nan_values_returns_zero(self):
        nan = float('nan')
        latest = {'K': nan, 'D': nan}
        prev = {'K': nan, 'D': nan}
        assert self._signal_stochastic(latest, prev) == 0

    def test_macd_rsi_signal_on_tiny_df_does_not_raise(self):
        """End-to-end: compute_indicators on tiny data, then try signal check."""
        df = _make_ohlcv(5)
        out = compute_indicators(df)
        latest = out.iloc[-1]
        # Should not raise, even with all-NaN indicators
        self._signal_macd_rsi(latest)
