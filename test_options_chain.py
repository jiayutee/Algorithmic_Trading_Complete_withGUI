"""Phase 4.2 — Options chain + Greeks retrieval tests.

All tests use the shared ``ib_env`` fixture from conftest.py (fake ib_insync module,
no real TWS/Gateway connection opened).  The AST live-connection audit in
test_ibkr_manager.py::test_no_live_connection_is_ever_attempted_by_the_suite
continues to pass because this file contains no ``something.ib.connect(...)`` calls.
"""

from __future__ import annotations

import math
from datetime import date
from types import SimpleNamespace as NS
from unittest import mock

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

# A realistic-looking future expiry set (all past dates compared to test run
# date, but that's fine — the function picks nearest or first, still exercises
# the logic; we use frozen dates so tests are deterministic).
_EXPIRY_A = "20991016"   # far-future so _select_expiry picks it as "nearest"
_EXPIRY_B = "20991115"
_EXPIRY_C = "20991218"

_STRIKES = [140.0, 145.0, 150.0, 155.0, 160.0]
_SPOT = 150.0


def _make_greeks(
    delta=0.52, gamma=0.01, theta=-0.05, vega=0.15,
    impliedVol=0.25, undPrice=_SPOT,
) -> NS:
    return NS(
        delta=delta, gamma=gamma, theta=theta, vega=vega,
        impliedVol=impliedVol, undPrice=undPrice,
    )


def _make_opt_contract(
    symbol="AAPL", expiry=_EXPIRY_A, strike=150.0, right="C",
) -> NS:
    return NS(
        symbol=symbol,
        lastTradeDateOrContractMonth=expiry,
        strike=strike,
        right=right,
        secType="OPT",
        currency="USD",
        conId=0,
    )


def _make_ticker(contract: NS, bid=5.2, ask=5.4, greeks=None) -> NS:
    return NS(
        contract=contract,
        bid=bid,
        ask=ask,
        modelGreeks=greeks if greeks is not None else _make_greeks(),
    )


def _make_chain(
    expirations=None, strikes=None, exchange="SMART",
) -> NS:
    return NS(
        exchange=exchange,
        expirations=expirations or [_EXPIRY_A, _EXPIRY_B, _EXPIRY_C],
        strikes=strikes or _STRIKES,
    )


def _stock_contract(symbol="AAPL", conid=12345) -> NS:
    return NS(
        symbol=symbol, exchange="SMART", currency="USD",
        secType="STK", conId=conid,
    )


def _setup_ib(fake_ib: mock.MagicMock, *, symbol="AAPL", strikes=None,
              expirations=None, spot=_SPOT):
    """Wire *fake_ib* for a standard chain request and return the stock contract."""
    stock_c = _stock_contract(symbol)
    strikes = strikes or _STRIKES
    expirations = expirations or [_EXPIRY_A, _EXPIRY_B, _EXPIRY_C]

    # qualifyContracts: give the stock a conId; return options unchanged
    def _qualify(*contracts):
        out = []
        for c in contracts:
            if getattr(c, "secType", "") == "STK":
                # Copy the NS object with a real conId
                c_dict = dict(vars(c))
                c_dict["conId"] = 12345
                out.append(NS(**c_dict))
            else:
                out.append(c)
        return out

    fake_ib.qualifyContracts.side_effect = _qualify

    # reqSecDefOptParams: return a single chain
    fake_ib.reqSecDefOptParams.return_value = [
        _make_chain(expirations=expirations, strikes=strikes)
    ]

    # reqTickers: first call is for the underlying (spot), subsequent calls
    # are for options; distinguish by secType of the first contract.
    spot_ticker = NS(
        contract=stock_c, last=spot, close=spot - 0.2, bid=spot - 0.1, ask=spot + 0.1,
        modelGreeks=None,
    )

    def _req_tickers(*contracts):
        if not contracts:
            return []
        if getattr(contracts[0], "secType", "") == "STK":
            return [spot_ticker]
        # One ticker per option contract, with synthetic but realistic values
        result = []
        for i, c in enumerate(contracts):
            strike_val = getattr(c, "strike", 150.0)
            right = getattr(c, "right", "C")
            moneyness = strike_val - spot
            delta = (0.5 - moneyness * 0.01) if right == "C" else (-0.5 + moneyness * 0.01)
            result.append(
                _make_ticker(
                    c,
                    bid=round(max(0.05, 5.0 - moneyness * 0.1 + i * 0.02), 2),
                    ask=round(max(0.10, 5.2 - moneyness * 0.1 + i * 0.02), 2),
                    greeks=_make_greeks(delta=delta, undPrice=spot),
                )
            )
        return result

    fake_ib.reqTickers.side_effect = _req_tickers
    return stock_c


# ---------------------------------------------------------------------------
# Tests: core.options_chain._ibkr_to_nan
# ---------------------------------------------------------------------------

class TestIbkrToNan:
    """Unit tests for the _ibkr_to_nan helper — no IB connection needed."""

    def setup_method(self):
        from core.options_chain import _ibkr_to_nan
        self.f = _ibkr_to_nan

    def test_none_returns_none(self):
        assert self.f(None) is None

    def test_nan_returns_none(self):
        assert self.f(float("nan")) is None

    def test_inf_returns_none(self):
        assert self.f(float("inf")) is None
        assert self.f(float("-inf")) is None

    def test_sentinel_minus_one_returns_none(self):
        assert self.f(-1.0) is None
        assert self.f(-1) is None

    def test_zero_is_valid(self):
        assert self.f(0.0) == 0.0
        assert self.f(0) == 0.0

    def test_positive_float_passes_through(self):
        assert self.f(0.25) == pytest.approx(0.25)
        assert self.f(150.0) == pytest.approx(150.0)

    def test_negative_non_sentinel_passes_through(self):
        # -0.05 is a valid theta value
        assert self.f(-0.05) == pytest.approx(-0.05)

    def test_non_numeric_string_returns_none(self):
        assert self.f("n/a") is None

    def test_numeric_string_is_converted(self):
        assert self.f("0.52") == pytest.approx(0.52)


# ---------------------------------------------------------------------------
# Tests: core.options_chain._parse_expiry
# ---------------------------------------------------------------------------

class TestParseExpiry:
    """Unit tests for expiry normalisation — no IB connection needed."""

    def setup_method(self):
        from core.options_chain import _parse_expiry
        self.f = _parse_expiry

    def test_none_returns_none(self):
        assert self.f(None) is None

    def test_yyyymmdd_string_passes_through(self):
        assert self.f("20261016") == "20261016"

    def test_dash_separated_string_normalised(self):
        assert self.f("2026-10-16") == "20261016"

    def test_date_object_converted(self):
        assert self.f(date(2026, 10, 16)) == "20261016"

    def test_invalid_string_raises_value_error(self):
        with pytest.raises(ValueError):
            self.f("October 2026")

    def test_wrong_type_raises_type_error(self):
        with pytest.raises(TypeError):
            self.f(20261016)   # int, not str/date/None


# ---------------------------------------------------------------------------
# Tests: core.options_chain._select_expiry
# ---------------------------------------------------------------------------

class TestSelectExpiry:
    """Unit tests for expiry selection logic."""

    def setup_method(self):
        from core.options_chain import _select_expiry
        self.f = _select_expiry

    def test_exact_match_used(self):
        available = ["20991016", "20991115"]
        assert self.f(available, "20991016") == "20991016"

    def test_none_target_picks_nearest_future(self):
        # All entries are far-future; should return the first (earliest)
        available = ["20991016", "20991115", "20991218"]
        result = self.f(available, None)
        assert result == "20991016"

    def test_empty_available_returns_none(self):
        assert self.f([], None) is None
        assert self.f([], "20991016") is None

    def test_target_not_listed_falls_through_to_nearest(self):
        available = ["20991016", "20991115"]
        # "20991020" is not listed; should fall through and pick nearest future
        result = self.f(available, "20991020")
        assert result == "20991016"

    def test_all_past_expirations_returns_last(self):
        # All in 2000 (definitely past)
        available = ["20000101", "20000201", "20000301"]
        result = self.f(available, None)
        assert result == "20000301"   # most recent past


# ---------------------------------------------------------------------------
# Tests: core.options_chain._select_strikes
# ---------------------------------------------------------------------------

class TestSelectStrikes:
    """Unit tests for strike window selection."""

    def setup_method(self):
        from core.options_chain import _select_strikes
        self.f = _select_strikes

    def test_empty_input_returns_empty(self):
        assert self.f([], 150.0, 10) == []

    def test_none_spot_returns_all_strikes(self):
        strikes = [140.0, 145.0, 150.0, 155.0, 160.0]
        result = self.f(strikes, None, 2)
        assert result == strikes

    def test_zero_spot_returns_all_strikes(self):
        strikes = [140.0, 145.0, 150.0, 155.0, 160.0]
        assert self.f(strikes, 0.0, 2) == strikes

    def test_window_centred_on_atm_strike(self):
        strikes = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0]
        # spot=150 → ATM at index 5; n=2 → indices 3..7 → [130, 140, 150, 160, 170]
        result = self.f(strikes, 150.0, 2)
        assert result == [130.0, 140.0, 150.0, 160.0, 170.0]

    def test_window_clipped_at_edges(self):
        strikes = [140.0, 145.0, 150.0, 155.0, 160.0]
        # spot=140 → closest is index 0; n=10 → all strikes included
        result = self.f(strikes, 140.0, 10)
        assert result == strikes

    def test_output_sorted_ascending(self):
        strikes = [160.0, 140.0, 150.0]
        result = self.f(strikes, 150.0, 1)
        assert result == sorted(result)


# ---------------------------------------------------------------------------
# Tests: get_options_chain (main integration — uses ib_env from conftest.py)
# ---------------------------------------------------------------------------

class TestGetOptionsChain:
    """Integration tests for get_options_chain against a mocked IB object."""

    def test_happy_path_returns_dataframe_with_correct_columns(self, ib_env):
        """Normal AAPL chain returns a non-empty DataFrame with all required columns."""
        _setup_ib(ib_env.ib)
        from core.options_chain import CHAIN_COLUMNS, get_options_chain
        df = get_options_chain(ib_env.ib, "AAPL")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == CHAIN_COLUMNS
        assert len(df) > 0

    def test_right_column_contains_only_c_and_p(self, ib_env):
        """Each strike produces both a call ('C') and put ('P') row."""
        _setup_ib(ib_env.ib, strikes=[145.0, 150.0, 155.0])
        from core.options_chain import get_options_chain
        df = get_options_chain(ib_env.ib, "AAPL")
        assert set(df["right"].unique()) == {"C", "P"}

    def test_two_rows_per_strike(self, ib_env):
        """Each selected strike has exactly one call row and one put row."""
        strikes = [145.0, 150.0, 155.0]
        _setup_ib(ib_env.ib, strikes=strikes)
        from core.options_chain import get_options_chain
        df = get_options_chain(ib_env.ib, "AAPL")
        # With n=10 and spot=150, all 3 strikes are within the window
        assert len(df) == len(strikes) * 2

    def test_mid_equals_bid_plus_ask_over_two(self, ib_env):
        """mid == (bid + ask) / 2 for every row where bid and ask are finite."""
        _setup_ib(ib_env.ib)
        from core.options_chain import get_options_chain
        df = get_options_chain(ib_env.ib, "AAPL")
        valid = df.dropna(subset=["bid", "ask", "mid"])
        for _, row in valid.iterrows():
            assert row["mid"] == pytest.approx((row["bid"] + row["ask"]) / 2.0, abs=1e-9)

    def test_expiry_none_picks_nearest_listed(self, ib_env):
        """expiry=None selects the first (nearest future) expiry from the chain."""
        _setup_ib(ib_env.ib)
        from core.options_chain import get_options_chain
        df = get_options_chain(ib_env.ib, "AAPL", expiry=None)
        assert (df["expiry"] == _EXPIRY_A).all()

    def test_expiry_string_yyyymmdd_selects_correct_expiry(self, ib_env):
        """A YYYYMMDD string selects the matching expiry from the chain."""
        _setup_ib(ib_env.ib)
        from core.options_chain import get_options_chain
        df = get_options_chain(ib_env.ib, "AAPL", expiry=_EXPIRY_B)
        assert (df["expiry"] == _EXPIRY_B).all()

    def test_expiry_date_object_accepted(self, ib_env):
        """A datetime.date object is accepted and normalised to YYYYMMDD."""
        _setup_ib(ib_env.ib, expirations=[_EXPIRY_A, "20991116"])
        from core.options_chain import get_options_chain
        # _EXPIRY_A = "20991016" → date(2099, 10, 16)
        d = date(2099, 10, 16)
        df = get_options_chain(ib_env.ib, "AAPL", expiry=d)
        assert (df["expiry"] == _EXPIRY_A).all()

    def test_expiry_dash_separated_string_accepted(self, ib_env):
        """A YYYY-MM-DD string is accepted and normalised."""
        _setup_ib(ib_env.ib)
        from core.options_chain import get_options_chain
        df = get_options_chain(ib_env.ib, "AAPL", expiry="2099-10-16")
        assert (df["expiry"] == _EXPIRY_A).all()

    def test_invalid_expiry_returns_empty_dataframe(self, ib_env):
        """A badly-formatted expiry string returns an empty DataFrame, not a crash."""
        _setup_ib(ib_env.ib)
        from core.options_chain import CHAIN_COLUMNS, get_options_chain
        df = get_options_chain(ib_env.ib, "AAPL", expiry="October 2026")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == CHAIN_COLUMNS
        assert len(df) == 0

    def test_strikes_near_spot_limits_row_count(self, ib_env):
        """strikes_near_spot=1 limits the chain to at most 3 strikes × 2 rights = 6 rows."""
        _setup_ib(ib_env.ib,
                  strikes=[130.0, 135.0, 140.0, 145.0, 150.0, 155.0, 160.0, 165.0, 170.0])
        from core.options_chain import get_options_chain
        df = get_options_chain(ib_env.ib, "AAPL", strikes_near_spot=1)
        # n=1 → closest ± 1 → at most 3 strikes → 6 rows
        assert len(df) <= 6

    def test_numeric_columns_are_float64(self, ib_env):
        """Numeric chain columns are float64 (not object)."""
        _setup_ib(ib_env.ib)
        from core.options_chain import get_options_chain
        df = get_options_chain(ib_env.ib, "AAPL")
        for col in ["strike", "bid", "ask", "mid", "iv", "delta", "gamma", "theta", "vega", "underlying_price"]:
            assert df[col].dtype == float, f"column {col!r} should be float, got {df[col].dtype}"

    def test_ibkr_sentinel_minus_one_becomes_nan_not_zero(self, ib_env):
        """IBKR's -1 sentinel for bid/ask/Greeks is converted to NaN, not 0."""
        _setup_ib(ib_env.ib, strikes=[150.0])
        from core.options_chain import get_options_chain

        # Override reqTickers to return -1 for all numeric fields
        def _bad_tickers(*contracts):
            if getattr(contracts[0], "secType", "") == "STK":
                return [NS(contract=contracts[0], last=_SPOT, close=_SPOT, bid=_SPOT, modelGreeks=None)]
            return [
                _make_ticker(
                    c,
                    bid=-1.0,
                    ask=-1.0,
                    greeks=_make_greeks(delta=-1.0, gamma=-1.0, theta=-1.0,
                                        vega=-1.0, impliedVol=-1.0, undPrice=-1.0),
                )
                for c in contracts
            ]

        ib_env.ib.reqTickers.side_effect = _bad_tickers
        df = get_options_chain(ib_env.ib, "AAPL")
        # All -1 sentinels must be NaN
        for col in ["bid", "ask", "iv", "delta", "gamma", "theta", "vega"]:
            assert df[col].isna().all(), (
                f"column {col!r}: expected all NaN (sentinel -1 converted), "
                f"got {df[col].tolist()}"
            )

    def test_missing_model_greeks_gives_nan_not_zero(self, ib_env):
        """When modelGreeks is None, Greek columns are NaN — not 0."""
        _setup_ib(ib_env.ib, strikes=[150.0])
        from core.options_chain import get_options_chain

        def _no_greeks_tickers(*contracts):
            if getattr(contracts[0], "secType", "") == "STK":
                return [NS(contract=contracts[0], last=_SPOT, bid=_SPOT, close=_SPOT, modelGreeks=None)]
            return [
                NS(contract=c, bid=5.2, ask=5.4, modelGreeks=None)
                for c in contracts
            ]

        ib_env.ib.reqTickers.side_effect = _no_greeks_tickers
        df = get_options_chain(ib_env.ib, "AAPL")
        for col in ["iv", "delta", "gamma", "theta", "vega"]:
            assert df[col].isna().all(), (
                f"column {col!r}: expected NaN when modelGreeks is None, "
                f"got {df[col].tolist()}"
            )

    def test_underlying_price_falls_back_to_spot_when_undprice_missing(self, ib_env):
        """When undPrice is -1 in modelGreeks, underlying_price falls back to spot."""
        _setup_ib(ib_env.ib, strikes=[150.0])
        from core.options_chain import get_options_chain

        def _no_undprice(*contracts):
            if getattr(contracts[0], "secType", "") == "STK":
                return [NS(contract=contracts[0], last=_SPOT, bid=_SPOT, close=_SPOT, modelGreeks=None)]
            return [
                _make_ticker(c, greeks=_make_greeks(undPrice=-1.0))
                for c in contracts
            ]

        ib_env.ib.reqTickers.side_effect = _no_undprice
        df = get_options_chain(ib_env.ib, "AAPL")
        # underlying_price should be filled from spot when undPrice sentinel
        assert df["underlying_price"].notna().all(), "underlying_price must not be NaN when spot is available"
        assert all(abs(v - _SPOT) < 1e-6 for v in df["underlying_price"])

    def test_qualify_contracts_failure_falls_back_to_unqualified(self, ib_env):
        """When qualifyContracts raises for options, unqualified contracts are tried."""
        _setup_ib(ib_env.ib, strikes=[150.0])
        from core.options_chain import get_options_chain

        call_count = [0]
        original_side_effect = ib_env.ib.qualifyContracts.side_effect

        def _qualify_fails_for_options(*contracts):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call is for the underlying; let it succeed
                return original_side_effect(*contracts)
            raise RuntimeError("simulated qualifyContracts failure")

        ib_env.ib.qualifyContracts.side_effect = _qualify_fails_for_options
        # Should still return data (using unqualified contracts)
        df = get_options_chain(ib_env.ib, "AAPL")
        assert isinstance(df, pd.DataFrame)

    def test_qualify_underlying_failure_returns_empty(self, ib_env):
        """When qualifyContracts fails for the underlying, an empty DataFrame is returned."""
        ib_env.ib.qualifyContracts.side_effect = RuntimeError("simulated failure")
        from core.options_chain import CHAIN_COLUMNS, get_options_chain
        df = get_options_chain(ib_env.ib, "AAPL")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == CHAIN_COLUMNS
        assert len(df) == 0

    def test_req_sec_def_opt_params_returns_empty_gives_empty_df(self, ib_env):
        """Empty reqSecDefOptParams response → empty DataFrame."""
        _setup_ib(ib_env.ib)
        ib_env.ib.reqSecDefOptParams.return_value = []
        from core.options_chain import CHAIN_COLUMNS, get_options_chain
        df = get_options_chain(ib_env.ib, "AAPL")
        assert list(df.columns) == CHAIN_COLUMNS
        assert len(df) == 0

    def test_req_tickers_exception_returns_empty(self, ib_env):
        """reqTickers raising an exception → empty DataFrame, not a crash."""
        _setup_ib(ib_env.ib)
        from core.options_chain import CHAIN_COLUMNS, get_options_chain

        call_count = [0]
        original = ib_env.ib.reqTickers.side_effect

        def _tickers_fail_for_options(*contracts):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call is for spot price (underlying); let it succeed
                return original(*contracts)
            raise RuntimeError("market data lines exhausted")

        ib_env.ib.reqTickers.side_effect = _tickers_fail_for_options
        df = get_options_chain(ib_env.ib, "AAPL")
        assert list(df.columns) == CHAIN_COLUMNS
        assert len(df) == 0

    def test_reqsecdefoptparams_called_with_correct_conid(self, ib_env):
        """reqSecDefOptParams is called with the conId returned by qualifyContracts."""
        _setup_ib(ib_env.ib)
        from core.options_chain import get_options_chain
        get_options_chain(ib_env.ib, "AAPL")
        args = ib_env.ib.reqSecDefOptParams.call_args[0]
        # args: (symbol, futFopExchange, underlyingSecType, underlyingConId)
        assert args[0] == "AAPL"
        assert args[2] == "STK"
        assert args[3] == 12345   # conId from our _qualify side_effect

    def test_trailing_whitespace_in_right_is_stripped(self, ib_env):
        """IBKR sometimes returns 'C '/'P ' with trailing whitespace; both are normalised."""
        _setup_ib(ib_env.ib, strikes=[150.0])
        from core.options_chain import get_options_chain

        def _whitespace_right(*contracts):
            if getattr(contracts[0], "secType", "") == "STK":
                return [NS(contract=contracts[0], last=_SPOT, bid=_SPOT, close=_SPOT, modelGreeks=None)]
            result = []
            for i, c in enumerate(contracts):
                # Mutate right to have trailing space
                c_copy = NS(**vars(c))
                c_copy.right = "C " if i % 2 == 0 else "P "
                result.append(_make_ticker(c_copy))
            return result

        ib_env.ib.reqTickers.side_effect = _whitespace_right
        df = get_options_chain(ib_env.ib, "AAPL")
        assert set(df["right"].unique()).issubset({"C", "P"})


# ---------------------------------------------------------------------------
# Tests: DataLoader.get_options_chain (thin entry point)
# ---------------------------------------------------------------------------

class TestDataLoaderGetOptionsChain:
    """Tests for the DataLoader.get_options_chain entry point."""

    @pytest.fixture
    def loader(self):
        """Minimal DataLoader with a mocked CCXT exchange."""
        from core.data_loader import DataLoader
        from unittest.mock import MagicMock
        dl = DataLoader()
        mock_ex = MagicMock()
        mock_ex.milliseconds.return_value = 0
        mock_ex.rateLimit = 0
        dl.binance_public = mock_ex
        dl.binance_connector = mock_ex
        return dl

    def test_no_broker_manager_returns_empty(self, loader):
        """Passing broker_manager=None returns an empty chain DataFrame."""
        from core.options_chain import CHAIN_COLUMNS
        df = loader.get_options_chain("AAPL", broker_manager=None)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == CHAIN_COLUMNS
        assert len(df) == 0

    def test_ibkr_not_configured_returns_empty(self, loader):
        """When IBKR connector is None in the BrokerManager, an empty chain is returned."""
        from core.options_chain import CHAIN_COLUMNS
        mock_bm = mock.MagicMock()
        mock_bm.brokers = {"IBKR": None, "Simulator": mock.MagicMock()}
        df = loader.get_options_chain("AAPL", broker_manager=mock_bm)
        assert list(df.columns) == CHAIN_COLUMNS
        assert len(df) == 0

    def test_ibkr_connector_missing_ib_attribute_returns_empty(self, loader):
        """When the IBKR connector has no .ib attribute, an empty chain is returned."""
        from core.options_chain import CHAIN_COLUMNS
        mock_connector = mock.MagicMock(spec=[])   # no .ib attribute
        mock_bm = mock.MagicMock()
        mock_bm.brokers = {"IBKR": mock_connector}
        df = loader.get_options_chain("AAPL", broker_manager=mock_bm)
        assert list(df.columns) == CHAIN_COLUMNS
        assert len(df) == 0

    def test_ibkr_connected_delegates_to_options_chain_module(self, loader, ib_env):
        """With a connected IBKR connector, the DataLoader delegates to core.options_chain."""
        _setup_ib(ib_env.ib)

        # Build a minimal fake connector that exposes .ib = ib_env.ib
        fake_connector = NS(ib=ib_env.ib)
        # Use a plain object (not MagicMock) so that .brokers.get is the dict's own .get
        mock_bm = NS(brokers={"IBKR": fake_connector})

        from core.options_chain import CHAIN_COLUMNS
        df = loader.get_options_chain("AAPL", broker_manager=mock_bm)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == CHAIN_COLUMNS
        assert len(df) > 0

    def test_return_columns_always_match_chain_columns(self, loader):
        """The column schema is always CHAIN_COLUMNS, whether the chain is empty or not."""
        from core.options_chain import CHAIN_COLUMNS
        df_empty = loader.get_options_chain("AAPL", broker_manager=None)
        assert list(df_empty.columns) == CHAIN_COLUMNS


# ---------------------------------------------------------------------------
# Tests: empty DataFrame column schema invariant
# ---------------------------------------------------------------------------

def test_empty_chain_has_correct_schema():
    """_empty_chain() always returns a DataFrame with the CHAIN_COLUMNS schema."""
    from core.options_chain import CHAIN_COLUMNS, _empty_chain
    df = _empty_chain("test reason")
    assert list(df.columns) == CHAIN_COLUMNS
    assert len(df) == 0


def test_chain_columns_exported():
    """CHAIN_COLUMNS is importable and contains the minimum required fields."""
    from core.options_chain import CHAIN_COLUMNS
    required = {"strike", "right", "expiry", "bid", "ask", "mid",
                "iv", "delta", "gamma", "theta", "vega", "underlying_price"}
    assert required.issubset(set(CHAIN_COLUMNS))
