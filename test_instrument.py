"""Unit tests for core/instrument.py — Phase 4.0 Instrument data model.

All tests are offline (no network or broker dependency).  They cover:

- Direct construction of each asset class (EQUITY, CRYPTO, OPTION, FUTURE).
- ``Instrument.from_symbol()`` factory for equities and crypto pairs.
- Consistency check: ``from_symbol`` USDT heuristic must agree with
  ``core/chart_builder.is_crypto_symbol()`` so the two modules stay in sync.
- Validation that asset-class-specific optional fields are accessible and
  correctly typed.

Style follows ``test_data_loading.py`` (plain pytest functions, no fixtures
required, imports at top level).
"""

import pytest
from datetime import date

from core.instrument import AssetClass, Instrument, OptionType


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _assert_common_fields(inst: Instrument, symbol: str, asset_class: AssetClass) -> None:
    """Assert the four common fields are present and correctly typed."""
    assert isinstance(inst.symbol, str)
    assert inst.symbol == symbol
    assert isinstance(inst.asset_class, AssetClass)
    assert inst.asset_class == asset_class
    assert isinstance(inst.exchange, str)
    assert isinstance(inst.currency, str)


# ---------------------------------------------------------------------------
# AssetClass enum
# ---------------------------------------------------------------------------


def test_asset_class_values():
    """AssetClass enum has the expected four members."""
    assert AssetClass.EQUITY == "EQUITY"
    assert AssetClass.CRYPTO == "CRYPTO"
    assert AssetClass.OPTION == "OPTION"
    assert AssetClass.FUTURE == "FUTURE"


def test_option_type_values():
    """OptionType enum has CALL and PUT members."""
    assert OptionType.CALL == "CALL"
    assert OptionType.PUT == "PUT"


# ---------------------------------------------------------------------------
# EQUITY construction
# ---------------------------------------------------------------------------


def test_equity_minimal_construction():
    """Instrument with only symbol + EQUITY asset class is valid."""
    inst = Instrument(symbol="AAPL", asset_class=AssetClass.EQUITY)
    _assert_common_fields(inst, "AAPL", AssetClass.EQUITY)
    assert inst.currency == "USD"


def test_equity_optional_fields_are_none():
    """EQUITY instrument has all asset-class-specific optional fields as None."""
    inst = Instrument(symbol="MSFT", asset_class=AssetClass.EQUITY)
    assert inst.base_asset is None
    assert inst.quote_asset is None
    assert inst.underlying is None
    assert inst.expiry is None
    assert inst.strike is None
    assert inst.option_type is None
    assert inst.contract_size is None


def test_equity_with_exchange():
    """EQUITY instrument accepts an explicit exchange string."""
    inst = Instrument(symbol="TSLA", asset_class=AssetClass.EQUITY, exchange="NASDAQ")
    assert inst.exchange == "NASDAQ"
    assert inst.currency == "USD"


# ---------------------------------------------------------------------------
# CRYPTO construction
# ---------------------------------------------------------------------------


def test_crypto_construction():
    """CRYPTO instrument stores base_asset and quote_asset correctly."""
    inst = Instrument(
        symbol="BTCUSDT",
        asset_class=AssetClass.CRYPTO,
        exchange="BINANCE",
        currency="USDT",
        base_asset="BTC",
        quote_asset="USDT",
    )
    _assert_common_fields(inst, "BTCUSDT", AssetClass.CRYPTO)
    assert inst.base_asset == "BTC"
    assert inst.quote_asset == "USDT"
    assert inst.exchange == "BINANCE"
    assert inst.currency == "USDT"


def test_crypto_non_option_fields_are_none():
    """CRYPTO instrument leaves OPTION/FUTURE-specific fields as None."""
    inst = Instrument(
        symbol="ETHUSDT",
        asset_class=AssetClass.CRYPTO,
        base_asset="ETH",
        quote_asset="USDT",
    )
    assert inst.underlying is None
    assert inst.strike is None
    assert inst.expiry is None
    assert inst.option_type is None
    assert inst.contract_size is None


# ---------------------------------------------------------------------------
# OPTION construction
# ---------------------------------------------------------------------------


def test_option_call_construction():
    """OPTION (call) instrument stores all option-specific fields."""
    exp = date(2026, 1, 17)
    inst = Instrument(
        symbol="AAPL260117C00150000",
        asset_class=AssetClass.OPTION,
        exchange="CBOE",
        currency="USD",
        underlying="AAPL",
        strike=150.0,
        expiry=exp,
        option_type=OptionType.CALL,
    )
    _assert_common_fields(inst, "AAPL260117C00150000", AssetClass.OPTION)
    assert inst.underlying == "AAPL"
    assert inst.strike == pytest.approx(150.0)
    assert inst.expiry == exp
    assert inst.option_type == OptionType.CALL
    assert inst.exchange == "CBOE"


def test_option_put_construction():
    """OPTION (put) instrument stores OptionType.PUT correctly."""
    exp = date(2026, 3, 21)
    inst = Instrument(
        symbol="TSLA260321P00200000",
        asset_class=AssetClass.OPTION,
        underlying="TSLA",
        strike=200.0,
        expiry=exp,
        option_type=OptionType.PUT,
    )
    assert inst.option_type == OptionType.PUT
    assert inst.strike == pytest.approx(200.0)
    assert inst.expiry == exp
    assert inst.underlying == "TSLA"


def test_option_crypto_fields_are_none():
    """OPTION instrument has CRYPTO-specific optional fields as None."""
    inst = Instrument(
        symbol="AAPL260117C00150000",
        asset_class=AssetClass.OPTION,
        underlying="AAPL",
        strike=150.0,
        expiry=date(2026, 1, 17),
        option_type=OptionType.CALL,
    )
    assert inst.base_asset is None
    assert inst.quote_asset is None
    assert inst.contract_size is None


# ---------------------------------------------------------------------------
# FUTURE construction
# ---------------------------------------------------------------------------


def test_future_construction():
    """FUTURE instrument stores underlying, expiry, and contract_size."""
    exp = date(2026, 12, 19)
    inst = Instrument(
        symbol="ESZ26",
        asset_class=AssetClass.FUTURE,
        exchange="CME",
        currency="USD",
        underlying="ES",
        expiry=exp,
        contract_size=50.0,
    )
    _assert_common_fields(inst, "ESZ26", AssetClass.FUTURE)
    assert inst.underlying == "ES"
    assert inst.expiry == exp
    assert inst.contract_size == pytest.approx(50.0)
    assert inst.exchange == "CME"


def test_future_crypto_and_option_fields_are_none():
    """FUTURE instrument leaves CRYPTO- and OPTION-specific optional fields as None."""
    inst = Instrument(
        symbol="CLF26",
        asset_class=AssetClass.FUTURE,
        underlying="CL",
        expiry=date(2026, 1, 16),
        contract_size=1000.0,
    )
    assert inst.base_asset is None
    assert inst.quote_asset is None
    assert inst.strike is None
    assert inst.option_type is None


# ---------------------------------------------------------------------------
# Instrument.from_symbol() factory — equity
# ---------------------------------------------------------------------------


def test_from_symbol_equity_aapl():
    """from_symbol('AAPL') returns an EQUITY Instrument."""
    inst = Instrument.from_symbol("AAPL")
    assert inst.asset_class == AssetClass.EQUITY
    assert inst.symbol == "AAPL"
    assert inst.currency == "USD"


def test_from_symbol_equity_msft():
    """from_symbol('MSFT') returns an EQUITY Instrument."""
    inst = Instrument.from_symbol("MSFT")
    assert inst.asset_class == AssetClass.EQUITY
    assert inst.symbol == "MSFT"


def test_from_symbol_equity_uppercase_normalisation():
    """from_symbol normalises the symbol to uppercase."""
    inst = Instrument.from_symbol("aapl")
    assert inst.symbol == "AAPL"
    assert inst.asset_class == AssetClass.EQUITY


def test_from_symbol_equity_spy():
    """from_symbol('SPY') returns an EQUITY Instrument (ETF treated as equity)."""
    inst = Instrument.from_symbol("SPY")
    assert inst.asset_class == AssetClass.EQUITY


# ---------------------------------------------------------------------------
# Instrument.from_symbol() factory — crypto
# ---------------------------------------------------------------------------


def test_from_symbol_crypto_btcusdt():
    """from_symbol('BTCUSDT') returns a CRYPTO Instrument with correct base/quote."""
    inst = Instrument.from_symbol("BTCUSDT")
    assert inst.asset_class == AssetClass.CRYPTO
    assert inst.symbol == "BTCUSDT"
    assert inst.base_asset == "BTC"
    assert inst.quote_asset == "USDT"
    assert inst.exchange == "BINANCE"
    assert inst.currency == "USDT"


def test_from_symbol_crypto_ethusdt():
    """from_symbol('ETHUSDT') returns a CRYPTO Instrument."""
    inst = Instrument.from_symbol("ETHUSDT")
    assert inst.asset_class == AssetClass.CRYPTO
    assert inst.base_asset == "ETH"
    assert inst.quote_asset == "USDT"


def test_from_symbol_crypto_solusdt():
    """from_symbol('SOLUSDT') returns a CRYPTO Instrument."""
    inst = Instrument.from_symbol("SOLUSDT")
    assert inst.asset_class == AssetClass.CRYPTO
    assert inst.base_asset == "SOL"


def test_from_symbol_crypto_lowercase_ethusdt():
    """from_symbol is case-insensitive — 'ethusdt' is classified as CRYPTO."""
    inst = Instrument.from_symbol("ethusdt")
    assert inst.asset_class == AssetClass.CRYPTO
    assert inst.symbol == "ETHUSDT"
    assert inst.base_asset == "ETH"


def test_from_symbol_crypto_adausdt():
    """from_symbol('ADAUSDT') returns a CRYPTO Instrument (matches Yahoo fallback map)."""
    inst = Instrument.from_symbol("ADAUSDT")
    assert inst.asset_class == AssetClass.CRYPTO
    assert inst.base_asset == "ADA"


# ---------------------------------------------------------------------------
# Consistency with core/chart_builder.is_crypto_symbol()
# ---------------------------------------------------------------------------


def test_from_symbol_consistent_with_is_crypto_symbol():
    """from_symbol() USDT heuristic is consistent with chart_builder.is_crypto_symbol().

    Both must agree on every symbol in the set so that Phase 4.x consumers
    can trust either function without the codebase diverging.
    """
    from core.chart_builder import is_crypto_symbol

    symbols = [
        # Crypto — both must return True / CRYPTO
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "ADAUSDT",
        "LINKUSDT",
        "XYZUSDT",          # hypothetical unknown pair
        # Equity — both must return False / EQUITY
        "AAPL",
        "MSFT",
        "TSLA",
        "SPY",
        "AMZN",
    ]

    for sym in symbols:
        inst = Instrument.from_symbol(sym)
        chart_says_crypto = is_crypto_symbol(sym)
        inst_says_crypto = inst.asset_class == AssetClass.CRYPTO

        assert inst_says_crypto == chart_says_crypto, (
            f"Inconsistency for '{sym}': Instrument says "
            f"{'CRYPTO' if inst_says_crypto else 'EQUITY'} but "
            f"is_crypto_symbol() returned {chart_says_crypto}"
        )


# ---------------------------------------------------------------------------
# Misc: equality and repr
# ---------------------------------------------------------------------------


def test_instrument_equality():
    """Two Instruments with identical fields compare equal (dataclass default)."""
    a = Instrument(symbol="AAPL", asset_class=AssetClass.EQUITY)
    b = Instrument(symbol="AAPL", asset_class=AssetClass.EQUITY)
    assert a == b


def test_instrument_inequality():
    """Instruments with different symbols are not equal."""
    a = Instrument(symbol="AAPL", asset_class=AssetClass.EQUITY)
    b = Instrument(symbol="MSFT", asset_class=AssetClass.EQUITY)
    assert a != b


def test_instrument_repr_contains_symbol():
    """repr(Instrument) contains the symbol string (dataclass auto-repr)."""
    inst = Instrument(symbol="BTCUSDT", asset_class=AssetClass.CRYPTO)
    assert "BTCUSDT" in repr(inst)
