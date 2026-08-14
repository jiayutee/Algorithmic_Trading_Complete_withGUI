"""Phase 4.0 — Unified Instrument data model.

This module is the foundation for the multi-asset roadmap:

* Phase 4.1  — IBKR connector (consumes ``Instrument`` to build TWS contract objects)
* Phase 4.2  — Options data layer (relies on ``OPTION`` asset class + strike/expiry fields)
* Phase 4.3  — Futures data layer (relies on ``FUTURE`` asset class + contract_size field)
* Phase 4.4  — Cross-asset portfolio analytics (uses ``AssetClass`` for bucketing)

**Design principles**

1. ADDITIVE ONLY — nothing in this file touches existing data paths.
   ``core/data_loader.py``, ``core/chart_builder.py``, ``brokers/``, and
   ``strategies/`` continue to operate identically.

2. Naming conventions deliberately mirror those already used in the codebase:
   - Crypto classification: ``"USDT" in symbol.upper()`` — the same heuristic
     used in ``core/data_loader._get_historical_data()`` and
     ``core/chart_builder.is_crypto_symbol()``.
   - Crypto exchange default: ``"BINANCE"`` — the primary exchange used by
     ``core/data_loader._get_binance_historical()``.
   - Symbol format for crypto: ``"BTCUSDT"`` (Binance notation), not ``"BTC-USD"``.

3. All asset-class-specific optional fields default to ``None`` so a plain
   ``Instrument(symbol="AAPL", asset_class=AssetClass.EQUITY)`` is valid
   without supplying irrelevant fields.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AssetClass(str, Enum):
    """Top-level asset classification.

    ``str`` mixin allows ``AssetClass.EQUITY == "EQUITY"`` comparisons and
    clean JSON serialisation without extra conversion.
    """

    EQUITY = "EQUITY"
    CRYPTO = "CRYPTO"
    OPTION = "OPTION"
    FUTURE = "FUTURE"


class OptionType(str, Enum):
    """Put/call flag for OPTION instruments."""

    CALL = "CALL"
    PUT = "PUT"


# ---------------------------------------------------------------------------
# Instrument dataclass
# ---------------------------------------------------------------------------


@dataclass
class Instrument:
    """Unified descriptor for any tradeable instrument.

    Common fields
    -------------
    symbol : str
        The canonical ticker string (e.g. ``"AAPL"``, ``"BTCUSDT"``,
        ``"AAPL260117C00150000"`` for an option).
    asset_class : AssetClass
        Broad classification — EQUITY, CRYPTO, OPTION, or FUTURE.
    exchange : str
        Exchange identifier (e.g. ``"NASDAQ"``, ``"BINANCE"``, ``"CME"``).
        Defaults to ``""`` (unknown / not applicable).
    currency : str
        Settlement currency ISO code (e.g. ``"USD"``, ``"USDT"``).
        Defaults to ``"USD"``.

    CRYPTO-specific (optional)
    --------------------------
    base_asset : str or None
        The base/traded coin (e.g. ``"BTC"`` for BTCUSDT).
    quote_asset : str or None
        The quote/settlement coin (e.g. ``"USDT"`` for BTCUSDT).

    OPTION-specific (optional)
    --------------------------
    underlying : str or None
        Underlying ticker (e.g. ``"AAPL"`` for an AAPL option).
    strike : float or None
        Strike price.
    expiry : date or None
        Expiry / expiration date.
    option_type : OptionType or None
        CALL or PUT.

    FUTURE-specific (optional)
    --------------------------
    underlying : str or None
        Underlying commodity / index ticker (shared with OPTION field).
    expiry : date or None
        Delivery / expiry date (shared with OPTION field).
    contract_size : float or None
        Number of units per contract (e.g. ``100.0`` for ES futures).

    Notes
    -----
    - EQUITY instruments need only the four common fields.
    - ``underlying`` and ``expiry`` are shared between OPTION and FUTURE so
      Phase 4.1+ can inspect them uniformly without branching on asset class.
    """

    # ------------------------------------------------------------------
    # Common fields
    # ------------------------------------------------------------------
    symbol: str
    asset_class: AssetClass
    exchange: str = ""
    currency: str = "USD"

    # ------------------------------------------------------------------
    # CRYPTO-specific
    # ------------------------------------------------------------------
    base_asset: Optional[str] = field(default=None)
    quote_asset: Optional[str] = field(default=None)

    # ------------------------------------------------------------------
    # OPTION / FUTURE shared
    # ------------------------------------------------------------------
    underlying: Optional[str] = field(default=None)
    expiry: Optional[date] = field(default=None)

    # ------------------------------------------------------------------
    # OPTION-specific
    # ------------------------------------------------------------------
    strike: Optional[float] = field(default=None)
    option_type: Optional[OptionType] = field(default=None)

    # ------------------------------------------------------------------
    # FUTURE-specific
    # ------------------------------------------------------------------
    contract_size: Optional[float] = field(default=None)

    # ------------------------------------------------------------------
    # Factory / parsing helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_symbol(cls, symbol: str) -> "Instrument":
        """Build an ``Instrument`` from a raw symbol string using codebase conventions.

        Classification rules (in priority order):

        1. **CRYPTO** — ``"USDT"`` appears anywhere in the uppercased symbol.
           This mirrors ``core/data_loader._get_historical_data()`` and
           ``core/chart_builder.is_crypto_symbol()`` exactly.
           - ``base_asset`` is derived by stripping the ``USDT`` suffix.
           - ``quote_asset`` is set to ``"USDT"``.
           - ``exchange`` defaults to ``"BINANCE"`` (primary crypto venue).
           - ``currency`` defaults to ``"USDT"``.

        2. **EQUITY** — everything else defaults to equity.
           - ``exchange`` and ``currency`` are left at their dataclass defaults
             (``""`` and ``"USD"`` respectively).

        Note: OPTION and FUTURE instruments have complex symbol structures that
        vary by venue.  Use the ``Instrument(...)`` constructor directly for
        those asset classes — the ``from_symbol`` factory only handles
        equity/crypto auto-detection.

        Parameters
        ----------
        symbol:
            Raw ticker string (case-insensitive for the USDT check).

        Returns
        -------
        Instrument
            A fully populated (for the detected asset class) ``Instrument``.

        Examples
        --------
        >>> Instrument.from_symbol("AAPL")
        Instrument(symbol='AAPL', asset_class=<AssetClass.EQUITY: 'EQUITY'>, ...)

        >>> Instrument.from_symbol("BTCUSDT")
        Instrument(symbol='BTCUSDT', asset_class=<AssetClass.CRYPTO: 'CRYPTO'>,
                   base_asset='BTC', quote_asset='USDT', ...)

        >>> Instrument.from_symbol("ethusdt")   # case-insensitive
        Instrument(symbol='ETHUSDT', asset_class=<AssetClass.CRYPTO: 'CRYPTO'>,
                   base_asset='ETH', quote_asset='USDT', ...)
        """
        upper = symbol.upper()

        # --- Crypto detection (mirrors the canonical codebase heuristic) ---
        if "USDT" in upper:
            # Strip USDT suffix to get the base asset (e.g. "BTC" from "BTCUSDT")
            base = re.sub(r"USDT$", "", upper)
            return cls(
                symbol=upper,
                asset_class=AssetClass.CRYPTO,
                exchange="BINANCE",
                currency="USDT",
                base_asset=base,
                quote_asset="USDT",
            )

        # --- Default: equity ---
        return cls(
            symbol=upper,
            asset_class=AssetClass.EQUITY,
        )
