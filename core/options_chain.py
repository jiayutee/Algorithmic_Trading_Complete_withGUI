"""Phase 4.2 — Options chain + Greeks retrieval via IBKR (READ-ONLY market data; no order path).

Public API
----------
get_options_chain(ib, symbol, expiry=None, strikes_near_spot=10, exchange='SMART', currency='USD')
    Returns a pandas DataFrame, one row per (strike × right) combination.
    Columns: strike, right, expiry, bid, ask, mid, iv, delta, gamma, theta, vega, underlying_price.
    All unavailable quotes/Greeks are NaN — never 0.
    On any unrecoverable error, returns an empty DataFrame (correct schema, zero rows);
    the reason is logged at WARNING level and never raised.

Usage example (from the desktop or Dash front-end)
---------------------------------------------------
    # broker_manager is an existing BrokerManager with IBKR connected:
    ibkr = broker_manager.brokers.get("IBKR")
    if ibkr:
        df = get_options_chain(ibkr.ib, "AAPL")
    # Or via the DataLoader thin entry point:
    df = data_loader.get_options_chain("AAPL", broker_manager=broker_manager)

Design notes
------------
* ib_insync.Stock / ib_insync.Option contracts are imported lazily (inside the
  function) so this module can be imported without ib_insync installed.
* expiry=None → nearest listed future expiry (from reqSecDefOptParams.expirations).
  When all listed expirations are in the past the most-recent past expiry is used
  and a warning is logged.
* strikes_near_spot caps the number of market-data lines requested; IBKR enforces
  per-account limits.  Default 10 yields ≤ 2×10+1 = 21 strikes × 2 rights = ≤ 42
  lines per call.
* IBKR uses -1 as a sentinel for unavailable numeric fields; _ibkr_to_nan()
  converts those (plus NaN/±Inf) to None before they enter the DataFrame.
* This module is entirely read-only: it calls only qualifyContracts,
  reqSecDefOptParams, and reqTickers — no order path whatsoever.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output schema — callers may reference this to build empty frames of the
# correct shape when IBKR is unavailable.
# ---------------------------------------------------------------------------

CHAIN_COLUMNS: list = [
    "strike", "right", "expiry", "bid", "ask", "mid",
    "iv", "delta", "gamma", "theta", "vega", "underlying_price",
]

# IBKR returns -1 for many unavailable numeric fields instead of NaN/None.
_IBKR_UNAVAILABLE = -1.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ibkr_to_nan(value) -> Optional[float]:
    """Convert a raw IBKR numeric field to float or None.

    Conversions
    -----------
    * None / non-numeric  → None
    * NaN or ±Inf         → None
    * -1.0 (IBKR sentinel)→ None  (IBKR uses -1 to mean "unavailable")
    * any other float     → float (passed through)
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    if f == _IBKR_UNAVAILABLE:
        return None
    return f


def _empty_chain(reason: str = "") -> pd.DataFrame:
    """Return an empty DataFrame with the canonical column schema."""
    if reason:
        logger.warning("get_options_chain: %s — returning empty chain", reason)
    return pd.DataFrame(columns=CHAIN_COLUMNS)


def _parse_expiry(expiry_input) -> Optional[str]:
    """Normalise *expiry_input* to a YYYYMMDD string, or raise on bad input.

    Accepted inputs
    ---------------
    * ``None``           → None (caller picks nearest)
    * ``"20261016"``     → ``"20261016"``
    * ``"2026-10-16"``   → ``"20261016"``  (dashes stripped)
    * ``datetime.date``  → ``"20261016"``

    Raises
    ------
    ValueError
        If a string is given but cannot be parsed.
    TypeError
        If the type is not str, date, or None.
    """
    if expiry_input is None:
        return None
    if isinstance(expiry_input, date):
        return expiry_input.strftime("%Y%m%d")
    if isinstance(expiry_input, str):
        cleaned = expiry_input.replace("-", "")
        if len(cleaned) == 8 and cleaned.isdigit():
            return cleaned
        raise ValueError(
            f"Invalid expiry format '{expiry_input}'; "
            "expected YYYYMMDD or YYYY-MM-DD"
        )
    raise TypeError(
        f"expiry must be str, datetime.date, or None; got {type(expiry_input)!r}"
    )


def _select_expiry(available: list, target: Optional[str]) -> Optional[str]:
    """Pick the expiry to use from the chain's listed expirations.

    Rules (evaluated in order)
    --------------------------
    1. *target* is in *available*             → use it.
    2. *target* is not in *available* (or None) → fall through to nearest-future logic.
    3. Nearest future: first expiry ≥ today in YYYYMMDD lexicographic order.
    4. All past: most-recent past expiry, with a warning.
    5. *available* is empty                   → return None.
    """
    if not available:
        return None
    if target is not None and target in available:
        return target
    if target is not None:
        logger.warning(
            "get_options_chain: expiry %s not listed; available=%s ... selecting nearest",
            target, list(available)[:4],
        )
    today = datetime.now().strftime("%Y%m%d")
    future = [e for e in available if e >= today]
    if future:
        return future[0]
    logger.warning(
        "get_options_chain: all listed expirations are in the past; "
        "using most recent: %s", available[-1],
    )
    return available[-1]


def _select_strikes(all_strikes: list, spot: Optional[float], n: int) -> list:
    """Return at most 2*n+1 strikes centred on *spot*.

    Parameters
    ----------
    all_strikes:
        Full list of available strikes from reqSecDefOptParams.
    spot:
        Current underlying price, used to anchor the window.  If None or ≤ 0,
        all strikes are returned with a warning (no data to filter on).
    n:
        Number of strikes above *and* below the ATM strike to include.

    Returns
    -------
    list
        Sorted ascending strike list, capped at 2*n+1 entries.
    """
    if not all_strikes:
        return []
    sorted_strikes = sorted(all_strikes)
    if spot is None or spot <= 0:
        logger.warning(
            "get_options_chain: spot price unavailable; "
            "returning all %d strikes", len(sorted_strikes),
        )
        return sorted_strikes
    # Index of the strike closest to spot
    closest_idx = min(
        range(len(sorted_strikes)), key=lambda i: abs(sorted_strikes[i] - spot)
    )
    lo = max(0, closest_idx - n)
    hi = min(len(sorted_strikes), closest_idx + n + 1)
    return sorted_strikes[lo:hi]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_options_chain(
    ib,
    symbol: str,
    expiry: Union[str, date, None] = None,
    strikes_near_spot: int = 10,
    exchange: str = "SMART",
    currency: str = "USD",
) -> pd.DataFrame:
    """Retrieve the options chain for *symbol* from IBKR (read-only).

    Parameters
    ----------
    ib : ib_insync.IB
        A live, connected IB instance — the raw object, NOT the IBKRConnector
        wrapper.  Obtain it as ``broker_manager.brokers['IBKR'].ib``.
    symbol : str
        Underlying ticker (equity / ETF), e.g. ``"AAPL"``, ``"SPY"``.
        Crypto symbols are not supported by IBKR options data.
    expiry : str, datetime.date, or None
        Target expiry.  Accepted formats: YYYYMMDD or YYYY-MM-DD strings, or a
        ``datetime.date`` object.  When ``None`` (default), the nearest listed
        future expiry is selected automatically from the chain's expiration list
        returned by ``reqSecDefOptParams``.
    strikes_near_spot : int
        Number of strikes above *and* below the current spot price to include.
        Keeping this small avoids hitting IBKR's per-account market-data line
        limits.  Default: 10 (yields ≤ 2×10+1 = 21 strikes × 2 rights = ≤ 42
        market-data lines per call).
    exchange : str
        Routing exchange for option contracts.  Default: ``"SMART"``.
    currency : str
        Settlement currency.  Default: ``"USD"``.

    Returns
    -------
    pd.DataFrame
        One row per (strike × right) combination.  Columns:

        ========================  ============================================
        strike                    float — strike price
        right                     str   — ``'C'`` (call) or ``'P'`` (put)
        expiry                    str   — YYYYMMDD
        bid                       float or NaN
        ask                       float or NaN
        mid                       float or NaN — (bid + ask) / 2
        iv                        float or NaN — implied volatility
        delta                     float or NaN
        gamma                     float or NaN
        theta                     float or NaN
        vega                      float or NaN
        underlying_price          float or NaN
        ========================  ============================================

        Returns an empty DataFrame (correct schema, zero rows) when the chain
        cannot be fetched; the reason is logged at WARNING level.

    Notes
    -----
    * ``expiry=None`` picks the **nearest listed future expiry** from the
      chain's ``.expirations`` list.  When all listed expirations are in the
      past, the most-recent past expiry is used and a warning is logged.
    * IBKR returns ``-1`` for many unavailable numeric fields; this function
      normalises those — and NaN / ±Inf — to ``NaN`` in float columns.
      Missing quotes or Greeks are never filled with ``0``.
    * This function is **read-only**: it calls only ``qualifyContracts``,
      ``reqSecDefOptParams``, and ``reqTickers`` — no order path whatsoever.
    """
    # --- Normalise expiry input ---
    try:
        expiry_str = _parse_expiry(expiry)
    except (TypeError, ValueError) as exc:
        return _empty_chain(f"invalid expiry argument: {exc}")

    # ------------------------------------------------------------------ #
    #  Step 1: Qualify the underlying stock to obtain its conId.          #
    #  reqSecDefOptParams requires the numeric conId, not just the        #
    #  ticker string.                                                      #
    # ------------------------------------------------------------------ #
    try:
        from ib_insync import Stock  # lazy: ib_insync may not be installed
        underlying = Stock(symbol, exchange, currency)
        qualified_underlying = ib.qualifyContracts(underlying)
        if not qualified_underlying:
            return _empty_chain(
                f"qualifyContracts returned empty list for underlying {symbol!r}"
            )
        underlying_contract = qualified_underlying[0]
        underlying_conid = underlying_contract.conId
        logger.debug(
            "get_options_chain: %s qualified — conId=%s", symbol, underlying_conid
        )
    except Exception as exc:
        return _empty_chain(f"failed to qualify underlying {symbol!r}: {exc}")

    # ------------------------------------------------------------------ #
    #  Step 2: Request option chain parameters (all expiries + strikes).  #
    # ------------------------------------------------------------------ #
    try:
        chains = ib.reqSecDefOptParams(symbol, "", "STK", underlying_conid)
        if not chains:
            return _empty_chain(
                f"reqSecDefOptParams returned no data for {symbol!r}"
            )
    except Exception as exc:
        return _empty_chain(f"reqSecDefOptParams failed for {symbol!r}: {exc}")

    # Prefer the SMART-routed chain when available; fall back to the first entry.
    smart_chains = [c for c in chains if getattr(c, "exchange", "") == "SMART"]
    chain = smart_chains[0] if smart_chains else chains[0]
    all_expirations = sorted(getattr(chain, "expirations", []))
    all_strikes = sorted(getattr(chain, "strikes", []))

    if not all_expirations:
        return _empty_chain(f"no expirations found in chain for {symbol!r}")

    chosen_expiry = _select_expiry(all_expirations, expiry_str)
    if chosen_expiry is None:
        return _empty_chain(f"could not determine expiry for {symbol!r}")
    logger.info(
        "get_options_chain: %s — chosen expiry %s", symbol, chosen_expiry
    )

    # ------------------------------------------------------------------ #
    #  Step 3: Fetch the underlying spot price for strike filtering.      #
    # ------------------------------------------------------------------ #
    spot_price: Optional[float] = None
    try:
        spot_tickers = ib.reqTickers(underlying_contract)
        if spot_tickers:
            t = spot_tickers[0]
            # Prefer 'last', then 'close', then 'bid' as a fallback proxy
            for field in ("last", "close", "bid"):
                candidate = _ibkr_to_nan(getattr(t, field, None))
                if candidate is not None:
                    spot_price = candidate
                    break
    except Exception as exc:
        logger.warning(
            "get_options_chain: spot price fetch for %s failed (%s); "
            "all strikes will be included", symbol, exc,
        )

    # ------------------------------------------------------------------ #
    #  Step 4: Select strikes near spot.                                  #
    # ------------------------------------------------------------------ #
    selected_strikes = _select_strikes(all_strikes, spot_price, strikes_near_spot)
    if not selected_strikes:
        return _empty_chain(
            f"no strikes available for {symbol!r} expiry {chosen_expiry}"
        )
    logger.info(
        "get_options_chain: %s %s — %d strikes selected "
        "(spot=%s, strikes_near_spot=%d)",
        symbol, chosen_expiry, len(selected_strikes),
        f"{spot_price:.2f}" if spot_price else "n/a", strikes_near_spot,
    )

    # ------------------------------------------------------------------ #
    #  Step 5: Build Option contracts, qualify them, request tickers.     #
    # ------------------------------------------------------------------ #
    try:
        from ib_insync import Option  # lazy: ib_insync may not be installed
        option_contracts = [
            Option(symbol, chosen_expiry, strike, right, exchange, currency=currency)
            for strike in selected_strikes
            for right in ("C", "P")
        ]
    except Exception as exc:
        return _empty_chain(f"failed to construct Option contracts: {exc}")

    try:
        qualified_opts = ib.qualifyContracts(*option_contracts)
        # qualifyContracts may return fewer contracts than requested if some are
        # unrecognised by IBKR; proceed with what was qualified rather than failing.
        if not qualified_opts:
            return _empty_chain(
                f"qualifyContracts returned empty for {symbol!r} options"
            )
        logger.debug(
            "get_options_chain: %d / %d option contracts qualified",
            len(qualified_opts), len(option_contracts),
        )
    except Exception as exc:
        logger.warning(
            "get_options_chain: qualifyContracts failed for options (%s); "
            "proceeding with unqualified contracts", exc,
        )
        qualified_opts = option_contracts

    try:
        tickers = ib.reqTickers(*qualified_opts)
    except Exception as exc:
        return _empty_chain(
            f"reqTickers failed for {symbol!r} {chosen_expiry} options: {exc}"
        )

    # ------------------------------------------------------------------ #
    #  Step 6: Build the output DataFrame from ticker data.               #
    # ------------------------------------------------------------------ #
    rows = []
    for ticker in tickers:
        contract = getattr(ticker, "contract", None)
        if contract is None:
            continue

        strike = _ibkr_to_nan(getattr(contract, "strike", None))
        right_raw = getattr(contract, "right", "") or ""
        # IBKR occasionally returns 'C ' or 'P ' with trailing whitespace
        right = right_raw.strip().upper()[:1]
        exp = getattr(contract, "lastTradeDateOrContractMonth", chosen_expiry)

        bid = _ibkr_to_nan(getattr(ticker, "bid", None))
        ask = _ibkr_to_nan(getattr(ticker, "ask", None))
        mid: Optional[float] = None
        if bid is not None and ask is not None:
            mid = (bid + ask) / 2.0

        # Greeks come from modelGreeks (model-implied; populated after
        # reqTickers with generic tick type 13 / market-data type 4).
        greeks = getattr(ticker, "modelGreeks", None)
        if greeks is not None:
            iv        = _ibkr_to_nan(getattr(greeks, "impliedVol", None))
            delta     = _ibkr_to_nan(getattr(greeks, "delta", None))
            gamma     = _ibkr_to_nan(getattr(greeks, "gamma", None))
            theta     = _ibkr_to_nan(getattr(greeks, "theta", None))
            vega      = _ibkr_to_nan(getattr(greeks, "vega", None))
            und_price = _ibkr_to_nan(getattr(greeks, "undPrice", None))
        else:
            iv = delta = gamma = theta = vega = und_price = None

        # Fall back to the spot fetched from the underlying ticker when
        # modelGreeks.undPrice is absent — ensures the column is populated.
        if und_price is None and spot_price is not None:
            und_price = spot_price

        rows.append({
            "strike":           strike,
            "right":            right,
            "expiry":           exp,
            "bid":              bid,
            "ask":              ask,
            "mid":              mid,
            "iv":               iv,
            "delta":            delta,
            "gamma":            gamma,
            "theta":            theta,
            "vega":             vega,
            "underlying_price": und_price,
        })

    if not rows:
        return _empty_chain(
            f"reqTickers returned no usable rows for {symbol!r} {chosen_expiry}"
        )

    df = pd.DataFrame(rows, columns=CHAIN_COLUMNS)

    # Coerce numeric columns to float64 (None → NaN); string columns stay as-is.
    float_cols = [
        "strike", "bid", "ask", "mid", "iv",
        "delta", "gamma", "theta", "vega", "underlying_price",
    ]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(
        "get_options_chain: %s %s — %d rows returned", symbol, chosen_expiry, len(df)
    )
    return df
