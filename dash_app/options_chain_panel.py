"""Phase 5.1 - Options chain panel helpers (read-only, no order path).

The panel shows a live options chain table with Greeks sourced from IBKR.
It degrades gracefully when IBKR is unavailable:

    IBKR_ENABLED not set         -> "IBKR not enabled (set IBKR_ENABLED=1)"
    broker_manager is None       -> "IBKR not connected"
    IBKR connector missing/None  -> "IBKR not connected"
    IBKR ib not connected        -> "IBKR not connected"
    Empty chain result           -> "No options data returned for {symbol}"

CRITICAL -- NO ORDER PATH:  This module contains no buy/sell buttons, no
submit_order calls, and no imports from brokers/*.py order methods.  It is
entirely read-only.  Phase 5.2 (order flow) remains blocked per specification.
"""

from __future__ import annotations

import math
import os
import logging
from typing import Optional

from core.options_chain import CHAIN_COLUMNS

logger = logging.getLogger(__name__)

# Human-readable column headers for the DataTable (order matches CHAIN_COLUMNS).
CHAIN_TABLE_COLUMNS = [
    {"name": "Strike",     "id": "strike"},
    {"name": "C/P",        "id": "right"},
    {"name": "Expiry",     "id": "expiry"},
    {"name": "Bid",        "id": "bid"},
    {"name": "Ask",        "id": "ask"},
    {"name": "Mid",        "id": "mid"},
    {"name": "IV",         "id": "iv"},
    {"name": "Delta",      "id": "delta"},
    {"name": "Gamma",      "id": "gamma"},
    {"name": "Theta",      "id": "theta"},
    {"name": "Vega",       "id": "vega"},
    {"name": "Underlying", "id": "underlying_price"},
]

# Sanity-check: IDs must match CHAIN_COLUMNS in order.
assert [c["id"] for c in CHAIN_TABLE_COLUMNS] == list(CHAIN_COLUMNS), (
    "CHAIN_TABLE_COLUMNS IDs diverged from CHAIN_COLUMNS -- update options_chain_panel.py"
)


# ---------------------------------------------------------------------------
# Internal formatting helpers
# ---------------------------------------------------------------------------

def _fmt_float(val, ndigits: int = 4) -> str:
    """Format a numeric value as a string; returns '--' for None / NaN / Inf."""
    if val is None:
        return "--"
    try:
        f = float(val)
    except (TypeError, ValueError):
        return "--"
    if math.isnan(f) or math.isinf(f):
        return "--"
    return f"{f:.{ndigits}f}"


def _format_chain_row(row) -> dict:
    """Convert one row of the chain DataFrame to a display dict.

    Parameters
    ----------
    row : pandas.Series or dict-like
        One row of the DataFrame returned by ``get_options_chain``.

    Returns
    -------
    dict
        Keys match the ``id`` fields in ``CHAIN_TABLE_COLUMNS``.
    """
    def _get(col):
        try:
            return row[col]
        except (KeyError, TypeError):
            return None

    return {
        "strike":           _fmt_float(_get("strike"), 2),
        "right":            str(_get("right") or "--"),
        "expiry":           str(_get("expiry") or "--"),
        "bid":              _fmt_float(_get("bid"), 4),
        "ask":              _fmt_float(_get("ask"), 4),
        "mid":              _fmt_float(_get("mid"), 4),
        "iv":               _fmt_float(_get("iv"), 4),
        "delta":            _fmt_float(_get("delta"), 4),
        "gamma":            _fmt_float(_get("gamma"), 4),
        "theta":            _fmt_float(_get("theta"), 4),
        "vega":             _fmt_float(_get("vega"), 4),
        "underlying_price": _fmt_float(_get("underlying_price"), 2),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_options_chain_data(
    symbol: str,
    broker_manager=None,
    expiry: Optional[str] = None,
) -> tuple:
    """Build ``(rows, status_msg)`` for the options chain DataTable.

    Degrades gracefully for every unavailable state.  Never raises into the
    Dash callback layer.

    Parameters
    ----------
    symbol : str
        Underlying ticker, e.g. ``"AAPL"``.
    broker_manager : BrokerManager or object-with-.brokers dict, or None
        Must expose ``broker_manager.brokers.get("IBKR")`` -> connector with
        ``connector.ib.isConnected()``.  When None (or when the IBKR broker
        slot is not populated), the function returns a degraded message.
    expiry : str or None
        Target expiry in YYYYMMDD format.  Passed directly to
        ``core.options_chain.get_options_chain``.  Defaults to the nearest
        listed future expiry when ``None``.

    Returns
    -------
    tuple[list[dict], str]
        ``(rows, status_msg)`` -- rows is a list of dicts keyed by the column
        IDs in ``CHAIN_TABLE_COLUMNS``, ready for ``dash_table.DataTable``.
        ``status_msg`` is a human-readable one-line summary.
    """
    # --- State 1: IBKR_ENABLED env var not set --------------------------------
    if not os.environ.get("IBKR_ENABLED"):
        return [], "IBKR not enabled (set IBKR_ENABLED=1)"

    # --- State 2: No broker_manager supplied ----------------------------------
    if broker_manager is None:
        return [], "IBKR not connected"

    # --- State 3: IBKR connector missing from the broker map -----------------
    brokers_dict = getattr(broker_manager, "brokers", None) or {}
    ibkr_connector = brokers_dict.get("IBKR") if hasattr(brokers_dict, "get") else None
    if ibkr_connector is None:
        return [], "IBKR not connected"

    # --- State 4: ib object missing or not connected -------------------------
    ib = getattr(ibkr_connector, "ib", None)
    if ib is None or not ib.isConnected():
        return [], "IBKR not connected"

    # --- Fetch the chain (returns empty DataFrame on any error) --------------
    try:
        from core.options_chain import get_options_chain
        df = get_options_chain(ib, symbol, expiry=expiry or None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Options chain panel] fetch error for %s: %s", symbol, exc)
        return [], f"Options chain error: {exc}"

    # --- State 5: Empty chain result -----------------------------------------
    if df is None or df.empty:
        return [], f"No options data returned for {symbol}"

    # --- Format rows for display ---------------------------------------------
    rows = [_format_chain_row(row) for _, row in df.iterrows()]

    expiry_tag = f" (expiry {expiry})" if expiry else ""
    status_msg = f"Options chain: {len(rows)} contracts for {symbol}{expiry_tag}"
    logger.info("[Options chain panel] %s", status_msg)
    return rows, status_msg
