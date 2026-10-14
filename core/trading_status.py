"""
core/trading_status.py

Display-side safety state shared by both front ends (desktop ``ui/main_window.py`` and Dash ``dash_app/``):

* ``account_mode``    -- which account an order would hit right now: PAPER, DRY-RUN, LIVE, LIVE BLOCKED or NOT CONNECTED.
                         Mirrors ``brokers/execution_guard.py`` exactly; it never decides anything itself.
* ``price_freshness`` -- how old the price on screen is, using the simulator's own stale limit (300 s).
* ``preview_order``   -- what an order will cost before it is sent (notional, fee, cash after) plus warnings
                         and blocking problems, for the Review -> Confirm step.
* ``engage_kill_switch`` / ``release_kill_switch`` -- create / remove the kill-switch FILE the guard already
                         watches, so the switch holds across processes (desktop, Dash, scripts) until released.

Pure functions, no Qt or Dash imports, so they are unit-tested directly (tests/test_trading_status.py).
"""

from __future__ import annotations

import datetime as _dt
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional

# Mode identifiers (also used as CSS / style keys by the front ends).
MODE_PAPER = "paper"
MODE_DRY_RUN = "dry_run"
MODE_LIVE = "live"
MODE_BLOCKED = "blocked"
MODE_UNAVAILABLE = "unavailable"

# Price-age thresholds. STALE matches the simulator's max_price_age_s (core/broker_manager.py, dash_app/callbacks.py):
# past it, a strict paper account rejects market orders.
PRICE_FRESH_S = 60.0
PRICE_STALE_S = 300.0

SIMULATOR_NAME = "Simulator"

# Presentation colors shared by both front ends (design language v2). Each mode differs in lightness as well as hue,
# and the label text always names the mode, so color is never the only cue.
MODE_COLORS = {          # mode -> (background, foreground, border)
    MODE_PAPER:       ("#0f3d3a", "#56d4c9", "#0f3d3a"),
    MODE_DRY_RUN:     ("#3d2e00", "#e3b341", "#3d2e00"),
    MODE_LIVE:        ("#b62324", "#ffffff", "#ff7b72"),
    MODE_BLOCKED:     ("#21262d", "#e6edf3", "#ff7b72"),
    MODE_UNAVAILABLE: ("#21262d", "#8b949e", "#30363d"),
}
FRESHNESS_COLORS = {"fresh": "#8b949e", "aging": "#e3b341", "stale": "#ff7b72", "unknown": "#8b949e"}
UI_FONT_FAMILIES = ["IBM Plex Sans", "SF Pro Text", "Helvetica Neue", "Segoe UI", "Arial", "sans-serif"]
MONO_FONT_FAMILIES = ["IBM Plex Mono", "SF Mono", "Menlo", "Consolas", "monospace"]


@dataclass(frozen=True)
class AccountMode:
    mode: str      # one of the MODE_* constants
    label: str     # short pill text, e.g. "PAPER · Simulator"
    detail: str    # one sentence for a tooltip / status line

    @property
    def real_money(self) -> bool:
        """True only when an order would actually be sent to a real account."""
        return self.mode == MODE_LIVE


def _is_paper_broker(broker_name: str, broker) -> bool:
    if broker_name == SIMULATOR_NAME:
        return True
    if broker is None:
        return False
    if type(broker).__name__ == "SimulatedBroker":
        return True
    return bool(getattr(broker, "paper_mode", False))


def account_mode(broker_name: str, broker=None, guard=None) -> AccountMode:
    """Which account an order placed now would reach. ``broker`` is the connector instance (None = not connected)."""
    name = broker_name or SIMULATOR_NAME
    if _is_paper_broker(name, broker):
        if name == SIMULATOR_NAME:
            return AccountMode(MODE_PAPER, "PAPER · Simulator", "Paper account: simulated fills at real prices, no real money.")
        return AccountMode(MODE_PAPER, f"PAPER · {name}", f"{name} paper/testnet account: no real money.")
    if broker is None:
        return AccountMode(MODE_UNAVAILABLE, f"NOT CONNECTED · {name}",
                           f"{name} is not connected (missing keys or library). Orders cannot be sent.")
    if guard is None:
        from brokers.execution_guard import get_guard
        guard = get_guard()
    reason = guard.kill_switch_reason()
    if reason:
        return AccountMode(MODE_BLOCKED, f"LIVE BLOCKED · {name}", f"Live orders are blocked: {reason}.")
    if guard.dry_run():
        return AccountMode(MODE_DRY_RUN, f"DRY-RUN · {name}",
                           f"{name} live connector in dry-run: orders are logged, not sent (LIVE_DRY_RUN=false to send).")
    return AccountMode(MODE_LIVE, f"LIVE · {name}", f"{name} LIVE: real orders with real money will be sent.")


# ---------------------------------------------------------------------------------------------------- kill switch

def kill_switch_engaged(guard=None) -> bool:
    """True when the kill-switch file exists (the cross-process switch). Live trading being disabled is separate."""
    if guard is None:
        from brokers.execution_guard import get_guard
        guard = get_guard()
    return os.path.exists(guard.kill_switch_file())


def engage_kill_switch(reason: str = "engaged from the UI", guard=None) -> str:
    """Create the kill-switch file the execution guard checks on every live order. Returns its path."""
    if guard is None:
        from brokers.execution_guard import get_guard
        guard = get_guard()
    path = guard.kill_switch_file()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{_dt.datetime.now().isoformat(timespec='seconds')} {reason}\n")
    return path


def release_kill_switch(guard=None) -> bool:
    """Remove the kill-switch file. Returns True if a file was removed."""
    if guard is None:
        from brokers.execution_guard import get_guard
        guard = get_guard()
    path = guard.kill_switch_file()
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return False


# ------------------------------------------------------------------------------------------------- price freshness

@dataclass(frozen=True)
class PriceFreshness:
    state: str     # "fresh" | "aging" | "stale" | "unknown"
    label: str     # e.g. "12s ago", "4m ago · stale"


def format_age(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


def price_freshness(age_s: Optional[float]) -> PriceFreshness:
    if age_s is None or not math.isfinite(age_s):
        return PriceFreshness("unknown", "no price yet")
    if age_s <= PRICE_FRESH_S:
        return PriceFreshness("fresh", f"{format_age(age_s)} ago")
    if age_s <= PRICE_STALE_S:
        return PriceFreshness("aging", f"{format_age(age_s)} ago")
    return PriceFreshness("stale", f"{format_age(age_s)} ago · stale")


# ---------------------------------------------------------------------------------------------------- order preview

@dataclass
class OrderPreview:
    side: str
    qty: Optional[float]
    symbol: str
    order_type: str
    price: Optional[float]          # price the estimate uses (limit/stop price, else the last price)
    notional: Optional[float]
    fee_rate: Optional[float]
    fee: Optional[float]
    cash: Optional[float]
    cash_after: Optional[float]
    mode: Optional[AccountMode] = None
    warnings: List[str] = field(default_factory=list)   # shown, but the user may still confirm
    blocking: List[str] = field(default_factory=list)   # the order cannot be sent until fixed

    @property
    def ok(self) -> bool:
        return not self.blocking

    @property
    def title(self) -> str:
        qty = "?" if self.qty is None else f"{self.qty:g}"
        return f"{self.side.upper()} {qty} {self.symbol} · {self.order_type.capitalize()}"

    def lines(self) -> List[str]:
        """Plain-text summary rows (label, value) joined for dialogs that only take text."""
        rows = [
            ("Price used", money(self.price)),
            ("Est. notional", money(self.notional)),
            ("Fee" + (f" ({self.fee_rate * 100:.2f}%)" if self.fee_rate is not None else ""), money(self.fee)),
        ]
        if self.cash is not None:
            rows.append(("Cash after", money(self.cash_after)))
        if self.mode is not None:
            rows.append(("Account", self.mode.label))
        return [f"{k}: {v}" for k, v in rows]


def money(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    sign = "−" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def _num(value) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def preview_order(side: str, qty, symbol: str, order_type: str = "market", *,
                  last_price=None, order_price=None, fee_rate=None, cash=None, position_qty=None,
                  price_age_s: Optional[float] = None, mode: Optional[AccountMode] = None) -> OrderPreview:
    """Estimate an order before it is sent. Never raises; problems are reported in ``warnings`` / ``blocking``."""
    side = (side or "").lower()
    order_type = (order_type or "market").lower()
    qty_f = _num(qty)
    last = _num(last_price)
    px_in = _num(order_price)
    fee_rate_f = _num(fee_rate)
    cash_f = _num(cash)
    held = _num(position_qty) or 0.0

    blocking: List[str] = []
    warnings: List[str] = []

    if not symbol:
        blocking.append("Load a symbol first.")
    if side not in ("buy", "sell"):
        blocking.append(f"Unknown side {side!r}.")
    if qty_f is None or qty_f <= 0:
        blocking.append("Quantity must be greater than 0.")
    if order_type in ("limit", "stop") and (px_in is None or px_in <= 0):
        blocking.append(f"{order_type.capitalize()} orders need a {order_type} price.")
    if mode is not None and mode.mode in (MODE_BLOCKED, MODE_UNAVAILABLE):
        blocking.append(mode.detail)

    price = px_in if order_type in ("limit", "stop") and px_in else last
    if order_type == "market" and last is None:
        blocking.append(f"No current price for {symbol or 'this symbol'}: load data first.")

    fresh = price_freshness(price_age_s)
    if order_type == "market" and fresh.state == "stale":
        warnings.append(f"The price is {format_age(price_age_s)} old. Reload data so the fill uses a current price.")

    notional = qty_f * price if (qty_f and price) else None
    fee = notional * fee_rate_f if (notional is not None and fee_rate_f is not None) else None
    cash_after = None
    if cash_f is not None and notional is not None:
        cost = notional + (fee or 0.0)
        cash_after = cash_f - cost if side == "buy" else cash_f + notional - (fee or 0.0)
        if side == "buy" and cash_after < 0:
            warnings.append("Costs more than the cash available: the paper account shrinks it to what you can afford."
                            if mode is None or mode.mode == MODE_PAPER
                            else "Costs more than the cash available: the broker may reject it.")
    if side == "sell" and qty_f and qty_f > held + 1e-12:
        if held > 0:
            warnings.append(f"Sells more than you hold ({held:g}): the paper account caps it at your holding.")
        else:
            warnings.append("You hold none of this: the sell opens (or adds to) a short position.")
    if mode is not None and mode.mode == MODE_LIVE:
        warnings.append("LIVE account: this sends a real order with real money.")

    return OrderPreview(side=side, qty=qty_f, symbol=symbol or "", order_type=order_type, price=price,
                        notional=notional, fee_rate=fee_rate_f, fee=fee, cash=cash_f, cash_after=cash_after,
                        mode=mode, warnings=warnings, blocking=blocking)


def position_rows(positions, prices=None) -> List[dict]:
    """Open positions as display rows: symbol, side, qty, avg, last, value, upnl, upnl_pct (None when unknown).

    ``positions`` is a broker's ``{symbol: Position}`` mapping (qty, avg_price, pnl, last_price). Flat ones are skipped.
    ``prices`` (the broker's ``market_data``) fills ``last`` when a position has not been re-marked since it opened.
    """
    rows = []
    for symbol, pos in sorted((positions or {}).items()):
        qty = _num(getattr(pos, "qty", None)) or 0.0
        if qty == 0:
            continue
        avg = _num(getattr(pos, "avg_price", None))
        last = _num(getattr(pos, "last_price", None)) or _num((prices or {}).get(symbol)) or None
        upnl = _num(getattr(pos, "pnl", None))
        cost = abs(qty) * avg if avg else None
        rows.append({
            "symbol": symbol,
            "side": "Long" if qty > 0 else "Short",
            "qty": qty,
            "avg": avg,
            "last": last,
            "value": qty * last if last else None,
            "upnl": upnl,
            "upnl_pct": (upnl / cost * 100.0) if (upnl is not None and cost) else None,
        })
    return rows


def broker_fee_rate(broker, order_type: str) -> Optional[float]:
    """The simulator's fee for this order type (limit_fee for limits, else market_fee); None for unknown connectors."""
    attr = "limit_fee" if (order_type or "").lower() == "limit" else "market_fee"
    return _num(getattr(broker, attr, None))
