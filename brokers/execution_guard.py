"""Shared pre-trade safety for every live broker connector (Phase 11.2).

Every connector that can reach a real exchange decorates its ``submit_order`` with
:func:`guarded_live_order`, so the checks below run *inside the connector* -- a strategy,
the desktop UI, a script or a future agent cannot route around them by calling the
connector directly.

Order of evaluation for each live order
  1. KILL SWITCH -- blocks everything. Trading is blocked unless ``LIVE_TRADING_ENABLED``
     is explicitly true, and also whenever the kill-switch file exists (``touch .kill_switch``
     stops all live trading at once, without restarting the app) or the switch was engaged in code.
  2. Order validity -- positive finite quantity, side buy/sell/long/short.
  3. Pre-trade limits -- max notional per order, max cumulative notional per broker per session,
     max orders per minute. If the order's value cannot be determined the order is blocked
     (fail closed).
  4. DRY RUN -- on by default: the order is logged as "would submit" and NOT sent. Set
     ``LIVE_DRY_RUN=false`` as well as ``LIVE_TRADING_ENABLED=true`` to really trade, so a
     live order needs two deliberate opt-ins.

Configuration (environment variables, read on every order so changes apply immediately)
  LIVE_TRADING_ENABLED     default false   master switch (kill switch is ON unless true)
  LIVE_DRY_RUN             default true    log instead of submit
  KILL_SWITCH_FILE         default <repo>/.kill_switch
  MAX_ORDER_NOTIONAL_USD   default 100
  MAX_SESSION_NOTIONAL_USD default 500     per broker, since the process started
  MAX_ORDERS_PER_MINUTE    default 6
Malformed numbers fall back to these conservative defaults.
"""
from __future__ import annotations

import functools
import inspect
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional

from core.logger import get_logger

logger = get_logger(__name__)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}

DEFAULTS = {
    "MAX_ORDER_NOTIONAL_USD": 100.0,
    "MAX_SESSION_NOTIONAL_USD": 500.0,
    "MAX_ORDERS_PER_MINUTE": 6,
}


class OrderBlockedError(RuntimeError):
    """Raised instead of submitting an order the guard refuses."""


class _DryRunStatus(Enum):
    DRY_RUN = "dry_run"


@dataclass
class DryRunResult:
    """Returned in place of a broker order when dry-run is on. Quacks enough like an order for the UI."""
    broker: str
    symbol: str
    qty: float
    side: str
    order_type: str
    notional: Optional[float]
    would_be_rejected_by: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: f"dryrun_{int(time.time() * 1000)}")
    status: _DryRunStatus = _DryRunStatus.DRY_RUN
    filled_avg_price: float = 0.0
    filled_qty: float = 0.0


@dataclass
class GuardDecision:
    action: str                                  # "allow" | "dry_run" | "block"
    reasons: List[str] = field(default_factory=list)
    notional: Optional[float] = None


def _env_float(env, name: str) -> float:
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return float(DEFAULTS[name])
    try:
        v = float(raw)
        if math.isfinite(v) and v > 0:
            return v
    except ValueError:
        pass
    logger.warning("Invalid %s=%r; using the conservative default %s", name, raw, DEFAULTS[name])
    return float(DEFAULTS[name])


def _default_price_provider(symbol: str) -> Optional[float]:
    """Best-effort latest price (USD) for the notional check; None if unavailable."""
    try:
        from core.data_loader import DataLoader
        clean = symbol.replace("/", "").replace(":USDT", "")
        price = DataLoader().get_latest_price(clean)
        return float(price) if price else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Guard could not fetch a reference price for %s: %s", symbol, exc)
        return None


class ExecutionGuard:
    def __init__(self, env=None, clock: Callable[[], float] = time.monotonic,
                 price_provider: Optional[Callable[[str], Optional[float]]] = None):
        self._env = env if env is not None else os.environ
        self._clock = clock
        self._price_provider = price_provider or _default_price_provider
        self._lock = threading.Lock()
        self._engaged_reason: Optional[str] = None
        self._session_notional: Dict[str, float] = {}
        self._recent: Dict[str, Deque[float]] = {}
        self.audit: Deque[dict] = deque(maxlen=200)

    # ------------------------------------------------------------------ switches

    def live_trading_enabled(self) -> bool:
        return str(self._env.get("LIVE_TRADING_ENABLED", "")).strip().lower() in _TRUE

    def dry_run(self) -> bool:
        return str(self._env.get("LIVE_DRY_RUN", "true")).strip().lower() not in _FALSE

    def kill_switch_file(self) -> str:
        return self._env.get("KILL_SWITCH_FILE") or os.path.join(_REPO_ROOT, ".kill_switch")

    def engage_kill_switch(self, reason: str = "engaged in code") -> None:
        self._engaged_reason = reason
        logger.error("KILL SWITCH ENGAGED: %s", reason)

    def release_kill_switch(self) -> None:
        self._engaged_reason = None

    def kill_switch_reason(self) -> Optional[str]:
        if self._engaged_reason:
            return f"kill switch engaged ({self._engaged_reason})"
        if os.path.exists(self.kill_switch_file()):
            return f"kill-switch file present ({self.kill_switch_file()})"
        if not self.live_trading_enabled():
            return "live trading is disabled (set LIVE_TRADING_ENABLED=true to allow it)"
        return None

    def describe(self) -> str:
        reason = self.kill_switch_reason()
        if reason:
            return f"LIVE TRADING BLOCKED: {reason}"
        return ("LIVE TRADING ENABLED in DRY-RUN mode (orders are logged, not sent)" if self.dry_run()
                else "LIVE TRADING ENABLED -- REAL ORDERS WILL BE SENT")

    # ------------------------------------------------------------------ evaluation

    def evaluate(self, broker: str, symbol: str, qty: Any, side: Any, order_type: str = "market",
                 price: Optional[float] = None) -> GuardDecision:
        reason = self.kill_switch_reason()
        if reason:
            return self._record(broker, symbol, qty, side, None, GuardDecision("block", [reason]))

        problems: List[str] = []
        try:
            q = float(qty)
            if not math.isfinite(q) or q <= 0:
                problems.append(f"quantity must be a positive number (got {qty!r})")
        except (TypeError, ValueError):
            q = float("nan")
            problems.append(f"quantity must be a positive number (got {qty!r})")
        if str(side).lower() not in ("buy", "sell", "long", "short"):
            problems.append(f"side must be buy/sell/long/short (got {side!r})")

        notional = None
        if not problems:
            px = price if price else self._price_provider(symbol)
            if px and math.isfinite(float(px)) and float(px) > 0:
                notional = q * float(px)
            else:
                problems.append("cannot determine the order's value (no price) -- refusing to send an unverifiable order")

        if notional is not None:
            max_order = _env_float(self._env, "MAX_ORDER_NOTIONAL_USD")
            if notional > max_order:
                problems.append(f"order value ${notional:,.2f} exceeds MAX_ORDER_NOTIONAL_USD ${max_order:,.2f}")
            max_session = _env_float(self._env, "MAX_SESSION_NOTIONAL_USD")
            with self._lock:
                so_far = self._session_notional.get(broker, 0.0)
            if so_far + notional > max_session:
                problems.append(f"session total would be ${so_far + notional:,.2f}, over "
                                f"MAX_SESSION_NOTIONAL_USD ${max_session:,.2f}")
        max_rate = int(_env_float(self._env, "MAX_ORDERS_PER_MINUTE"))
        now = self._clock()
        with self._lock:
            recent = self._recent.setdefault(broker, deque())
            while recent and now - recent[0] > 60:
                recent.popleft()
            if len(recent) >= max_rate:
                problems.append(f"rate limit: {len(recent)} orders in the last minute (MAX_ORDERS_PER_MINUTE={max_rate})")

        if problems and not self.dry_run():
            return self._record(broker, symbol, qty, side, notional, GuardDecision("block", problems, notional))
        if self.dry_run():
            return self._record(broker, symbol, qty, side, notional, GuardDecision("dry_run", problems, notional))
        return self._record(broker, symbol, qty, side, notional, GuardDecision("allow", [], notional))

    def note_submitted(self, broker: str, notional: Optional[float]) -> None:
        with self._lock:
            self._recent.setdefault(broker, deque()).append(self._clock())
            if notional:
                self._session_notional[broker] = self._session_notional.get(broker, 0.0) + notional

    def _record(self, broker, symbol, qty, side, notional, decision: GuardDecision) -> GuardDecision:
        self.audit.append({"time": time.time(), "broker": broker, "symbol": symbol, "qty": qty, "side": str(side),
                           "notional": notional, "action": decision.action, "reasons": list(decision.reasons)})
        return decision


_guard: Optional[ExecutionGuard] = None
_guard_lock = threading.Lock()


def get_guard() -> ExecutionGuard:
    global _guard
    with _guard_lock:
        if _guard is None:
            _guard = ExecutionGuard()
        return _guard


def set_guard(guard: Optional[ExecutionGuard]) -> None:
    """Replace the process-wide guard (tests)."""
    global _guard
    with _guard_lock:
        _guard = guard


def guarded_live_order(broker_name: str):
    """Decorate a connector's ``submit_order`` so it cannot reach the exchange without passing the guard.

    Connectors with ``paper_mode`` true never touch an exchange (local ledger or a paper
    account) and are passed straight through.
    """
    def decorator(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            if getattr(self, "paper_mode", False):
                return fn(self, *args, **kwargs)
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            a = bound.arguments
            symbol, qty, side = a.get("symbol"), a.get("qty"), a.get("side")
            order_type = str(a.get("order_type", "market"))
            guard = get_guard()
            decision = guard.evaluate(broker_name, symbol, qty, side, order_type, price=a.get("price"))
            if decision.action == "block":
                msg = f"{broker_name} order BLOCKED: " + "; ".join(decision.reasons)
                logger.error("%s | %s %s %s", msg, side, qty, symbol)
                raise OrderBlockedError(msg)
            if decision.action == "dry_run":
                logger.warning("DRY RUN (not sent): %s %s %s %s on %s%s", side, qty, symbol, order_type, broker_name,
                               f" -- would be REJECTED: {'; '.join(decision.reasons)}" if decision.reasons else "")
                return DryRunResult(broker_name, symbol, float(qty) if isinstance(qty, (int, float)) else 0.0,
                                    str(side), order_type, decision.notional, list(decision.reasons))
            result = fn(self, *args, **kwargs)
            guard.note_submitted(broker_name, decision.notional)
            return result

        wrapper.__guarded_live_order__ = True
        return wrapper
    return decorator
