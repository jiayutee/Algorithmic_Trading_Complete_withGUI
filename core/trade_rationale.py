"""Structured "why" records for trades (Phase 11.1).

A rationale is a plain JSON-serialisable dict attached to every order/signal:

    {
      "schema": 1,
      "source": "strategy" | "manual" | "unspecified",
      "action": "open_long" | "close_long" | "open_short" | "close_short" | "buy" | "sell",
      "strategy": "MACD_RSI" | None,
      "signal": "rsi_oversold_macd_bullish" | None,
      "summary": "RSI 24.1 < 30 (oversold) and MACD ... ",   # one human line
      "features": {"rsi": 24.1, ...},          # values at decision time
      "thresholds": {"rsi_oversold": 30, ...}, # rule parameters that were compared
      "confidence": None,                      # model probability, when a model decided
      "feature_importance": None,              # {feature: weight}, filled by ML strategies
    }

`confidence` / `feature_importance` are deliberately part of the schema now so
the Phase 6 ML strategies can populate real values without a format change.
"""
from __future__ import annotations

import inspect
from typing import Any, Optional

from core.logger import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = 1


def _clean(value: Any) -> Any:
    """Coerce numpy / backtrader scalars to plain JSON-safe Python values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f != f or f in (float("inf"), float("-inf")):  # NaN / inf are not valid JSON
        return None
    return round(f, 6)


def build_rationale(
    *,
    source: str,
    action: str,
    summary: str,
    strategy: Optional[str] = None,
    signal: Optional[str] = None,
    features: Optional[dict] = None,
    thresholds: Optional[dict] = None,
    confidence: Optional[float] = None,
    feature_importance: Optional[dict] = None,
) -> dict:
    return {
        "schema": SCHEMA_VERSION,
        "source": source,
        "action": action,
        "strategy": strategy,
        "signal": signal,
        "summary": summary,
        "features": _clean(features or {}),
        "thresholds": _clean(thresholds or {}),
        "confidence": _clean(confidence),
        "feature_importance": _clean(feature_importance),
    }


def manual_rationale(side: str, symbol: str, order_type: str = "market",
                     price: Optional[float] = None, origin: str = "Order Entry panel") -> dict:
    side = str(side).lower()
    where = f" @ ${price:,.2f}" if price else ""
    return build_rationale(
        source="manual",
        action=side,
        summary=f"Manual {side.upper()} {symbol} ({order_type}){where} -- placed by the user from the {origin}",
        features={"price_at_decision": price} if price else None,
    )


def unspecified_rationale() -> dict:
    return build_rationale(
        source="unspecified",
        action="unknown",
        summary="No rationale was supplied by the caller",
    )


def format_rationale(rationale: Optional[dict]) -> str:
    """One-line summary for table cells / marker hovers."""
    if not rationale:
        return "—"
    return str(rationale.get("summary") or "—")


def format_rationale_detail(rationale: Optional[dict]) -> str:
    """Multi-line detail for tooltips: summary + decision-time values."""
    if not rationale:
        return "No rationale recorded"
    lines = [str(rationale.get("summary") or "—")]
    strategy, signal = rationale.get("strategy"), rationale.get("signal")
    if strategy or signal:
        lines.append(f"Strategy: {strategy or '—'}  |  Signal: {signal or '—'}")
    if rationale.get("features"):
        lines.append("Values: " + ", ".join(f"{k}={v}" for k, v in rationale["features"].items()))
    if rationale.get("thresholds"):
        lines.append("Rules: " + ", ".join(f"{k}={v}" for k, v in rationale["thresholds"].items()))
    if rationale.get("confidence") is not None:
        lines.append(f"Model confidence: {rationale['confidence']:.0%}")
    if rationale.get("feature_importance"):
        top = sorted(rationale["feature_importance"].items(), key=lambda kv: -abs(kv[1]))[:5]
        lines.append("Top drivers: " + ", ".join(f"{k} ({v:+.2f})" for k, v in top))
    return "\n".join(lines)


class RationaleMixin:
    """Mixin for backtrader strategies: capture the reason at decision time
    (in ``next()``), attach it to the signal record when the order completes
    (in ``notify_order()``).

    The order fills on a later bar than the one that decided it, so the reason
    must be stashed at decision time rather than recomputed at fill time.
    """

    _pending_rationale: Optional[dict] = None

    def _set_rationale(self, **kwargs) -> None:
        self._pending_rationale = build_rationale(source="strategy", **kwargs)

    def _attach_rationale_to_last_signal(self) -> None:
        signals = getattr(self, "signals", None)
        if signals:
            signals[-1]["rationale"] = self._pending_rationale or unspecified_rationale()
        self._pending_rationale = None


def submit_with_rationale(broker: Any, rationale: dict, **order_kwargs):
    """Submit an order, attaching *rationale* when the broker supports it.

    Only ``SimulatedBroker.submit_order`` accepts a ``rationale`` argument; the
    live connectors (Binance/KuCoin/MEXC/Alpaca/IBKR) have their own fixed
    signatures and would raise ``TypeError`` if handed an unknown keyword. For
    those, the rationale is written to the application log so there is still
    an audit trail of why the order was sent.
    """
    try:
        supports = "rationale" in inspect.signature(broker.submit_order).parameters
    except (TypeError, ValueError):
        supports = False
    if supports:
        return broker.submit_order(rationale=rationale, **order_kwargs)
    logger.info(
        "Order rationale (%s not rationale-aware, logged only): %s | %s",
        type(broker).__name__, order_kwargs, rationale.get("summary"),
    )
    return broker.submit_order(**order_kwargs)
