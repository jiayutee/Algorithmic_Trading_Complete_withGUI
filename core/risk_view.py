"""
core/risk_view.py

Read-only risk dashboard rows for both front ends: where the paper account stands against the paper execution
service's risk limits (``core/execution/risk.py``).

* Limits: the ones the runner last published (journal status ``risk``), else ``RiskConfig`` defaults.
* Day start / peak equity and today's runner orders: from the execution journal, exactly as the runner records them.
  Just looking never creates the journal database (same rule as ``core/execution/view.py``).
* Exposure: from the broker you pass (positions marked at the broker's last price, else the average price).

These limits gate the RUNNER's entries only; manual orders are not gated by them (the panel says so).
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

WARN_AT = 0.75     # a row turns amber at 75% of its limit


def _state(used: Optional[float]) -> str:
    if used is None:
        return "unknown"
    if used >= 1.0:
        return "breach"
    return "warn" if used >= WARN_AT else "ok"


def _pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v * 100:+.2f}%" if v < 0 else f"{v * 100:.1f}%"


def _journal_facts(path: Optional[str], now: float) -> Dict:
    """Runner state from the journal; {} if it does not exist. Never raises, never creates the file."""
    try:
        from core.execution.journal import ExecutionJournal, default_journal_path
        path = path or default_journal_path()
        if not os.path.exists(path):
            return {}
        j = ExecutionJournal(path)
        try:
            day = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d")
            day_ts = datetime.fromtimestamp(now, tz=timezone.utc).replace(hour=0, minute=0, second=0,
                                                                           microsecond=0).timestamp()
            base = j.get("day_baseline") or {}
            status = j.get("status") or {}
            return {
                "day_start": float(base["equity"]) if base.get("day") == day and base.get("equity") else None,
                "peak": float(j.get("peak_equity")) if j.get("peak_equity") else None,
                "orders_today": j.submitted_today(day_ts),
                "halted": j.halted(),
                "limits": status.get("risk") or {},
            }
        finally:
            j.close()
    except Exception as exc:  # noqa: BLE001 -- a status panel must never break a UI
        return {"error": str(exc)}


def risk_snapshot(broker, journal_path: Optional[str] = None, now: Optional[float] = None) -> Dict:
    """{equity, rows:[{label, value_text, limit_text, used, state}], halted, limits_source, note}. Never raises."""
    from core.execution.risk import RiskConfig
    now = time.time() if now is None else now
    facts = _journal_facts(journal_path, now)
    defaults = vars(RiskConfig())
    limits = {k: facts.get("limits", {}).get(k, defaults[k]) for k in
              ("max_position_pct", "max_gross_pct", "max_daily_loss_pct", "max_drawdown_pct", "max_orders_per_day")}
    limits_source = "paper runner settings" if facts.get("limits") else "defaults (runner has not published limits)"

    equity = None
    exposures: Dict[str, float] = {}
    if broker is not None:
        try:
            equity = float(broker.get_account_info().get("portfolio_value"))
            prices = getattr(broker, "market_data", {}) or {}
            for sym, pos in (getattr(broker, "positions", {}) or {}).items():
                qty = float(getattr(pos, "qty", 0) or 0)
                if qty:
                    px = prices.get(sym) or getattr(pos, "last_price", 0) or getattr(pos, "avg_price", 0) or 0
                    exposures[sym] = abs(qty) * float(px)
        except Exception as exc:  # noqa: BLE001
            return {"equity": None, "rows": [], "halted": facts.get("halted"), "limits_source": limits_source,
                    "note": f"Account unavailable: {exc}"}

    rows: List[Dict] = []

    def row(label, value, limit, used, value_text=None, limit_text=None):
        rows.append({"label": label, "value": value, "used": used, "state": _state(used),
                     "value_text": value_text if value_text is not None else _pct(value),
                     "limit_text": limit_text if limit_text is not None else _pct(limit)})

    if equity and equity > 0:
        gross = sum(exposures.values()) / equity
        row("Gross exposure", gross, limits["max_gross_pct"], gross / limits["max_gross_pct"])
        if exposures:
            sym, top = max(exposures.items(), key=lambda kv: kv[1])
            frac = top / equity
            row(f"Largest position ({sym})", frac, limits["max_position_pct"], frac / limits["max_position_pct"])
        else:
            row("Largest position", 0.0, limits["max_position_pct"], 0.0)
        day_start = facts.get("day_start")
        day = (equity / day_start - 1) if day_start else None
        row("Day P&L vs start", day, -limits["max_daily_loss_pct"],
            None if day is None else max(0.0, -day) / limits["max_daily_loss_pct"],
            limit_text=f"−{limits['max_daily_loss_pct'] * 100:.1f}%")
        peak = facts.get("peak")
        dd = (equity / max(peak, equity) - 1) if peak else None
        row("Drawdown from peak", dd, -limits["max_drawdown_pct"],
            None if dd is None else max(0.0, -dd) / limits["max_drawdown_pct"],
            limit_text=f"−{limits['max_drawdown_pct'] * 100:.1f}%")
    n = facts.get("orders_today")
    row("Runner orders today", n, limits["max_orders_per_day"],
        None if n is None else n / limits["max_orders_per_day"],
        value_text="n/a" if n is None else str(n), limit_text=str(limits["max_orders_per_day"]))

    note = "Limits gate the paper runner's entries; manual orders are not gated by them."
    if "error" in facts:
        note += f" Journal unreadable: {facts['error']}"
    elif not facts:
        note += " The runner has never run here, so day start, peak and order count are n/a."
    return {"equity": equity, "rows": rows, "halted": facts.get("halted"), "limits_source": limits_source, "note": note}
