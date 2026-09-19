"""Read-only view of the execution service for the UIs (desktop tab and Dash tab render the SAME data).

Reads only the journal, so it works from any process: a UI can show a service that is running in another process, and it
reports a stale heartbeat as "not running" rather than trusting the last thing the service wrote.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, Optional

import os

from core.execution.journal import ExecutionJournal, default_journal_path


def _fmt_ts(ts: Optional[float]) -> str:
    return datetime.fromtimestamp(ts).strftime("%m-%d %H:%M:%S") if ts else "—"


def _detail(d: Dict) -> str:
    if d["action"] == "blocked":
        risk = d.get("risk") or {}
        return "BLOCKED: " + "; ".join(risk.get("reasons", [])) if isinstance(risk, dict) else "BLOCKED"
    if d["action"] == "error":
        return f"ERROR: {d.get('error') or ''}"
    if d["action"] in ("submitted", "submitting"):
        base = f"{d.get('side', '')} {d.get('order_qty') or 0:g} @ {d.get('price') or 0:g} -> {d.get('status') or 'sent'}"
        return base + (f" ({d['error']})" if d.get("error") else "")
    rat = d.get("rationale")
    return (rat.get("summary") if isinstance(rat, dict) else "") or "no order needed"


def execution_view(journal: Optional[ExecutionJournal] = None, now: Optional[float] = None, limit: int = 40) -> dict:
    """{headline, running, halted, symbols:[...], decisions:[...], issues:[...], account:{...}}. Never raises."""
    now = time.time() if now is None else now
    try:
        if journal is None and not os.path.exists(default_journal_path()):     # just looking must not create a database
            return {"headline": "Paper execution STOPPED | never run on this machine", "running": False, "halted": None,
                    "symbols": [], "decisions": [], "issues": [], "account": {}, "risk": {}}
        j = journal or ExecutionJournal()
        st = j.get("status") or {}
        holder = j.lease_holder(now)
        hb_age = now - st["heartbeat"] if st.get("heartbeat") else None
        fresh = hb_age is not None and hb_age < max(3 * st.get("poll_seconds", 30), 60)
        running = bool(st.get("running") and fresh and holder)
        halted = j.halted()
        issues = list((st.get("reconciliation") or {}).get("issues", []))
        if st.get("last_error"):
            issues.append(f"last error: {st['last_error']}")
        if st.get("running") and not fresh:
            issues.append(f"heartbeat is {hb_age:.0f}s old -- the service stopped without saying so")
        state = "RUNNING" if running else "STOPPED"
        headline = (f"Paper execution {state} | {st.get('strategy', '—')} {st.get('interval', '')} on "
                    f"{', '.join(st.get('symbols', [])) or '—'} | last heartbeat {_fmt_ts(st.get('heartbeat'))}"
                    + (f" | HALTED: {halted['reason']}" if halted else ""))
        symbols = [{"symbol": s, **{k: (str(v) if not isinstance(v, (int, float, str)) else v) for k, v in info.items()}}
                   for s, info in (st.get("per_symbol") or {}).items()]
        decisions = [{"when": _fmt_ts(d["ts"]), "symbol": d["symbol"], "action": d["action"], "status": d.get("status") or "",
                      "detail": _detail(d), "id": d["decision_id"]} for d in j.decisions(limit=limit)]
        return {"headline": headline, "running": running, "halted": halted, "symbols": symbols, "decisions": decisions,
                "issues": issues, "account": st.get("account", {}), "risk": st.get("risk", {})}
    except Exception as exc:  # noqa: BLE001 -- a status view must never break a UI
        return {"headline": f"Execution status unavailable: {exc}", "running": False, "halted": None, "symbols": [],
                "decisions": [], "issues": [str(exc)], "account": {}, "risk": {}}
