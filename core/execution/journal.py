"""Durable execution journal (SQLite): every decision, the service heartbeat, a single-runner lease and the halt flag.

Why durable: an execution loop that forgets what it did on restart will do it again. The journal makes decisions
idempotent (one row per decision id, enforced by a UNIQUE constraint) so a restart, a second process, or a re-run of the
same bar can never submit the same order twice, and it lets the desktop app and Dash show the SAME status.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

ENV = "EXECUTION_DB_PATH"


def default_journal_path() -> str:
    env = os.environ.get(ENV)
    if env:
        return env
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "training_ground", "paper", "execution.sqlite3")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions(
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  decision_id TEXT UNIQUE NOT NULL,
  ts REAL NOT NULL, symbol TEXT NOT NULL, bar_ts TEXT, strategy TEXT,
  signal INTEGER, target_qty REAL, current_qty REAL, order_qty REAL, side TEXT, price REAL,
  action TEXT NOT NULL,            -- hold | submitted | blocked | error
  status TEXT,                     -- order status from the broker (filled/rejected/pending/...)
  order_id TEXT, filled_qty REAL, fill_price REAL,
  risk TEXT, rationale TEXT, error TEXT);
CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class ExecutionJournal:
    def __init__(self, path: Optional[str] = None):
        self.path = path or default_journal_path()
        if os.path.dirname(self.path):
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=15, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._lock = threading.RLock()
        self.owner_id = uuid.uuid4().hex[:12]

    # ------------------------------------------------------------------ key/value (status, halt, day baseline, peak)
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            r = self._conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return default if r is None else json.loads(r[0])

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute("INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                               (key, json.dumps(value)))

    # ------------------------------------------------------------------ halt flag (persistent: survives restarts)
    def halt(self, reason: str) -> None:
        self.set("halt", {"reason": reason, "at": time.time()})

    def resume(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM kv WHERE key='halt'")

    def halted(self) -> Optional[Dict]:
        return self.get("halt")

    # ------------------------------------------------------------------ single-runner lease
    def acquire_lease(self, ttl_s: float, now: Optional[float] = None) -> bool:
        """Become THE runner unless another owner's lease is still fresh. Atomic (BEGIN IMMEDIATE)."""
        now = time.time() if now is None else now
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                r = self._conn.execute("SELECT value FROM kv WHERE key='lease'").fetchone()
                cur = json.loads(r[0]) if r else None
                if cur and cur["owner"] != self.owner_id and now - cur["renewed"] < cur["ttl"]:
                    self._conn.execute("ROLLBACK")
                    return False
                self._conn.execute("INSERT INTO kv(key,value) VALUES('lease',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                                   (json.dumps({"owner": self.owner_id, "renewed": now, "ttl": ttl_s}),))
                self._conn.execute("COMMIT")
                return True
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def release_lease(self) -> None:
        with self._lock:
            cur = self.get("lease")
            if cur and cur["owner"] == self.owner_id:
                self._conn.execute("DELETE FROM kv WHERE key='lease'")

    def lease_holder(self, now: Optional[float] = None) -> Optional[Dict]:
        cur = self.get("lease")
        now = time.time() if now is None else now
        if cur and now - cur["renewed"] < cur["ttl"]:
            return cur
        return None

    # ------------------------------------------------------------------ decisions
    def has_decision(self, decision_id: str) -> bool:
        with self._lock:
            return self._conn.execute("SELECT 1 FROM decisions WHERE decision_id=?", (decision_id,)).fetchone() is not None

    def record(self, decision_id: str, **fields) -> bool:
        """Insert a decision. Returns False (and writes nothing) if this decision id already exists -- the idempotency guard."""
        cols = ["decision_id", "ts"] + list(fields)
        vals = [decision_id, time.time()] + [json.dumps(v) if isinstance(v, (dict, list)) else v for v in fields.values()]
        with self._lock:
            try:
                self._conn.execute(f"INSERT INTO decisions({','.join(cols)}) VALUES({','.join('?' * len(cols))})", vals)
                return True
            except sqlite3.IntegrityError:
                return False

    def update(self, decision_id: str, **fields) -> None:
        sets = ",".join(f"{k}=?" for k in fields)
        vals = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in fields.values()]
        with self._lock:
            self._conn.execute(f"UPDATE decisions SET {sets} WHERE decision_id=?", vals + [decision_id])

    def decisions(self, limit: int = 100, symbol: Optional[str] = None) -> List[Dict]:
        q, args = "SELECT * FROM decisions", []
        if symbol:
            q += " WHERE symbol=?"; args.append(symbol)
        q += " ORDER BY seq DESC LIMIT ?"; args.append(limit)
        with self._lock:
            cur = self._conn.execute(q, args)
            cols = [c[0] for c in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        for r in rows:
            for k in ("risk", "rationale"):
                if r.get(k):
                    try:
                        r[k] = json.loads(r[k])
                    except ValueError:
                        pass
        return rows

    def submitted_today(self, day_start_ts: float) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM decisions WHERE action='submitted' AND ts>=?", (day_start_ts,)).fetchone()[0]

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass
