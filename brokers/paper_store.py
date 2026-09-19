"""Durable storage for the paper (simulated) account: balance, positions and every order, in one SQLite file.

Why: the simulator used to keep everything in memory, so a restart silently reset the account. With a store the account
survives restarts, and the desktop app and the Dash view can share ONE account: each process notices commits made by the
other (``PRAGMA data_version`` changes only when a *different* connection commits) and reloads before it acts.

Writes happen inside ``BEGIN IMMEDIATE`` transactions so two processes cannot interleave a read-modify-write.
Prices are deliberately NOT stored: a stale saved price would be exactly the kind of fake mark this replaces.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Dict, List, Optional

DEFAULT_ENV = "PAPER_ACCOUNT_PATH"


def default_account_path() -> str:
    """``$PAPER_ACCOUNT_PATH`` or ``training_ground/paper/paper_account.sqlite3`` under the repo (gitignored)."""
    env = os.environ.get(DEFAULT_ENV)
    if env:
        return env
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "training_ground", "paper", "paper_account.sqlite3")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS positions(symbol TEXT PRIMARY KEY, qty REAL NOT NULL, avg_price REAL NOT NULL, leverage REAL NOT NULL);
CREATE TABLE IF NOT EXISTS orders(seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL, json TEXT NOT NULL);
"""


class PaperStore:
    def __init__(self, path: str):
        self.path = path
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self._conn = sqlite3.connect(path, timeout=15, check_same_thread=False, isolation_level=None)  # autocommit; we BEGIN ourselves
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._seen_version = self._data_version()

    # ------------------------------------------------------------------ change detection
    def _data_version(self) -> int:
        return int(self._conn.execute("PRAGMA data_version").fetchone()[0])

    def changed_externally(self) -> bool:
        """True (once) if another connection committed since we last looked."""
        v = self._data_version()
        if v != self._seen_version:
            self._seen_version = v
            return True
        return False

    # ------------------------------------------------------------------ transactions
    @contextmanager
    def transaction(self):
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        finally:
            self._seen_version = self._data_version()      # our own commit must not look like someone else's

    # ------------------------------------------------------------------ read / write
    def has_state(self) -> bool:
        return self._conn.execute("SELECT 1 FROM meta WHERE key='balance'").fetchone() is not None

    def load(self) -> Dict:
        meta = {k: json.loads(v) for k, v in self._conn.execute("SELECT key, value FROM meta")}
        positions = [dict(zip(("symbol", "qty", "avg_price", "leverage"), r))
                     for r in self._conn.execute("SELECT symbol, qty, avg_price, leverage FROM positions")]
        orders = [json.loads(r[0]) for r in self._conn.execute("SELECT json FROM orders ORDER BY seq")]
        return {"meta": meta, "positions": positions, "orders": orders}

    def save(self, meta: Dict, positions: List[Dict], orders: List[Dict]) -> None:
        """Call inside ``transaction()``. ``orders`` are the created/changed ones only (upsert by id)."""
        for k, v in meta.items():
            self._conn.execute("INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                               (k, json.dumps(v)))
        self._conn.execute("DELETE FROM positions")
        for p in positions:
            self._conn.execute("INSERT INTO positions(symbol, qty, avg_price, leverage) VALUES(?,?,?,?)",
                               (p["symbol"], p["qty"], p["avg_price"], p["leverage"]))
        for o in orders:
            self._conn.execute("INSERT INTO orders(id, json) VALUES(?, ?) ON CONFLICT(id) DO UPDATE SET json=excluded.json",
                               (o["id"], json.dumps(o)))

    def wipe(self) -> None:
        """Call inside ``transaction()``."""
        for t in ("meta", "positions", "orders"):
            self._conn.execute(f"DELETE FROM {t}")

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass
