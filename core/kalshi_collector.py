"""Daily Kalshi snapshot collector (read-only) -- builds the multi-week dataset a calibration study needs.

Phase 9.3a could only use the most recent ~9 hours of settled markets. This stores what the live book looked like
*before* markets settle, then records outcomes once they do, so calibration at a fixed lead time can be studied
without depending on the API's short settled-history window.

    python -m core.kalshi_collector collect     # snapshot open, liquid, two-sided markets closing soon
    python -m core.kalshi_collector resolve     # fetch results for snapshotted markets that have settled
    python -m core.kalshi_collector status

Storage: SQLite, default ``training_ground/datasets/kalshi_snapshots.sqlite3`` (env ``KALSHI_DB_PATH``).
No credentials, no orders.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from contextlib import closing
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from core.kalshi_data import KalshiClient, KalshiError, Market
from core.logger import logger

DEFAULT_DB = os.path.join("training_ground", "datasets", "kalshi_snapshots.sqlite3")
MIN_VOLUME = 100.0
MIN_BID, MAX_SPREAD = 0.01, 0.10        # same real-two-sided-book rule as Phase 9.3a (Amendment 2)
HORIZON_H = 72                           # only markets closing within this window (cheap, and near-settlement is the interesting part)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots(
  ts TEXT NOT NULL, ticker TEXT NOT NULL, event_ticker TEXT, close_time TEXT,
  yes_bid REAL, yes_ask REAL, last_price REAL, volume REAL, open_interest REAL,
  PRIMARY KEY(ts, ticker));
CREATE TABLE IF NOT EXISTS outcomes(ticker TEXT PRIMARY KEY, result INTEGER, resolved_at TEXT);
CREATE INDEX IF NOT EXISTS ix_snap_ticker ON snapshots(ticker);
"""


def _db_path(path: Optional[str]) -> str:
    return path or os.environ.get("KALSHI_DB_PATH") or DEFAULT_DB


def _connect(path: Optional[str]) -> sqlite3.Connection:
    p = _db_path(path)
    if os.path.dirname(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
    con = sqlite3.connect(p)
    con.executescript(_SCHEMA)
    return con


def _worth_snapshotting(m: Market, now: datetime, horizon_h: float) -> bool:
    if m.volume < MIN_VOLUME or m.yes_bid is None or m.yes_ask is None:
        return False
    if m.yes_bid < MIN_BID or m.yes_ask - m.yes_bid > MAX_SPREAD or m.yes_ask >= 1:
        return False
    try:
        close = datetime.fromisoformat((m.close_time or "").replace("Z", "+00:00"))
    except ValueError:
        return False
    return now <= close <= now + timedelta(hours=horizon_h)


def collect(client: KalshiClient, db_path: Optional[str] = None, *, max_markets: int = 3000,
            horizon_h: float = HORIZON_H, now: Optional[datetime] = None) -> Dict[str, int]:
    now = now or datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    scanned = kept = 0
    with closing(_connect(db_path)) as con:
        for m in client.iter_markets(status="open", max_items=max_markets, page_size=1000):
            scanned += 1
            if not _worth_snapshotting(m, now, horizon_h):
                continue
            con.execute("INSERT OR IGNORE INTO snapshots VALUES (?,?,?,?,?,?,?,?,?)",
                        (ts, m.ticker, m.event_ticker, m.close_time, m.yes_bid, m.yes_ask, m.last_price, m.volume, m.open_interest))
            kept += 1
        con.commit()
    logger.info("Kalshi collect: scanned %d open markets, stored %d snapshots at %s", scanned, kept, ts)
    return {"scanned": scanned, "stored": kept}


def resolve(client: KalshiClient, db_path: Optional[str] = None, *, max_lookups: int = 500,
            now: Optional[datetime] = None) -> Dict[str, int]:
    """For snapshotted markets already past close and without a stored outcome, ask the API for the result."""
    now = now or datetime.now(timezone.utc)
    cutoff = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    resolved = pending = errors = 0
    with closing(_connect(db_path)) as con:
        rows = con.execute("""SELECT DISTINCT s.ticker FROM snapshots s LEFT JOIN outcomes o ON o.ticker = s.ticker
                              WHERE o.ticker IS NULL AND s.close_time <= ? LIMIT ?""", (cutoff, max_lookups)).fetchall()
        for (ticker,) in rows:
            try:
                m = client.get_market(ticker)
            except KalshiError as exc:
                errors += 1
                logger.warning("Kalshi resolve: %s (%s)", ticker, exc)
                continue
            if m.result is None:
                pending += 1                       # closed but not settled yet: try again next run
                continue
            con.execute("INSERT OR REPLACE INTO outcomes VALUES (?,?,?)", (ticker, int(m.result), cutoff))
            resolved += 1
        con.commit()
    return {"resolved": resolved, "pending": pending, "errors": errors}


def status(db_path: Optional[str] = None) -> Dict[str, object]:
    with closing(_connect(db_path)) as con:
        n, mk, t0, t1 = con.execute("SELECT COUNT(*), COUNT(DISTINCT ticker), MIN(ts), MAX(ts) FROM snapshots").fetchone()
        days = con.execute("SELECT COUNT(DISTINCT substr(ts,1,10)) FROM snapshots").fetchone()[0]
        res = con.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
        labelled = con.execute("SELECT COUNT(*) FROM snapshots s JOIN outcomes o ON o.ticker = s.ticker").fetchone()[0]
    return {"snapshots": n, "markets": mk, "first": t0, "last": t1, "days": days, "resolved_markets": res, "labelled_snapshots": labelled}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["collect", "resolve", "status"])
    ap.add_argument("--db", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "status":
        for k, v in status(args.db).items():
            print(f"{k:>18}: {v}")
        return 0
    client = KalshiClient()
    out = collect(client, args.db) if args.cmd == "collect" else resolve(client, args.db)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
