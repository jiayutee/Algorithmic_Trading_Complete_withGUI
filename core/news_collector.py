"""Accumulate news history so news-sentiment hypotheses become testable (Phase 6.5 / H3).

Historical news cannot be back-filled from the free sources, so the only way to get a sentiment
history is to collect it going forward. ``collect`` runs the (time-budgeted, cached, filtered)
news pipeline for each symbol and stores what it finds; ``status`` reports how close each symbol
is to the pre-registered threshold for testing H3: at least ``TARGET_DAYS`` distinct days that each
have at least ``MIN_ITEMS_PER_DAY`` headlines (docs/PHASE_6_5_PREREGISTRATION.md).

    python -m core.news_collector collect     # fetch + store now (safe to run as often as you like)
    python -m core.news_collector status      # coverage vs the H3 threshold

Run ``collect`` once a day (cron / scheduled task) and H3 becomes testable in about ``TARGET_DAYS`` days.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

from core.logger import logger
from core.news_store import DEFAULT_DB

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "LTCUSDT", "DOGEUSDT", "SOLUSDT"]
TARGET_DAYS = 300
MIN_ITEMS_PER_DAY = 3


def coverage(symbols: List[str] = SYMBOLS, db_path: Optional[str] = None,
             target_days: int = TARGET_DAYS, min_items: int = MIN_ITEMS_PER_DAY, today: Optional[date] = None) -> Dict[str, dict]:
    """Per-symbol news coverage read straight from the store (items tagged with the symbol)."""
    path = db_path or DEFAULT_DB
    out: Dict[str, dict] = {}
    if not os.path.exists(path):
        return {s: {"items": 0, "days": 0, "qualifying_days": 0, "first": None, "last": None, "pct_of_target": 0.0,
                    "days_per_week_rate": 0.0, "eta_days": None} for s in symbols}
    conn = sqlite3.connect(path)
    try:
        for sym in symbols:
            rows = conn.execute(
                "SELECT substr(datetime_utc, 1, 10) AS d, COUNT(*) FROM news WHERE tickers LIKE ? GROUP BY d ORDER BY d",
                (f'%"{sym}"%',)).fetchall()
            days = [(d, n) for d, n in rows if d]
            qualifying = [d for d, n in days if n >= min_items]
            first, last = (days[0][0], days[-1][0]) if days else (None, None)
            rate, eta = 0.0, None
            if qualifying:
                span = (date.fromisoformat(qualifying[-1]) - date.fromisoformat(qualifying[0])).days + 1
                rate = len(qualifying) / max(1, span) * 7
                remaining = max(0, target_days - len(qualifying))
                eta = 0 if remaining == 0 else (round(remaining / (len(qualifying) / span)) if len(qualifying) / span > 0 else None)
            out[sym] = {"items": sum(n for _, n in days), "days": len(days), "qualifying_days": len(qualifying),
                        "first": first, "last": last, "pct_of_target": round(100 * len(qualifying) / target_days, 1),
                        "days_per_week_rate": round(rate, 2), "eta_days": eta}
    finally:
        conn.close()
    return out


def collect(symbols: List[str] = SYMBOLS, pipeline=None, limit: int = 25, progress=print,
            pause_seconds: float = 2.0) -> Dict[str, dict]:
    """Fetch, filter, score (cached) and store news for each symbol. One failing symbol never stops the rest.

    ``pause_seconds`` between symbols keeps us under free-tier rate limits (Brave allows ~1 request/second
    and each symbol makes two); without it later symbols came back empty."""
    if pipeline is None:
        from core.news_pipeline import NewsPipeline
        pipeline = NewsPipeline.from_env()
    results: Dict[str, dict] = {}
    for n, sym in enumerate(symbols):
        if n and pause_seconds:
            time.sleep(pause_seconds)
        try:
            items = pipeline.fetch_news_items(sym, limit=limit)
            results[sym] = {"items": len(items), "error": None}
            progress(f"  {sym}: {len(items)} items")
        except Exception as exc:  # noqa: BLE001
            logger.warning("news collector: %s failed: %s", sym, exc)
            results[sym] = {"items": 0, "error": str(exc)}
            progress(f"  {sym}: ERROR {exc}")
    return results


def format_status(cov: Dict[str, dict], target_days: int = TARGET_DAYS, min_items: int = MIN_ITEMS_PER_DAY) -> str:
    lines = [f"News history vs the H3 threshold ({target_days} days with >= {min_items} headlines each)",
             f"{'symbol':<10}{'items':>7}{'days':>6}{'qualifying':>12}{'progress':>10}{'first':>12}{'last':>12}{'ETA':>10}"]
    for sym, c in cov.items():
        eta = "reached" if c["eta_days"] == 0 else (f"~{c['eta_days']}d" if c["eta_days"] else "n/a")
        lines.append(f"{sym:<10}{c['items']:>7}{c['days']:>6}{c['qualifying_days']:>12}{c['pct_of_target']:>9.1f}%"
                     f"{(c['first'] or '-'):>12}{(c['last'] or '-'):>12}{eta:>10}")
    ready = [s for s, c in cov.items() if c["qualifying_days"] >= target_days]
    lines.append(f"\nSymbols ready for the H3 test: {', '.join(ready) if ready else 'none yet'}")
    lines.append("*ETA extrapolates the rate seen so far (sparse until collection runs daily); treat it as an upper bound.")
    return "\n".join(lines)


def _cli(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m core.news_collector", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["collect", "status"])
    ap.add_argument("--symbols", default=",".join(SYMBOLS))
    args = ap.parse_args(argv)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if args.cmd == "collect":
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
        print(f"Collecting news for {len(symbols)} symbols ...")
        collect(symbols)
    print(format_status(coverage(symbols)))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
