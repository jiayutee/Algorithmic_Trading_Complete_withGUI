#!/usr/bin/env python3
"""Probe the app's news sources without scoring sentiment or writing the news store.

Exit 0: each requested symbol received at least one raw item; 1: at least one did not.
Individual sources can still be degraded when the exit code is zero: inspect source rows.
Disabled optional sources are configuration information, not failed network probes.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.news_pipeline import NewsPipeline, _query_variants
from core.logger import logger


def run_report(pipeline: NewsPipeline, symbols: list[str], limit: int = 5) -> dict:
    """Use the same query routing, deadline and circuit breaker as a normal refresh.

    Reports raw delivery only: no claim about relevance, sentiment quality or profitability.
    Raw source errors, request URLs, keys and article bodies are deliberately not exported.
    """
    started = time.monotonic()
    probes = []
    for symbol in symbols:
        query = _query_variants(symbol)[0]
        t0 = time.monotonic()
        items = pipeline._fetch_all_sources(query, limit, ticker_query=symbol)
        rows = pipeline.source_status()
        enabled = [r for r in rows if r["enabled"]]
        status = "unavailable" if not items else (
            "ok" if all(r["status"] == "ok" for r in enabled) else "degraded")
        probes.append({"symbol": symbol, "text_query": query, "status": status,
                       "raw_item_count": len(items), "elapsed_seconds": round(time.monotonic() - t0, 3),
                       "sources": rows})
    return {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
            "scope": "raw source fetch only; excludes sentiment, filtering and persistence",
            "fetch_deadline_seconds": pipeline.deadline_seconds,
            "elapsed_seconds": round(time.monotonic() - started, 3), "probes": probes,
            "usable": bool(probes) and all(p["raw_item_count"] > 0 for p in probes)}


def positive_seconds(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("deadline must be a finite positive number")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "AAPL"])
    parser.add_argument("--deadline", type=positive_seconds, default=6.0,
                        help="shared fetch budget per symbol in seconds (not sentiment latency)")
    parser.add_argument("--env-file", type=Path, help="explicit dotenv file; existing environment wins")
    parser.add_argument("--output", type=Path, help="save the same JSON printed on stdout")
    args = parser.parse_args(argv)
    if args.env_file:
        if not args.env_file.is_file():
            parser.error("env file does not exist")
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)

    # Keep stdout machine-readable; providers can print during optional dependency imports.
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setStream(sys.stderr)
    with redirect_stdout(sys.stderr):
        pipeline = NewsPipeline.from_env()
        pipeline.deadline_seconds = args.deadline
        report = run_report(pipeline, args.symbols)
    try:
        report["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        report["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True))
    except (OSError, subprocess.CalledProcessError):
        report["git_commit"] = None
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if report["usable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
