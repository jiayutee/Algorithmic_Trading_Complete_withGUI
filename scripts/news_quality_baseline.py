"""News quality baseline (report only): duplicates, freshness, source coverage and lag from a news store file.

    python scripts/news_quality_baseline.py --db /path/to/COPY_of_news_store.sqlite3 [--json]

Read-only by construction: the database is opened with SQLite's ``mode=ro&immutable=1`` URI, so nothing (not even a
journal or WAL file) is written. Point it at a scratch copy anyway, and compare the canonical file's SHA-256 before/after.

What it measures (CONTINUATION_PLAN item 2, first slice):
  * counts per source family (metadata.source_api; "openbb:*" source names map to "openbb") and per symbol (tickers)
  * duplicate rate: exact (identical headline) and near (identical after normalisation: lowercase, " - Publisher"
    suffix removed, punctuation stripped). URLs are UNIQUE in the schema, so URL duplicates cannot occur.
  * timestamps: share whose publication time is within 60 s of the fetch time (the source gave no publication time and
    the fetch time was substituted), share at exactly 00:00:00 (date-only), and publication-to-fetch lag in hours for
    the rest (median / p90), plus negative lags (published "after" fetch = clock or parse problem)
  * daily coverage: distinct ingest days over the collection span and the longest gap without any ingest

What it does NOT measure: symbol relevance (needs labels), sentiment, fetch/pipeline latency (not stored), paid feeds.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timezone

SUBSTITUTED_SECONDS = 60


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00").replace(" ", "T", 1))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)  # created_at is SQLite datetime('now') = UTC


def normalise_title(title: str | None) -> str:
    t = (title or "").lower()
    t = re.sub(r"\s+[-|–—]\s+[^-|–—]{1,60}$", "", t)  # trailing " - Publisher"
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def source_family(source: str | None, metadata: str | None) -> str:
    try:
        api = (json.loads(metadata) or {}).get("source_api") if metadata else None
    except (ValueError, AttributeError):
        api = None
    if api:
        return str(api)
    if (source or "").startswith("openbb:"):
        return "openbb"
    return "unknown"


def symbols(tickers: str | None) -> list[str]:
    try:
        vals = json.loads(tickers) if tickers else []
    except ValueError:
        vals = []
    vals = [str(v) for v in vals if v] if isinstance(vals, list) else []
    return vals or ["(none)"]


def _pct(p: list[float], q: float) -> float | None:
    if not p:
        return None
    s = sorted(p)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def _group_stats(rows: list[dict]) -> dict:
    n = len(rows)
    heads = [r["headline"] or "" for r in rows]
    norms = [normalise_title(h) for h in heads]
    exact_dups = n - len(set(heads))
    near_dups = n - len(set(norms))
    substituted = midnight = negative = 0
    lags = []
    for r in rows:
        pub, fetched = r["pub"], r["fetched"]
        if pub is None or fetched is None:
            continue
        if pub.time() == datetime.min.time():
            midnight += 1
        lag = (fetched - pub).total_seconds()
        if abs(lag) < SUBSTITUTED_SECONDS:
            substituted += 1
        elif lag < 0:
            negative += 1
        else:
            lags.append(lag / 3600.0)
    unparsed = sum(1 for r in rows if r["pub"] is None or r["fetched"] is None)
    rate = (lambda k: round(k / n, 4) if n else None)
    return {
        "items": n,
        "exact_duplicate_rate": rate(exact_dups),
        "near_duplicate_rate": rate(near_dups),
        "unparseable_timestamp_share": rate(unparsed),
        "fetch_time_substituted_share": rate(substituted),
        "date_only_midnight_share": rate(midnight),
        "negative_lag_count": negative,
        "lag_hours_n": len(lags),
        "lag_hours_median": round(statistics.median(lags), 2) if lags else None,
        "lag_hours_p90": round(_pct(lags, 0.9), 2) if lags else None,
    }


def load_rows(db_path: str) -> list[dict]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    try:
        cur = con.execute("SELECT datetime_utc, created_at, source, headline, tickers, metadata FROM news")
        return [
            {"pub": _parse(p), "fetched": _parse(c), "source": s, "headline": h,
             "family": source_family(s, m), "symbols": symbols(t)}
            for p, c, s, h, t, m in cur.fetchall()
        ]
    finally:
        con.close()


def coverage(rows: list[dict]) -> dict:
    days = sorted({r["fetched"].date() for r in rows if r["fetched"]})
    if not days:
        return {"first_ingest_day": None, "last_ingest_day": None, "span_days": 0, "ingest_days": 0, "longest_gap_days": 0}
    gaps = [(b - a).days - 1 for a, b in zip(days, days[1:])]
    return {
        "first_ingest_day": days[0].isoformat(), "last_ingest_day": days[-1].isoformat(),
        "span_days": (days[-1] - days[0]).days + 1, "ingest_days": len(days),
        "longest_gap_days": max(gaps) if gaps else 0,
    }


def build_report(rows: list[dict]) -> dict:
    by_family, by_symbol = defaultdict(list), defaultdict(list)
    for r in rows:
        by_family[r["family"]].append(r)
        for s in r["symbols"]:
            by_symbol[s].append(r)
    return {
        "overall": _group_stats(rows),
        "coverage": coverage(rows),
        "by_source_family": {k: {**_group_stats(v), "coverage": coverage(v)}
                             for k, v in sorted(by_family.items(), key=lambda kv: -len(kv[1]))},
        "by_symbol": {k: _group_stats(v) for k, v in sorted(by_symbol.items(), key=lambda kv: -len(kv[1]))},
    }


def _fmt(v) -> str:
    return "n/a" if v is None else (f"{v:.1%}" if isinstance(v, float) and v <= 1 and not isinstance(v, bool) else str(v))


def to_markdown(rep: dict) -> str:
    cols = ["items", "exact_duplicate_rate", "near_duplicate_rate", "fetch_time_substituted_share",
            "date_only_midnight_share", "negative_lag_count", "lag_hours_n", "lag_hours_median", "lag_hours_p90"]
    head = "| group | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n"

    def row(name, st):
        return f"| {name} | " + " | ".join(
            str(st[c]) if c.startswith("lag_hours_") or c in ("items", "negative_lag_count") else _fmt(st[c])
            for c in cols) + " |\n"

    out = ["## Overall\n", head, row("all", rep["overall"]), "\n", f"Coverage: {json.dumps(rep['coverage'])}\n\n",
           "## By source family\n", head]
    out += [row(k, v) for k, v in rep["by_source_family"].items()]
    out += ["\nCoverage per family:\n\n"] + [f"- {k}: {json.dumps(v['coverage'])}\n" for k, v in rep["by_source_family"].items()]
    out += ["\n## By symbol (tickers field)\n", head] + [row(k, v) for k, v in rep["by_symbol"].items()]
    return "".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", required=True, help="path to a (scratch copy of a) news store sqlite file")
    ap.add_argument("--json", action="store_true", help="print JSON instead of Markdown tables")
    a = ap.parse_args(argv)
    rep = build_report(load_rows(a.db))
    print(json.dumps(rep, indent=2) if a.json else to_markdown(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
