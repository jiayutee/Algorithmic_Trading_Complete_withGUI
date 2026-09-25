"""Phase 9.3 on the COLLECTED snapshots -- the pre-registered test (docs/PHASE_9_3_PREREGISTRATION.md) re-run on
``core.kalshi_collector`` data instead of the API's short settled-history window.

    python training_ground/experiments_9_3_collected.py [--db PATH] [--out PATH]

No protocol change: the hypotheses (L, F), the two-sided-book filter (Amendment 2), the event-clustered bootstrap, the
97.5% intervals and the three finding criteria are the ones in ``experiments_9_3`` (``run`` is reused as-is).
Only the *source* of the snapshot differs, and that difference is reported, not hidden:

* one snapshot per market: the LAST collected snapshot at or before ``close - 2h`` (the pre-registered lead rule);
* the collector snapshots a few times a day, so the realised lead time is much longer than the ~2h of the original
  study (it is printed and stored as ``lead_h``). This is a calibration study at a longer lead, not a replica of 9.3a;
* ``open >= 3h`` and the random 800-market sample cannot be re-applied (open time is not stored): every resolved,
  eligible market is used.

A hypothesis that fails criterion 3 (>= 100 markets and >= 30 events) is reported as UNDERPOWERED, which is different
from "no evidence of bias": the data cannot say either way. Read-only: public data already on disk, no orders.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.kalshi_collector import _db_path  # noqa: E402
from training_ground import experiments_9_3 as e  # noqa: E402

LEAD_H = e.LEAD_H


def load_collected(db_path: Optional[str] = None, lead_h: float = LEAD_H) -> pd.DataFrame:
    """One row per resolved market: the last snapshot at or before ``close - lead_h`` (never a later one)."""
    path = _db_path(db_path)
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as con:
        raw = pd.read_sql_query(
            "SELECT s.ts, s.ticker, s.event_ticker, s.close_time, s.yes_bid, s.yes_ask, o.result "
            "FROM snapshots s JOIN outcomes o ON o.ticker = s.ticker", con)
    cols = ["ticker", "event", "close_ts", "outcome", "yes_ask", "yes_bid", "lead_h"]
    if raw.empty:
        return pd.DataFrame(columns=cols)
    raw["ts_"] = pd.to_datetime(raw["ts"], utc=True)
    raw["close_"] = pd.to_datetime(raw["close_time"], utc=True)
    raw["lead_h"] = (raw["close_"] - raw["ts_"]).dt.total_seconds() / 3600.0
    usable = raw[(raw["lead_h"] >= lead_h) & (raw["yes_ask"] > 0) & (raw["yes_ask"] < 1)]
    last = usable.sort_values("ts_").groupby("ticker", as_index=False).tail(1)
    out = pd.DataFrame({
        "ticker": last["ticker"].to_numpy(),
        "event": last["event_ticker"].where(last["event_ticker"].notna(), last["ticker"]).to_numpy(),  # unknown event -> its own cluster
        "close_ts": (last["close_"].astype("int64") // 10**9).to_numpy(),
        "outcome": last["result"].astype(int).to_numpy(),
        "yes_ask": last["yes_ask"].to_numpy(), "yes_bid": last["yes_bid"].to_numpy(),
        "lead_h": last["lead_h"].to_numpy()})
    out.attrs["resolved_markets"] = int(raw["ticker"].nunique())
    return out.reset_index(drop=True)


def poisson_binomial_pmf(p) -> np.ndarray:
    """Exact distribution of the number of YES outcomes when market i resolves YES with probability p[i]."""
    pmf = np.array([1.0])
    for pi in np.asarray(p, dtype=float):
        pmf = np.convolve(pmf, [1.0 - pi, pi])
    return pmf


def calibrated_tail(asks, hits: int, direction: str) -> float:
    """P(result at least as extreme as observed | every market is fairly priced at its ask).

    direction 'negative' (L): P(hits' <= observed); 'positive' (F): P(hits' >= observed). EXPLORATORY: unlike the
    pre-registered bootstrap it stays honest when the hit rate is 0% or 100% (the bootstrap interval collapses to the
    price spread there), but it treats markets as independent, which understates the probability for clustered events.
    """
    pmf = poisson_binomial_pmf(asks)
    return float(pmf[: hits + 1].sum() if direction == "negative" else pmf[hits:].sum())


def evaluate(df: pd.DataFrame, n_boot: int = e.N_BOOT) -> dict:
    res = e.run(df, n_boot)
    res["source"] = "core.kalshi_collector snapshots (one per resolved market, last at or before close - 2h)"
    res["resolved_markets"] = int(df.attrs.get("resolved_markets", len(df)))
    res["markets_without_usable_snapshot"] = res["resolved_markets"] - int(len(df))
    lead = df["lead_h"] if len(df) else pd.Series(dtype=float)
    res["lead_h"] = ({k: float(v) for k, v in lead.describe().items() if k in ("min", "25%", "50%", "75%", "max", "mean")}
                     if len(lead) else None)
    for h in res["hypotheses"]:
        crit = h.get("criteria", {})
        h["underpowered"] = not crit.get("3_enough_data", False)
        h["verdict"] = ("FINDING" if crit.get("FINDING") else
                        "UNDERPOWERED (criterion 3 unmet: cannot support or refute)" if h["underpowered"] else "no evidence")
    ts = e.two_sided(df)
    for h, (mask, direction) in zip(res["hypotheses"], [(ts["yes_ask"] <= e.LONGSHOT, "negative"), (ts["yes_ask"] >= e.FAVORITE, "positive")]):
        sub = ts[mask]
        if len(sub):
            hits = int(sub["outcome"].sum())
            h["exploratory_calibrated_tail"] = {
                "hits": hits, "n": int(len(sub)), "expected_hits_if_calibrated": float(sub["yes_ask"].sum()),
                "p_at_least_as_extreme": calibrated_tail(sub["yes_ask"], hits, direction),
                "note": "exploratory; assumes independent markets (they are event-clustered); not a pre-registered test"}
    res["protocol_deviations"] = [
        f"lead time is {res['lead_h']['min']:.1f}-{res['lead_h']['max']:.1f}h before close (median {res['lead_h']['50%']:.1f}h), "
        "not ~2h: a longer-lead calibration, not a replica of 9.3a" if res["lead_h"] else "no usable snapshots",
        "no random 800-market sample and no open>=3h filter (open time not stored): all resolved markets used",
        "one clustered window: markets closing 2026-09-20..21, mostly short-lived sports/commodity markets"]
    return res


def _fmt(res: dict) -> str:
    with_snapshot = res["resolved_markets"] - res["markets_without_usable_snapshot"]
    lines = [f"resolved markets {res['resolved_markets']}, with a snapshot >= 2h before close {with_snapshot}, "
             f"after two-sided filter {res['n_markets']} ({res['n_events']} events)"]
    if res["lead_h"]:
        lines.append("lead time (h before close): " + ", ".join(f"{k} {v:.1f}" for k, v in res["lead_h"].items()))
    lines += ["", e._fmt(res).split("\n", 1)[1].replace(" -> no evidence", "")]
    lines.append("")
    for h in res["hypotheses"]:
        lines.append(f"{h['id']}: {h['verdict']}")
        t = h.get("exploratory_calibrated_tail")
        if t:
            lines.append(f"   exploratory: {t['hits']} YES of {t['n']} vs {t['expected_hits_if_calibrated']:.1f} expected if fairly priced; "
                         f"P(as extreme | fair) = {t['p_at_least_as_extreme']:.3f} (independence assumed)")
    lines.append("\nDeviations (disclosed): " + "; ".join(res["protocol_deviations"]))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=None)
    ap.add_argument("--out", default=os.path.join("training_ground", "results", "phase_9_3_collected.json"))
    args = ap.parse_args(argv)
    res = evaluate(load_collected(args.db))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"run_at": datetime.now(timezone.utc).isoformat(), **res}, fh, indent=2, default=float)
    print("=== PHASE 9.3 ON COLLECTED SNAPSHOTS (pre-registered protocol, source differs) ===")
    print(_fmt(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
