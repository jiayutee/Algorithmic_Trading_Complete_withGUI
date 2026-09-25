"""Phase 13.1: do the Market Context event readings predict later price moves?  (docs/PHASE_13_1_PREREGISTRATION.md)

Frozen protocol -- read the pre-registration before changing anything here. Re-run as the news collector accumulates history:

    python training_ground/experiments_13_1.py            # writes training_ground/results/phase_13_1.json, prints the verdict
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ml_validation import _block_resample_dates  # noqa: E402
from core.news_context import keep_for_context  # noqa: E402
from core.news_sources import NewsItem  # noqa: E402

HORIZONS = (1, 3, 7)
PRIMARY = 3
BLOCK = 7
N_BOOT = 5000
N_PLACEBO = 2000
LEVEL = 1 - 0.05 / len(HORIZONS)          # Bonferroni across horizons (one-sided claim, two-sided interval, as in 9.3)
MIN_READS, MIN_EACH_SIDE = 30, 10
RESULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "phase_13_1.json")


def load_reads(db_path: str, symbols) -> pd.DataFrame:
    """One row per symbol-day with a read: columns symbol, date, read (+1 bullish / -1 bearish), n_bull, n_bear.

    Events are filtered and read exactly as Market Context does it (rules only, no model, no price)."""
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=["symbol", "date", "read", "n_bull", "n_bear"])
    conn = sqlite3.connect(db_path)
    rows = []
    try:
        for sym in symbols:
            for dt, source, headline, url, summary in conn.execute(
                    "SELECT datetime_utc, source, headline, url, summary FROM news WHERE tickers LIKE ?", (f'%"{sym}"%',)):
                ts = pd.to_datetime(dt, utc=True, errors="coerce")
                if pd.isna(ts):
                    continue
                item = NewsItem(datetime_utc=ts.to_pydatetime(), source=source or "", headline=headline or "",
                                url=url or "", summary=summary or "")
                interp = keep_for_context(item, sym)
                if interp is None:
                    continue
                bias = interp["conditional_bias"]
                rows.append((sym, ts.strftime("%Y-%m-%d"), 1 if bias == "bullish" else -1 if bias == "bearish" else 0))
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame(columns=["symbol", "date", "read", "n_bull", "n_bear"])
    df = pd.DataFrame(rows, columns=["symbol", "date", "b"])
    g = df.groupby(["symbol", "date"])["b"]
    out = pd.DataFrame({"n_bull": g.apply(lambda s: int((s > 0).sum())), "n_bear": g.apply(lambda s: int((s < 0).sum()))}).reset_index()
    out["read"] = np.sign(out["n_bull"] - out["n_bear"]).astype(int)
    return out[out["read"] != 0].reset_index(drop=True)


def directional_excess(reads: pd.DataFrame, closes: dict, horizon: int) -> pd.DataFrame:
    """Per read: entry = close of the publication day, exit = close ``horizon`` days later; the symbol's mean h-day log
    return is subtracted, then the sign is applied (+ for bullish, - for bearish)."""
    out = []
    for sym, grp in reads.groupby("symbol"):
        c = closes.get(sym)
        if c is None or len(c) <= horizon:
            continue
        lr = np.log(c.shift(-horizon) / c).dropna()
        drift = float(lr.mean())
        for _, r in grp.iterrows():
            d = pd.Timestamp(r["date"])
            if d in lr.index:
                out.append({"symbol": sym, "date": d, "read": int(r["read"]),
                            "value": int(r["read"]) * (float(lr.loc[d]) - drift)})
    return pd.DataFrame(out, columns=["symbol", "date", "read", "value"])


def block_ci(values: pd.DataFrame, level: float, seed: int = 0):
    """Moving-block bootstrap over dates (whole dates resampled together across symbols)."""
    if values.empty:
        return (float("nan"), float("nan"))
    dates = np.array(sorted(values["date"].unique()))
    by_date = [values.loc[values["date"] == d, "value"].to_numpy() for d in dates]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(N_BOOT):
        picked = np.concatenate([by_date[i] for i in _block_resample_dates(len(dates), BLOCK, rng)])
        means.append(picked.mean())
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def placebo_p(values: pd.DataFrame, seed: int = 0) -> float:
    """Share of label shuffles (reads permuted across the same symbol-days) whose mean directional excess >= observed."""
    if values.empty:
        return float("nan")
    raw = (values["value"] / values["read"]).to_numpy()        # the excess return itself, before the sign
    reads = values["read"].to_numpy()
    observed = float(values["value"].mean())
    rng = np.random.default_rng(seed)
    hits = sum(float((rng.permutation(reads) * raw).mean()) >= observed for _ in range(N_PLACEBO))
    return hits / N_PLACEBO


def evaluate(reads: pd.DataFrame, closes: dict) -> dict:
    per_h = {}
    for h in HORIZONS:
        v = directional_excess(reads, closes, h)
        lo, hi = block_ci(v, LEVEL)
        per_h[h] = {"n": int(len(v)), "n_bullish": int((v["read"] > 0).sum()) if len(v) else 0,
                    "n_bearish": int((v["read"] < 0).sum()) if len(v) else 0,
                    "mean": float(v["value"].mean()) if len(v) else float("nan"), "ci": [lo, hi]}
        if h == PRIMARY:
            primary = v
    p = per_h[PRIMARY]
    enough = p["n"] >= MIN_READS and p["n_bullish"] >= MIN_EACH_SIDE and p["n_bearish"] >= MIN_EACH_SIDE
    halves_ok, placebo = None, float("nan")
    if len(primary):
        cut = np.median(primary["date"].astype("int64"))
        first, second = primary[primary["date"].astype("int64") <= cut], primary[primary["date"].astype("int64") > cut]
        halves_ok = bool(len(first) and len(second) and first["value"].mean() > 0 and second["value"].mean() > 0)
        placebo = placebo_p(primary)
    checks = {"ci_lower_above_zero": bool(p["ci"][0] > 0), "positive_in_both_halves": bool(halves_ok),
              "placebo_p_below_0_05": bool(placebo < 0.05), "enough_reads": bool(enough)}
    if not enough:
        verdict = "INCONCLUSIVE (too few directional reads to say anything)"
    elif all(checks.values()):
        verdict = "SUPPORTED (one sample; replicate on a later period before trusting)"
    else:
        verdict = "NOT SUPPORTED"
    return {"horizons": per_h, "primary_horizon": PRIMARY, "checks": checks, "placebo_p": placebo, "verdict": verdict,
            "level": LEVEL}


def _nan_to_none(obj):
    """Strict JSON has no NaN: an empty statistic is written as null."""
    if isinstance(obj, dict):
        return {str(k): _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_to_none(v) for v in obj]
    return None if isinstance(obj, float) and obj != obj else obj


def main() -> int:
    from core.news_collector import SYMBOLS
    from core.news_store import DEFAULT_DB
    from core.order_flow_data import fetch_klines
    reads = load_reads(DEFAULT_DB, SYMBOLS)
    closes = {}
    for sym in SYMBOLS:
        try:
            kl = fetch_klines(sym, days=900)
            closes[sym] = kl["Close"].astype(float)
        except Exception as exc:  # noqa: BLE001 -- one symbol's data outage must not void the rest
            print(f"  price fetch failed for {sym}: {exc}")
    result = evaluate(reads, closes)
    result.update({"run_at": datetime.now(timezone.utc).isoformat(), "symbols": SYMBOLS,
                   "n_symbol_days_with_read": int(len(reads)),
                   "reads_by_symbol": reads.groupby("symbol").size().to_dict() if len(reads) else {}})
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w") as fh:
        json.dump(_nan_to_none(result), fh, indent=2, default=str, allow_nan=False)
    print(f"Phase 13.1 -- {result['n_symbol_days_with_read']} symbol-days with a bullish/bearish read")
    for h, r in result["horizons"].items():
        print(f"  {h}d: n={r['n']} ({r['n_bullish']} bull / {r['n_bearish']} bear)  mean directional excess {r['mean']:+.4f}  "
              f"{LEVEL:.2%} CI [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]")
    print("VERDICT:", result["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
