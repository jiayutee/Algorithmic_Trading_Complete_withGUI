"""Phase 9.3a -- Kalshi favorite-longshot calibration. Implements docs/PHASE_9_3_PREREGISTRATION.md exactly.

    python training_ground/experiments_9_3.py
Read-only: public market data only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.kalshi_arbitrage import taker_fee  # noqa: E402
from core.kalshi_data import KalshiClient, KalshiError, Market  # noqa: E402

MAX_LOAD = 4000
MIN_VOLUME = 100.0
MIN_LIFE_H = 3.0
SAMPLE = 800
LEAD_H = 2.0
LONGSHOT, FAVORITE = 0.10, 0.90
N_BOOT = 5000
K = 2
LEVEL = 1 - 0.05 / K
MIN_MARKETS, MIN_EVENTS = 100, 30
MIN_BID, MAX_SPREAD = 0.01, 0.10     # amendment 2: two-sided book required (empty books show ask ~0.99 / bid ~0)
SEED = 0


def _ts(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def snapshot_quotes(candles: list, cutoff_ts: int) -> Optional[dict]:
    """Last candle ending at/before cutoff with usable quotes -> taker entry prices."""
    usable = [c for c in candles if c.get("end_period_ts", 0) <= cutoff_ts]
    if not usable:
        return None
    c = max(usable, key=lambda c: c["end_period_ts"])
    try:
        ask = float(c["yes_ask"]["close_dollars"])
        bid = float(c["yes_bid"]["close_dollars"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (0 < ask < 1) or not (0 <= bid < 1) or bid > ask:
        return None
    return {"yes_ask": ask, "yes_bid": bid}


def collect(client: KalshiClient, seed: int = SEED, sample: int = SAMPLE, log=print) -> pd.DataFrame:
    markets: List[Market] = list(client.iter_markets(status="settled", max_items=MAX_LOAD, page_size=1000))
    eligible = [m for m in markets if m.result is not None and m.volume >= MIN_VOLUME and m.close_time and m.raw.get("open_time")
                and _ts(m.close_time) - _ts(m.raw["open_time"]) >= MIN_LIFE_H * 3600]
    log(f"loaded {len(markets)} settled; {len(eligible)} eligible")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(eligible))[:sample]
    rows, dropped = [], 0
    for i in idx:
        m = eligible[i]
        close, opened = _ts(m.close_time), _ts(m.raw["open_time"])
        try:
            candles = client.get_candlesticks(m.ticker, opened, close, 60)
        except KalshiError:
            dropped += 1
            continue
        q = snapshot_quotes(candles, int(close - LEAD_H * 3600))
        if q is None:
            dropped += 1
            continue
        rows.append({"ticker": m.ticker, "event": m.event_ticker, "close_ts": close, "outcome": int(m.result), **q})
    log(f"snapshots: {len(rows)} kept, {dropped} dropped")
    df = pd.DataFrame(rows)
    df.attrs["dropped"] = dropped
    return df


def edge_yes(df: pd.DataFrame) -> pd.Series:
    """Pre-fee mispricing edge of buying YES at the ask."""
    return df["outcome"] - df["yes_ask"]


def net_no(df: pd.DataFrame) -> pd.Series:
    """Net return of buying NO at 1 - yes_bid (the tradable side of a 'longshots are overpriced' finding)."""
    price = 1 - df["yes_bid"]
    return (1 - df["outcome"]) - price - price.map(lambda p: taker_fee(p))


def net_yes(df: pd.DataFrame) -> pd.Series:
    fee = df["yes_ask"].map(lambda p: taker_fee(p))
    return df["outcome"] - df["yes_ask"] - fee


def event_bootstrap_mean(df: pd.DataFrame, values: pd.Series, n_boot: int, level: float, seed: int = SEED):
    groups = [g.to_numpy() for _, g in values.groupby(df["event"])]
    rng = np.random.default_rng(seed)
    n = len(groups)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n, n)
        sel = np.concatenate([groups[j] for j in pick])
        means[b] = sel.mean()
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def test_bucket(df: pd.DataFrame, name: str, mask: pd.Series, direction: str, n_boot: int = N_BOOT) -> dict:
    sub = df[mask]
    out = {"id": name, "n_markets": int(len(sub)), "n_events": int(sub["event"].nunique())}
    if len(sub) < 2:
        out.update({"mean_net": None, "ci": None, "criteria": {"FINDING": False}, "note": "too few markets"})
        return out
    v = edge_yes(sub)                                            # criteria use the PRE-fee edge (see amendment)
    lo, hi = event_bootstrap_mean(sub, v, n_boot, LEVEL)
    tv = net_no(sub) if direction == "negative" else net_yes(sub)   # tradable side, after fees
    tlo, thi = event_bootstrap_mean(sub, tv, n_boot, LEVEL)
    half = sub["close_ts"].median()
    h1, h2 = v[sub["close_ts"] <= half], v[sub["close_ts"] > half]
    sign = -1 if direction == "negative" else 1
    crit = {"1_ci_excludes_zero_in_claimed_direction": bool(hi < 0) if direction == "negative" else bool(lo > 0),
            "2_same_sign_both_halves": bool(len(h1) and len(h2) and sign * h1.mean() > 0 and sign * h2.mean() > 0),
            "3_enough_data": bool(len(sub) >= MIN_MARKETS and sub["event"].nunique() >= MIN_EVENTS)}
    crit["FINDING"] = all(crit.values())
    tradable = bool(crit["FINDING"] and tlo > 0)
    out.update({"mean_edge": float(v.mean()), "mean_net": float(v.mean()), "tradable_mean_net": float(tv.mean()),
                "tradable_ci": [tlo, thi], "TRADABLE": tradable, "mean_ask": float(sub["yes_ask"].mean()), "hit_rate": float(sub["outcome"].mean()),
                "ci": [lo, hi], "halves": [float(h1.mean()) if len(h1) else None, float(h2.mean()) if len(h2) else None],
                "criteria": crit})
    return out


def calibration_table(df: pd.DataFrame) -> list:
    bins = np.linspace(0, 1, 11)
    cut = pd.cut(df["yes_ask"], bins, include_lowest=True)
    t = df.groupby(cut, observed=True).agg(n=("outcome", "size"), mean_ask=("yes_ask", "mean"), hit_rate=("outcome", "mean"))
    return [{"bucket": str(k), "n": int(r.n), "mean_ask": float(r.mean_ask), "hit_rate": float(r.hit_rate)} for k, r in t.iterrows()]


def two_sided(df: pd.DataFrame) -> pd.DataFrame:
    """Amendment 2: keep only snapshots with a real two-sided book (bid >= 0.01 and ask - bid <= 0.10)."""
    return df[(df["yes_bid"] >= MIN_BID) & (df["yes_ask"] - df["yes_bid"] <= MAX_SPREAD)].reset_index(drop=True)


def run(df: pd.DataFrame, n_boot: int = N_BOOT) -> dict:
    n_before = len(df)
    df = two_sided(df)
    hyps = [test_bucket(df, "L longshots (ask<=0.10): mean edge < 0", df["yes_ask"] <= LONGSHOT, "negative", n_boot),
            test_bucket(df, "F favorites (ask>=0.90): mean edge > 0", df["yes_ask"] >= FAVORITE, "positive", n_boot)]
    return {"n_markets": int(len(df)), "n_events": int(df["event"].nunique()) if len(df) else 0,
            "window": [str(pd.to_datetime(df["close_ts"].min(), unit="s")), str(pd.to_datetime(df["close_ts"].max(), unit="s"))] if len(df) else None,
            "dropped": int(df.attrs.get("dropped", 0)), "dropped_one_sided": int(n_before - len(df)), "hypotheses": hyps, "calibration": calibration_table(df) if len(df) else [],
            "protocol": {"min_volume": MIN_VOLUME, "min_life_h": MIN_LIFE_H, "sample": SAMPLE, "lead_h": LEAD_H,
                         "longshot": LONGSHOT, "favorite": FAVORITE, "level": LEVEL, "n_boot": n_boot, "seed": SEED}}


def _fmt(res: dict) -> str:
    lines = [f"{res['n_markets']} markets / {res['n_events']} events, window {res['window']}, dropped {res['dropped']} (+{res['dropped_one_sided']} one-sided books)", ""]
    for h in res["hypotheses"]:
        if h.get("mean_edge") is None and h.get("mean_net") is None:
            lines.append(f"{h['id']}: n={h['n_markets']} -- {h.get('note')}")
            continue
        lines.append(f"{h['id']}: n={h['n_markets']} ({h['n_events']} events) mean ask {h['mean_ask']:.3f} hit {h['hit_rate']:.3f} "
                     f"edge {h['mean_edge']:+.4f} CI{LEVEL*100:.1f} [{h['ci'][0]:+.4f},{h['ci'][1]:+.4f}] "
                     f"tradable-side net {h['tradable_mean_net']:+.4f} [{h['tradable_ci'][0]:+.4f},{h['tradable_ci'][1]:+.4f}] "
                     f"halves {h['halves'][0]:+.3f}/{h['halves'][1]:+.3f} -> {'*** FINDING ***' if h['criteria']['FINDING'] else 'no evidence'}")
    lines.append("\ncalibration (ask bucket -> hit rate):")
    for r in res["calibration"]:
        lines.append(f"  {r['bucket']:<14} n={r['n']:<4} ask {r['mean_ask']:.3f}  hit {r['hit_rate']:.3f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join("training_ground", "results", "phase_9_3.json"))
    ap.add_argument("--cache", default=os.path.join("training_ground", "datasets", "cache", "kalshi_9_3.csv"))
    args = ap.parse_args(argv)
    if os.path.exists(args.cache):
        df = pd.read_csv(args.cache)
    else:
        df = collect(KalshiClient())
        os.makedirs(os.path.dirname(args.cache), exist_ok=True)
        df.to_csv(args.cache, index=False)
    res = run(df)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"run_at": datetime.now(timezone.utc).isoformat(), **res}, fh, indent=2, default=float)
    print("=== PHASE 9.3a RESULTS (pre-registered protocol) ===")
    print(_fmt(res))
    print("\nFINDINGS:", "; ".join(h["id"] for h in res["hypotheses"] if h["criteria"]["FINDING"]) or "none.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
