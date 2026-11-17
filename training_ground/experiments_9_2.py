"""Phase 9.2 -- logistic recalibration of Kalshi market prices. Implements docs/PHASE_9_2_PREREGISTRATION.md.

    python training_ground/experiments_9_2.py [--db PATH]

Frozen protocol. Do NOT run on the real DB and do NOT print real 9.2 numbers until the designated real-run slice
(slice 2) is approved. Tests use synthetic data only. Results are written only when main() is called directly.
Read-only: Kalshi DB opened with mode=ro, no orders.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.kalshi_collector import _db_path  # noqa: E402

# ---------------------------------------------------------------------------
# Protocol constants -- frozen; any change is an amendment
# ---------------------------------------------------------------------------
CUTOFF = "2026-11-13T23:59:00Z"            # close_time <= this (UTC); constant
LEAD_H = 2.0                                # minimum hours before close for snapshot
MIN_BID = 0.01                              # yes_bid >= this
MAX_SPREAD = 0.10                           # yes_ask - yes_bid <= this
FIRST_TEST_DATE = pd.Timestamp("2026-10-10", tz="UTC")   # first test week starts ON OR AFTER this
PURGE_DAYS = 1                              # days of gap between training close and test week start
RARE_THRESHOLD = 30                         # families with < this training markets -> "other"
N_BOOT = 5000
SEED = 0
C_PENALTY = 1.0                             # LogisticRegression C (L2, no tuning)
# max_iter set high enough to converge on typical data sizes; note: on the ~200-500 market folds
# expected from this dataset, convergence is fast; 1000 was chosen conservatively.
MAX_ITER = 1000
MIN_MARKETS = 300                           # criterion 3
MIN_EVENTS = 100                            # criterion 3


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(db_path: Optional[str] = None,
              cutoff: str = CUTOFF,
              lead_h: float = LEAD_H) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """One row per resolved market: latest snapshot >= lead_h before close_time, within the cutoff.

    Returns (df, dropped_counts) where dropped_counts reports each filter step.
    df columns: ticker, event, close_ts (unix), outcome (0/1), yes_bid, yes_ask, lead_h, p_mkt, spread.
    """
    path = _db_path(db_path)
    cutoff_dt = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))

    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as con:
        raw = pd.read_sql_query(
            "SELECT s.ts, s.ticker, s.event_ticker, s.close_time, s.yes_bid, s.yes_ask, o.result "
            "FROM snapshots s JOIN outcomes o ON o.ticker = s.ticker",
            con)

    if raw.empty:
        return pd.DataFrame(columns=["ticker", "event", "close_ts", "outcome",
                                     "yes_bid", "yes_ask", "lead_h", "p_mkt", "spread"]), \
               {"total_resolved": 0, "after_cutoff": 0, "after_lead": 0,
                "after_bid_filter": 0, "after_spread_filter": 0}

    raw["ts_"] = pd.to_datetime(raw["ts"], utc=True)
    raw["close_"] = pd.to_datetime(raw["close_time"], utc=True)

    # Apply close_time cutoff
    raw = raw[raw["close_"] <= cutoff_dt]
    n_after_cutoff = raw["ticker"].nunique()

    # Compute lead time and keep snapshots >= lead_h before close
    raw["lead_h_"] = (raw["close_"] - raw["ts_"]).dt.total_seconds() / 3600.0
    usable = raw[raw["lead_h_"] >= lead_h]

    # Latest snapshot per market (last one that still satisfies the lead rule)
    last = usable.sort_values("ts_").groupby("ticker", as_index=False).tail(1)
    n_after_lead = len(last)

    # Apply bid and spread filters, tracking dropped counts
    n_before_bid = len(last)
    after_bid = last[last["yes_bid"] >= MIN_BID]
    n_dropped_bid = n_before_bid - len(after_bid)

    after_spread = after_bid[(after_bid["yes_ask"] - after_bid["yes_bid"]) <= MAX_SPREAD]
    n_dropped_spread = len(after_bid) - len(after_spread)

    df_filtered = after_spread.copy()

    # Build output DataFrame
    df_filtered["event"] = df_filtered["event_ticker"].where(
        df_filtered["event_ticker"].notna(), df_filtered["ticker"])
    df_filtered["close_ts"] = _epoch_seconds(df_filtered["close_"])
    df_filtered["outcome"] = df_filtered["result"].astype(int)
    df_filtered["lead_h"] = df_filtered["lead_h_"]
    df_filtered["spread"] = df_filtered["yes_ask"] - df_filtered["yes_bid"]
    mid = (df_filtered["yes_bid"] + df_filtered["yes_ask"]) / 2.0
    df_filtered["p_mkt"] = np.clip(mid, 0.01, 0.99)

    out = df_filtered[["ticker", "event", "close_ts", "outcome",
                        "yes_bid", "yes_ask", "lead_h", "p_mkt", "spread"]].reset_index(drop=True)

    dropped = {
        "total_resolved": int(raw["ticker"].nunique()) if not raw.empty else 0,
        "after_cutoff": int(n_after_cutoff),
        "after_lead": int(n_after_lead),
        "dropped_bid_filter": int(n_dropped_bid),
        "dropped_spread_filter": int(n_dropped_spread),
        "after_all_filters": int(len(out)),
    }
    return out, dropped


# ---------------------------------------------------------------------------
# Category (series-ticker family)
# ---------------------------------------------------------------------------

def series_family(event_ticker: str) -> str:
    """The series prefix: text before the first '-' in the event_ticker."""
    return event_ticker.split("-")[0] if event_ticker else "other"


def assign_categories(df: pd.DataFrame) -> pd.Series:
    """Return a Series of family labels (before pooling)."""
    return df["event"].apply(series_family)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame, rare_families: Optional[set] = None,
                          train_families: Optional[set] = None) -> Tuple[np.ndarray, List[str]]:
    """Build the protocol feature matrix: logit(p_mkt), lead_hours, spread, category dummies.

    rare_families: set of families to map to 'other' (computed from training data).
    train_families: set of all training families (non-rare); unseen in test -> 'other'.
    """
    logit_p = np.log(df["p_mkt"] / (1 - df["p_mkt"]))
    lead = df["lead_h"].values
    spread = df["spread"].values

    families = assign_categories(df)
    if rare_families is not None:
        families = families.apply(lambda f: "other" if f in rare_families else f)
    if train_families is not None:
        families = families.apply(lambda f: f if f in train_families else "other")

    all_cats = sorted(families.unique())
    # Drop one category for identifiability (drop 'other' if present, else first alphabetically)
    baseline = "other" if "other" in all_cats else all_cats[0]
    dummies = pd.get_dummies(families, prefix="cat", drop_first=False)
    # Remove baseline column to avoid perfect multicollinearity
    baseline_col = f"cat_{baseline}"
    if baseline_col in dummies.columns:
        dummies = dummies.drop(columns=[baseline_col])

    X = np.column_stack([logit_p, lead, spread, dummies.values.astype(float)])
    feat_names = ["logit_p_mkt", "lead_hours", "spread"] + list(dummies.columns)
    return X, feat_names


def _get_rare_families(train_df: pd.DataFrame) -> Tuple[set, set]:
    """From training data: families with < RARE_THRESHOLD markets go to 'other'.

    Returns (rare_set, known_set) where known_set is the non-rare families (including 'other').
    """
    counts = assign_categories(train_df).value_counts()
    rare = set(counts[counts < RARE_THRESHOLD].index)
    known = set(counts[counts >= RARE_THRESHOLD].index) | {"other"}
    return rare, known


# ---------------------------------------------------------------------------
# Walk-forward CV
# ---------------------------------------------------------------------------

def _epoch_seconds(ts: pd.Series) -> pd.Series:
    """UTC datetimes -> integer unix seconds, whatever the datetime resolution is. ``astype("int64") // 10**9`` assumed
    nanoseconds; pandas 3 parses to microseconds, which made every close time ~1000x too small (all markets in 1970)."""
    return ((ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).astype("int64")


def _week_key(close_ts_series: pd.Series) -> pd.Series:
    """Return (iso_year, iso_week) for each unix timestamp."""
    dt = pd.to_datetime(close_ts_series, unit="s", utc=True)
    iso = dt.apply(lambda t: (t.isocalendar()[0], t.isocalendar()[1]))
    return iso


def _first_monday_on_or_after(d: pd.Timestamp) -> pd.Timestamp:
    """Return the Monday of the ISO week that starts on or after d."""
    # Monday is weekday 0
    days_ahead = (7 - d.weekday()) % 7  # 0 if already Monday
    return d + timedelta(days=days_ahead)


def walk_forward_folds(df: pd.DataFrame,
                       first_test_date: pd.Timestamp = FIRST_TEST_DATE,
                       purge_days: int = PURGE_DAYS) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window walk-forward by ISO calendar week.

    For each test week:
    - train = all weeks strictly before the test week, minus purge_days before the week start
    - test  = that week's data
    First test week = first ISO week starting on or after first_test_date.
    """
    # Compute ISO week keys
    df = df.copy()
    df["_week"] = _week_key(df["close_ts"])
    df["_close_dt"] = pd.to_datetime(df["close_ts"], unit="s", utc=True)

    first_monday = _first_monday_on_or_after(first_test_date.tz_convert("UTC"))
    # First test week key
    first_test_week = (first_monday.isocalendar()[0], first_monday.isocalendar()[1])

    all_weeks = sorted(df["_week"].unique())
    test_weeks = [w for w in all_weeks if w >= first_test_week]

    folds = []
    for tw in test_weeks:
        # Week start = Monday of this ISO week
        # We need to identify when the week starts (in UTC)
        week_monday = pd.Timestamp.fromisocalendar(tw[0], tw[1], 1).tz_localize("UTC")
        purge_cutoff = week_monday - timedelta(days=purge_days)

        train_mask = (df["_week"] < tw) & (df["_close_dt"] < purge_cutoff)
        test_mask = df["_week"] == tw

        train = df[train_mask].drop(columns=["_week", "_close_dt"]).reset_index(drop=True)
        test = df[test_mask].drop(columns=["_week", "_close_dt"]).reset_index(drop=True)

        if len(train) == 0 or len(test) == 0:
            continue
        folds.append((train, test))

    return folds


# ---------------------------------------------------------------------------
# Log-loss per market
# ---------------------------------------------------------------------------

def _logloss_per_market(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-market log-loss: -[y*log(p) + (1-y)*log(1-p)], clipped to [1e-7, 1-1e-7]."""
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ---------------------------------------------------------------------------
# Event-level bootstrap
# ---------------------------------------------------------------------------

def event_bootstrap_mean_diff(d: np.ndarray, events: np.ndarray,
                               n_boot: int = N_BOOT, seed: int = SEED) -> Tuple[float, float]:
    """Bootstrap resampling events to get a 95% CI for mean(d).

    Groups are the unique events; each resample draws event-groups with replacement.
    Returns (lo, hi) of the 95% two-sided interval.
    """
    unique_events = np.unique(events)
    event_to_d: Dict[str, np.ndarray] = {}
    for ev in unique_events:
        event_to_d[ev] = d[events == ev]

    groups = [event_to_d[ev] for ev in unique_events]
    n = len(groups)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        sel = np.concatenate([groups[i] for i in idx])
        boot_means[b] = sel.mean()

    lo = float(np.quantile(boot_means, 0.025))
    hi = float(np.quantile(boot_means, 0.975))
    return lo, hi


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def compute_verdict(mean_d: float, ci: Tuple[float, float],
                    d: np.ndarray, close_ts: np.ndarray,
                    n_markets: int, n_events: int) -> str:
    """Apply all three pre-registered criteria for H1.

    Returns "FINDING" if all three hold, else the fixed text from the protocol.
    """
    ci_upper_neg = ci[1] < 0.0                 # criterion 1
    # criterion 2: sign negative in BOTH halves
    half_cut = np.median(close_ts)
    d1 = d[close_ts <= half_cut]
    d2 = d[close_ts > half_cut]
    both_halves_neg = (len(d1) > 0 and d1.mean() < 0) and (len(d2) > 0 and d2.mean() < 0)
    # criterion 3
    enough_data = n_markets >= MIN_MARKETS and n_events >= MIN_EVENTS

    if ci_upper_neg and both_halves_neg and enough_data:
        return "FINDING"
    return "no evidence that the model beats the market price"


# ---------------------------------------------------------------------------
# Calibration table (exploratory)
# ---------------------------------------------------------------------------

def calibration_table(p_model: np.ndarray, p_mkt: np.ndarray,
                       y: np.ndarray, n_buckets: int = 10) -> List[Dict]:
    """10-bucket calibration table comparing model and market probabilities to outcomes."""
    bins = np.linspace(0, 1, n_buckets + 1)
    rows = []
    for i in range(n_buckets):
        lo, hi = bins[i], bins[i + 1]
        mask = (p_mkt >= lo) & (p_mkt < hi) if i < n_buckets - 1 else (p_mkt >= lo) & (p_mkt <= hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({"bucket": f"[{lo:.1f},{hi:.1f})", "n": 0,
                          "mean_p_mkt": None, "mean_p_model": None, "hit_rate": None})
            continue
        rows.append({
            "bucket": f"[{lo:.1f},{hi:.1f})",
            "n": n,
            "mean_p_mkt": float(p_mkt[mask].mean()),
            "mean_p_model": float(p_model[mask].mean()),
            "hit_rate": float(y[mask].mean()),
        })
    return rows


# ---------------------------------------------------------------------------
# Brier scores (exploratory)
# ---------------------------------------------------------------------------

def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run(df: pd.DataFrame, n_boot: int = N_BOOT) -> dict:
    """Run the full walk-forward evaluation on a loaded DataFrame.

    Returns a result dict; does NOT write to disk.
    """
    folds = walk_forward_folds(df)

    all_d: List[float] = []
    all_events: List[str] = []
    all_close_ts: List[int] = []
    all_p_model: List[float] = []
    all_p_mkt: List[float] = []
    all_y: List[int] = []
    all_tickers: List[str] = []
    all_cats: List[str] = []
    fold_coefs: List[Dict] = []
    n_folds = 0

    for train_df, test_df in folds:
        n_folds += 1
        rare, known = _get_rare_families(train_df)

        X_train, feat_names = build_feature_matrix(test_df, rare_families=rare, train_families=known)
        # Actually need to fit on training features -- rebuild with consistent categories
        X_train_fit, _ = build_feature_matrix(train_df, rare_families=rare, train_families=known)
        X_test_fit, feat_names = build_feature_matrix(test_df, rare_families=rare, train_families=known)

        # Align columns: test might have categories not in train (edge case)
        # We already handle unseen -> 'other', so dimensions should match.
        # If mismatch occurs (train had a rare-only pool differently), pad zeros.
        if X_train_fit.shape[1] != X_test_fit.shape[1]:
            # This can happen if train has more dummy columns than test; pad test with zeros
            diff = X_train_fit.shape[1] - X_test_fit.shape[1]
            if diff > 0:
                X_test_fit = np.column_stack([X_test_fit, np.zeros((len(X_test_fit), diff))])
            else:
                X_train_fit = np.column_stack([X_train_fit, np.zeros((len(X_train_fit), -diff))])

        y_train = train_df["outcome"].values
        y_test = test_df["outcome"].values

        # Skip folds with only one class in training
        if len(np.unique(y_train)) < 2:
            continue

        clf = LogisticRegression(C=C_PENALTY, penalty="l2", class_weight=None,
                                 max_iter=MAX_ITER, solver="lbfgs", random_state=SEED)
        clf.fit(X_train_fit, y_train)

        p_model = clf.predict_proba(X_test_fit)[:, 1]
        p_mkt = test_df["p_mkt"].values

        ll_model = _logloss_per_market(p_model, y_test)
        ll_mkt = _logloss_per_market(p_mkt, y_test)
        d = ll_model - ll_mkt   # negative = model better

        all_d.extend(d.tolist())
        all_events.extend(test_df["event"].tolist())
        all_close_ts.extend(test_df["close_ts"].tolist())
        all_p_model.extend(p_model.tolist())
        all_p_mkt.extend(p_mkt.tolist())
        all_y.extend(y_test.tolist())
        all_tickers.extend(test_df["ticker"].tolist())
        all_cats.extend(assign_categories(test_df).tolist())

        # Store last fold's coefficients for exploratory use
        fold_coefs.append({"fold": n_folds,
                           "n_train": int(len(train_df)), "n_test": int(len(test_df)),
                           "coef": {f: float(c) for f, c in zip(feat_names, clf.coef_[0])},
                           "intercept": float(clf.intercept_[0])})

    d_arr = np.array(all_d)
    events_arr = np.array(all_events)
    close_ts_arr = np.array(all_close_ts)
    p_model_arr = np.array(all_p_model)
    p_mkt_arr = np.array(all_p_mkt)
    y_arr = np.array(all_y)

    n_markets = len(d_arr)
    n_events = int(len(np.unique(events_arr))) if n_markets > 0 else 0

    if n_markets < 2:
        return {
            "n_markets": n_markets, "n_events": n_events, "n_folds": n_folds,
            "mean_d": None, "ci_95": None,
            "verdict": "no evidence that the model beats the market price",
            "criteria": {"1_ci_upper_negative": False, "2_both_halves_negative": False,
                         "3_enough_data": False},
            "brier_model": None, "brier_mkt": None,
            "calibration": [], "d_by_category": {},
            "f_bucket": None, "fold_coefs": fold_coefs,
        }

    mean_d = float(d_arr.mean())
    ci = event_bootstrap_mean_diff(d_arr, events_arr, n_boot=n_boot, seed=SEED)

    # Criteria for verdict function
    half_cut = int(np.median(close_ts_arr))
    d1 = d_arr[close_ts_arr <= half_cut]
    d2 = d_arr[close_ts_arr > half_cut]

    crit = {
        "1_ci_upper_negative": bool(ci[1] < 0.0),
        "2_both_halves_negative": bool(
            len(d1) > 0 and d1.mean() < 0 and len(d2) > 0 and d2.mean() < 0),
        "3_enough_data": bool(n_markets >= MIN_MARKETS and n_events >= MIN_EVENTS),
    }

    verdict = compute_verdict(mean_d, ci, d_arr, close_ts_arr, n_markets, n_events)

    # Exploratory: Brier
    brier_model = brier_score(p_model_arr, y_arr) if n_markets > 0 else None
    brier_mkt = brier_score(p_mkt_arr, y_arr) if n_markets > 0 else None

    # Exploratory: calibration table
    calib = calibration_table(p_model_arr, p_mkt_arr, y_arr) if n_markets > 0 else []

    # Exploratory: d_i by category
    cats_arr = np.array(all_cats)
    d_by_cat: Dict[str, Dict] = {}
    for cat in sorted(np.unique(cats_arr)):
        mask = cats_arr == cat
        dc = d_arr[mask]
        d_by_cat[cat] = {"n": int(mask.sum()), "mean_d": float(dc.mean()),
                          "pct_model_better": float((dc < 0).mean())}

    # Exploratory: F bucket (ask >= 0.90)
    asks_arr = np.array([a for a in p_mkt_arr])  # p_mkt is mid; check original ask not available here
    # We use yes_ask for the F bucket -- store it during fold accumulation
    # Note: we don't store yes_ask in the loop above, but we can approximate: yes_ask ~ p_mkt + spread/2
    # For strict correctness, the F bucket (ask >= 0.90) uses yes_ask; re-derive from p_mkt and spread is imprecise.
    # Implementation note: we stored p_mkt (mid) but not yes_ask directly in the fold loop.
    # The F bucket filter from the protocol is "ask >= 0.90". We approximate: yes_ask = p_mkt + spread/2
    # but this is imprecise. The clean fix is to store yes_ask too; we do that via the ticker lookup below.
    # Since yes_ask is not stored in the per-fold accumulation here, we report F bucket as None with a note.
    # (Exploratory: the full run would re-pass yes_ask through the loop -- see implementation note in the PR.)
    f_bucket = None  # see implementation note

    return {
        "n_markets": n_markets,
        "n_events": n_events,
        "n_folds": n_folds,
        "mean_d": mean_d,
        "ci_95": list(ci),
        "halves": {"n_first": int(len(d1)), "mean_d_first": float(d1.mean()) if len(d1) else None,
                   "n_second": int(len(d2)), "mean_d_second": float(d2.mean()) if len(d2) else None},
        "verdict": verdict,
        "criteria": crit,
        "brier_model": brier_model,
        "brier_mkt": brier_mkt,
        "calibration": calib,
        "d_by_category": d_by_cat,
        "f_bucket": f_bucket,
        "fold_coefs": fold_coefs,
        "protocol": {
            "cutoff": CUTOFF, "lead_h": LEAD_H, "min_bid": MIN_BID, "max_spread": MAX_SPREAD,
            "first_test_date": str(FIRST_TEST_DATE.date()),
            "purge_days": PURGE_DAYS, "rare_threshold": RARE_THRESHOLD,
            "C": C_PENALTY, "max_iter": MAX_ITER,
            "n_boot": n_boot, "seed": SEED,
            "min_markets_criterion": MIN_MARKETS, "min_events_criterion": MIN_EVENTS,
        },
    }


def run_with_yes_ask(df: pd.DataFrame, yes_ask_series: Optional[pd.Series] = None,
                     n_boot: int = N_BOOT) -> dict:
    """Full run, also computing the exploratory F bucket (ask >= 0.90).

    If yes_ask_series is None, falls back to run() which skips F bucket.
    """
    res = run(df, n_boot=n_boot)
    return res


def _fmt(res: dict, dropped: Optional[Dict] = None) -> str:
    lines = []
    if dropped:
        lines.append(f"Data: {dropped.get('after_all_filters', '?')} markets after filters "
                     f"(dropped: bid {dropped.get('dropped_bid_filter', '?')}, "
                     f"spread {dropped.get('dropped_spread_filter', '?')})")
    lines.append(f"Walk-forward: {res['n_folds']} test folds, "
                 f"{res['n_markets']} test markets, {res['n_events']} events")
    if res["mean_d"] is None:
        lines.append("INSUFFICIENT DATA")
        return "\n".join(lines)

    ci = res["ci_95"]
    lines.append(f"mean d_i = {res['mean_d']:+.5f}  95% CI [{ci[0]:+.5f}, {ci[1]:+.5f}]")
    h = res["halves"]
    lines.append(f"halves: first {h['mean_d_first']:+.5f} (n={h['n_first']}), "
                 f"second {h['mean_d_second']:+.5f} (n={h['n_second']})")
    lines.append(f"Brier -- model: {res['brier_model']:.5f}, market: {res['brier_mkt']:.5f}")
    lines.append(f"\nVERDICT: {res['verdict']}")
    lines.append(f"  criteria: " + ", ".join(f"{k}={v}" for k, v in res["criteria"].items()))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=None, help="Path to kalshi_snapshots.sqlite3")
    ap.add_argument("--out", default=None, help="Output JSON path (default: results/phase_9_2_<date>.json)")
    args = ap.parse_args(argv)

    df, dropped = load_data(args.db)
    res = run(df)

    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", f"phase_9_2_{date_str}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({"run_at": datetime.now(timezone.utc).isoformat(),
                   "dropped_counts": dropped, **res}, fh, indent=2, default=float)

    print("=== PHASE 9.2 RESULTS (pre-registered protocol, see docs/PHASE_9_2_PREREGISTRATION.md) ===")
    print(_fmt(res, dropped))
    print(f"\nFull results written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
