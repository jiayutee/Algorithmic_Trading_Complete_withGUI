"""Phase 6.5 experiments -- implements docs/PHASE_6_5_PREREGISTRATION.md exactly.

    python training_ground/experiments_6_5.py                # all experiments
    python training_ground/experiments_6_5.py --only H2a,H5  # a subset

Results go to training_ground/results/phase_6_5.json and a table on stdout. The protocol
(symbols, model, validation, intervals, pass/fail rules) is fixed in the pre-registration;
nothing here is tuned.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.experiment_log import ExperimentLog  # noqa: E402
from core.feature_engineering import build_features, make_target  # noqa: E402
from core.ml_validation import (  # noqa: E402
    _auc, block_bootstrap_auc_ci, paired_block_bootstrap_auc_diff, walk_forward_predict,
    walk_forward_predict_panel,
)
from core.order_flow_data import (  # noqa: E402
    fetch_funding_rates, fetch_klines, funding_features, taker_flow_features,
)
from strategies.gbm_strategy import make_lgbm_classifier  # noqa: E402

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "LTCUSDT", "DOGEUSDT", "SOLUSDT"]
DAYS = 1500
MIN_BARS = 1100
TRAIN = 400
RETRAIN = 20
FEE = 0.001
LONG_TH, SHORT_TH = 0.55, 0.45
BLOCK = 10
N_BOOT = 2000
K_DIRECTIONAL = 7
LEVEL_ADJ = 1 - 0.05 / K_DIRECTIONAL          # Bonferroni: 99.29% interval

# ------------------------------------------------------------------------- data

def load_universe(symbols=SYMBOLS, days=DAYS, need_funding=True) -> dict:
    """{symbol: {"klines": df, "funding": series}} -- symbols with too little history are dropped and reported."""
    uni, dropped = {}, {}
    for s in symbols:
        kl = fetch_klines(s, days=days)
        if kl is None or len(kl) < MIN_BARS:
            dropped[s] = 0 if kl is None else len(kl)
            continue
        uni[s] = {"klines": kl, "funding": fetch_funding_rates(s, days=days) if need_funding else pd.Series(dtype=float)}
    if dropped:
        print(f"  dropped (< {MIN_BARS} bars): {dropped}")
    return uni


def _panel_pieces(kl: pd.DataFrame, funding: pd.Series, extra: str, horizon: int) -> tuple:
    feats = build_features(kl[["Open", "High", "Low", "Close", "Volume"]])
    if extra in ("taker", "both"):
        feats = feats.join(taker_flow_features(kl))
    if extra in ("funding", "both"):
        feats = feats.join(funding_features(funding, kl.index))
    return feats, make_target(kl, horizon=horizon), make_target(kl, horizon=horizon, kind="return")


def _stack(parts: dict) -> pd.Series | pd.DataFrame:
    frames = []
    for sym, obj in parts.items():
        o = obj.copy()
        o.index = pd.MultiIndex.from_arrays([[sym] * len(o), o.index], names=["symbol", "timestamp"])
        frames.append(o)
    return pd.concat(frames)


def build_panel(uni: dict, horizon: int, extra: str = "none") -> tuple:
    feats, ys, rets = {}, {}, {}
    for sym, d in uni.items():
        feats[sym], ys[sym], rets[sym] = _panel_pieces(d["klines"], d["funding"], extra, horizon)
    return _stack(feats), _stack(ys), _stack(rets)

# ------------------------------------------------------------------- evaluation

def economic_check(p: pd.Series, fwd_ret: pd.Series, horizon: int) -> dict:
    """Equal-weight long/short/flat rule vs equal-weight buy-and-hold, on the SAME out-of-sample bars.

    For horizon h > 1 only every h-th date is a decision point, so overlapping returns are not
    double counted; fees are charged on every change of position between decision points.
    """
    df = pd.concat([p.rename("p"), fwd_ret.rename("r")], axis=1).dropna().reset_index()
    dates = np.sort(df["timestamp"].unique())
    keep = set(dates[::horizon])
    df = df[df["timestamp"].isin(keep)]
    pos = pd.Series(0.0, index=df.index)
    pos[df["p"] >= LONG_TH] = 1.0
    pos[df["p"] <= SHORT_TH] = -1.0
    df = df.assign(pos=pos).sort_values(["symbol", "timestamp"])
    df["turn"] = df.groupby("symbol")["pos"].diff().abs().fillna(df["pos"].abs())
    df["net"] = df["pos"] * df["r"] - FEE * df["turn"]
    port = df.groupby("timestamp")["net"].mean()
    hold = df.groupby("timestamp")["r"].mean()
    per_year = 365.0 / horizon
    sharpe = lambda x: float(x.mean() / x.std() * np.sqrt(per_year)) if x.std() > 0 else float("nan")
    return {
        "rule_sharpe": sharpe(port), "hold_sharpe": sharpe(hold),
        "rule_return_pct": float((1 + port).prod() - 1) * 100, "hold_return_pct": float((1 + hold).prod() - 1) * 100,
        "decisions": int(len(port)), "turnover_per_decision": float(df["turn"].mean()),
    }


def evaluate_direction(name: str, p: pd.Series, y: pd.Series, fwd_ret: pd.Series, horizon: int,
                       n_boot: int = N_BOOT) -> dict:
    d = pd.concat([y.rename("y"), p.rename("p")], axis=1).dropna().reset_index()
    if d.empty:
        return {"name": name, "error": "no out-of-sample predictions"}
    lo, hi = block_bootstrap_auc_ci(d["y"], d["p"], d["timestamp"], block=BLOCK, n_boot=n_boot, level=LEVEL_ADJ)
    per_symbol = {s: _auc(g["y"].to_numpy(), g["p"].to_numpy()) for s, g in d.groupby("symbol")}
    median_date = d["timestamp"].sort_values().iloc[len(d) // 2]
    first, second = d[d["timestamp"] < median_date], d[d["timestamp"] >= median_date]
    res = {
        "name": name, "horizon": horizon, "n_oos": int(len(d)), "symbols": len(per_symbol),
        "auc": _auc(d["y"].to_numpy(), d["p"].to_numpy()),
        "auc_ci_bonferroni": [lo, hi], "ci_level": LEVEL_ADJ,
        "auc_per_symbol": per_symbol,
        "symbols_above_half": int(sum(v > 0.5 for v in per_symbol.values())),
        "auc_first_half": _auc(first["y"].to_numpy(), first["p"].to_numpy()),
        "auc_second_half": _auc(second["y"].to_numpy(), second["p"].to_numpy()),
        "up_rate": float(d["y"].mean()),
        "economics": economic_check(p, fwd_ret, horizon),
    }
    res["criteria"] = criteria_verdict(res)
    return res


def criteria_verdict(r: dict) -> dict:
    """The four pre-registered conditions. A directional hypothesis is a 'finding' only if all hold."""
    need_symbols = max(1, int(np.ceil(0.75 * r["symbols"])))        # 6 of 8
    c = {
        "1_ci_lower_above_half": r["auc_ci_bonferroni"][0] > 0.5,
        f"2_at_least_{need_symbols}_of_{r['symbols']}_symbols_above_half": r["symbols_above_half"] >= need_symbols,
        "3_rule_sharpe_beats_buy_hold": bool(r["economics"]["rule_sharpe"] > r["economics"]["hold_sharpe"]),
        "4_both_halves_above_half": bool(r["auc_first_half"] > 0.5 and r["auc_second_half"] > 0.5),
    }
    c["FINDING"] = all(c.values())
    return c

# ------------------------------------------------------------------ experiments

def _per_symbol_predictions(uni: dict, horizon: int) -> tuple:
    ps, ys, rets = {}, {}, {}
    for sym, d in uni.items():
        X, y, r = _panel_pieces(d["klines"], d["funding"], "none", horizon)
        wf = walk_forward_predict(make_lgbm_classifier, X, y, train_size=TRAIN, retrain_every=RETRAIN, horizon=horizon)
        ps[sym], ys[sym], rets[sym] = wf.predictions["p_up"], y, r
    return _stack(ps), _stack(ys), _stack(rets)


def run_h1(uni, horizon, name, n_boot):
    p, y, r = _per_symbol_predictions(uni, horizon)
    return evaluate_direction(name, p, y, r, horizon, n_boot)


def run_pooled(uni, horizon, extra, name, n_boot):
    X, y, r = build_panel(uni, horizon, extra)
    p = walk_forward_predict_panel(make_lgbm_classifier, X, y, train_dates=TRAIN, retrain_every=RETRAIN, horizon=horizon)
    return evaluate_direction(name, p, y, r, horizon, n_boot)


def run_h5(uni, n_boot):
    """Volatility: 'tomorrow's range is above its trailing-60-bar median' -- GBM vs the naive one-line rule."""
    feats, ys, naive = {}, {}, {}
    for sym, d in uni.items():
        kl = d["klines"]
        rng = (kl["High"] - kl["Low"]) / kl["Close"]
        med60 = rng.rolling(60).median()
        y = (rng.shift(-1) > med60).astype(float).where(rng.shift(-1).notna() & med60.notna())
        feats[sym], ys[sym], naive[sym] = build_features(kl[["Open", "High", "Low", "Close", "Volume"]]), y, rng / med60
    X, y, nv = _stack(feats), _stack(ys), _stack(naive)
    p = walk_forward_predict_panel(make_lgbm_classifier, X, y, train_dates=TRAIN, retrain_every=RETRAIN, horizon=1)
    d = pd.concat([y.rename("y"), p.rename("gbm"), nv.rename("naive")], axis=1).dropna().reset_index()
    lo, hi = block_bootstrap_auc_ci(d["y"], d["gbm"], d["timestamp"], block=BLOCK, n_boot=n_boot)
    dlo, dhi = paired_block_bootstrap_auc_diff(d["y"], d["gbm"], d["naive"], d["timestamp"], block=BLOCK, n_boot=n_boot)
    res = {
        "name": "H5 volatility", "n_oos": int(len(d)), "high_vol_rate": float(d["y"].mean()),
        "auc_gbm": _auc(d["y"].to_numpy(), d["gbm"].to_numpy()), "auc_gbm_ci95": [lo, hi],
        "auc_naive": _auc(d["y"].to_numpy(), d["naive"].to_numpy()),
        "auc_diff_gbm_minus_naive_ci95": [dlo, dhi],
    }
    res["gbm_adds_value"] = bool(dlo > 0)
    return res


def run_all(only=None, n_boot=N_BOOT, universe=None) -> dict:
    want = lambda k: only is None or k in only
    uni = universe or load_universe(need_funding=any(want(k) for k in ("H4b", "H4c")))
    results = {}
    plan = [
        ("H1a", lambda: run_h1(uni, 5, "H1a per-symbol h=5", n_boot)),
        ("H1b", lambda: run_h1(uni, 10, "H1b per-symbol h=10", n_boot)),
        ("H2a", lambda: run_pooled(uni, 1, "none", "H2a pooled h=1", n_boot)),
        ("H2b", lambda: run_pooled(uni, 5, "none", "H2b pooled h=5", n_boot)),
        ("H4a", lambda: run_pooled(uni, 1, "taker", "H4a pooled h=1 + taker flow", n_boot)),
        ("H4b", lambda: run_pooled(uni, 1, "funding", "H4b pooled h=1 + funding", n_boot)),
        ("H4c", lambda: run_pooled(uni, 1, "both", "H4c pooled h=1 + both", n_boot)),
        ("H5", lambda: run_h5(uni, n_boot)),
    ]
    for key, fn in plan:
        if want(key):
            print(f"running {key} ...", flush=True)
            results[key] = fn()
    return results


def _fmt(key: str, r: dict) -> str:
    if key == "H5":
        return (f"{key}: vol AUC gbm {r['auc_gbm']:.3f} {tuple(round(x, 3) for x in r['auc_gbm_ci95'])} | naive {r['auc_naive']:.3f} | "
                f"gbm-naive {tuple(round(x, 3) for x in r['auc_diff_gbm_minus_naive_ci95'])} -> "
                f"{'GBM ADDS VALUE' if r['gbm_adds_value'] else 'no gain over naive rule'}")
    if "error" in r:
        return f"{key}: {r['error']}"
    e, c = r["economics"], r["criteria"]
    return (f"{key}: AUC {r['auc']:.3f} CI99.3 [{r['auc_ci_bonferroni'][0]:.3f},{r['auc_ci_bonferroni'][1]:.3f}] "
            f"| {r['symbols_above_half']}/{r['symbols']} syms>0.5 | halves {r['auc_first_half']:.3f}/{r['auc_second_half']:.3f} "
            f"| Sharpe rule {e['rule_sharpe']:.2f} vs hold {e['hold_sharpe']:.2f} | "
            f"{'*** FINDING ***' if c['FINDING'] else 'no evidence'}")


def log_results(results: dict, n_boot: int) -> list:
    """Record every experiment as one run in the local experiment log. Never raises."""
    ids = []
    try:
        log = ExperimentLog()
        protocol = {"days": DAYS, "train": TRAIN, "retrain_every": RETRAIN, "fee": FEE, "block": BLOCK,
                    "n_boot": n_boot, "ci_level_directional": LEVEL_ADJ, "symbols": SYMBOLS}
        for key, r in results.items():
            metrics = {k: v for k, v in r.items() if k not in ("name", "auc_per_symbol")}
            metrics["FINDING"] = bool(r.get("criteria", {}).get("FINDING", r.get("gbm_adds_value", False)))
            ids.append(log.log_run(
                name=r.get("name", key), model_type="lightgbm", params={**protocol, "experiment": key,
                                                                         "horizon": r.get("horizon", 1)},
                metrics=metrics, dataset={"symbols": SYMBOLS, "days": DAYS, "source": "binance spot 1d"},
                tags=["phase-6.5", key], notes="pre-registered: docs/PHASE_6_5_PREREGISTRATION.md"))
    except Exception as exc:  # noqa: BLE001
        print(f"(experiment log unavailable: {exc})")
    return ids


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="comma list, e.g. H2a,H5")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out", default=os.path.join("training_ground", "results", "phase_6_5.json"))
    ap.add_argument("--no-log", action="store_true", help="do not record runs in the experiment log")
    args = ap.parse_args(argv)
    only = set(args.only.split(",")) if args.only else None
    results = run_all(only, args.n_boot)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"run_at": datetime.now(timezone.utc).isoformat(), "symbols": SYMBOLS, "protocol": {
            "days": DAYS, "train": TRAIN, "retrain_every": RETRAIN, "fee": FEE, "ci_level_directional": LEVEL_ADJ,
            "block": BLOCK, "n_boot": args.n_boot}, "results": results}, fh, indent=2, default=float)
    if not args.no_log:
        log_results(results, args.n_boot)
    print("\n=== PHASE 6.5 RESULTS (pre-registered protocol) ===")
    for k, r in results.items():
        print(_fmt(k, r))
    findings = [k for k, r in results.items() if k != "H5" and r.get("criteria", {}).get("FINDING")]
    print("\nFINDINGS:", ", ".join(findings) if findings else "none -- no directional hypothesis met all four criteria.")
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
