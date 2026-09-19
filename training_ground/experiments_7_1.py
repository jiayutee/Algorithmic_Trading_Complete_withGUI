"""Phase 7.1 -- risk-based allocation vs equal weight. Implements docs/PHASE_7_1_PREREGISTRATION.md.

    python training_ground/experiments_7_1.py
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
from core.ml_validation import paired_block_bootstrap_stat_diff  # noqa: E402
from core.portfolio_optimizer import weights as optimal_weights  # noqa: E402
from core.risk_sizing import perf_stats, sharpe  # noqa: E402
from training_ground.experiments_6_5 import SYMBOLS, DAYS, BLOCK, load_universe  # noqa: E402

LOOKBACK = 250
REBALANCE_EVERY = 30
FEE = 0.001
METHODS = ["equal", "inverse_vol", "min_variance", "risk_parity", "hrp", "max_sharpe"]
CHALLENGERS = [m for m in METHODS if m != "equal"]
LEVEL = 1 - 0.05 / len(CHALLENGERS)                  # 99%


def simulate(returns: pd.DataFrame, method: str, lookback: int = LOOKBACK, every: int = REBALANCE_EVERY,
             fee: float = FEE) -> tuple:
    """Daily net portfolio returns for one allocator (weights drift between rebalances) plus diagnostics."""
    idx = returns.index
    w = pd.Series(0.0, index=returns.columns)
    out, turnovers, eff = [], [], []
    for i in range(lookback, len(returns)):
        cost = 0.0
        if (i - lookback) % every == 0:                                   # rebalance at the close of day i-1
            target = optimal_weights(returns.iloc[i - lookback:i], method)   # trailing window only: rows < i
            cost = fee * float((target - w).abs().sum())
            turnovers.append(float((target - w).abs().sum()))
            eff.append(float(1.0 / (target ** 2).sum()))
            w = target
        r = returns.iloc[i]
        day = float((w * r).sum()) - cost
        out.append((idx[i], day))
        w = w * (1 + r)
        w = w / w.sum()                                                    # drift
    s = pd.Series(dict(out))
    return s, {"rebalances": len(turnovers), "avg_turnover_per_rebalance": float(np.mean(turnovers)),
               "annual_turnover": float(np.sum(turnovers) / (len(s) / 365.0)), "avg_effective_assets": float(np.mean(eff))}


def run(uni: dict | None = None, n_boot: int = 2000, lookback: int = LOOKBACK, every: int = REBALANCE_EVERY) -> dict:
    uni = uni or load_universe(need_funding=False)
    closes = pd.DataFrame({s: d["klines"]["Close"] for s, d in uni.items()}).dropna()
    returns = closes.pct_change().dropna()
    series, diag = {}, {}
    for m in METHODS:
        series[m], diag[m] = simulate(returns, m, lookback, every)
    stats = {m: {**perf_stats(series[m]), **diag[m]} for m in METHODS}
    dates = series["equal"].index.to_numpy()
    half = len(dates) // 2
    comps = []
    for m in CHALLENGERS:
        lo, hi = paired_block_bootstrap_stat_diff(series[m].to_numpy(), series["equal"].to_numpy(), dates, sharpe,
                                                  block=BLOCK, n_boot=n_boot, level=LEVEL)
        halves = [sharpe(series[m].iloc[:half].to_numpy()) - sharpe(series["equal"].iloc[:half].to_numpy()),
                  sharpe(series[m].iloc[half:].to_numpy()) - sharpe(series["equal"].iloc[half:].to_numpy())]
        crit = {"1_ci_lower_above_zero": bool(lo > 0), "2_positive_in_both_halves": bool(all(h > 0 for h in halves)),
                "3_max_drawdown_not_worse": bool(stats[m]["max_drawdown"] >= stats["equal"]["max_drawdown"])}
        crit["FINDING"] = all(crit.values())
        comps.append({"method": m, "sharpe_diff": sharpe(series[m].to_numpy()) - sharpe(series["equal"].to_numpy()),
                      "ci": [lo, hi], "ci_level": LEVEL, "sharpe_diff_halves": halves, "criteria": crit})
    return {"window": [str(series["equal"].index[0].date()), str(series["equal"].index[-1].date())],
            "days": int(len(series["equal"])), "symbols": list(closes.columns), "stats": stats, "comparisons": comps,
            "protocol": {"lookback": lookback, "rebalance_every": every, "fee": FEE, "block": BLOCK, "n_boot": n_boot,
                         "ci_level": LEVEL}}


def _fmt(res: dict) -> str:
    lines = [f"window {res['window'][0]} -> {res['window'][1]} ({res['days']} days, {len(res['symbols'])} assets)",
             f"{'method':<14}{'Sharpe':>8}{'AnnRet%':>9}{'AnnVol%':>9}{'MaxDD%':>9}{'EffN':>7}{'Turn/yr':>9}"]
    for m, s in res["stats"].items():
        lines.append(f"{m:<14}{s['sharpe']:>8.2f}{s['ann_return']*100:>9.1f}{s['ann_vol']*100:>9.1f}{s['max_drawdown']*100:>9.1f}"
                     f"{s['avg_effective_assets']:>7.1f}{s['annual_turnover']:>9.1f}")
    lines.append("")
    for c in res["comparisons"]:
        lines.append(f"{c['method']:<13} vs equal: Sharpe diff {c['sharpe_diff']:+.3f}  CI{c['ci_level']*100:.0f} "
                     f"[{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}]  halves {c['sharpe_diff_halves'][0]:+.2f}/{c['sharpe_diff_halves'][1]:+.2f}"
                     f"  -> {'*** FINDING ***' if c['criteria']['FINDING'] else 'no evidence'}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join("training_ground", "results", "phase_7_1.json"))
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args(argv)
    res = run(n_boot=args.n_boot)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"run_at": datetime.now(timezone.utc).isoformat(), **res}, fh, indent=2, default=float)
    if not args.no_log:
        try:
            log = ExperimentLog()
            for c in res["comparisons"]:
                log.log_run(name=f"7.1 {c['method']} vs equal", model_type="allocation", params=res["protocol"],
                            metrics={"sharpe_diff": c["sharpe_diff"], "ci_low": c["ci"][0], "ci_high": c["ci"][1],
                                     "FINDING": c["criteria"]["FINDING"], **{f"criteria.{k}": v for k, v in c["criteria"].items()}},
                            dataset={"symbols": res["symbols"], "window": res["window"], "days": res["days"]},
                            tags=["phase-7.1"], notes="pre-registered: docs/PHASE_7_1_PREREGISTRATION.md")
        except Exception as exc:  # noqa: BLE001
            print(f"(experiment log unavailable: {exc})")
    print("\n=== PHASE 7.1 RESULTS (pre-registered protocol) ===")
    print(_fmt(res))
    found = [c["method"] for c in res["comparisons"] if c["criteria"]["FINDING"]]
    print("\nFINDINGS:", ", ".join(found) if found else "none.")
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
