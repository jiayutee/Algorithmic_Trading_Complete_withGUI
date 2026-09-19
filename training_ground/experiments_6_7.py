"""Phase 6.7 -- crypto momentum. Implements docs/PHASE_6_7_PREREGISTRATION.md exactly.

    python training_ground/experiments_6_7.py
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
from core.risk_sizing import perf_stats, sharpe  # noqa: E402
from training_ground.experiments_6_5 import BLOCK, load_universe  # noqa: E402

LOOKBACK, TOP_K, REBALANCE, FEE = 28, 3, 7, 0.001
K = 2
LEVEL = 1 - 0.05 / K
DD_TOLERANCE = 0.05


def target_weights(hist: pd.DataFrame, mode: str, top_k: int = TOP_K) -> pd.Series:
    """Target weights from trailing returns ``hist`` (rows strictly before the day they are applied to)."""
    n = hist.shape[1]
    trailing = (1 + hist).prod() - 1
    if mode == "ew":
        return pd.Series(1.0 / n, index=hist.columns)
    if mode == "xsmom":
        w = pd.Series(0.0, index=hist.columns)
        w[trailing.sort_values(ascending=False, kind="mergesort").index[:top_k]] = 1.0 / top_k
        return w
    if mode == "tsmom":
        return (trailing > 0).astype(float) / n
    raise ValueError(mode)


def simulate(returns: pd.DataFrame, mode: str, lookback: int = LOOKBACK, every: int = REBALANCE, fee: float = FEE,
             top_k: int = TOP_K) -> tuple:
    w = pd.Series(0.0, index=returns.columns)
    out, turns = {}, []
    for i in range(lookback, len(returns)):
        cost = 0.0
        if (i - lookback) % every == 0:
            target = target_weights(returns.iloc[i - lookback:i], mode, top_k)
            t = float((target - w).abs().sum())
            cost, w = fee * t, target
            turns.append(t)
        r = returns.iloc[i]
        day = float((w * r).sum()) - cost
        out[returns.index[i]] = day
        w = w * (1 + r) / (1 + day + cost) if (1 + day + cost) != 0 else w      # drift; the remainder is cash
    s = pd.Series(out)
    return s, {"rebalances": len(turns), "annual_turnover": float(np.sum(turns) / (len(s) / 365.0))}


def compare(name: str, a: pd.Series, b: pd.Series, n_boot: int) -> dict:
    lo, hi = paired_block_bootstrap_stat_diff(a.to_numpy(), b.to_numpy(), a.index.to_numpy(), sharpe, block=BLOCK,
                                              n_boot=n_boot, level=LEVEL)
    half = len(a) // 2
    halves = [sharpe(a.iloc[:half].to_numpy()) - sharpe(b.iloc[:half].to_numpy()),
              sharpe(a.iloc[half:].to_numpy()) - sharpe(b.iloc[half:].to_numpy())]
    sa, sb = perf_stats(a), perf_stats(b)
    crit = {"1_ci_lower_above_zero": bool(lo > 0), "2_positive_in_both_halves": bool(all(h > 0 for h in halves)),
            "3_drawdown_within_5pp": bool(sa["max_drawdown"] >= sb["max_drawdown"] - DD_TOLERANCE)}
    crit["FINDING"] = all(crit.values())
    return {"comparison": name, "sharpe_diff": sharpe(a.to_numpy()) - sharpe(b.to_numpy()), "ci": [lo, hi], "ci_level": LEVEL,
            "sharpe_diff_halves": halves, "criteria": crit}


def run(returns: pd.DataFrame | None = None, n_boot: int = 2000) -> dict:
    if returns is None:
        uni = load_universe(need_funding=False)
        returns = pd.DataFrame({s: d["klines"]["Close"] for s, d in uni.items()}).dropna().pct_change().dropna()
    sims = {m: simulate(returns, m) for m in ("ew", "xsmom", "tsmom")}
    series = {m: s for m, (s, _) in sims.items()}
    stats = {m: {**perf_stats(series[m]), **sims[m][1]} for m in series}
    comps = [compare("M1 XSMOM vs equal weight", series["xsmom"], series["ew"], n_boot),
             compare("M2 TSMOM vs equal weight", series["tsmom"], series["ew"], n_boot)]
    context = {}                                       # untested sensitivity, NOT used to choose anything
    for L in (14, 56):
        ctx = {m: simulate(returns, m, lookback=L)[0] for m in ("ew", "xsmom", "tsmom")}
        idx = ctx["ew"].index
        context[f"L={L}"] = {m: float(sharpe(ctx[m].reindex(idx).to_numpy())) for m in ctx}
    return {"window": [str(series["ew"].index[0].date()), str(series["ew"].index[-1].date())], "days": int(len(series["ew"])),
            "symbols": list(returns.columns), "stats": stats, "comparisons": comps, "context_sharpe_other_lookbacks": context,
            "protocol": {"lookback": LOOKBACK, "top_k": TOP_K, "rebalance": REBALANCE, "fee": FEE, "block": BLOCK,
                         "n_boot": n_boot, "ci_level": LEVEL}}


def _fmt(res: dict) -> str:
    lines = [f"window {res['window'][0]} -> {res['window'][1]} ({res['days']} days, {len(res['symbols'])} symbols)",
             f"{'strategy':<8}{'Sharpe':>8}{'AnnRet%':>9}{'AnnVol%':>9}{'MaxDD%':>9}{'Turn/yr':>9}"]
    for k, s in res["stats"].items():
        lines.append(f"{k:<8}{s['sharpe']:>8.2f}{s['ann_return']*100:>9.1f}{s['ann_vol']*100:>9.1f}{s['max_drawdown']*100:>9.1f}{s['annual_turnover']:>9.1f}")
    lines.append("")
    for c in res["comparisons"]:
        lines.append(f"{c['comparison']:<28} Sharpe diff {c['sharpe_diff']:+.3f}  CI{c['ci_level']*100:.1f} [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}]"
                     f"  halves {c['sharpe_diff_halves'][0]:+.2f}/{c['sharpe_diff_halves'][1]:+.2f}  -> "
                     f"{'*** FINDING ***' if c['criteria']['FINDING'] else 'no evidence'}")
    lines.append("context (untested) Sharpe at other lookbacks: " + json.dumps(res["context_sharpe_other_lookbacks"]))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join("training_ground", "results", "phase_6_7.json"))
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
                log.log_run(name=c["comparison"], model_type="momentum", params=res["protocol"],
                            metrics={"sharpe_diff": c["sharpe_diff"], "ci_low": c["ci"][0], "ci_high": c["ci"][1],
                                     "FINDING": c["criteria"]["FINDING"]},
                            dataset={"symbols": res["symbols"], "window": res["window"], "days": res["days"]},
                            tags=["phase-6.7"], notes="pre-registered: docs/PHASE_6_7_PREREGISTRATION.md")
        except Exception as exc:  # noqa: BLE001
            print(f"(experiment log unavailable: {exc})")
    print("=== PHASE 6.7 RESULTS (pre-registered protocol) ===")
    print(_fmt(res))
    findings = [c["comparison"] for c in res["comparisons"] if c["criteria"]["FINDING"]]
    print("\nFINDINGS:", "; ".join(findings) if findings else "none.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
