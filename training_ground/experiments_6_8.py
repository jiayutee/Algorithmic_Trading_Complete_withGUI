"""Phase 6.8 -- trend-filter drawdown reduction on held-out symbols. Implements docs/PHASE_6_8_PREREGISTRATION.md.

    python training_ground/experiments_6_8.py
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
from training_ground.experiments_6_5 import load_universe  # noqa: E402
from training_ground.experiments_6_7 import simulate  # noqa: E402

SYMBOLS = ["LINKUSDT", "DOTUSDT", "AVAXUSDT", "ATOMUSDT", "TRXUSDT", "NEARUSDT", "UNIUSDT", "FILUSDT"]
MIN_SYMBOLS = 5
BLOCK = 30
K = 2
LEVEL = 1 - 0.05 / K
SHARPE_NONINFERIORITY = -0.50


def max_drawdown(r) -> float:
    eq = np.cumprod(1 + np.asarray(r, float))
    return float(np.min(eq / np.maximum.accumulate(eq) - 1))


def run(returns: pd.DataFrame | None = None, n_boot: int = 2000) -> dict:
    if returns is None:
        uni = load_universe(symbols=SYMBOLS, need_funding=False)
        if len(uni) < MIN_SYMBOLS:
            return {"void": True, "reason": f"only {len(uni)} symbols had enough history (need {MIN_SYMBOLS})", "symbols": list(uni)}
        returns = pd.DataFrame({s: d["klines"]["Close"] for s, d in uni.items()}).dropna().pct_change().dropna()
    ts, _ = simulate(returns, "tsmom")
    ew, _ = simulate(returns, "ew")
    a, b, dates = ts.to_numpy(), ew.reindex(ts.index).to_numpy(), ts.index.to_numpy()
    dd_lo, dd_hi = paired_block_bootstrap_stat_diff(a, b, dates, max_drawdown, block=BLOCK, n_boot=n_boot, level=LEVEL)
    sh_lo, sh_hi = paired_block_bootstrap_stat_diff(a, b, dates, sharpe, block=BLOCK, n_boot=n_boot, level=LEVEL)
    half = len(a) // 2
    dd_halves = [max_drawdown(a[:half]) - max_drawdown(b[:half]), max_drawdown(a[half:]) - max_drawdown(b[half:])]
    crit = {"1_D1_ci_lower_above_zero": bool(dd_lo > 0), "2_D1_shallower_in_both_halves": bool(all(d > 0 for d in dd_halves)),
            "3_D2_sharpe_not_materially_worse": bool(sh_lo > SHARPE_NONINFERIORITY)}
    crit["CONFIRMED"] = all(crit.values())
    return {"void": False, "window": [str(ts.index[0].date()), str(ts.index[-1].date())], "days": int(len(ts)),
            "symbols": list(returns.columns),
            "stats": {"tsmom": perf_stats(ts), "ew": perf_stats(pd.Series(b, index=ts.index))},
            "D1_maxdd_diff": {"value": max_drawdown(a) - max_drawdown(b), "ci": [dd_lo, dd_hi], "halves": dd_halves},
            "D2_sharpe_diff": {"value": sharpe(a) - sharpe(b), "ci": [sh_lo, sh_hi]},
            "criteria": crit, "protocol": {"block": BLOCK, "level": LEVEL, "n_boot": n_boot, "noninferiority": SHARPE_NONINFERIORITY}}


def _fmt(res: dict) -> str:
    if res.get("void"):
        return f"VOID: {res['reason']}"
    s = res["stats"]
    lines = [f"window {res['window'][0]} -> {res['window'][1]} ({res['days']} days, {len(res['symbols'])} symbols: {', '.join(res['symbols'])})",
             f"{'':8}{'Sharpe':>8}{'AnnRet%':>9}{'AnnVol%':>9}{'MaxDD%':>9}"]
    for k in ("ew", "tsmom"):
        lines.append(f"{k:<8}{s[k]['sharpe']:>8.2f}{s[k]['ann_return']*100:>9.1f}{s[k]['ann_vol']*100:>9.1f}{s[k]['max_drawdown']*100:>9.1f}")
    d1, d2 = res["D1_maxdd_diff"], res["D2_sharpe_diff"]
    lines += ["", f"D1 maxDD(TSMOM)-maxDD(EW) {d1['value']*100:+.1f}pp  CI97.5 [{d1['ci'][0]*100:+.1f},{d1['ci'][1]*100:+.1f}]pp  halves {d1['halves'][0]*100:+.1f}/{d1['halves'][1]*100:+.1f}pp",
              f"D2 Sharpe(TSMOM)-Sharpe(EW) {d2['value']:+.3f}  CI97.5 [{d2['ci'][0]:+.3f},{d2['ci'][1]:+.3f}]  (non-inferiority bound {SHARPE_NONINFERIORITY})",
              "criteria: " + json.dumps(res["criteria"])]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join("training_ground", "results", "phase_6_8.json"))
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args(argv)
    res = run(n_boot=args.n_boot)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"run_at": datetime.now(timezone.utc).isoformat(), **res}, fh, indent=2, default=float)
    if not args.no_log and not res.get("void"):
        try:
            ExperimentLog().log_run(name="6.8 TSMOM drawdown reduction (held-out symbols)", model_type="momentum", params=res["protocol"],
                                    metrics={"maxdd_diff": res["D1_maxdd_diff"]["value"], "sharpe_diff": res["D2_sharpe_diff"]["value"],
                                             "CONFIRMED": res["criteria"]["CONFIRMED"]},
                                    dataset={"symbols": res["symbols"], "window": res["window"], "days": res["days"]},
                                    tags=["phase-6.8"], notes="pre-registered: docs/PHASE_6_8_PREREGISTRATION.md")
        except Exception as exc:  # noqa: BLE001
            print(f"(experiment log unavailable: {exc})")
    print("=== PHASE 6.8 RESULTS (pre-registered protocol) ===")
    print(_fmt(res))
    if not res.get("void"):
        print("\nDRAWDOWN REDUCTION:", "CONFIRMED" if res["criteria"]["CONFIRMED"] else "not confirmed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
