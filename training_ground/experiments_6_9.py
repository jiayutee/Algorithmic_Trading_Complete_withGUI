"""Phase 6.9 -- trend-filter drawdown reduction in an earlier, disjoint period. Implements docs/PHASE_6_9_PREREGISTRATION.md.

    python training_ground/experiments_6_9.py
Reuses the Phase 6.8 test unchanged; only the universe and the (earlier) window differ.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.experiment_log import ExperimentLog  # noqa: E402
from training_ground import experiments_6_8 as e68  # noqa: E402
from training_ground.experiments_6_5 import load_universe  # noqa: E402

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "LTCUSDT", "LINKUSDT", "TRXUSDT"]
CUTOFF = pd.Timestamp("2022-09-09")
DAYS = 3000                                  # reaches back to ~2018
MIN_SYMBOLS = 6
MIN_DAYS = 800


def load_returns() -> tuple:
    uni = load_universe(symbols=SYMBOLS, days=DAYS, need_funding=False)
    closes = pd.DataFrame({s: d["klines"]["Close"] for s, d in uni.items()})
    closes = closes[closes.index <= CUTOFF]
    return closes, uni


def run(closes: pd.DataFrame | None = None, n_boot: int = 2000) -> dict:
    if closes is None:
        closes, _ = load_returns()
    closes = closes[closes.index <= CUTOFF].dropna(how="any")
    if closes.shape[1] < MIN_SYMBOLS or len(closes) < MIN_DAYS:
        return {"void": True, "reason": f"{closes.shape[1]} symbols x {len(closes)} common days (need {MIN_SYMBOLS} x {MIN_DAYS})",
                "symbols": list(closes.columns)}
    res = e68.run(closes.pct_change().dropna(), n_boot=n_boot)
    res["cutoff"] = str(CUTOFF.date())
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join("training_ground", "results", "phase_6_9.json"))
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args(argv)
    res = run(n_boot=args.n_boot)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"run_at": datetime.now(timezone.utc).isoformat(), **res}, fh, indent=2, default=float)
    if not args.no_log and not res.get("void"):
        try:
            ExperimentLog().log_run(name="6.9 TSMOM drawdown reduction (earlier period)", model_type="momentum", params=res["protocol"],
                                    metrics={"maxdd_diff": res["D1_maxdd_diff"]["value"], "sharpe_diff": res["D2_sharpe_diff"]["value"],
                                             "CONFIRMED": res["criteria"]["CONFIRMED"]},
                                    dataset={"symbols": res["symbols"], "window": res["window"], "days": res["days"]},
                                    tags=["phase-6.9"], notes="pre-registered: docs/PHASE_6_9_PREREGISTRATION.md")
        except Exception as exc:  # noqa: BLE001
            print(f"(experiment log unavailable: {exc})")
    print("=== PHASE 6.9 RESULTS (pre-registered protocol) ===")
    print(e68._fmt(res))
    if not res.get("void"):
        print("\nDRAWDOWN REDUCTION:", "CONFIRMED" if res["criteria"]["CONFIRMED"] else "not confirmed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
