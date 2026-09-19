"""Phase 6.6 -- volatility-targeted sizing. Implements docs/PHASE_6_6_PREREGISTRATION.md exactly.

    python training_ground/experiments_6_6.py

Compares, on identical out-of-sample dates and with costs:
    NAIVE-VT   exposure from an EWMA volatility forecast
    GBM-VT     exposure from a LightGBM volatility forecast (walk-forward, pooled)
    FIXED      constant exposure matched to each variant's own average exposure
Nothing is tuned; see the pre-registration for the pass/fail rules.
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
from core.feature_engineering import build_features  # noqa: E402
from core.ml_validation import paired_block_bootstrap_stat_diff, walk_forward_predict_panel  # noqa: E402
from core.risk_sizing import (  # noqa: E402
    apply_no_trade_band, log_range, naive_vol_forecast, perf_stats, portfolio_returns, sharpe,
    vol_target_exposure,
)
from strategies.gbm_strategy import DEFAULT_LGBM_PARAMS  # noqa: E402
from training_ground.experiments_6_5 import SYMBOLS, DAYS, TRAIN, RETRAIN, FEE, BLOCK, load_universe, _stack  # noqa: E402

K_EXPOSURE = 0.7
CAP = 1.0
BAND = 0.10
MIN_HISTORY = 30
N_COMPARISONS = 3
LEVEL = 1 - 0.05 / N_COMPARISONS        # 98.3% intervals


def make_lgbm_regressor():
    import lightgbm as lgb
    return lgb.LGBMRegressor(**DEFAULT_LGBM_PARAMS)


def _unstack(s: pd.Series) -> pd.DataFrame:
    return s.unstack(level="symbol").sort_index()


def build_inputs(uni: dict) -> dict:
    """Per-symbol features, next-day log-range target, naive forecast, and next-day returns (dates x symbols)."""
    feats, ys, naive, fwd = {}, {}, {}, {}
    for sym, d in uni.items():
        kl = d["klines"]
        lr = log_range(kl)
        feats[sym] = build_features(kl[["Open", "High", "Low", "Close", "Volume"]])
        ys[sym] = np.log(lr.shift(-1))                                   # target: log of TOMORROW's range
        naive[sym] = naive_vol_forecast(lr)                              # uses bars <= t
        fwd[sym] = kl["Close"].shift(-1) / kl["Close"] - 1               # return earned holding from close t to t+1
    return {"X": _stack(feats), "y": _stack(ys), "naive": pd.DataFrame(naive), "fwd": pd.DataFrame(fwd)}


def exposures_from_sigma(sigma: pd.DataFrame) -> pd.DataFrame:
    """Identical rule for every forecast: expanding-median-relative exposure, then the no-trade band."""
    out = {}
    for col in sigma.columns:
        out[col] = apply_no_trade_band(vol_target_exposure(sigma[col], k=K_EXPOSURE, cap=CAP, min_history=MIN_HISTORY), BAND)
    return pd.DataFrame(out)


def _stats_block(ret: pd.Series, exposure: pd.DataFrame) -> dict:
    s = perf_stats(ret)
    s["avg_exposure"] = float(exposure.mean().mean())
    s["annual_turnover"] = float(exposure.diff().abs().mean(axis=1).sum() / (len(exposure) / 365.0))
    return s


def compare(name: str, a: pd.Series, b: pd.Series, a_stats: dict, b_stats: dict, n_boot: int) -> dict:
    """Sharpe(a) - Sharpe(b) with a paired block-bootstrap interval, plus the pre-registered criteria."""
    dates = a.index.to_numpy()
    lo, hi = paired_block_bootstrap_stat_diff(a.to_numpy(), b.to_numpy(), dates, sharpe, block=BLOCK,
                                              n_boot=n_boot, level=LEVEL)
    half = len(a) // 2
    diff_halves = [sharpe(a.iloc[:half].to_numpy()) - sharpe(b.iloc[:half].to_numpy()),
                   sharpe(a.iloc[half:].to_numpy()) - sharpe(b.iloc[half:].to_numpy())]
    crit = {
        "1_ci_lower_above_zero": bool(lo > 0),
        "2_positive_in_both_halves": bool(all(d > 0 for d in diff_halves)),
        "3_max_drawdown_not_worse": bool(a_stats["max_drawdown"] >= b_stats["max_drawdown"]),
    }
    crit["FINDING"] = all(crit.values())
    return {"comparison": name, "sharpe_diff": sharpe(a.to_numpy()) - sharpe(b.to_numpy()), "ci": [lo, hi],
            "ci_level": LEVEL, "sharpe_diff_halves": diff_halves, "criteria": crit}


def run(uni: dict | None = None, n_boot: int = 2000) -> dict:
    uni = uni or load_universe(need_funding=False)
    inp = build_inputs(uni)

    print(f"Walk-forward GBM volatility regressor on {len(uni)} symbols ...", flush=True)
    pred = walk_forward_predict_panel(make_lgbm_regressor, inp["X"], inp["y"], train_dates=TRAIN,
                                      retrain_every=RETRAIN, horizon=1, task="regress")
    gbm_sigma = np.exp(_unstack(pred))
    gbm_sigma = gbm_sigma.dropna(how="any")                              # window where every symbol has a forecast
    start = gbm_sigma.index[0]
    naive_sigma = inp["naive"].reindex(gbm_sigma.index)
    fwd = inp["fwd"].reindex(gbm_sigma.index)

    exp_gbm = exposures_from_sigma(gbm_sigma)
    exp_naive = exposures_from_sigma(naive_sigma)
    valid = exp_gbm.dropna().index.intersection(exp_naive.dropna().index).intersection(fwd.dropna().index)
    exp_gbm, exp_naive, fwd = exp_gbm.loc[valid], exp_naive.loc[valid], fwd.loc[valid]

    fixed_gbm = pd.DataFrame(float(exp_gbm.mean().mean()), index=valid, columns=exp_gbm.columns)
    fixed_naive = pd.DataFrame(float(exp_naive.mean().mean()), index=valid, columns=exp_naive.columns)
    hold = pd.DataFrame(1.0, index=valid, columns=exp_gbm.columns)

    rets = {"GBM-VT": portfolio_returns(exp_gbm, fwd, FEE), "NAIVE-VT": portfolio_returns(exp_naive, fwd, FEE),
            "FIXED (matched to GBM-VT)": portfolio_returns(fixed_gbm, fwd, FEE),
            "FIXED (matched to NAIVE-VT)": portfolio_returns(fixed_naive, fwd, FEE),
            "BUY&HOLD (100%)": portfolio_returns(hold, fwd, FEE)}
    exposures = {"GBM-VT": exp_gbm, "NAIVE-VT": exp_naive, "FIXED (matched to GBM-VT)": fixed_gbm,
                 "FIXED (matched to NAIVE-VT)": fixed_naive, "BUY&HOLD (100%)": hold}
    common = rets["GBM-VT"].index
    for k in rets:
        rets[k] = rets[k].reindex(common).dropna()
    common = rets["GBM-VT"].index
    for k in rets:
        rets[k] = rets[k].loc[common]
    stats = {k: _stats_block(rets[k], exposures[k].loc[common]) for k in rets}

    comparisons = [
        compare("C1 NAIVE-VT vs FIXED matched", rets["NAIVE-VT"], rets["FIXED (matched to NAIVE-VT)"],
                stats["NAIVE-VT"], stats["FIXED (matched to NAIVE-VT)"], n_boot),
        compare("C2 GBM-VT vs FIXED matched", rets["GBM-VT"], rets["FIXED (matched to GBM-VT)"],
                stats["GBM-VT"], stats["FIXED (matched to GBM-VT)"], n_boot),
        compare("C3 GBM-VT vs NAIVE-VT", rets["GBM-VT"], rets["NAIVE-VT"], stats["GBM-VT"], stats["NAIVE-VT"], n_boot),
    ]
    forecast_corr = float(np.corrcoef(np.log(gbm_sigma.loc[common].to_numpy().ravel()),
                                      np.log(naive_sigma.loc[common].to_numpy().ravel()))[0, 1])
    return {"window": [str(common[0].date()), str(common[-1].date())], "days": int(len(common)),
            "symbols": list(uni), "stats": stats, "comparisons": comparisons,
            "gbm_naive_forecast_log_correlation": forecast_corr,
            "protocol": {"k": K_EXPOSURE, "cap": CAP, "band": BAND, "fee": FEE, "train": TRAIN, "retrain": RETRAIN,
                         "block": BLOCK, "n_boot": n_boot, "ci_level": LEVEL}}


def _fmt(res: dict) -> str:
    lines = [f"window {res['window'][0]} -> {res['window'][1]} ({res['days']} days, {len(res['symbols'])} symbols)",
             f"{'strategy':<30}{'Sharpe':>8}{'AnnRet%':>9}{'AnnVol%':>9}{'MaxDD%':>9}{'Calmar':>8}{'AvgExp':>8}{'Turn/yr':>9}"]
    for k, s in res["stats"].items():
        lines.append(f"{k:<30}{s['sharpe']:>8.2f}{s['ann_return']*100:>9.1f}{s['ann_vol']*100:>9.1f}"
                     f"{s['max_drawdown']*100:>9.1f}{s['calmar']:>8.2f}{s['avg_exposure']:>8.2f}{s['annual_turnover']:>9.1f}")
    lines.append("")
    for c in res["comparisons"]:
        lines.append(f"{c['comparison']:<32} Sharpe diff {c['sharpe_diff']:+.3f}  CI{c['ci_level']*100:.1f} "
                     f"[{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}]  halves {c['sharpe_diff_halves'][0]:+.2f}/{c['sharpe_diff_halves'][1]:+.2f}"
                     f"  -> {'*** FINDING ***' if c['criteria']['FINDING'] else 'no evidence'}")
    lines.append(f"forecast agreement (corr of log GBM vs log naive vol): {res['gbm_naive_forecast_log_correlation']:.2f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join("training_ground", "results", "phase_6_6.json"))
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
                log.log_run(name=c["comparison"], model_type="lightgbm", params=res["protocol"],
                            metrics={"sharpe_diff": c["sharpe_diff"], "ci_low": c["ci"][0], "ci_high": c["ci"][1],
                                     "FINDING": c["criteria"]["FINDING"], **{f"criteria.{k}": v for k, v in c["criteria"].items()}},
                            dataset={"symbols": res["symbols"], "window": res["window"], "days": res["days"]},
                            tags=["phase-6.6"], notes="pre-registered: docs/PHASE_6_6_PREREGISTRATION.md")
        except Exception as exc:  # noqa: BLE001
            print(f"(experiment log unavailable: {exc})")
    print("\n=== PHASE 6.6 RESULTS (pre-registered protocol) ===")
    print(_fmt(res))
    findings = [c["comparison"] for c in res["comparisons"] if c["criteria"]["FINDING"]]
    print("\nFINDINGS:", "; ".join(findings) if findings else "none.")
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
