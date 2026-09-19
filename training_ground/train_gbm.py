"""Train and honestly evaluate the LightGBM direction model (Phase 6.2).

Reproduce from scratch:

    python training_ground/train_gbm.py --symbol BTCUSDT --days 1500 --interval 1d

What it does
  1. downloads OHLCV (no news: historical news can't be back-filled),
  2. builds the Phase 6.0 feature matrix and the next-bar direction label,
  3. runs the Phase 6.1 walk-forward loop -- the model is always fit on the past only --
     and reports OUT-OF-SAMPLE accuracy / AUC with a bootstrap confidence interval,
  4. converts the predictions into a long/short/flat rule and reports its return, Sharpe
     and turnover after fees next to buy-and-hold,
  5. fits a final model on all labelled data and saves it (+ metadata) under --out.

Read the verdict at the end before believing any number: on daily crypto/equity data an
AUC whose confidence interval includes 0.50 means "no evidence of skill".
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

from core.feature_engineering import build_features, clean_xy, make_target  # noqa: E402
from core.ml_validation import evaluate_predictions, walk_forward_predict  # noqa: E402
from strategies.gbm_strategy import DEFAULT_LGBM_PARAMS, make_lgbm_classifier  # noqa: E402


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 500, seed: int = 0) -> tuple:
    """95% percentile CI for AUC (iid resampling: slightly optimistic for autocorrelated data)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            aucs.append(roc_auc_score(y[i], p[i]))
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))) if aucs else (float("nan"),) * 2


def rule_performance(p_up: pd.Series, fwd_ret: pd.Series, long_th: float, short_th: float,
                     fee: float, periods_per_year: int) -> dict:
    """Return of: long if P(up)>=long_th, short if <=short_th, else flat -- fees on every position change."""
    df = pd.concat([p_up.rename("p"), fwd_ret.rename("r")], axis=1).dropna()
    pos = pd.Series(0.0, index=df.index)
    pos[df["p"] >= long_th] = 1.0
    pos[df["p"] <= short_th] = -1.0
    gross = pos * df["r"]
    turnover = pos.diff().abs().fillna(pos.abs())          # includes opening the first position
    cost = turnover * fee
    net = gross - cost
    sd = net.std()
    return {
        "bars": int(len(df)),
        "strategy_return_pct": float((1 + net).prod() - 1) * 100,
        "buy_hold_return_pct": float((1 + df["r"]).prod() - 1) * 100,
        "sharpe": float(net.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan"),
        "position_changes": int((turnover > 0).sum()),
        "time_in_market_pct": float((pos != 0).mean()) * 100,
        "fees_paid_pct_of_capital": float(cost.sum()) * 100,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--horizon", type=int, default=1, help="label looks this many bars ahead (= purge gap)")
    ap.add_argument("--train-size", type=int, default=400)
    ap.add_argument("--retrain-every", type=int, default=20)
    ap.add_argument("--long-threshold", type=float, default=0.55)
    ap.add_argument("--short-threshold", type=float, default=0.45)
    ap.add_argument("--fee", type=float, default=0.001, help="per-side fee as a fraction (0.001 = 0.1%%)")
    ap.add_argument("--out", default=None, help="output prefix (default trained_models/gbm_<symbol>_<interval>)")
    args = ap.parse_args(argv)

    from core.data_loader import DataLoader
    print(f"Loading {args.days} days of {args.interval} {args.symbol} ...", flush=True)
    df = DataLoader().load_data(args.symbol, source="Historical", days=args.days,
                                interval=args.interval, include_news=False)
    if df is None or len(df) < args.train_size + 100:
        print(f"Not enough data ({0 if df is None else len(df)} bars) for train_size={args.train_size}.")
        return 1
    print(f"  {len(df)} bars, {df.index[0].date()} -> {df.index[-1].date()}")

    X = build_features(df, include_news=False)
    y = make_target(df, horizon=args.horizon)
    fwd_ret = make_target(df, horizon=args.horizon, kind="return")

    print(f"Walk-forward (train_size={args.train_size}, retrain_every={args.retrain_every}, "
          f"purge gap={args.horizon}) ...", flush=True)
    wf = walk_forward_predict(make_lgbm_classifier, X, y, train_size=args.train_size,
                              retrain_every=args.retrain_every, horizon=args.horizon)
    p = wf.predictions["p_up"]
    m = evaluate_predictions(y, p)
    if not m.get("n"):
        print("No out-of-sample predictions were produced (too little data).")
        return 1
    both = pd.concat([y, p], axis=1).dropna()
    lo, hi = bootstrap_auc_ci(both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy())

    ppy = 365 if "USDT" in args.symbol.upper() else 252
    if args.interval != "1d":
        ppy = int(ppy * {"1h": 24, "15m": 96, "5m": 288, "1m": 1440}.get(args.interval, 1))
    perf = rule_performance(p, fwd_ret, args.long_threshold, args.short_threshold, args.fee, ppy)

    print("\n=== OUT-OF-SAMPLE (every prediction made by a model that had only seen the past) ===")
    print(f"  retrains: {len(wf.folds)}   predicted bars: {int(m['n'])}")
    print(f"  accuracy: {m['accuracy']:.3f}   (always-'up' would score {max(m['base_rate'], 1 - m['base_rate']):.3f}; "
          f"up-rate {m['base_rate']:.3f})")
    print(f"  AUC     : {m.get('auc', float('nan')):.3f}   95% CI [{lo:.3f}, {hi:.3f}]   (0.500 = no skill)")
    print(f"  log-loss: {m.get('logloss', float('nan')):.4f}   brier: {m['brier']:.4f}")
    print(f"\n=== LONG/SHORT/FLAT RULE (long>={args.long_threshold}, short<={args.short_threshold}, fee {args.fee:.2%}/side) ===")
    print(f"  strategy return : {perf['strategy_return_pct']:+8.1f}%   buy&hold: {perf['buy_hold_return_pct']:+8.1f}%   "
          f"Sharpe {perf['sharpe']:.2f}")
    print(f"  position changes: {perf['position_changes']}   time in market: {perf['time_in_market_pct']:.0f}%   "
          f"fees paid: {perf['fees_paid_pct_of_capital']:.1f}% of capital")

    # final model on everything that has a label
    Xc, yc = clean_xy(X, y)
    final = make_lgbm_classifier().fit(Xc, yc)
    gain = pd.Series(final.booster_.feature_importance("gain"), index=Xc.columns)
    gain = (gain / gain.sum()).sort_values(ascending=False)
    print("\n  top features (share of total gain):", ", ".join(f"{k} {v:.0%}" for k, v in gain.head(6).items()))

    prefix = args.out or os.path.join("trained_models", f"gbm_{args.symbol}_{args.interval}")
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    final.booster_.save_model(prefix + ".txt")
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "symbol": args.symbol, "interval": args.interval, "bars": len(df),
        "data_range": [str(df.index[0]), str(df.index[-1])],
        "horizon": args.horizon, "train_rows": int(len(Xc)),
        "features": list(Xc.columns), "lgbm_params": DEFAULT_LGBM_PARAMS,
        "walk_forward": {"train_size": args.train_size, "retrain_every": args.retrain_every,
                         "retrains": len(wf.folds), "metrics": m, "auc_ci95": [lo, hi]},
        "rule": {"long_threshold": args.long_threshold, "short_threshold": args.short_threshold,
                 "fee": args.fee, **perf},
        "top_features": {k: float(v) for k, v in gain.head(10).items()},
    }
    with open(prefix + ".json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"\n  saved {prefix}.txt and {prefix}.json")

    skill = lo > 0.5
    print("\nVERDICT:", "AUC confidence interval is above 0.50 -- some evidence of predictive skill (verify on other periods/symbols)."
          if skill else "AUC confidence interval includes 0.50 -- NO evidence of predictive skill on this sample. "
                        "Do not trade this; it is the honest baseline the next iteration must beat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
