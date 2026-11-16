"""Phase 14.0 -- best long-only crypto portfolio rule, selection and test kept apart. Implements docs/PHASE_14_0_PREREGISTRATION.md.

    python training_ground/experiments_14_0.py            # writes training_ground/results/phase_14_0.json and prints the verdict
Frozen protocol: read the pre-registration before changing anything here.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ml_validation import paired_block_bootstrap_stat_diff  # noqa: E402
from core.portfolio_rules import CANDIDATES, HISTORY_NEEDED, target_weights  # noqa: E402
from core.risk_sizing import perf_stats, sharpe  # noqa: E402

SET1 = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "LTCUSDT", "LINKUSDT", "TRXUSDT"]
SET2 = ["DOTUSDT", "AVAXUSDT", "ATOMUSDT", "NEARUSDT", "UNIUSDT", "FILUSDT", "SOLUSDT", "DOGEUSDT"]
A_START, A_END = pd.Timestamp("2019-02-14"), pd.Timestamp("2022-09-09")
H_START = pd.Timestamp("2022-09-10")
FEE, REBALANCE = 0.001, 7
MIN_SYMBOLS, MIN_DAYS = 6, 800
BLOCK, N_BOOT, K = 30, 5000, 2
LEVEL = 1 - 0.05 / K
DD_MARGIN = 0.10


def simulate(returns: pd.DataFrame, rule: str, fee: float = FEE, every: int = REBALANCE) -> pd.Series:
    """Daily net returns of ``rule``: weights decided from returns strictly before each day, rebalanced every ``every`` days,
    drifting in between, fee on turnover, remainder in cash. Starts after HISTORY_NEEDED days."""
    w = pd.Series(0.0, index=returns.columns)
    out = {}
    for i in range(HISTORY_NEEDED, len(returns)):
        cost = 0.0
        if (i - HISTORY_NEEDED) % every == 0:
            target = target_weights(rule, returns.iloc[:i])
            cost, w = fee * float((target - w).abs().sum()), target
        r = returns.iloc[i]
        day = float((w * r).sum()) - cost
        out[returns.index[i]] = day
        denom = 1 + day + cost
        w = w * (1 + r) / denom if denom != 0 else w
    return pd.Series(out)


def max_drawdown(r) -> float:
    eq = np.cumprod(1 + np.asarray(r, float))
    return float(np.min(eq / np.maximum.accumulate(eq) - 1))


def load_closes(symbols) -> pd.DataFrame:
    from core.order_flow_data import fetch_klines
    cols, dropped = {}, []
    for s in symbols:
        try:
            cols[s] = fetch_klines(s, days=3000)["Close"].astype(float)
        except Exception as exc:  # noqa: BLE001
            dropped.append(f"{s}: {type(exc).__name__}")
    if dropped:
        print("  dropped:", dropped)
    return pd.DataFrame(cols)


def window_returns(closes: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp | None):
    """Daily returns for the coins that have data for the whole evaluation window (+ look-back before it), or None if void."""
    c = closes[closes.index <= end] if end is not None else closes
    need_from = c.index[c.index >= start][0] - pd.Timedelta(days=HISTORY_NEEDED + 5) if (c.index >= start).any() else None
    if need_from is None:
        return None
    c = c[c.index >= need_from]
    c = c.loc[:, c[c.index >= start].notna().all()]                # coins present for the evaluation days
    c = c.dropna(how="any")
    if c.shape[1] < MIN_SYMBOLS or len(c[c.index >= start]) < MIN_DAYS:
        return None
    return c.pct_change().dropna()


def evaluate(returns: pd.DataFrame, start: pd.Timestamp, rules, n_boot: int = N_BOOT) -> dict:
    series = {r: simulate(returns, r) for r in rules}
    series = {r: s[s.index >= start] for r, s in series.items()}
    return {"window": [str(series["EW"].index[0].date()), str(series["EW"].index[-1].date())], "days": int(len(series["EW"])),
            "symbols": list(returns.columns),
            "stats": {r: {**perf_stats(s), "sharpe_boot": sharpe(s.to_numpy())} for r, s in series.items()}}, series


def bootstrap_diff(a: pd.Series, b: pd.Series, n_boot: int = N_BOOT) -> list:
    b = b.reindex(a.index)
    return list(paired_block_bootstrap_stat_diff(a.to_numpy(), b.to_numpy(), a.index.to_numpy(), sharpe, block=BLOCK,
                                                 n_boot=n_boot, level=LEVEL))


def verdict(selected: str, holdouts: dict) -> dict:
    """Apply the pre-registered criteria. ``holdouts``: {name: {"stats": {...}, "ci": [lo, hi] or None}} or None for a void window."""
    if any(h is None for h in holdouts.values()):
        return {"verdict": "INCONCLUSIVE (a holdout window was void)", "checks": {}}
    checks = {}
    for name, h in holdouts.items():
        st = h["stats"]
        checks[name] = {"a_sharpe_above_EW": bool(st[selected]["sharpe"] > st["EW"]["sharpe"]),
                        "b_maxdd_10pp_shallower_than_EW": bool(st[selected]["max_drawdown"] >= st["EW"]["max_drawdown"] + DD_MARGIN),
                        "c_sharpe_at_least_TS28": bool(st[selected]["sharpe"] >= st["TS28"]["sharpe"]),
                        "d_sharpe_diff_ci_lower_above_0": bool(h["ci"][0] > 0)}
    promising = all(c[k] for c in checks.values() for k in ("a_sharpe_above_EW", "b_maxdd_10pp_shallower_than_EW", "c_sharpe_at_least_TS28"))
    supported = promising and all(c["d_sharpe_diff_ci_lower_above_0"] for c in checks.values())
    return {"verdict": "SUPPORTED" if supported else "PROMISING (not statistically supported)" if promising else "NOT SUPPORTED",
            "checks": checks}


def run(closes1: pd.DataFrame | None = None, closes2: pd.DataFrame | None = None, n_boot: int = N_BOOT) -> dict:
    closes1 = load_closes(SET1) if closes1 is None else closes1
    closes2 = load_closes(SET2) if closes2 is None else closes2
    rules = ["EW", "TS28", *CANDIDATES]
    ra = window_returns(closes1, A_START, A_END)
    if ra is None:
        return {"void": True, "reason": "window A (selection) has too few symbols/days"}
    res_a, _ = evaluate(ra, A_START, rules)
    selected = max(CANDIDATES, key=lambda r: res_a["stats"][r]["sharpe"])
    out = {"void": False, "selected": selected, "A": res_a, "holdouts": {}}
    verdict_in = {}
    for name, closes in (("H1", closes1), ("H2", closes2)):
        rh = window_returns(closes, H_START, None)
        if rh is None:
            out["holdouts"][name] = {"void": True}
            verdict_in[name] = None
            continue
        res, series = evaluate(rh, H_START, rules)
        res["ci_selected_vs_EW"] = bootstrap_diff(series[selected], series["EW"], n_boot)
        res["ci_selected_vs_TS28"] = bootstrap_diff(series[selected], series["TS28"], n_boot)
        out["holdouts"][name] = res
        verdict_in[name] = {"stats": res["stats"], "ci": res["ci_selected_vs_EW"]}
    out.update(verdict(selected, verdict_in))
    out["protocol"] = {"fee": FEE, "rebalance_days": REBALANCE, "block": BLOCK, "n_boot": n_boot, "ci_level": LEVEL, "dd_margin": DD_MARGIN}
    return out


def _fmt(res: dict) -> str:
    if res.get("void"):
        return "VOID: " + res["reason"]
    lines = [f"selected on window A: {res['selected']}", ""]

    def table(title, block):
        lines.append(f"{title}  {block['window'][0]} -> {block['window'][1]}  ({block['days']} days, {len(block['symbols'])} coins)")
        lines.append(f"  {'rule':<5}{'Sharpe':>8}{'AnnRet%':>9}{'AnnVol%':>9}{'MaxDD%':>9}")
        for r, st in block["stats"].items():
            lines.append(f"  {r:<5}{st['sharpe']:>8.2f}{st['ann_return']*100:>9.1f}{st['ann_vol']*100:>9.1f}{st['max_drawdown']*100:>9.1f}")
    table("A  (selection)", res["A"])
    for n, h in res["holdouts"].items():
        lines.append("")
        if h.get("void"):
            lines.append(f"{n}: VOID")
            continue
        table(f"{n} (holdout)", h)
        lines.append(f"  {res['selected']} - EW Sharpe diff CI{(1-LEVEL)*100:.1f}% ... {h['ci_selected_vs_EW'][0]:+.2f} .. {h['ci_selected_vs_EW'][1]:+.2f}")
    lines += ["", "VERDICT: " + res["verdict"]]
    return "\n".join(lines)


def main() -> int:
    res = run()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "phase_14_0.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"run_at": datetime.now(timezone.utc).isoformat(), **res}, fh, indent=2, default=float)
    print(_fmt(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
