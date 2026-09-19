"""Autonomous strategy research loop, v1 (Phase 12.0).

Each run does four things, all explainable and all recorded:

  1. EVALUATE every candidate strategy on the same recent history (equal-weight across
     symbols, with fees) against buy-and-hold, and test whether the difference in Sharpe is real
     (paired block bootstrap, Bonferroni across candidates).
  2. DECIDE with rules fixed in advance (below) whether each candidate is promoted to paper
     trading or retired. Nothing is tuned to make a strategy pass.
  3. PAPER-TRADE the promoted ones forward: every run stores what each paper strategy wants to
     hold at the latest close; the next run marks those positions to the new prices. That is
     genuinely unseen data, unlike any backtest.
  4. REPORT status, evidence and paper P&L.

Promotion (candidate -> paper) requires ALL of:
  * the Sharpe difference vs buy-and-hold has a confidence-interval lower bound > 0
  * the difference is positive in both halves of the evaluation window
  * at least ``MIN_TRADES`` closed trades in total
  * max drawdown no more than ``MAX_DD_WORSE`` (5 points) worse than buy-and-hold
Retirement (paper -> retired): the interval's UPPER bound is below 0 (significantly worse than
buy-and-hold), or ``MAX_CONSECUTIVE_FAILS`` evaluations in a row fail the promotion test.
Retired strategies never come back automatically -- reviving one is a new hypothesis.

Run:  python -m core.research_loop run | status | report
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sqlite3
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from core.experiment_log import ExperimentLog, default_path
from core.logger import logger
from core.ml_validation import paired_block_bootstrap_stat_diff
from core.risk_sizing import perf_stats, sharpe

FEE = 0.001
HISTORY_DAYS = 1000            # bars each strategy is run over (warm-up + evaluation)
EVAL_DAYS = 700                # the last N bars are what gets judged
BLOCK = 10
N_BOOT = 2000
MIN_TRADES = 30
MAX_DD_WORSE = 0.05
MAX_CONSECUTIVE_FAILS = 3
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "LTCUSDT", "DOGEUSDT", "SOLUSDT"]


@dataclass(frozen=True)
class Candidate:
    name: str
    strategy: str                         # "module:Class"
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def load(self):
        mod, cls = self.strategy.split(":")
        return getattr(importlib.import_module(mod), cls)


# All-in per trade so each strategy is comparable with a 100%-invested buy-and-hold.
_ALLIN = {"risk_per_trade": 0.95}
DEFAULT_CANDIDATES: List[Candidate] = [
    Candidate("MACD/RSI", "strategies.simple_strategies:MACD_RSI_Strategy", dict(_ALLIN),
              "RSI oversold/overbought filtered by MACD direction"),
    Candidate("EMA Crossover", "strategies.simple_strategies:EMACrossoverStrategy", dict(_ALLIN),
              "12/26 EMA crossover trend following"),
    Candidate("Stochastic", "strategies.simple_strategies:StochasticStrategy", dict(_ALLIN),
              "Stochastic %K/%D cross in extreme zones"),
    Candidate("GBM (LightGBM)", "strategies.gbm_strategy:GBMStrategy", dict(_ALLIN),
              "Walk-forward LightGBM next-day direction"),
]

# ------------------------------------------------------------------------- evaluation


def run_strategy(candidate: Candidate, df: pd.DataFrame, fee: float = FEE) -> Dict[str, Any]:
    """Run one strategy on one symbol with backtrader. Returns daily returns, trade count and the position it wants next."""
    import backtrader as bt
    frame = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    frame.index.name = "datetime"
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(bt.feeds.PandasData(dataname=frame))
    cerebro.addstrategy(candidate.load(), **candidate.params)
    cerebro.broker.setcash(100_000.0)
    cerebro.broker.setcommission(commission=fee)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="tr", timeframe=bt.TimeFrame.Days)
    strat = cerebro.run()[0]
    raw = strat.analyzers.tr.get_analysis()
    rets = pd.Series({pd.Timestamp(k).normalize(): v for k, v in raw.items()}).sort_index()

    target = int(np.sign(strat.position.size))
    pending = getattr(strat, "_pending_rationale", None)          # a decision made on the last bar, not yet filled
    if pending:
        action = pending.get("action", "")
        target = {"open_long": 1, "open_short": -1, "close_long": 0, "close_short": 0}.get(action, target)
    return {"returns": rets, "trades": len(getattr(strat, "closed_trades", [])) or len(getattr(strat, "signals", [])) // 2,
            "target_position": target}


def evaluate_candidate(candidate: Candidate, data: Dict[str, pd.DataFrame], *, fee: float = FEE,
                       eval_days: int = EVAL_DAYS, level: float = 0.95, n_boot: int = N_BOOT) -> Dict[str, Any]:
    """Equal-weight portfolio of the strategy across symbols vs equal-weight buy-and-hold, same dates."""
    per_symbol, holds, targets, trades = {}, {}, {}, 0
    for sym, df in data.items():
        out = run_strategy(candidate, df, fee)
        per_symbol[sym] = out["returns"]
        holds[sym] = df["Close"].pct_change().rename(sym)
        holds[sym].index = pd.to_datetime(holds[sym].index).normalize()
        targets[sym] = out["target_position"]
        trades += out["trades"]
    strat = pd.concat(per_symbol, axis=1).sort_index()
    hold = pd.concat(holds, axis=1).sort_index()
    idx = strat.index.intersection(hold.index)[-eval_days:]
    s_ret = strat.loc[idx].fillna(0.0).mean(axis=1)
    h_ret = hold.loc[idx].fillna(0.0).mean(axis=1)

    s_stats, h_stats = perf_stats(s_ret), perf_stats(h_ret)
    lo, hi = paired_block_bootstrap_stat_diff(s_ret.to_numpy(), h_ret.to_numpy(), idx.to_numpy(), sharpe,
                                              block=BLOCK, n_boot=n_boot, level=level)
    half = len(idx) // 2
    halves = [sharpe(s_ret.iloc[:half].to_numpy()) - sharpe(h_ret.iloc[:half].to_numpy()),
              sharpe(s_ret.iloc[half:].to_numpy()) - sharpe(h_ret.iloc[half:].to_numpy())]
    return {"candidate": candidate.name, "days": int(len(idx)), "window": [str(idx[0].date()), str(idx[-1].date())],
            "symbols": list(data), "trades": int(trades), "strategy": s_stats, "buy_hold": h_stats,
            "sharpe_diff": sharpe(s_ret.to_numpy()) - sharpe(h_ret.to_numpy()), "ci": [lo, hi], "ci_level": level,
            "sharpe_diff_halves": halves, "target_positions": targets}


def promotion_test(ev: Dict[str, Any]) -> Dict[str, bool]:
    c = {
        "ci_lower_above_zero": bool(ev["ci"][0] > 0),
        "positive_in_both_halves": bool(all(d > 0 for d in ev["sharpe_diff_halves"])),
        f"at_least_{MIN_TRADES}_trades": bool(ev["trades"] >= MIN_TRADES),
        "drawdown_not_much_worse": bool(ev["strategy"]["max_drawdown"] >= ev["buy_hold"]["max_drawdown"] - MAX_DD_WORSE),
    }
    c["PASS"] = all(c.values())
    return c


# ------------------------------------------------------------------------------- state


class ResearchState:
    """Candidate status + the forward paper ledger, in the same SQLite file as the experiment log."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or default_path()
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS research_state (
                    candidate TEXT PRIMARY KEY, status TEXT NOT NULL, since TEXT NOT NULL,
                    consecutive_fails INTEGER NOT NULL DEFAULT 0, evaluations INTEGER NOT NULL DEFAULT 0,
                    last_run_id INTEGER, note TEXT NOT NULL DEFAULT '');
                CREATE TABLE IF NOT EXISTS paper_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, run_date TEXT NOT NULL, candidate TEXT NOT NULL,
                    symbol TEXT NOT NULL, position INTEGER NOT NULL, price REAL NOT NULL,
                    UNIQUE(run_date, candidate, symbol));
            """)
            self._conn.commit()

    def status(self, candidate: str) -> str:
        with self._lock:
            row = self._conn.execute("SELECT status FROM research_state WHERE candidate=?", (candidate,)).fetchone()
        return row["status"] if row else "candidate"

    def all_states(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute("SELECT * FROM research_state ORDER BY candidate").fetchall()]

    def apply(self, candidate: str, ev: Dict[str, Any], test: Dict[str, bool], run_id: Optional[int]) -> Dict[str, Any]:
        """Advance the state machine for one evaluation and return what happened."""
        with self._lock:
            row = self._conn.execute("SELECT * FROM research_state WHERE candidate=?", (candidate,)).fetchone()
            status = row["status"] if row else "candidate"
            fails = row["consecutive_fails"] if row else 0
            n_eval = (row["evaluations"] if row else 0) + 1
            since = row["since"] if row else datetime.now(timezone.utc).isoformat(timespec="seconds")
            note, changed = "", False
            fails = 0 if test["PASS"] else fails + 1
            if status == "candidate" and test["PASS"]:
                status, changed, note = "paper", True, "passed every promotion test"
            elif status == "paper":
                if ev["ci"][1] < 0:
                    status, changed, note = "retired", True, "significantly worse than buy-and-hold (interval upper bound < 0)"
                elif fails >= MAX_CONSECUTIVE_FAILS:
                    status, changed, note = "retired", True, f"failed the promotion test {fails} evaluations in a row"
            if changed:
                since = datetime.now(timezone.utc).isoformat(timespec="seconds")
            if status == "retired" and not changed:
                note = "retired (kept; reviving is a new hypothesis)"
            self._conn.execute(
                "INSERT INTO research_state (candidate,status,since,consecutive_fails,evaluations,last_run_id,note) VALUES (?,?,?,?,?,?,?) "
                "ON CONFLICT(candidate) DO UPDATE SET status=excluded.status, since=excluded.since, "
                "consecutive_fails=excluded.consecutive_fails, evaluations=excluded.evaluations, "
                "last_run_id=excluded.last_run_id, note=excluded.note",
                (candidate, status, since, fails, n_eval, run_id, note))
            self._conn.commit()
        return {"candidate": candidate, "status": status, "changed": changed, "note": note, "consecutive_fails": fails}

    # ---- paper ledger (forward test) ----
    def record_paper_positions(self, run_date: str, candidate: str, positions: Dict[str, int], prices: Dict[str, float]) -> int:
        n = 0
        with self._lock:
            for sym, pos in positions.items():
                cur = self._conn.execute(
                    "INSERT OR IGNORE INTO paper_ledger (run_date,candidate,symbol,position,price) VALUES (?,?,?,?,?)",
                    (run_date, candidate, sym, int(pos), float(prices[sym])))
                n += cur.rowcount
            self._conn.commit()
        return n

    def paper_pnl(self, fee: float = FEE) -> pd.DataFrame:
        """Forward P&L per candidate: the position recorded at date d earns the return from d's price to the next recorded price."""
        with self._lock:
            rows = [dict(r) for r in self._conn.execute("SELECT * FROM paper_ledger ORDER BY candidate, symbol, run_date").fetchall()]
        if not rows:
            return pd.DataFrame(columns=["candidate", "days", "return_pct", "trades"])
        df = pd.DataFrame(rows)
        out = []
        for (cand, sym), g in df.groupby(["candidate", "symbol"]):
            g = g.sort_values("run_date")
            ret = g["price"].shift(-1) / g["price"] - 1
            turnover = g["position"].diff().abs().fillna(g["position"].abs())
            net = (g["position"] * ret - fee * turnover).iloc[:-1]        # last row has no following price yet
            out.append(pd.DataFrame({"candidate": cand, "symbol": sym, "run_date": g["run_date"].iloc[:-1], "net": net.values,
                                     "changed": (turnover.iloc[:-1] > 0).astype(int).values}))
        if not out:
            return pd.DataFrame(columns=["candidate", "days", "return_pct", "trades"])
        allr = pd.concat(out)
        daily = allr.groupby(["candidate", "run_date"])["net"].mean()
        res = []
        for cand, s in daily.groupby(level=0):
            res.append({"candidate": cand, "days": int(len(s)), "return_pct": float((1 + s).prod() - 1) * 100,
                        "trades": int(allr[allr["candidate"] == cand]["changed"].sum())})
        return pd.DataFrame(res)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# --------------------------------------------------------------------------------- loop


def load_default_data(symbols=SYMBOLS, days: int = HISTORY_DAYS) -> Dict[str, pd.DataFrame]:
    from core.order_flow_data import fetch_klines
    data = {}
    for s in symbols:
        kl = fetch_klines(s, days=days)
        if kl is not None and len(kl) > 300:
            data[s] = kl
        else:
            logger.warning("research loop: skipping %s (not enough data)", s)
    return data


def run_cycle(data: Optional[Dict[str, pd.DataFrame]] = None, candidates: Optional[List[Candidate]] = None, *,
              state: Optional[ResearchState] = None, log: Optional[ExperimentLog] = None,
              n_boot: int = N_BOOT, eval_days: int = EVAL_DAYS, progress: Callable[[str], None] = print) -> Dict[str, Any]:
    """One full pass: evaluate -> decide -> record -> paper-trade. Returns a structured report."""
    data = data if data is not None else load_default_data()
    candidates = candidates if candidates is not None else DEFAULT_CANDIDATES
    state = state or ResearchState()
    log = log or ExperimentLog()
    level = 1 - 0.05 / max(1, len(candidates))                    # Bonferroni across candidates
    run_date = str(max(df.index[-1] for df in data.values()).date())
    prices = {s: float(df["Close"].iloc[-1]) for s, df in data.items()}
    report: Dict[str, Any] = {"run_date": run_date, "symbols": list(data), "candidates": []}

    for cand in candidates:
        if state.status(cand.name) == "retired":
            report["candidates"].append({"candidate": cand.name, "status": "retired", "skipped": True})
            continue
        progress(f"evaluating {cand.name} ...")
        try:
            ev = evaluate_candidate(cand, data, level=level, n_boot=n_boot, eval_days=eval_days)
        except Exception as exc:  # noqa: BLE001 -- one broken candidate must not stop the loop
            logger.error("research loop: %s failed: %s", cand.name, exc)
            report["candidates"].append({"candidate": cand.name, "status": state.status(cand.name), "error": str(exc)})
            continue
        test = promotion_test(ev)
        run_id = log.log_run(
            name=f"research-loop {cand.name}", model_type="strategy",
            params={"candidate": cand.name, "strategy": cand.strategy, **cand.params, "fee": FEE, "eval_days": eval_days,
                    "block": BLOCK, "ci_level": level},
            metrics={"sharpe": ev["strategy"]["sharpe"], "bh_sharpe": ev["buy_hold"]["sharpe"], "sharpe_diff": ev["sharpe_diff"],
                     "ci_low": ev["ci"][0], "ci_high": ev["ci"][1], "trades": ev["trades"],
                     "max_drawdown": ev["strategy"]["max_drawdown"], "bh_max_drawdown": ev["buy_hold"]["max_drawdown"],
                     "PASS": test["PASS"]},
            dataset={"symbols": ev["symbols"], "window": ev["window"], "days": ev["days"]}, tags=["research-loop", cand.name])
        transition = state.apply(cand.name, ev, test, run_id)
        if transition["status"] == "paper":                         # forward-test whatever it wants to hold now
            state.record_paper_positions(run_date, cand.name, ev["target_positions"], prices)
        report["candidates"].append({**transition, "evaluation": ev, "test": test, "run_id": run_id})
    report["paper_pnl"] = state.paper_pnl().to_dict("records")
    return report


def format_report(rep: Dict[str, Any]) -> str:
    lines = [f"Research loop report -- data through {rep['run_date']} ({len(rep['symbols'])} symbols)",
             f"{'candidate':<18}{'status':<10}{'Sharpe':>8}{'B&H':>7}{'diff':>8}{'CI':>20}{'trades':>8}{'maxDD%':>8}  promotion test"]
    for c in rep["candidates"]:
        if c.get("skipped"):
            lines.append(f"{c['candidate']:<18}{'retired':<10} (skipped)")
            continue
        if "error" in c:
            lines.append(f"{c['candidate']:<18}{c['status']:<10} ERROR: {c['error'][:60]}")
            continue
        ev, t = c["evaluation"], c["test"]
        failed = [k for k, v in t.items() if k != "PASS" and not v]
        lines.append(f"{c['candidate']:<18}{c['status']:<10}{ev['strategy']['sharpe']:>8.2f}{ev['buy_hold']['sharpe']:>7.2f}"
                     f"{ev['sharpe_diff']:>+8.2f}{'[%+.2f, %+.2f]' % tuple(ev['ci']):>20}{ev['trades']:>8}"
                     f"{ev['strategy']['max_drawdown']*100:>8.0f}  {'PASS' if t['PASS'] else 'fail: ' + ', '.join(failed)}"
                     f"{'   -> ' + c['note'] if c.get('changed') else ''}")
    if rep.get("paper_pnl"):
        lines.append("\nForward paper P&L (positions recorded by earlier runs, marked to newer prices):")
        for r in rep["paper_pnl"]:
            lines.append(f"  {r['candidate']:<18}{r['days']:>4} days  {r['return_pct']:+7.2f}%  {r['trades']} position changes")
    else:
        lines.append("\nNo strategy is in paper trading yet (none has passed every promotion test).")
    return "\n".join(lines)


def _cli(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m core.research_loop", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "status", "report"])
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        rep = run_cycle(n_boot=args.n_boot)
        print("\n" + format_report(rep))
    else:
        st = ResearchState()
        for r in st.all_states():
            print(f"{r['candidate']:<18}{r['status']:<10} since {r['since']}  evals={r['evaluations']}  fails={r['consecutive_fails']}  {r['note']}")
        pnl = st.paper_pnl()
        print("\n" + (pnl.to_string(index=False) if len(pnl) else "no paper positions recorded yet"))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())


def research_loop_view(path: Optional[str] = None) -> dict:
    """Rows for the Research Loop tab's three tables plus a status line. Never raises.

    Shared by the Dash tab and the desktop tab so both always show the same data."""
    empty = {"candidates": [], "paper": [], "runs": [],
             "message": "No research-loop runs yet. Run  python -m core.research_loop run  in a terminal."}
    try:
        from core.experiment_log import ExperimentLog
        from core.research_loop import ResearchState
        log, state = ExperimentLog(path), ResearchState(path)
        states = {r["candidate"]: r for r in state.all_states()}
        latest = {}
        for run in log.list_runs(tag="research-loop", limit=500):            # newest first
            latest.setdefault(run["params"].get("candidate"), run)
        candidates = []
        for name in sorted(set(states) | set(latest)):
            st, run = states.get(name), latest.get(name)
            m = run["metrics"] if run else {}
            f = lambda v, fmt="{:.2f}": fmt.format(v) if isinstance(v, (int, float)) else "—"
            candidates.append({
                "candidate": name, "status": st["status"] if st else "candidate",
                "sharpe": f(m.get("sharpe")), "bh_sharpe": f(m.get("bh_sharpe")), "diff": f(m.get("sharpe_diff"), "{:+.2f}"),
                "ci": f"[{m['ci_low']:+.2f}, {m['ci_high']:+.2f}]" if "ci_low" in m and "ci_high" in m else "—",
                "trades": int(m["trades"]) if "trades" in m else "—",
                "max_dd": f(m.get("max_drawdown") * 100 if "max_drawdown" in m else None, "{:.0f}%"),
                "verdict": ("PASS" if m.get("PASS") else "fail") if m else "—",
                "evaluated": run["created_at"][:16].replace("T", " ") if run else "—"})
        paper = [{"candidate": r["candidate"], "days": r["days"], "return_pct": f"{r['return_pct']:+.2f}", "trades": r["trades"]}
                 for r in state.paper_pnl().to_dict("records")]
        runs = []
        for r in log.list_runs(limit=15):
            flat = {}
            for k, v in r["metrics"].items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    flat[k] = v
            key = next((k for k in ("auc", "auc_gbm", "sharpe_diff", "sharpe") if k in flat), next(iter(flat), None))
            runs.append({"id": r["id"], "when": r["created_at"][:16].replace("T", " "), "name": r["name"][:44],
                         "model": r["model_type"], "result": f"{key} = {flat[key]:.3f}" if key else "—",
                         "commit": (r["git_commit"] or "")[:7] + ("*" if r["git_dirty"] else "")})
        n_paper = sum(1 for c in candidates if c["status"] == "paper")
        n_ret = sum(1 for c in candidates if c["status"] == "retired")
        msg = (f"{len(candidates)} candidate(s): {n_paper} in paper trading, {n_ret} retired, "
               f"{len(candidates) - n_paper - n_ret} still on trial. "
               + ("" if n_paper else "None has yet passed every promotion test (interval lower bound > 0, both halves positive, "
                                     ">= 30 trades, drawdown not much worse) -- which is the honest expected outcome for most ideas."))
        if not candidates and not runs:
            return empty
        return {"candidates": candidates, "paper": paper, "runs": runs, "message": msg}
    except Exception as exc:  # noqa: BLE001 -- a view problem must never break the page
        logger.warning("[Dash] research loop view failed: %s", exc)
        return {**empty, "message": f"Research loop data unavailable: {exc}"}


