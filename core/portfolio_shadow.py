"""Order-free forward shadow ledger of the Phase 14.0 portfolio rules (docs/PHASE_14_1_SHADOW_PREREGISTRATION.md).

Each day it records what EW, TS28, C2 (the pre-registered pick), C3 and C5 WOULD have held and earned, using only completed UTC daily
closes of the fixed SET1 universe. No broker, no orders, no money. It is the one piece of evidence that is genuinely unseen: every
backtest so far used coins that survived to today.

    python -m core.portfolio_shadow update    # process every completed day not yet in the ledger (safe to run any time, backfills gaps)
    python -m core.portfolio_shadow status    # NAV of each rule, current target weights, days of data, what can and cannot be concluded

Mechanics are Phase 14.0's (experiments_14_0.simulate), made incremental: weights for day t come from returns before t, weekly rebalance
from the start day, 0.1% fee on turnover, drifting weights in between, remainder cash. State is persisted so a past day is never recomputed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from core.portfolio_rules import HISTORY_NEEDED, target_weights
from core.risk_sizing import perf_stats

UNIVERSE = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "LTCUSDT", "LINKUSDT", "TRXUSDT"]
RULES = ["EW", "TS28", "C2", "C3", "C5"]
PRIMARY, DESCRIPTIVE = "C2", ("C3", "C5")
FEE, REBALANCE_DAYS = 0.001, 7
CONCLUSION_DAYS, STAT_DAYS = 180, 730
HISTORY_DAYS = 400                                  # download window: look-back (112) plus room
DEFAULT_PATH = os.path.join("training_ground", "paper", "portfolio_shadow.sqlite3")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS state(rule TEXT PRIMARY KEY, nav REAL NOT NULL, w TEXT NOT NULL, pending TEXT, last_decision TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS daily(day TEXT NOT NULL, rule TEXT NOT NULL, nav REAL NOT NULL, ret REAL NOT NULL, cost REAL NOT NULL,
    turnover REAL NOT NULL, held TEXT NOT NULL, PRIMARY KEY(day, rule));
CREATE TABLE IF NOT EXISTS prices(day TEXT NOT NULL, symbol TEXT NOT NULL, close REAL NOT NULL, PRIMARY KEY(day, symbol));
"""

ClosesFn = Callable[[], pd.DataFrame]


class ShadowError(RuntimeError):
    pass


def closes_from_binance(symbols=UNIVERSE, days: int = HISTORY_DAYS, now: Optional[datetime] = None) -> pd.DataFrame:
    """Completed UTC daily closes (index = day open time, naive UTC) from the cached Binance kline store."""
    from core.kline_cache import get_klines
    now = now or datetime.now(timezone.utc)
    today = pd.Timestamp(now).tz_convert("UTC").tz_localize(None).normalize()
    end_ms = int(today.value // 10**6)                                   # today's candle is still forming: excluded
    start_ms = end_ms - days * 86_400_000
    cols = {}
    for s in symbols:
        df = get_klines(s, "1d", start_ms, end_ms)
        if df is None or df.empty:
            raise ShadowError(f"no daily candles for {s}")
        cols[s] = df["Close"].astype(float)
    return pd.DataFrame(cols).sort_index()


class ShadowLedger:
    def __init__(self, path: Optional[str] = None, closes_fn: Optional[ClosesFn] = None):
        self.path = path or os.getenv("PORTFOLIO_SHADOW_PATH") or DEFAULT_PATH
        self.closes_fn = closes_fn or closes_from_binance
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        with closing(self._connect()) as con:
            con.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    # ------------------------------------------------------------------ update
    def update(self) -> dict:
        """Process every completed day not yet in the ledger, oldest first, in ONE transaction (all or nothing)."""
        closes = self.closes_fn()
        closes = closes[[c for c in UNIVERSE if c in closes.columns]]
        if list(closes.columns) != UNIVERSE:
            raise ShadowError(f"missing coins: {sorted(set(UNIVERSE) - set(closes.columns))}")
        closes = closes.sort_index()
        if closes.iloc[-HISTORY_NEEDED - 2:].isna().any().any():
            raise ShadowError("a coin has no close for a recent completed day; refusing to invent one")
        closes = closes.dropna(how="any")
        if len(closes) < HISTORY_NEEDED + 1:                                # HISTORY_NEEDED daily returns need one more close
            raise ShadowError(f"need {HISTORY_NEEDED + 1} days of closes, got {len(closes)}")
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            last = self._get(con, "last_day")
            if last is None:                                              # first run: start day = latest completed day
                start = closes.index[-1]
                self._init(con, closes, start)
                return {"started": str(start.date()), "processed": 0}
            todo = [d for d in closes.index if str(d.date()) > last]
            for d in todo:
                self._process_day(con, closes, d)
            return {"processed": len(todo), "last_day": str(closes.index[-1].date())}

    @staticmethod
    def _get(con, key):
        r = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r[0] if r else None

    @staticmethod
    def _put(con, key, value):
        con.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?,?)", (key, str(value)))

    def _init(self, con, closes: pd.DataFrame, start: pd.Timestamp) -> None:
        rets = closes.pct_change().loc[:start].dropna()
        for rule in RULES:
            pending = target_weights(rule, rets)                          # decided at the start day's close, applied from the next day
            con.execute("INSERT INTO state(rule, nav, w, pending, last_decision) VALUES(?,?,?,?,?)",
                        (rule, 1.0, json.dumps({s: 0.0 for s in UNIVERSE}), json.dumps(pending.to_dict()), str(start.date())))
            con.execute("INSERT INTO daily(day, rule, nav, ret, cost, turnover, held) VALUES(?,?,?,?,?,?,?)",
                        (str(start.date()), rule, 1.0, 0.0, 0.0, 0.0, json.dumps({s: 0.0 for s in UNIVERSE})))
        for s in UNIVERSE:
            con.execute("INSERT OR REPLACE INTO prices(day, symbol, close) VALUES(?,?,?)", (str(start.date()), s, float(closes.loc[start, s])))
        self._put(con, "start_day", start.date())
        self._put(con, "last_day", start.date())
        self._put(con, "created_at", datetime.now(timezone.utc).isoformat())

    def _process_day(self, con, closes: pd.DataFrame, d: pd.Timestamp) -> None:
        i = closes.index.get_loc(d)
        r = closes.iloc[i] / closes.iloc[i - 1] - 1.0
        rets_to_d = closes.pct_change().loc[:d].dropna()
        day = str(d.date())
        for rule in RULES:
            nav, w_json, pending_json, last_dec = con.execute("SELECT nav, w, pending, last_decision FROM state WHERE rule=?", (rule,)).fetchone()
            w = pd.Series(json.loads(w_json))[UNIVERSE]
            cost = turnover = 0.0
            if pending_json:
                target = pd.Series(json.loads(pending_json))[UNIVERSE]
                turnover = float((target - w).abs().sum())
                cost, w, pending_json = FEE * turnover, target, None
            held = w.copy()
            ret = float((w * r).sum()) - cost
            nav *= 1.0 + ret
            denom = 1.0 + ret + cost
            w = w * (1.0 + r) / denom if denom != 0 else w
            if (d - pd.Timestamp(last_dec)).days >= REBALANCE_DAYS:       # decide at today's close, apply from tomorrow
                pending_json = json.dumps(target_weights(rule, rets_to_d).to_dict())
                last_dec = day
            con.execute("INSERT INTO daily(day, rule, nav, ret, cost, turnover, held) VALUES(?,?,?,?,?,?,?)",
                        (day, rule, nav, ret, cost, turnover, json.dumps(held.to_dict())))
            con.execute("UPDATE state SET nav=?, w=?, pending=?, last_decision=? WHERE rule=?",
                        (nav, json.dumps(w.to_dict()), pending_json, last_dec, rule))
        for s in UNIVERSE:
            con.execute("INSERT OR REPLACE INTO prices(day, symbol, close) VALUES(?,?,?)", (day, s, float(closes.loc[d, s])))
        self._put(con, "last_day", day)

    # ------------------------------------------------------------------ reading
    def series(self) -> Dict[str, pd.Series]:
        with closing(self._connect()) as con:
            rows = con.execute("SELECT day, rule, ret FROM daily ORDER BY day").fetchall()
        out: Dict[str, dict] = {r: {} for r in RULES}
        for day, rule, ret in rows:
            out[rule][pd.Timestamp(day)] = ret
        return {r: pd.Series(v, dtype=float) for r, v in out.items()}

    def status(self) -> dict:
        with closing(self._connect()) as con:
            start, last = self._get(con, "start_day"), self._get(con, "last_day")
            if start is None:
                return {"started": False}
            state = {r: (nav, json.loads(w), json.loads(p) if p else None) for r, nav, w, p, _ in
                     con.execute("SELECT rule, nav, w, pending, last_decision FROM state").fetchall()}
        ser = self.series()
        days = max(0, len(ser["EW"]) - 1)                                   # the start day is the zero point
        stats = {}
        for r in RULES:
            s = ser[r].iloc[1:]
            st = perf_stats(s) if len(s) >= 30 else {}
            stats[r] = {"nav": state[r][0], "total_return": state[r][0] - 1.0, **{k: st.get(k) for k in ("sharpe", "ann_vol", "max_drawdown")}}
        age = (pd.Timestamp(datetime.now(timezone.utc)).tz_convert("UTC").tz_localize(None).normalize() - pd.Timestamp(last)).days - 1
        return {"started": True, "start_day": start, "last_day": last, "days": days, "days_behind": max(0, age), "stats": stats,
                "targets": {r: (state[r][2] or state[r][1]) for r in RULES}, "forward": self.forward_verdict(ser, days)}

    @staticmethod
    def forward_verdict(ser: Dict[str, pd.Series], days: int) -> dict:
        """The pre-registered forward criteria; no verdict before CONCLUSION_DAYS."""
        if days < CONCLUSION_DAYS:
            return {"verdict": f"NO CONCLUSION YET ({days} of {CONCLUSION_DAYS} days)", "stat_ready": False}
        s = {r: ser[r].iloc[1:] for r in ("EW", "TS28", PRIMARY)}
        st = {r: perf_stats(v) for r, v in s.items()}
        a = st[PRIMARY]["sharpe"] > st["EW"]["sharpe"]
        b = st[PRIMARY]["max_drawdown"] >= st["EW"]["max_drawdown"] + 0.10
        c = st[PRIMARY]["sharpe"] >= st["TS28"]["sharpe"]
        return {"verdict": "FORWARD-CONSISTENT" if (a and b and c) else "NOT CONFIRMED",
                "checks": {"a_sharpe_above_EW": bool(a), "b_maxdd_10pp_shallower": bool(b), "c_sharpe_at_least_TS28": bool(c)},
                "stat_ready": days >= STAT_DAYS,
                "note": ("enough days for the 14.0 statistical test" if days >= STAT_DAYS
                         else f"{STAT_DAYS - days} more days until a statistically meaningful look")}


def _fmt(st: dict) -> str:
    if not st.get("started"):
        return "Shadow ledger not started: run `python -m core.portfolio_shadow update`."
    lines = [f"Shadow ledger {st['start_day']} -> {st['last_day']}  ({st['days']} forward days"
             + (f", {st['days_behind']} day(s) behind: run update" if st["days_behind"] else "") + ")",
             f"  {'rule':<5}{'NAV':>8}{'return%':>9}{'Sharpe':>8}{'vol%':>7}{'maxDD%':>8}"]
    for r in RULES:
        x = st["stats"][r]
        tag = " <- pre-registered pick" if r == PRIMARY else " (descriptive)" if r in DESCRIPTIVE else ""
        f = lambda v, m=1.0, p=2: "   n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v * m:.{p}f}"
        lines.append(f"  {r:<5}{x['nav']:>8.3f}{x['total_return'] * 100:>9.1f}{f(x['sharpe']):>8}{f(x['ann_vol'], 100, 1):>7}{f(x['max_drawdown'], 100, 1):>8}{tag}")
    lines += ["", f"Current target weights of {PRIMARY} (what it would hold next):"]
    w = st["targets"][PRIMARY]
    lines.append("  " + ", ".join(f"{s.replace('USDT', '')} {v * 100:.1f}%" for s, v in w.items() if v > 0) + f"; cash {max(0.0, 1 - sum(w.values())) * 100:.1f}%"
                 if sum(w.values()) > 0 else "  all cash")
    fv = st["forward"]
    lines += ["", "Forward verdict (pre-registered, docs/PHASE_14_1_SHADOW_PREREGISTRATION.md): " + fv["verdict"]]
    if fv.get("checks"):
        lines.append("  " + json.dumps(fv["checks"]) + " -- " + fv["note"])
    lines.append("Sharpe/vol/drawdown are shown only after 30 days. No orders are placed by this ledger.")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["update", "status"])
    ap.add_argument("--path", default=None)
    args = ap.parse_args(argv)
    ledger = ShadowLedger(args.path)
    if args.cmd == "update":
        try:
            print(ledger.update())
        except ShadowError as exc:
            print(f"WARN shadow ledger not updated: {exc}")
            return 3
    print(_fmt(ledger.status()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
