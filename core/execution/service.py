"""Paper-only execution service: bar -> signal -> sizing -> risk approval -> order -> fill -> reconciliation, in a loop.

Guarantees (each has a test):
  * PAPER ONLY: refuses anything but a ``SimulatedBroker`` with strict prices, so it cannot place a real order.
  * ONE runner: a lease in the durable journal stops two processes (desktop + Dash + CLI) trading the same account.
  * IDEMPOTENT: one decision per (symbol, interval, completed bar, strategy). Restarts and re-runs of the same bar cannot
    place the order twice; the decision row is written BEFORE the order and updated after, so a crash mid-submit leaves an
    "ambiguous" row that is reported, never auto-retried.
  * NO LOOKAHEAD: the still-forming bar is dropped; decisions use completed bars only.
  * Exits are never blocked by risk limits (core/execution/risk.py); a halt is persistent until ``resume()``.
  * Every order carries a structured rationale (strategy signal, sizing, the risk verdict) stored in the journal.

CLI:  python -m core.execution.service run --symbols BTCUSDT --strategy "EMA Crossover" --interval 1h
      python -m core.execution.service status | halt "reason" | resume | flatten
"""
from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import pandas as pd

from brokers.simulatedbroker import OrderStatus, SimulatedBroker
from core.execution.journal import ExecutionJournal
from core.execution.risk import RiskConfig, RiskGate, RiskInput
from core.execution.signals import BacktraderReplaySignal
from core.logger import logger
from core.trade_rationale import build_rationale, submit_with_rationale


class PaperOnlyError(RuntimeError):
    """Raised when the service is handed anything that could move real money."""


_INTERVAL_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}
_DEFAULT_HISTORY_DAYS = {"1m": 3, "5m": 10, "15m": 20, "30m": 30, "1h": 60, "4h": 120, "1d": 500}


def paper_refusal(symbol: str) -> str:
    """Plain-words reason the service will not trade ``symbol`` in paper, or "" if it will. Indices, futures and FX are for
    charts and backtests only: they cannot be bought directly / need contract, margin or lot models the paper broker lacks.
    (Mirrors core/instruments.py; kept here so the service is safe on its own.)"""
    s = symbol.strip().upper()
    if s.startswith("^"):
        return f"{symbol} is chart/backtest-only: an index cannot be bought directly (use an ETF such as SPY or QQQ)"
    if s.endswith("=F"):
        return f"{symbol} is chart/backtest-only: futures need contract and margin handling that the paper broker does not model"
    if s.endswith("=X"):
        return f"{symbol} is chart/backtest-only: FX needs lot sizes, leverage and rollover that the paper broker does not model"
    return ""


def interval_seconds(interval: str) -> int:
    if interval not in _INTERVAL_SECONDS:
        raise ValueError(f"unsupported interval {interval!r}; use one of {sorted(_INTERVAL_SECONDS)}")
    return _INTERVAL_SECONDS[interval]


@dataclass
class ExecutionConfig:
    symbols: List[str]
    interval: str = "1d"
    history_days: Optional[int] = None
    poll_seconds: float = 30.0
    allocation_pct: float = 0.20          # target position notional per symbol, as a fraction of equity, when the signal is on
    allow_short: bool = False             # long/flat by default: shorting is unvalidated and the paper broker has no margin model
    min_bars: int = 60
    lease_ttl_s: Optional[float] = None
    risk: RiskConfig = field(default_factory=RiskConfig)

    def days(self) -> int:
        return self.history_days or _DEFAULT_HISTORY_DAYS.get(self.interval, 60)


def _to_utc_naive(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    return idx.tz_convert("UTC").tz_localize(None) if idx.tz is not None else idx


def drop_incomplete_bar(df: pd.DataFrame, interval: str, now_ts: float) -> pd.DataFrame:
    """Remove the last bar if it has not closed yet (its open time + one interval is still in the future)."""
    if df is None or len(df) == 0:
        return df
    last_open = _to_utc_naive(df.index)[-1]
    closes_at = last_open.to_pydatetime().replace(tzinfo=timezone.utc).timestamp() + interval_seconds(interval)
    return df.iloc[:-1] if now_ts < closes_at else df


class ExecutionService:
    def __init__(self, broker, data_loader, signal: BacktraderReplaySignal, config: ExecutionConfig,
                 journal: Optional[ExecutionJournal] = None, risk_gate: Optional[RiskGate] = None,
                 clock: Callable[[], float] = time.time):
        if not isinstance(broker, SimulatedBroker):
            raise PaperOnlyError(f"paper only: refusing to trade through {type(broker).__name__}")
        if not broker.strict_prices:
            raise PaperOnlyError("the paper broker must be created with strict_prices=True (otherwise it fills at invented prices)")
        for sym in config.symbols:
            reason = paper_refusal(sym)
            if reason:
                raise ValueError(reason)
        self.broker, self.loader, self.signal, self.cfg = broker, data_loader, signal, config
        self.journal = journal or ExecutionJournal()
        self.gate = risk_gate or RiskGate(config.risk)
        self._clock = clock
        self.interval_s = interval_seconds(config.interval)
        # short lease, renewed every ttl/3 even between ticks: after a hard crash another runner takes over within minutes,
        # not after a whole poll interval
        self.lease_ttl = config.lease_ttl_s or min(max(3 * config.poll_seconds, 60.0), 180.0)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._errors = 0
        self._per_symbol: Dict[str, dict] = {}
        self._last_error = ""

    # ------------------------------------------------------------------ control
    def halt(self, reason: str) -> None:
        """Persistent: blocks new entries until resume(), even across restarts. Exits stay allowed."""
        self.journal.halt(reason)
        logger.warning("execution HALTED: %s", reason)

    def resume(self) -> None:
        self.journal.resume()

    def flatten_all(self, reason: str = "flatten requested", halt: bool = True) -> List[str]:
        """Emergency: close every open position at the latest real price, and (by default) halt new entries."""
        closed = []
        if halt:
            self.halt(reason)
        for sym in list(self.broker.positions):
            pos = self.broker.get_position(sym)
            if pos is None or abs(pos.qty) < 1e-12:
                continue
            price = self._price(sym, fallback=None)
            if not price:
                self.journal.record(f"flatten|{int(self._clock())}|{sym}", symbol=sym, action="error", error="no price to flatten at")
                continue
            side = "sell" if pos.qty > 0 else "buy"
            did = f"flatten|{int(self._clock())}|{sym}"
            self.journal.record(did, symbol=sym, strategy="flatten", action="submitting", side=side, order_qty=abs(pos.qty), price=price)
            rationale = build_rationale(source="execution_service", action=f"close_{'long' if pos.qty > 0 else 'short'}",
                                        summary=f"Emergency flatten: {reason}", strategy="flatten", signal="flatten",
                                        features={"decision_id": did, "price": price})
            order = submit_with_rationale(self.broker, rationale, symbol=sym, qty=abs(pos.qty), side=side, order_type="market",
                                          execution_price=price)
            self.journal.update(did, action="submitted", status=order.status.value, order_id=order.id,
                                filled_qty=order.filled_qty, fill_price=order.filled_avg_price, rationale=rationale)
            closed.append(sym)
        return closed

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Start the loop on a daemon thread. Returns False if another runner holds the lease."""
        if self.running:
            return True
        if not self.journal.acquire_lease(self.lease_ttl, self._clock()):
            logger.warning("execution: another runner holds the lease; not starting")
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="execution-service", daemon=True)
        self._thread.start()
        return True

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self.journal.release_lease()
        self._publish_status(running=False)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
                self._errors = 0
                wait = self.cfg.poll_seconds
            except Exception as exc:  # noqa: BLE001 -- the loop must survive anything; back off and report
                self._errors += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
                logger.error("execution tick failed (%d in a row): %s", self._errors, exc)
                self._publish_status(running=True)
                wait = min(300.0, self.cfg.poll_seconds * (2 ** min(self._errors, 5)))
            self._wait_renewing_lease(wait)

    def _wait_renewing_lease(self, seconds: float) -> None:
        remaining = seconds
        while remaining > 0 and not self._stop.is_set():
            chunk = min(remaining, self.lease_ttl / 3.0)
            if self._stop.wait(chunk):
                return
            remaining -= chunk
            if not self.journal.acquire_lease(self.lease_ttl, self._clock()):
                logger.error("execution: lost the lease to another runner; stopping this one")
                self._stop.set()
                return

    # ------------------------------------------------------------------ one pass
    def tick(self) -> dict:
        now = self._clock()
        if not self.journal.acquire_lease(self.lease_ttl, now):
            self._publish_status(running=False, note="another runner holds the lease")
            return {"skipped": "lease held elsewhere"}
        self._resolve_ambiguous(now)
        for sym in self.cfg.symbols:
            try:
                self._per_symbol[sym] = self._process_symbol(sym, now)
            except Exception as exc:  # noqa: BLE001 -- one symbol failing must not stop the others
                logger.error("execution: %s failed: %s", sym, exc)
                self._per_symbol[sym] = {"state": "error", "detail": f"{type(exc).__name__}: {exc}"}
                self._last_error = self._per_symbol[sym]["detail"]
        self._publish_status(running=True)
        return dict(self._per_symbol)

    # ------------------------------------------------------------------ helpers
    def _price(self, symbol: str, fallback: Optional[float]) -> Optional[float]:
        try:
            p = self.loader.get_latest_price(symbol)
            if p and math.isfinite(float(p)) and float(p) > 0:
                return float(p)
        except Exception as exc:  # noqa: BLE001
            logger.debug("latest price failed for %s: %s", symbol, exc)
        return fallback

    def _baselines(self, equity: float, now: float) -> tuple:
        day = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d")
        rec = self.journal.get("day_baseline")
        if not rec or rec.get("day") != day:
            rec = {"day": day, "equity": equity}
            self.journal.set("day_baseline", rec)
        peak = max(float(self.journal.get("peak_equity", equity) or equity), equity)
        self.journal.set("peak_equity", peak)
        return float(rec["equity"]), peak

    @staticmethod
    def _is_crypto(symbol: str) -> bool:
        from core.chart_builder import is_crypto_symbol
        return is_crypto_symbol(symbol)

    def _round_qty(self, symbol: str, qty: float) -> float:
        return round(qty, 6) if self._is_crypto(symbol) else float(math.floor(qty))

    def _gross_exposure(self) -> float:
        with self.broker._lock:
            return sum(abs(p.qty) * self.broker.market_data.get(s, p.avg_price) for s, p in self.broker.positions.items())

    # ------------------------------------------------------------------ the pipeline for one symbol
    def _process_symbol(self, symbol: str, now: float) -> dict:
        # 1. market data (completed bars only)
        df = self.loader.load_data(symbol=symbol, source="Historical", live=False, days=self.cfg.days(),
                                   interval=self.cfg.interval, include_news=False)
        if df is None or len(df) == 0:
            return {"state": "no_data", "detail": "no bars returned"}
        bars = drop_incomplete_bar(df, self.cfg.interval, now)
        if len(bars) < self.cfg.min_bars:
            return {"state": "warming_up", "detail": f"{len(bars)}/{self.cfg.min_bars} completed bars"}
        bar_open = _to_utc_naive(bars.index)[-1]
        bar_close_ts = bar_open.to_pydatetime().replace(tzinfo=timezone.utc).timestamp() + self.interval_s
        age_bars = max(0.0, (now - bar_close_ts) / self.interval_s)

        # 2. idempotency: one decision per completed bar
        decision_id = f"{symbol}|{self.cfg.interval}|{bar_open.isoformat()}|{self.signal.name}"
        if self.journal.has_decision(decision_id):
            return {"state": "up_to_date", "bar": bar_open.isoformat(), "data_age_bars": round(age_bars, 2)}

        # 3. signal
        sig = self.signal.evaluate(bars)
        if not sig.ok:
            if sig.error == "insufficient history":
                return {"state": "warming_up", "detail": sig.summary}
            self.journal.record(decision_id, symbol=symbol, bar_ts=bar_open.isoformat(), strategy=self.signal.name,
                                action="error", error=sig.error)
            return {"state": "error", "detail": sig.error, "bar": bar_open.isoformat()}

        # 4. price + account state
        last_close = float(bars["Close"].iloc[-1])
        price = self._price(symbol, fallback=last_close)
        self.broker.update_price(symbol, price)
        info = self.broker.get_account_info()
        equity = float(info["portfolio_value"])
        pos = self.broker.get_position(symbol)
        current = float(pos.qty) if pos else 0.0

        # 5. portfolio decision: signal-change trading toward a target size
        direction = sig.direction
        note = ""
        if direction < 0 and not self.cfg.allow_short:
            direction, note = 0, "short signal ignored (long/flat mode)"
        cur_sign = (current > 1e-12) - (current < -1e-12)
        target_qty = direction * self._round_qty(symbol, self.cfg.allocation_pct * equity / price) if direction else 0.0
        if direction == cur_sign:
            self.journal.record(decision_id, symbol=symbol, bar_ts=bar_open.isoformat(), strategy=self.signal.name, signal=sig.direction,
                                target_qty=target_qty, current_qty=current, action="hold", price=price,
                                rationale={"summary": f"{sig.summary}; already {'long' if cur_sign > 0 else 'flat' if not cur_sign else 'short'}, no order. {note}".strip()})
            return {"state": "hold", "signal": sig.direction, "position": current, "bar": bar_open.isoformat(),
                    "data_age_bars": round(age_bars, 2), "note": note}
        # a flip (long<->short) closes first; the new side is entered on the next bar's decision
        closing = bool(cur_sign) and direction != cur_sign
        delta = -current if closing else target_qty - current
        order_qty = abs(current) if closing else self._round_qty(symbol, abs(delta))
        side = "buy" if delta > 0 else "sell"
        if order_qty <= 0:
            self.journal.record(decision_id, symbol=symbol, bar_ts=bar_open.isoformat(), strategy=self.signal.name, signal=sig.direction,
                                target_qty=target_qty, current_qty=current, action="hold", price=price,
                                rationale={"summary": "target size rounds to zero at this price/equity"})
            return {"state": "hold", "detail": "size rounds to zero", "bar": bar_open.isoformat()}

        # 6. risk approval
        day_start, peak = self._baselines(equity, now)
        day_ts = datetime.fromtimestamp(now, tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        verdict = self.gate.evaluate(RiskInput(
            symbol=symbol, side=side, qty=order_qty, price=price, current_qty=current, equity=equity,
            day_start_equity=day_start, peak_equity=peak, gross_exposure=self._gross_exposure(),
            orders_today=self.journal.submitted_today(day_ts), data_age_bars=age_bars, halted=self.journal.halted(),
            is_crypto=self._is_crypto(symbol)))
        if not verdict.approved:
            self.journal.record(decision_id, symbol=symbol, bar_ts=bar_open.isoformat(), strategy=self.signal.name, signal=sig.direction,
                                target_qty=target_qty, current_qty=current, order_qty=order_qty, side=side, price=price,
                                action="blocked", risk=verdict.as_dict(), rationale={"summary": f"{sig.summary}; blocked: {'; '.join(verdict.reasons)}"})
            return {"state": "blocked", "reasons": verdict.reasons, "bar": bar_open.isoformat(), "data_age_bars": round(age_bars, 2)}

        # 7. record intent BEFORE sending, then order, then update (crash between = "ambiguous", reported and never auto-retried)
        qty = verdict.qty if verdict.is_exit else self._round_qty(symbol, verdict.qty)
        if verdict.is_exit:
            action = "close_short" if side == "buy" else "close_long"
        else:
            action = "open_long" if side == "buy" else "open_short"
        rationale = build_rationale(
            source="execution_service", action=action,
            summary=f"{sig.summary}; {side} {qty:g} {symbol} @ {price:.6g} ({'; '.join(verdict.reasons)})",
            strategy=self.signal.name, signal=sig.signal,
            features={"decision_id": decision_id, "bar": bar_open.isoformat(), "price": price, "equity": equity,
                      "current_qty": current, "target_qty": target_qty, "data_age_bars": round(age_bars, 2), **sig.features},
            thresholds={"allocation_pct": self.cfg.allocation_pct, "max_position_pct": self.gate.cfg.max_position_pct,
                        "max_gross_pct": self.gate.cfg.max_gross_pct, "max_daily_loss_pct": self.gate.cfg.max_daily_loss_pct,
                        "max_drawdown_pct": self.gate.cfg.max_drawdown_pct})
        if not self.journal.record(decision_id, symbol=symbol, bar_ts=bar_open.isoformat(), strategy=self.signal.name, signal=sig.direction,
                                   target_qty=target_qty, current_qty=current, order_qty=qty, side=side, price=price,
                                   action="submitting", risk=verdict.as_dict(), rationale=rationale):
            return {"state": "up_to_date", "bar": bar_open.isoformat()}          # another runner won the race
        order = submit_with_rationale(self.broker, rationale, symbol=symbol, qty=qty, side=side, order_type="market",
                                      execution_price=price)
        self.journal.update(decision_id, action="submitted", status=order.status.value, order_id=order.id,
                            filled_qty=order.filled_qty, fill_price=order.filled_avg_price,
                            error=order.reject_reason or None)

        # 8. reconciliation of this order
        after = self.broker.get_position(symbol)
        after_qty = float(after.qty) if after else 0.0
        expected = current + (order.filled_qty if side == "buy" else -order.filled_qty)
        state = "traded" if order.status == OrderStatus.FILLED else "rejected"
        detail = {"state": state, "side": side, "qty": qty, "price": price, "bar": bar_open.isoformat(),
                  "order_status": order.status.value, "position": after_qty, "data_age_bars": round(age_bars, 2)}
        if abs(after_qty - expected) > 1e-6 * max(1.0, abs(expected)):
            msg = f"position mismatch after order: expected {expected:g}, broker has {after_qty:g}"
            self.journal.update(decision_id, error=msg)
            detail["reconciliation"] = msg
            logger.error("execution: %s %s", symbol, msg)
        return detail

    # ------------------------------------------------------------------ reconciliation
    def _resolve_ambiguous(self, now: float, grace_s: float = 60.0) -> None:
        """A decision left in 'submitting' means we died between recording intent and hearing back. Look for the order in the
        broker (the rationale carries the decision id); if found, adopt it, else flag it. NEVER re-submit automatically."""
        for d in self.journal.decisions(limit=200):
            if d["action"] != "submitting" or now - d["ts"] < grace_s:
                continue
            match = next((o for o in self.broker.get_orders()
                          if (o.rationale.get("features") or {}).get("decision_id") == d["decision_id"]), None)
            if match is not None:
                self.journal.update(d["decision_id"], action="submitted", status=match.status.value, order_id=match.id,
                                    filled_qty=match.filled_qty, fill_price=match.filled_avg_price,
                                    error="recovered after interruption")
            else:
                self.journal.update(d["decision_id"], action="error",
                                    error="ambiguous: interrupted before the broker confirmed; NOT retried, check the account")

    def reconcile(self) -> dict:
        """Compare what the journal says the service did with what the broker holds. Never mutates anything."""
        issues: List[str] = []
        net: Dict[str, float] = {}
        for d in self.journal.decisions(limit=5000):
            if d["action"] == "submitted" and d["status"] == "filled" and d["symbol"] in self.cfg.symbols:
                sign = 1.0 if d["side"] == "buy" else -1.0
                net[d["symbol"]] = net.get(d["symbol"], 0.0) + sign * float(d["filled_qty"] or 0)
        for sym in self.cfg.symbols:
            pos = self.broker.get_position(sym)
            actual = float(pos.qty) if pos else 0.0
            if abs(actual - net.get(sym, 0.0)) > 1e-6 * max(1.0, abs(actual)):
                issues.append(f"{sym}: service-filled net {net.get(sym, 0.0):g} but broker holds {actual:g} "
                              f"(manual trades in the same account, or a second system?)")
        for d in self.journal.decisions(limit=500):
            if d["action"] == "submitting":
                issues.append(f"{d['decision_id']}: ambiguous submission (interrupted)")
            if d.get("error") and str(d["error"]).startswith("position mismatch"):
                issues.append(f"{d['decision_id']}: {d['error']}")
        return {"ok": not issues, "issues": issues}

    # ------------------------------------------------------------------ status (also read by both UIs from the journal)
    def _publish_status(self, running: bool, note: str = "") -> None:
        rec = self.reconcile()
        try:
            info = self.broker.get_account_info()
            acct = {"equity": info["portfolio_value"], "cash": info["cash"], "realized_pnl": info["realized_pnl"],
                    "unrealized_pnl": info["unrealized_pnl"]}
        except Exception:  # noqa: BLE001
            acct = {}
        self.journal.set("status", {
            "running": running, "owner": self.journal.owner_id, "heartbeat": self._clock(), "poll_seconds": self.cfg.poll_seconds,
            "symbols": self.cfg.symbols, "interval": self.cfg.interval, "strategy": self.signal.name, "per_symbol": self._per_symbol,
            "halted": self.journal.halted(), "last_error": self._last_error, "note": note, "reconciliation": rec, "account": acct,
            "allocation_pct": self.cfg.allocation_pct, "allow_short": self.cfg.allow_short,
            "risk": {k: v for k, v in vars(self.gate.cfg).items() if k != "kill_switch_file"}})


# ---------------------------------------------------------------------- CLI
def _cli(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Paper-only execution service")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--symbols", nargs="+", required=True)
    r.add_argument("--strategy", default="EMA Crossover")
    r.add_argument("--interval", default="1d")
    r.add_argument("--poll", type=float, default=30.0)
    r.add_argument("--allocation", type=float, default=0.20)
    r.add_argument("--trend-overlay", action="store_true")
    r.add_argument("--allow-short", action="store_true")
    r.add_argument("--wait", action="store_true", help="if another runner holds the lease, keep retrying instead of exiting (for launchd)")
    for name in ("status", "resume", "flatten"):
        sub.add_parser(name)
    h = sub.add_parser("halt"); h.add_argument("reason", nargs="?", default="manual halt")
    args = ap.parse_args(argv)

    from brokers.paper_store import default_account_path
    journal = ExecutionJournal()
    if args.cmd == "status":
        from core.execution.view import execution_view
        v = execution_view(journal)
        print(v["headline"]); [print("  " + i) for i in v["issues"]]
        for d in v["decisions"][:15]:
            print(f"  {d['when']}  {d['symbol']:<9} {d['action']:<10} {d['detail']}")
        return 0
    if args.cmd in ("halt", "resume"):
        journal.halt(args.reason) if args.cmd == "halt" else journal.resume()
        print("halted" if args.cmd == "halt" else "resumed")
        return 0
    from core.data_loader import DataLoader
    import backtrader as bt
    from core.strategy_manager import StrategyManager

    def backtrader_strategies() -> dict:
        out = {}
        for name, cls in StrategyManager().strategies.items():
            try:
                if issubclass(cls, bt.Strategy):
                    out[name] = cls
            except TypeError:
                pass
        return out
    broker = SimulatedBroker(persist_path=default_account_path(), strict_prices=True, max_price_age_s=300.0)
    if args.cmd == "flatten":
        svc = ExecutionService(broker, DataLoader(), BacktraderReplaySignal(list(backtrader_strategies().values())[0]),
                               ExecutionConfig(symbols=list(broker.positions) or ["-"]), journal)
        print("closed:", svc.flatten_all("CLI flatten") or "nothing open")
        return 0
    strategies = backtrader_strategies()
    if args.strategy not in strategies:
        raise SystemExit(f"unknown strategy {args.strategy!r}; choose from {sorted(strategies)}")
    cfg = ExecutionConfig(symbols=args.symbols, interval=args.interval, poll_seconds=args.poll,
                          allocation_pct=args.allocation, allow_short=args.allow_short)
    svc = ExecutionService(broker, DataLoader(), BacktraderReplaySignal(strategies[args.strategy], name=args.strategy,
                                                                          trend_overlay=args.trend_overlay), cfg, journal)
    import signal as _signal
    _signal.signal(_signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))    # launchd stop -> clean shutdown, lease released
    try:
        while not svc.start():
            if not args.wait:
                raise SystemExit("another runner holds the lease (desktop app, Dash or another CLI is already trading this account)")
            print("another runner holds the lease; waiting...", flush=True)
            time.sleep(max(15.0, args.poll))
        print(f"paper execution running: {cfg.symbols} {cfg.interval} {svc.signal.name}; Ctrl+C to stop", flush=True)
        while svc.running:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        svc.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
