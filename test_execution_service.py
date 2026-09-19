"""Paper execution service: pipeline, safety guarantees, idempotency, risk, reconciliation."""
import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from brokers.simulatedbroker import OrderStatus, SimulatedBroker
from core.backtester import Backtester
from core.execution.journal import ExecutionJournal
from core.execution.risk import RiskConfig
from core.execution.service import ExecutionConfig, ExecutionService, PaperOnlyError, drop_incomplete_bar
from core.execution.signals import BacktraderReplaySignal
from core.execution.view import execution_view
from strategies.simple_strategies import EMACrossoverStrategy

logging.disable(logging.CRITICAL)
DAY = 86400.0


def _ts(d) -> float:
    return pd.Timestamp(d).to_pydatetime().replace(tzinfo=timezone.utc).timestamp()


def frame(kind="wavy", n=300, seed=3, start="2023-01-01"):
    r = np.random.default_rng(seed)
    if kind == "wavy":       # regimes flip every 40 bars
        drift = np.repeat(np.tile([0.01, -0.01], n // 80 + 1), 40)[:n]
    elif kind == "updown":   # down 60, up 80, down the rest: one long round trip
        drift = np.r_[np.full(60, -0.01), np.full(80, 0.012), np.full(n - 140, -0.012)]
    else:
        raise ValueError(kind)
    close = 20_000 * np.cumprod(1 + drift + r.normal(0, 0.006, n))
    op = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"Open": op, "High": np.maximum(op, close) * 1.002, "Low": np.minimum(op, close) * 0.998,
                         "Close": close, "Volume": 1e4}, index=pd.date_range(start, periods=n, freq="D"))


class FakeLoader:
    def __init__(self, df, price=None):
        self.df, self.k, self.price, self.fail, self.calls = df, len(df) - 1, price, False, 0

    def load_data(self, **kw):
        self.calls += 1
        if self.fail:
            raise RuntimeError("data source down")
        return self.df.iloc[: self.k + 1].copy()

    def get_latest_price(self, symbol):
        if self.price == "error":
            raise RuntimeError("no price")
        return self.price if self.price is not None else float(self.df["Close"].iloc[self.k])


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def make(tmp_path, df=None, symbols=("BTCUSDT",), allow_short=False, risk=None, allocation=0.2, name="j.sqlite3", broker=None,
         journal=None, clock=None, **cfg):
    df = df if df is not None else frame("updown")
    broker = broker or SimulatedBroker(strict_prices=True)
    loader = FakeLoader(df)
    clock = clock or Clock(_ts(df.index[-1]) + DAY + 60)        # just after the last bar closes
    journal = journal or ExecutionJournal(str(tmp_path / name))
    svc = ExecutionService(broker, loader, BacktraderReplaySignal(EMACrossoverStrategy, name="EMA"),
                           ExecutionConfig(symbols=list(symbols), interval="1d", allow_short=allow_short, allocation_pct=allocation,
                                           risk=risk or RiskConfig(), **cfg), journal, clock=clock)
    return svc, broker, loader, clock, journal


def step_through(svc, loader, clock, df, first=60):
    """Walk history forward one completed bar at a time, ticking after each bar closes."""
    trades = []
    for k in range(first, len(df)):
        loader.k = k
        clock.t = _ts(df.index[k]) + DAY + 60
        res = svc.tick()[svc.cfg.symbols[0]]
        if res.get("state") == "traded":
            trades.append((df.index[k].date(), res["side"]))
    return trades


# ------------------------------------------------------------------ paper-only guarantee

def test_refuses_anything_that_could_touch_real_money(tmp_path):
    class LiveLooking:                      # stands in for a Binance/Alpaca/IBKR connector
        def submit_order(self, *a, **k): raise AssertionError("must never be called")
    j = ExecutionJournal(str(tmp_path / "j.sqlite3"))
    cfg = ExecutionConfig(symbols=["BTCUSDT"])
    sig = BacktraderReplaySignal(EMACrossoverStrategy)
    with pytest.raises(PaperOnlyError):
        ExecutionService(LiveLooking(), FakeLoader(frame()), sig, cfg, j)
    with pytest.raises(PaperOnlyError):     # even the paper broker must be strict (no invented prices)
        ExecutionService(SimulatedBroker(strict_prices=False), FakeLoader(frame()), sig, cfg, j)


# ------------------------------------------------------------------ pipeline behaviour

def test_long_only_round_trip_buys_the_up_trend_and_exits_flat_never_short(tmp_path):
    df = frame("updown")
    svc, broker, loader, clock, _ = make(tmp_path, df)
    trades = step_through(svc, loader, clock, df)
    sides = [s for _, s in trades]
    assert sides[:2] == ["buy", "sell"], trades                        # entered the uptrend, left it
    assert not any(p.qty < 0 for p in broker.positions.values())      # never short in long/flat mode
    assert not broker.positions or all(abs(p.qty) < 1e-9 for p in broker.positions.values()) or sides[-1] == "buy"


def test_walk_forward_trades_land_on_the_same_signal_bars_as_a_backtest(tmp_path):
    """Feed the service history one completed bar at a time and compare with ONE backtest of the whole series: the service
    must trade on the signal bar (the bar before the backtest's next-open fill), for every trade after warm-up."""
    df = frame("wavy")
    svc, broker, loader, clock, _ = make(tmp_path, df, allow_short=True)
    trades = step_through(svc, loader, clock, df)
    b = Backtester()
    b.add_data(df.copy()); b.add_strategy(EMACrossoverStrategy)
    b.run_backtest(cash=100_000, benchmark_ticker=None, market_fee=0, limit_fee=0)
    bt = [((pd.Timestamp(s["date"]) - pd.Timedelta(days=1)).date(), "buy" if s["type"] in ("buy", "buy_cover") else "sell")
          for s in b.cerebro.runstrats[0][0].signals]
    bt_after_warmup = [t for t in bt if t[0] > df.index[60].date()]
    svc_after = [t for t in trades if t[0] > df.index[60].date()]
    assert len(bt_after_warmup) >= 5
    assert svc_after == bt_after_warmup


def test_incomplete_bar_is_dropped_so_decisions_never_use_a_bar_that_is_still_forming(tmp_path):
    df = frame("updown")
    now = _ts(df.index[-1]) + DAY / 2                                  # halfway through the last bar
    assert len(drop_incomplete_bar(df, "1d", now)) == len(df) - 1
    assert len(drop_incomplete_bar(df, "1d", now + DAY)) == len(df)
    tz = df.copy(); tz.index = tz.index.tz_localize("UTC")            # tz-aware indexes (yfinance) behave the same
    assert len(drop_incomplete_bar(tz, "1d", now)) == len(df) - 1
    svc, _, _, clock, journal = make(tmp_path, df, clock=Clock(now))
    svc.tick()
    ids = [d["decision_id"] for d in journal.decisions()]
    assert ids and all(str(df.index[-1].isoformat()) not in i for i in ids)


def test_short_signal_is_ignored_in_long_only_mode_and_recorded_as_such(tmp_path):
    df = frame("wavy")
    svc, broker, loader, clock, journal = make(tmp_path, df, allow_short=False)
    step_through(svc, loader, clock, df)
    assert not broker.positions or all(p.qty >= 0 for p in broker.positions.values())
    assert any("long/flat" in str((d.get("rationale") or {}).get("summary", "")) or d["action"] == "hold" for d in journal.decisions())


# ------------------------------------------------------------------ idempotency and restarts

def test_same_bar_twice_places_one_order_and_a_restart_does_not_repeat_it(tmp_path):
    df = frame("updown")
    k = next(i for i in range(60, len(df)) if True)
    svc, broker, loader, clock, journal = make(tmp_path, df)
    # find a bar where the service trades
    for kk in range(60, len(df)):
        loader.k = kk; clock.t = _ts(df.index[kk]) + DAY + 60
        if svc.tick()["BTCUSDT"].get("state") == "traded":
            break
    n_orders = len(broker.get_orders())
    assert n_orders >= 1
    assert svc.tick()["BTCUSDT"]["state"] == "up_to_date"              # same bar again: nothing happens
    assert len(broker.get_orders()) == n_orders
    svc2 = ExecutionService(broker, loader, BacktraderReplaySignal(EMACrossoverStrategy, name="EMA"),
                            ExecutionConfig(symbols=["BTCUSDT"], interval="1d"), ExecutionJournal(journal.path), clock=clock)
    assert svc2.tick() == {"skipped": "lease held elsewhere"}          # the first runner is still alive: no second trader
    clock.t += 3600                                                    # ...it crashed and its lease expired: a restart takes over
    assert svc2.tick()["BTCUSDT"]["state"] == "up_to_date"             # and the journal stops it repeating the order
    assert len(broker.get_orders()) == n_orders


def test_decision_row_is_unique_per_bar_at_the_database_level(tmp_path):
    j = ExecutionJournal(str(tmp_path / "j.sqlite3"))
    assert j.record("d1", symbol="X", action="submitting") is True
    assert j.record("d1", symbol="X", action="submitting") is False    # the UNIQUE constraint, not just an if-check


def test_interrupted_submission_is_flagged_not_retried_and_a_landed_order_is_adopted(tmp_path):
    svc, broker, loader, clock, journal = make(tmp_path)
    journal.record("lost|1", symbol="BTCUSDT", action="submitting", side="buy", order_qty=1)
    journal.record("landed|1", symbol="BTCUSDT", action="submitting", side="buy", order_qty=1)
    broker.update_price("BTCUSDT", 100.0)
    broker.submit_order("BTCUSDT", 1, "buy", execution_price=100.0, rationale={"features": {"decision_id": "landed|1"}})
    for d in ("lost|1", "landed|1"):
        journal.update(d, ts=1.0)                                       # old enough to be past the grace period
    before = len(broker.get_orders())
    svc._resolve_ambiguous(clock.t)
    rows = {d["decision_id"]: d for d in journal.decisions()}
    assert rows["lost|1"]["action"] == "error" and "NOT retried" in rows["lost|1"]["error"]
    assert rows["landed|1"]["action"] == "submitted" and rows["landed|1"]["status"] == "filled"
    assert len(broker.get_orders()) == before                           # nothing was re-submitted


# ------------------------------------------------------------------ risk gate through the pipeline

def _first_entry_bar(df):
    sig = BacktraderReplaySignal(EMACrossoverStrategy, name="EMA")
    for k in range(60, len(df)):
        if sig.evaluate(df.iloc[: k + 1]).direction > 0:
            return k
    raise AssertionError("fixture has no long signal")


def test_entry_is_blocked_after_a_daily_loss_but_the_exit_is_not(tmp_path):
    df = frame("updown")
    svc, broker, loader, clock, journal = make(tmp_path, df)
    k = _first_entry_bar(df)
    loader.k, clock.t = k, _ts(df.index[k]) + DAY + 60
    day = datetime.fromtimestamp(clock.t, tz=timezone.utc).strftime("%Y-%m-%d")
    journal.set("day_baseline", {"day": day, "equity": 100_000 * 1.10})   # equity is ~9% below the day's start
    res = svc.tick()["BTCUSDT"]
    assert res["state"] == "blocked" and "daily loss" in res["reasons"][0]
    assert not broker.positions
    # now hold a position and let the sell signal arrive while the loss limit is still tripped: exits are never blocked
    broker.update_price("BTCUSDT", 1000.0); broker.submit_order("BTCUSDT", 5, "buy", execution_price=1000.0)
    journal.set("day_baseline", {"day": day, "equity": 10_000_000})
    for kk in range(k + 1, len(df)):
        loader.k, clock.t = kk, _ts(df.index[kk]) + DAY + 60
        r = svc.tick()["BTCUSDT"]
        if r.get("state") == "traded" and r["side"] == "sell":
            break
    else:
        pytest.fail("no exit occurred")
    assert not broker.positions


def test_halt_survives_a_restart_blocks_entries_and_resume_lifts_it(tmp_path):
    df = frame("updown")
    svc, broker, loader, clock, journal = make(tmp_path, df)
    svc.halt("owner said stop")
    svc.journal.release_lease()
    svc2 = ExecutionService(broker, loader, BacktraderReplaySignal(EMACrossoverStrategy, name="EMA"),
                            ExecutionConfig(symbols=["BTCUSDT"], interval="1d"), ExecutionJournal(journal.path), clock=clock)
    k = _first_entry_bar(df)
    loader.k, clock.t = k, _ts(df.index[k]) + DAY + 60
    r = svc2.tick()["BTCUSDT"]
    assert r["state"] == "blocked" and "owner said stop" in r["reasons"][0] and not broker.positions
    svc2.resume()
    svc2.journal.release_lease()
    clock.t += 10
    svc3 = ExecutionService(broker, loader, BacktraderReplaySignal(EMACrossoverStrategy, name="EMA"),
                            ExecutionConfig(symbols=["BTCUSDT"], interval="1d"), ExecutionJournal(journal.path), clock=clock)
    # the blocked decision for this bar is final (idempotent); the NEXT bar with a signal can trade again
    loader.k = k + 1; clock.t = _ts(df.index[k + 1]) + DAY + 60
    assert svc3.tick()["BTCUSDT"]["state"] in ("traded", "hold")


def test_entry_size_is_shrunk_to_the_per_symbol_limit(tmp_path):
    df = frame("updown")
    svc, broker, loader, clock, journal = make(tmp_path, df, allocation=0.60, risk=RiskConfig(max_position_pct=0.25))
    k = _first_entry_bar(df)
    loader.k, clock.t = k, _ts(df.index[k]) + DAY + 60
    r = svc.tick()["BTCUSDT"]
    assert r["state"] == "traded"
    pos = broker.get_position("BTCUSDT")
    price = float(df["Close"].iloc[k])
    assert pos.qty * price == pytest.approx(0.25 * 100_000, rel=0.02)   # 25% of equity, not the 60% the strategy asked for
    d = journal.decisions()[0]
    assert any("reduced" in x for x in d["risk"]["reasons"])


def test_stale_data_blocks_entries(tmp_path):
    df = frame("updown")
    svc, broker, loader, clock, _ = make(tmp_path, df)
    k = _first_entry_bar(df)
    loader.k, clock.t = k, _ts(df.index[k]) + DAY * 12                  # the newest completed bar is 11 intervals old
    r = svc.tick()["BTCUSDT"]
    assert r["state"] == "blocked" and "stale" in r["reasons"][0] and not broker.positions


def test_broker_rejection_is_recorded_with_its_reason_and_leaves_no_position(tmp_path, monkeypatch):
    df = frame("updown")
    svc, broker, loader, clock, journal = make(tmp_path, df)

    def reject(order, fill_price):
        order.status = OrderStatus.REJECTED
        order.reject_reason = "simulated exchange rejection"
    monkeypatch.setattr(broker, "_fill_order", reject)
    k = _first_entry_bar(df)
    loader.k, clock.t = k, _ts(df.index[k]) + DAY + 60
    r = svc.tick()["BTCUSDT"]
    d = journal.decisions()[0]
    assert r["state"] == "rejected" and r["order_status"] == "rejected"
    assert d["status"] == "rejected" and d["error"] == "simulated exchange rejection"
    assert not broker.positions and "reconciliation" not in r          # nothing filled, nothing to reconcile


# ------------------------------------------------------------------ failures do not become trades

def test_data_failure_places_no_order_and_other_symbols_continue(tmp_path):
    df = frame("updown")
    svc, broker, loader, clock, _ = make(tmp_path, df, symbols=("BTCUSDT", "ETHUSDT"))
    orig = loader.load_data
    loader.load_data = lambda **kw: (_ for _ in ()).throw(RuntimeError("down")) if kw["symbol"] == "BTCUSDT" else orig(**kw)
    out = svc.tick()
    assert out["BTCUSDT"]["state"] == "error" and "down" in out["BTCUSDT"]["detail"]
    assert out["ETHUSDT"]["state"] in ("hold", "traded", "blocked", "up_to_date")     # the other symbol still ran
    assert not [o for o in broker.get_orders() if o.symbol == "BTCUSDT"]


def test_price_feed_failure_falls_back_to_the_last_completed_close(tmp_path):
    df = frame("updown")
    svc, broker, loader, clock, _ = make(tmp_path, df)
    loader.price = "error"
    k = _first_entry_bar(df)
    loader.k, clock.t = k, _ts(df.index[k]) + DAY + 60
    r = svc.tick()["BTCUSDT"]
    assert r["state"] == "traded" and r["price"] == pytest.approx(float(df["Close"].iloc[k]))


def test_too_little_history_waits_instead_of_guessing(tmp_path):
    df = frame("updown")
    svc, broker, loader, clock, _ = make(tmp_path, df)
    loader.k = 20
    clock.t = _ts(df.index[20]) + DAY + 60
    assert svc.tick()["BTCUSDT"]["state"] == "warming_up" and not broker.get_orders()


# ------------------------------------------------------------------ one runner at a time

def test_only_one_runner_holds_the_lease_and_a_stale_lease_can_be_taken_over(tmp_path):
    df = frame("updown")
    clock = Clock(1_000_000.0)
    a, _, _, _, ja = make(tmp_path, df, clock=clock, name="shared.sqlite3")
    b, _, _, _, jb = make(tmp_path, df, clock=clock, name="shared.sqlite3")
    assert a.journal.owner_id != b.journal.owner_id
    assert a.journal.acquire_lease(60, clock.t) is True
    assert b.journal.acquire_lease(60, clock.t + 10) is False           # a is fresh
    assert b.tick() == {"skipped": "lease held elsewhere"}
    assert b.journal.acquire_lease(60, clock.t + 120) is True           # a stopped renewing: b takes over
    assert a.journal.acquire_lease(60, clock.t + 130) is False


# ------------------------------------------------------------------ emergency + reconciliation

def test_flatten_all_closes_everything_records_it_and_halts_entries(tmp_path):
    df = frame("updown")
    svc, broker, loader, clock, journal = make(tmp_path, df)
    broker.update_price("BTCUSDT", 100.0); broker.submit_order("BTCUSDT", 3, "buy", execution_price=100.0)
    assert svc.flatten_all("test") == ["BTCUSDT"]
    assert not broker.positions
    assert journal.halted()["reason"] == "test"
    assert any(d["strategy"] == "flatten" and d["status"] == "filled" for d in journal.decisions())


def test_reconcile_reports_manual_trades_and_is_clean_otherwise(tmp_path):
    df = frame("updown")
    svc, broker, loader, clock, journal = make(tmp_path, df)
    for kk in range(60, len(df)):
        loader.k, clock.t = kk, _ts(df.index[kk]) + DAY + 60
        svc.tick()
    assert svc.reconcile()["ok"]
    broker.update_price("BTCUSDT", 100.0); broker.submit_order("BTCUSDT", 2, "buy", execution_price=100.0)   # a manual UI trade
    rec = svc.reconcile()
    assert not rec["ok"] and "manual trades" in rec["issues"][0]


def test_every_order_carries_a_structured_rationale_in_the_journal_and_the_broker(tmp_path):
    df = frame("updown")
    svc, broker, loader, clock, journal = make(tmp_path, df)
    k = _first_entry_bar(df)
    loader.k, clock.t = k, _ts(df.index[k]) + DAY + 60
    assert svc.tick()["BTCUSDT"]["state"] == "traded"
    d = journal.decisions()[0]
    assert d["rationale"]["source"] == "execution_service" and d["rationale"]["features"]["decision_id"] == d["decision_id"]
    assert d["risk"]["approved"] is True and d["rationale"]["thresholds"]["max_position_pct"] == 0.25
    assert broker.get_orders()[0].rationale["features"]["decision_id"] == d["decision_id"]


# ------------------------------------------------------------------ status view and the loop

def test_view_shows_running_only_while_the_heartbeat_is_fresh_and_flags_a_dead_service(tmp_path):
    df = frame("updown")
    clock = Clock(_ts(df.index[-1]) + DAY + 60)
    svc, broker, loader, _, journal = make(tmp_path, df, clock=clock)
    assert svc.journal.acquire_lease(svc.lease_ttl, clock.t)
    svc.tick()
    v = execution_view(journal, now=clock.t + 5)
    assert v["running"] and "RUNNING" in v["headline"]
    dead = execution_view(journal, now=clock.t + 10_000)
    assert not dead["running"] and "STOPPED" in dead["headline"] and any("heartbeat" in i for i in dead["issues"])
    svc.halt("x")
    assert "HALTED" in execution_view(journal, now=clock.t + 5)["headline"]


def test_view_never_raises_on_a_broken_journal(tmp_path):
    class Broken:
        def get(self, *a, **k): raise RuntimeError("db locked")
    v = execution_view(Broken())
    assert "unavailable" in v["headline"] and v["decisions"] == []


def test_background_loop_runs_ticks_and_stops_cleanly(tmp_path):
    df = frame("updown")
    svc, broker, loader, clock, journal = make(tmp_path, df, poll_seconds=0.05)
    assert svc.start() and svc.running
    time.sleep(0.4)
    svc.stop()
    assert not svc.running and journal.get("status")["running"] is False
    assert journal.get("lease") is None                                  # released, so another runner can start


def test_loop_survives_an_exception_and_backs_off(tmp_path):
    df = frame("updown")
    svc, broker, loader, clock, journal = make(tmp_path, df, poll_seconds=0.05)
    calls = {"n": 0}
    real = svc.tick
    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return real()
    svc.tick = flaky
    svc.start(); time.sleep(0.6); svc.stop()
    assert calls["n"] >= 2 and "boom" in journal.get("status")["last_error"]
