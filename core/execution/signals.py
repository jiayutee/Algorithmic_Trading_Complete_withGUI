"""Turn the latest COMPLETED bars into a desired position direction.

``BacktraderReplaySignal`` replays an existing Backtrader strategy over the trailing history and reads what it wants to
hold *after the last bar*: its open position plus any order it issued on that final bar (which a backtest would fill on
the next bar's open). So the service acts on the signal bar itself, one bar earlier than a backtest's fill timestamp,
at essentially the same price -- and paper results stay comparable to the backtest of the same strategy.

Caveat (documented, not hidden): the replay starts at the first bar of the window, so a strategy whose state depends on
its full history (e.g. an EMA still converging) can differ slightly if the window start moves. Use a long, fixed window.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import backtrader as bt
import pandas as pd

from core.logger import logger


@dataclass
class SignalDecision:
    direction: int                       # +1 long, 0 flat, -1 short
    strategy: str
    summary: str
    signal: str = ""
    features: Dict = field(default_factory=dict)
    ok: bool = True
    error: str = ""


def _instrument(strategy_cls):
    """Subclass that records every order the strategy issues, with the bar it was issued on. Order objects come back from
    buy/sell/close even though nothing has filled yet, so the LAST bar's intent can be read after the replay finishes."""
    class Instrumented(strategy_cls):
        _rec_depth = 0

        def _wrapped(self, fn, *a, **k):
            # close() calls buy()/sell() internally: record only the OUTERMOST call or the order is counted twice
            self._rec_depth += 1
            try:
                order = fn(*a, **k)
            finally:
                self._rec_depth -= 1
            if self._rec_depth == 0:
                if not hasattr(self, "_issued_orders"):
                    self._issued_orders = []
                try:
                    self._issued_orders.append((len(self.data), float(order.created.size)))
                except Exception:  # noqa: BLE001
                    pass
            return order

        def buy(self, *a, **k):
            return self._wrapped(super().buy, *a, **k)

        def sell(self, *a, **k):
            return self._wrapped(super().sell, *a, **k)

        def close(self, *a, **k):
            return self._wrapped(super().close, *a, **k)
    Instrumented.__name__ = strategy_cls.__name__
    return Instrumented


class BacktraderReplaySignal:
    def __init__(self, strategy_cls, name: Optional[str] = None, min_bars: int = 60, trend_overlay: bool = False, **params):
        if trend_overlay:
            from strategies.trend_filter_strategy import with_trend_overlay
            strategy_cls = with_trend_overlay(strategy_cls)
        self.strategy_cls, self.params, self.min_bars = strategy_cls, params, min_bars
        self.name = (name or strategy_cls.__name__) + (" + trend overlay" if trend_overlay else "")

    def evaluate(self, df: pd.DataFrame) -> SignalDecision:
        if df is None or len(df) < self.min_bars:
            return SignalDecision(0, self.name, f"only {0 if df is None else len(df)} bars, need {self.min_bars}", ok=False,
                                  error="insufficient history")
        try:
            cerebro = bt.Cerebro(stdstats=False)
            cerebro.adddata(bt.feeds.PandasData(dataname=df.copy()))
            cerebro.addstrategy(_instrument(self.strategy_cls), **self.params)
            cerebro.broker.setcash(100_000.0)
            strat = cerebro.run()[0]
            held = float(strat.position.size)
            last_bar = len(df)
            pending = sum(sz for bar, sz in getattr(strat, "_issued_orders", []) if bar == last_bar)   # buy>0, sell<0; fills next bar
            want = held + pending
        except Exception as exc:  # noqa: BLE001 -- a strategy bug must not crash the loop
            logger.error("signal evaluation failed for %s: %s", self.name, exc)
            return SignalDecision(0, self.name, f"signal error: {exc}", ok=False, error=str(exc))
        direction = 1 if want > 1e-12 else (-1 if want < -1e-12 else 0)
        last = df.iloc[-1]
        return SignalDecision(direction, self.name,
                              f"{self.name} wants {'LONG' if direction > 0 else 'SHORT' if direction < 0 else 'FLAT'} "
                              f"after the bar closing at {last['Close']:.4g}",
                              signal={1: "long", -1: "short", 0: "flat"}[direction],
                              features={"close": float(last["Close"]), "held_after_replay": held, "pending_on_last_bar": pending,
                                        "bars_used": int(len(df))})
