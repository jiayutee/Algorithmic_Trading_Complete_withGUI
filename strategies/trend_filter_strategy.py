"""Trend-filter strategy and overlay for backtrader (see core/trend_overlay.py for what the evidence does and does not say).

* ``TrendFilterStrategy``  -- standalone long/flat: long while the trailing 28-bar return is > 0, else cash.
* ``with_trend_overlay(cls)`` -- wrap ANY existing strategy so it is only allowed to hold positions while the trend is up:
  when the trend reads DOWN (or is unknown during warm-up) open positions are closed and the wrapped strategy is not
  consulted (no new entries, long or short). When it reads UP the wrapped strategy runs unchanged. The reading is
  refreshed weekly and lags, so a strategy can still open a short while the last reading was UP; it is then covered as
  soon as the reading flips. Cost: profits a strategy would make shorting a downtrend are given up.
"""
from __future__ import annotations

import backtrader as bt

from core.logger import logger
from core.trade_rationale import RationaleMixin
from core.trend_overlay import DEFAULT_LOOKBACK, DEFAULT_REBALANCE, describe, is_evaluation_bar, trend_is_up


def _recent_closes(data, lookback: int) -> list:
    """The last lookback+1 closes up to and including the current bar (fewer if not yet available)."""
    n = min(len(data), lookback + 1)
    return list(data.close.get(ago=0, size=n)) if n else []


class TrendFilterStrategy(RationaleMixin, bt.Strategy):
    params = (("lookback", DEFAULT_LOOKBACK), ("rebalance_every", DEFAULT_REBALANCE), ("size_fraction", 0.99))

    def __init__(self):
        self.signals = []
        self.order_count = 0
        self.closed_trades = []
        self._closing_long = False
        self._closing_short = False
        self._trend_up = None
        self._order = None

    def _state(self):
        return {"close": self.data.close[0], "lookback": self.params.lookback}

    def next(self):
        closes = _recent_closes(self.data, self.params.lookback)
        if is_evaluation_bar(len(self.data), self.params.lookback, self.params.rebalance_every):
            self._trend_up = trend_is_up(closes, self.params.lookback)
        if self._order is not None and self._order.alive():
            return
        if self._trend_up and not self.position:
            size = self.broker.getcash() * self.params.size_fraction / self.data.close[0]
            if size > 0.0001:
                self._set_rationale(action="open_long", strategy="Trend_Filter", signal="trend_up",
                                    summary=f"Opened LONG: {describe(closes, self.params.lookback)}",
                                    features=self._state(), thresholds={"lookback": self.params.lookback, "min_return": 0.0})
                self._order = self.buy(size=size)
                self.order_count += 1
        elif not self._trend_up and self.position.size > 0:
            self._set_rationale(action="close_long", strategy="Trend_Filter", signal="trend_down",
                                summary=f"Closed LONG: {describe(closes, self.params.lookback)}",
                                features=self._state(), thresholds={"lookback": self.params.lookback, "min_return": 0.0})
            self._closing_long = True
            self._order = self.close()
            self.order_count += 1

    def notify_order(self, order):
        if order.status == order.Completed:
            kind = ("buy" if order.isbuy() else ("sell" if self._closing_long else "sell_short"))
            if order.issell() and self._closing_long:
                self._closing_long = False
            self.signals.append({"date": self.data.datetime.datetime(0), "type": kind,
                                 "price": order.executed.price, "qty": order.executed.size})
            self._attach_rationale_to_last_signal()

    def notify_trade(self, trade):
        if trade.isclosed:
            self.closed_trades.append(trade)


def with_trend_overlay(strategy_cls, lookback: int = DEFAULT_LOOKBACK, rebalance_every: int = DEFAULT_REBALANCE):
    """Return a subclass of ``strategy_cls`` that only lets it hold positions while the trend is up."""
    if getattr(strategy_cls, "_trend_overlay_wrapped", False):
        return strategy_cls

    class TrendOverlaid(strategy_cls):
        _trend_overlay_wrapped = True
        _overlay_lookback = lookback
        _overlay_every = rebalance_every

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._overlay_up = None
            self._overlay_order = None
            if not hasattr(self, "_closing_long"):
                self._closing_long = False
                self._closing_short = False

        def _overlay_flatten(self, closes):
            long_side = self.position.size > 0
            if hasattr(self, "_set_rationale"):
                self._set_rationale(action="close_long" if long_side else "close_short", strategy="Trend_Overlay",
                                    signal="trend_down", summary=f"Closed {'LONG' if long_side else 'SHORT'} by trend overlay: "
                                    f"{describe(closes, self._overlay_lookback)}",
                                    features={"close": self.data.close[0]},
                                    thresholds={"lookback": self._overlay_lookback, "min_return": 0.0})
            if long_side:
                self._closing_long = True
            else:
                self._closing_short = True
            self._overlay_order = self.close()

        def next(self):
            closes = _recent_closes(self.data, self._overlay_lookback)
            if is_evaluation_bar(len(self.data), self._overlay_lookback, self._overlay_every):
                self._overlay_up = trend_is_up(closes, self._overlay_lookback)
                logger.debug("Trend overlay: %s", describe(closes, self._overlay_lookback))
            if self._overlay_up:
                return super().next()
            if self.position and not (self._overlay_order is not None and self._overlay_order.alive()):
                self._overlay_flatten(closes)

    TrendOverlaid.__name__ = f"{strategy_cls.__name__}WithTrendOverlay"
    TrendOverlaid.__qualname__ = TrendOverlaid.__name__
    return TrendOverlaid
