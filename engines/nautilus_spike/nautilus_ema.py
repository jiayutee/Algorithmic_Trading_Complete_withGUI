"""EMACrossoverStrategy ported to NautilusTrader's ``on_bar`` event-handler model.

Mirrors strategies/simple_strategies.py::EMACrossoverStrategy rule for rule so the two engines can be compared:
  * EMA(12) vs EMA(26) on the close; a cross is 'previous diff on one side, current diff on the other'.
  * Flat:  cross up   -> BUY  size; cross down -> SELL (open short) size.  size = 10% of cash / close.
  * Long:  cross down -> close (exit only, NO reversal).   Short: cross up -> close.
  * One decision per bar close; the market order fills on the simulated exchange afterwards.
Records a mark-to-market equity value on every bar so the equity curve can be scored with the same function as backtrader's.
"""
from __future__ import annotations

from decimal import Decimal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.indicators import ExponentialMovingAverage
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy


class EMACrossConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    ema_short: int = 12
    ema_long: int = 26
    risk_per_trade: float = 0.1


class EMACrossNautilus(Strategy):
    def __init__(self, config: EMACrossConfig) -> None:
        super().__init__(config)
        self.instrument = None
        self.ema_s = ExponentialMovingAverage(config.ema_short)
        self.ema_l = ExponentialMovingAverage(config.ema_long)
        self._prev_diff = None
        self.equity = []            # (bar ts_event ns, equity) per bar
        self.order_count = 0

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        self.register_indicator_for_bars(self.config.bar_type, self.ema_s)
        self.register_indicator_for_bars(self.config.bar_type, self.ema_l)
        self.subscribe_bars(self.config.bar_type)

    def _equity(self) -> float:
        acct = self.portfolio.account(self.instrument.id.venue)
        bal = float(acct.balance_total(USDT).as_decimal())
        upnl = self.portfolio.unrealized_pnl(self.instrument.id)
        return bal + (float(upnl.as_decimal()) if upnl is not None else 0.0)

    def on_bar(self, bar: Bar) -> None:
        self.equity.append((bar.ts_event, self._equity()))
        if not (self.ema_s.initialized and self.ema_l.initialized):
            return
        diff = self.ema_s.value - self.ema_l.value
        prev, self._prev_diff = self._prev_diff, diff
        if prev is None:
            return
        cross = 1 if (prev <= 0 < diff) else (-1 if (prev >= 0 > diff) else 0)
        if cross == 0:
            return
        net = self.portfolio.net_position(self.instrument.id)          # Decimal, signed
        if net == 0:
            acct = self.portfolio.account(self.instrument.id.venue)
            cash = float(acct.balance_free(USDT).as_decimal())
            size = cash * self.config.risk_per_trade / float(bar.close)
            qty = self.instrument.make_qty(size)
            if float(qty) <= 0.0001:
                return
            self.submit_order(self.order_factory.market(self.instrument.id, OrderSide.BUY if cross > 0 else OrderSide.SELL, qty))
            self.order_count += 1
        elif net > 0 and cross < 0:
            self.close_all_positions(self.instrument.id)
            self.order_count += 1
        elif net < 0 and cross > 0:
            self.close_all_positions(self.instrument.id)
            self.order_count += 1

    def on_stop(self) -> None:
        self.unsubscribe_bars(self.config.bar_type)
