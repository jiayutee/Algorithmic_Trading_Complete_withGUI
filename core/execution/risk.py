"""Pre-trade risk approval for the paper execution service.

Design rule: the gate can BLOCK or SHRINK an order that increases exposure ("entry"), but it never blocks an order that
reduces exposure ("exit"). "Stop trading" must never mean "cannot get out". Every verdict lists the checks that fired so
the reason is stored with the order.

This is the portfolio-level layer the live-order guard (brokers/execution_guard.py) does not have: per-symbol and gross
exposure, daily loss, drawdown from peak, order count and data staleness. All limits are fractions of current equity.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class RiskConfig:
    max_position_pct: float = 0.25        # one symbol's notional / equity
    max_gross_pct: float = 0.75           # sum of |notional| across symbols / equity
    max_daily_loss_pct: float = 0.03      # block entries once equity is this far below the day's starting equity
    max_drawdown_pct: float = 0.15        # block entries once equity is this far below its peak
    max_orders_per_day: int = 20
    max_data_age_bars: float = 3.0        # block entries if the newest COMPLETED bar is older than this many intervals (crypto, 24/7)
    max_data_age_bars_non_crypto: float = 5.0   # equities have weekends/holidays with no new bars, so allow longer
    min_order_notional: float = 10.0      # ignore dust
    kill_switch_file: Optional[str] = None   # if this file exists, entries are blocked (same idea as the live guard's .kill_switch)


@dataclass
class RiskInput:
    symbol: str
    side: str                              # "buy" | "sell"
    qty: float
    price: float
    current_qty: float                     # signed position in this symbol before the order
    equity: float
    day_start_equity: float
    peak_equity: float
    gross_exposure: float                  # sum |qty*price| over all open positions, before the order
    orders_today: int
    data_age_bars: float
    halted: Optional[Dict] = None
    is_crypto: bool = True


@dataclass
class RiskVerdict:
    approved: bool
    qty: float
    is_exit: bool
    reasons: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


class RiskGate:
    def __init__(self, config: Optional[RiskConfig] = None):
        self.cfg = config or RiskConfig()

    def evaluate(self, r: RiskInput) -> RiskVerdict:
        signed = r.qty if r.side == "buy" else -r.qty
        after = r.current_qty + signed
        is_exit = abs(after) < abs(r.current_qty) - 1e-12          # strictly reduces |position|
        if r.qty <= 0 or r.price <= 0:
            return RiskVerdict(False, 0.0, is_exit, ["non-positive quantity or price"])
        if is_exit:
            return RiskVerdict(True, r.qty, True, ["exit: risk limits never block reducing exposure"])

        cfg, reasons = self.cfg, []
        if r.halted:
            return RiskVerdict(False, 0.0, False, [f"halted: {r.halted.get('reason', 'no reason given')}"])
        if cfg.kill_switch_file:
            import os
            if os.path.exists(cfg.kill_switch_file):
                return RiskVerdict(False, 0.0, False, [f"kill switch file present ({cfg.kill_switch_file})"])
        limit = cfg.max_data_age_bars if r.is_crypto else cfg.max_data_age_bars_non_crypto
        if r.data_age_bars > limit:
            return RiskVerdict(False, 0.0, False, [f"data stale: newest completed bar is {r.data_age_bars:.1f} intervals old "
                                                   f"(limit {limit:g})"])
        if r.day_start_equity > 0 and r.equity / r.day_start_equity - 1 <= -cfg.max_daily_loss_pct:
            return RiskVerdict(False, 0.0, False, [f"daily loss limit hit: {(r.equity / r.day_start_equity - 1) * 100:.2f}% "
                                                   f"(limit -{cfg.max_daily_loss_pct * 100:.1f}%)"])
        if r.peak_equity > 0 and r.equity / r.peak_equity - 1 <= -cfg.max_drawdown_pct:
            return RiskVerdict(False, 0.0, False, [f"drawdown limit hit: {(r.equity / r.peak_equity - 1) * 100:.2f}% from peak "
                                                   f"(limit -{cfg.max_drawdown_pct * 100:.1f}%)"])
        if r.orders_today >= cfg.max_orders_per_day:
            return RiskVerdict(False, 0.0, False, [f"order limit reached: {r.orders_today} today (limit {cfg.max_orders_per_day})"])

        # size caps: shrink the entry to fit, block if nothing fits
        cur_notional = abs(r.current_qty) * r.price
        room_symbol = cfg.max_position_pct * r.equity - cur_notional
        room_gross = cfg.max_gross_pct * r.equity - r.gross_exposure
        room = min(room_symbol, room_gross)
        if room <= 0:
            which = "position" if room_symbol <= room_gross else "gross exposure"
            return RiskVerdict(False, 0.0, False, [f"{which} limit already reached"])
        qty = r.qty
        if qty * r.price > room:
            qty = room / r.price
            reasons.append(f"size reduced to ${room:,.2f} to respect the "
                           f"{'per-symbol' if room_symbol <= room_gross else 'gross exposure'} limit")
        if qty * r.price < cfg.min_order_notional:
            return RiskVerdict(False, 0.0, False, reasons + [f"order below minimum notional ${cfg.min_order_notional:g}"])
        return RiskVerdict(True, qty, False, reasons or ["all entry checks passed"])
