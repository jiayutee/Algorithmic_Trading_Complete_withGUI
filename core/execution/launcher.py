"""One way for BOTH front ends to start the paper execution service (so they behave identically and the checks live in one place)."""
from __future__ import annotations

from typing import Optional, Tuple

from core.execution.journal import ExecutionJournal
from core.execution.risk import RiskConfig
from core.execution.service import ExecutionConfig, ExecutionService, PaperOnlyError, interval_seconds
from core.execution.signals import BacktraderReplaySignal


def default_poll_seconds(interval: str) -> float:
    """Check often enough to catch a bar close promptly without hammering the data source."""
    return float(min(300.0, max(15.0, interval_seconds(interval) / 60.0)))


def build_service(broker, loader, strategy_cls, strategy_name: str, symbols, interval: str, trend_overlay: bool = False,
                  allocation_pct: float = 0.20, journal: Optional[ExecutionJournal] = None,
                  risk: Optional[RiskConfig] = None) -> ExecutionService:
    """Raises PaperOnlyError / ValueError with a message meant to be shown to the user."""
    import backtrader as bt
    try:
        is_bt = issubclass(strategy_cls, bt.Strategy)
    except TypeError:
        is_bt = False
    if not is_bt:
        raise ValueError(f"{strategy_name!r} is not a rule-based Backtrader strategy; the paper execution service replays "
                         f"Backtrader strategies only")
    symbols = [symbols] if isinstance(symbols, str) else list(symbols)
    cfg = ExecutionConfig(symbols=symbols, interval=interval, poll_seconds=default_poll_seconds(interval),
                          allocation_pct=allocation_pct, risk=risk or RiskConfig())
    signal = BacktraderReplaySignal(strategy_cls, name=strategy_name, trend_overlay=trend_overlay)
    return ExecutionService(broker, loader, signal, cfg, journal or ExecutionJournal())


def start_paper_execution(broker, loader, strategy_cls, strategy_name: str, symbols, interval: str, trend_overlay: bool = False,
                          allocation_pct: float = 0.20, journal: Optional[ExecutionJournal] = None) -> Tuple[Optional[ExecutionService], str]:
    """(service, message). service is None when it could not start; message says why, in plain words."""
    try:
        svc = build_service(broker, loader, strategy_cls, strategy_name, symbols, interval, trend_overlay, allocation_pct, journal)
    except PaperOnlyError as exc:
        return None, f"Not started: {exc}"
    except ValueError as exc:
        return None, f"Not started: {exc}"
    if not svc.start():
        return None, "Not started: another runner (the other app, or a command-line run) is already trading this paper account."
    label = svc.signal.name
    return svc, f"Paper trading started: {label} on {', '.join(svc.cfg.symbols)} ({interval} bars, checks every {svc.cfg.poll_seconds:g}s). No real orders are possible."


def flatten_paper(broker, loader, journal: Optional[ExecutionJournal] = None, reason: str = "flatten requested") -> list:
    """Emergency button for both UIs: close every open paper position at the latest real price and halt new entries."""
    from strategies.simple_strategies import EMACrossoverStrategy
    from core.execution.service import ExecutionConfig, ExecutionService
    from core.execution.signals import BacktraderReplaySignal
    svc = ExecutionService(broker, loader, BacktraderReplaySignal(EMACrossoverStrategy),
                           ExecutionConfig(symbols=list(broker.positions) or ["-"]), journal or ExecutionJournal())
    return svc.flatten_all(reason)
