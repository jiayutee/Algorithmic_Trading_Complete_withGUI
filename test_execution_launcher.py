import logging

import pytest

from brokers.simulatedbroker import SimulatedBroker
from core.execution.journal import ExecutionJournal
from core.execution.launcher import build_service, default_poll_seconds, start_paper_execution
from core.execution.service import PaperOnlyError
from strategies.simple_strategies import EMACrossoverStrategy

logging.disable(logging.CRITICAL)


class Loader:
    def load_data(self, **kw): return None
    def get_latest_price(self, s): return 100.0


def test_poll_interval_scales_with_the_bar_size():
    assert default_poll_seconds("1m") == 15 and default_poll_seconds("1h") == 60 and default_poll_seconds("1d") == 300


def test_only_paper_strict_brokers_and_backtrader_strategies_are_accepted(tmp_path):
    j = ExecutionJournal(str(tmp_path / "j.sqlite3"))
    with pytest.raises(PaperOnlyError):
        build_service(object(), Loader(), EMACrossoverStrategy, "EMA", "BTCUSDT", "1h", journal=j)
    with pytest.raises(PaperOnlyError):
        build_service(SimulatedBroker(), Loader(), EMACrossoverStrategy, "EMA", "BTCUSDT", "1h", journal=j)     # not strict
    with pytest.raises(ValueError, match="Backtrader"):
        build_service(SimulatedBroker(strict_prices=True), Loader(), object, "ML", "BTCUSDT", "1h", journal=j)


def test_start_reports_plain_english_reasons_and_a_second_start_is_refused(tmp_path):
    j1, j2 = ExecutionJournal(str(tmp_path / "j.sqlite3")), ExecutionJournal(str(tmp_path / "j.sqlite3"))
    broker = SimulatedBroker(strict_prices=True)
    svc, msg = start_paper_execution(SimulatedBroker(), Loader(), EMACrossoverStrategy, "EMA", "BTCUSDT", "1h", journal=j1)
    assert svc is None and msg.startswith("Not started") and "strict_prices" in msg
    svc, msg = start_paper_execution(broker, Loader(), EMACrossoverStrategy, "EMA", ["BTCUSDT"], "1h", trend_overlay=True, journal=j1)
    try:
        assert svc is not None and svc.running and "No real orders are possible" in msg and "trend overlay" in msg
        svc2, msg2 = start_paper_execution(broker, Loader(), EMACrossoverStrategy, "EMA", ["BTCUSDT"], "1h", journal=j2)
        assert svc2 is None and "already trading" in msg2
    finally:
        svc.stop()
