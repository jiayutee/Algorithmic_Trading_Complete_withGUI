"""Dash Execution tab + CLI. (The desktop half is in test_gui.py.)"""
import logging

import pytest

import dash_app.callbacks as cb
from brokers.simulatedbroker import SimulatedBroker
from core.execution.journal import ExecutionJournal
from core.execution.service import _cli

logging.disable(logging.CRITICAL)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("EXECUTION_DB_PATH", str(tmp_path / "exec.sqlite3"))
    monkeypatch.setattr(cb, "_exec_service", None)
    monkeypatch.setattr(cb, "_broker", None)
    monkeypatch.delenv("PAPER_ACCOUNT_PATH", raising=False)


class Loader:
    def load_data(self, **kw): return None
    def get_latest_price(self, s): return 100.0


def test_dash_layout_and_callback_expose_the_execution_tab():
    import dash_app.app as m
    text = str(m.app.layout)
    for needed in ("exec-start-btn", "exec-stop-btn", "exec-halt-btn", "exec-resume-btn", "exec-flatten-btn",
                   "exec-decisions-table", "exec-symbols-table", "exec-headline", "exec-interval"):
        assert needed in text, needed
    found = [c for k, c in m.app.callback_map.items() if "exec-headline" in k]
    assert found and {"exec-interval", "exec-start-btn", "exec-flatten-btn"} <= {i["id"] for i in found[0]["inputs"]}
    assert {"symbol-dropdown", "interval-dropdown", "strategy-dropdown"} <= {s["id"] for s in found[0]["state"]}


def test_halt_and_resume_buttons_change_the_shared_journal_flag():
    assert "Halted" in cb._execution_action("exec-halt-btn", "BTCUSDT", "1h", "EMA Crossover")
    assert ExecutionJournal().halted()["reason"].startswith("halted from the Dash")
    assert cb._execution_action("exec-resume-btn", "BTCUSDT", "1h", "EMA Crossover") == "Resumed."
    assert ExecutionJournal().halted() is None


def test_start_is_refused_for_a_non_strict_account_or_a_non_rule_strategy_with_plain_words(monkeypatch):
    monkeypatch.setattr(cb, "_get_broker", lambda: SimulatedBroker())              # not strict
    monkeypatch.setattr("core.data_loader.DataLoader", Loader)
    msg = cb._execution_action("exec-start-btn", "BTCUSDT", "1h", "EMA Crossover")
    assert msg.startswith("Not started") and "strict_prices" in msg
    monkeypatch.setattr(cb, "_get_broker", lambda: SimulatedBroker(strict_prices=True))
    assert cb._execution_action("exec-start-btn", "BTCUSDT", "1h", None).startswith("Not started: pick a rule-based strategy")
    assert "GBM" not in cb._execution_action("exec-start-btn", "BTCUSDT", "1h", "no such strategy")


def test_start_then_stop_then_flatten_paper_only(monkeypatch):
    broker = SimulatedBroker(strict_prices=True)
    monkeypatch.setattr(cb, "_get_broker", lambda: broker)
    monkeypatch.setattr("core.data_loader.DataLoader", Loader)
    msg = cb._execution_action("exec-start-btn", "BTCUSDT", "1h", "EMA Crossover")
    try:
        assert "Paper trading started" in msg and "No real orders are possible" in msg and cb._exec_service.running
        assert cb._execution_action("exec-start-btn", "BTCUSDT", "1h", "EMA Crossover") == "Already running."
    finally:
        assert "Stopped" in cb._execution_action("exec-stop-btn", "BTCUSDT", "1h", "EMA Crossover")
    assert cb._exec_service is None
    broker.update_price("BTCUSDT", 100.0); broker.submit_order("BTCUSDT", 2, "buy", execution_price=100.0)
    assert "BTCUSDT" in cb._execution_action("exec-flatten-btn", "BTCUSDT", "1h", "EMA Crossover")
    assert not broker.positions and ExecutionJournal().halted()


def test_cli_status_on_an_empty_journal_and_halt_resume_roundtrip(capsys):
    assert _cli(["status"]) == 0
    assert "STOPPED" in capsys.readouterr().out
    assert _cli(["halt", "cli test"]) == 0
    assert ExecutionJournal().halted()["reason"] == "cli test"
    _cli(["status"])
    assert "HALTED: cli test" in capsys.readouterr().out
    assert _cli(["resume"]) == 0 and ExecutionJournal().halted() is None


def test_looking_at_the_status_never_creates_a_database(tmp_path):
    import os
    from core.execution.journal import default_journal_path
    from core.execution.view import execution_view
    assert not os.path.exists(default_journal_path())
    v = execution_view()
    assert "never run" in v["headline"] and not os.path.exists(default_journal_path())
