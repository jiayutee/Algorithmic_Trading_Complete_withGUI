"""Tests added during the Paper Operations Audit (CONTINUATION_PLAN item 4, 2026-09-21).

All four tests use in-memory / tmp_path journals -- never the live
training_ground/paper/execution.sqlite3 file.  They verify that
``hb_age_s`` is present in every return path of ``execution_view``.
"""
import pytest

from core.execution.journal import ExecutionJournal
from core.execution.view import execution_view


def test_hb_age_s_is_none_when_journal_has_no_heartbeat(tmp_path):
    """An empty journal (no status key yet) must return hb_age_s=None, not raise."""
    j = ExecutionJournal(str(tmp_path / "exec.sqlite3"))
    v = execution_view(j, now=1000.0)
    assert "hb_age_s" in v
    assert v["hb_age_s"] is None


def test_hb_age_s_matches_elapsed_seconds_since_heartbeat(tmp_path):
    """When a heartbeat has been written, hb_age_s must equal now - heartbeat (rounded to 1dp)."""
    j = ExecutionJournal(str(tmp_path / "exec.sqlite3"))
    j.set("status", {"heartbeat": 900.0, "symbols": [], "per_symbol": {}, "running": False, "poll_seconds": 30})
    v = execution_view(j, now=942.3)
    assert "hb_age_s" in v
    assert v["hb_age_s"] == pytest.approx(42.3, abs=0.05)


def test_hb_age_s_is_none_in_the_never_run_fast_path(tmp_path, monkeypatch):
    """The early-return path (no journal file on disk) must include hb_age_s=None."""
    monkeypatch.setenv("EXECUTION_DB_PATH", str(tmp_path / "nonexistent.sqlite3"))
    # journal=None + file does not exist → the fast-path branch
    v = execution_view(journal=None, now=1000.0)
    assert "never run" in v["headline"]
    assert "hb_age_s" in v
    assert v["hb_age_s"] is None


def test_hb_age_s_is_none_on_broken_journal():
    """The exception-catch path (journal raises) must include hb_age_s=None and never re-raise."""

    class Broken:
        def get(self, *_a, **_k):
            raise RuntimeError("db locked")

    v = execution_view(Broken(), now=1000.0)
    assert "unavailable" in v["headline"]
    assert "hb_age_s" in v
    assert v["hb_age_s"] is None
