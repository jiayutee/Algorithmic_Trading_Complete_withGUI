"""A stale heartbeat from a sleeping host must read STALE, not STOPPED; a dead runner must still read STOPPED."""
import os
import socket
import subprocess
import sys

from core.execution.journal import ExecutionJournal
from core.execution.view import execution_view


def _journal(tmp_path, **status):
    j = ExecutionJournal(str(tmp_path / "exec.sqlite3"))
    base = {"heartbeat": 1000.0, "symbols": ["BTCUSDT"], "per_symbol": {}, "running": True, "poll_seconds": 300}
    j.set("status", {**base, **status})
    return j


def _dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid                       # exited and reaped, so the pid no longer exists


def test_stale_heartbeat_with_live_process_on_this_host_is_stale_not_stopped(tmp_path):
    j = _journal(tmp_path, pid=os.getpid(), host=socket.gethostname())
    v = execution_view(j, now=1000.0 + 16000)
    assert "STALE (host asleep?)" in v["headline"] and "STOPPED" not in v["headline"]
    assert v["running"] is False                      # still not claiming it is trading right now
    assert any("still alive" in i for i in v["issues"])


def test_stale_heartbeat_with_dead_process_is_still_stopped(tmp_path):
    j = _journal(tmp_path, pid=_dead_pid(), host=socket.gethostname())
    v = execution_view(j, now=1000.0 + 16000)
    assert "STOPPED" in v["headline"] and "STALE" not in v["headline"]
    assert any("stopped without saying so" in i for i in v["issues"])


def test_stale_heartbeat_without_pid_or_from_another_host_is_stopped(tmp_path):
    old = execution_view(_journal(tmp_path), now=1000.0 + 16000)              # status written by older code: no pid/host
    assert "STOPPED" in old["headline"]
    other = ExecutionJournal(str(tmp_path / "other.sqlite3"))
    other.set("status", {"heartbeat": 1000.0, "symbols": [], "per_symbol": {}, "running": True, "poll_seconds": 300,
                         "pid": os.getpid(), "host": "some-other-machine"})
    assert "STOPPED" in execution_view(other, now=1000.0 + 16000)["headline"]


def test_fresh_heartbeat_still_reads_running(tmp_path):
    j = _journal(tmp_path, pid=os.getpid(), host=socket.gethostname())
    assert j.acquire_lease(180, 1100.0)
    v = execution_view(j, now=1100.0)
    assert "RUNNING" in v["headline"] and v["running"] is True


def test_published_status_carries_pid_and_host():
    import inspect
    from core.execution import service
    src = inspect.getsource(service.ExecutionService._publish_status)
    assert "os.getpid()" in src and "socket.gethostname()" in src
