"""scripts/late_check.py: lateness maths, once-per-slot dedupe, and 'never blocks the caller'. Offline: fake clock and sender."""
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import late_check as lc  # noqa: E402

BERLIN = ZoneInfo("Europe/Berlin")
SLOTS = [(23, 20), (0, 20)]


def at(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=BERLIN)


class Sender:
    def __init__(self, ok=True):
        self.sent, self.ok = [], ok

    def __call__(self, text):
        self.sent.append(text)
        return self.ok


def test_minutes_late_uses_the_most_recent_slot_including_yesterdays():
    assert lc.minutes_late(at(2026, 11, 14, 23, 25), SLOTS)[0] == 5
    assert lc.minutes_late(at(2026, 11, 15, 2, 5), SLOTS)[0] == 105          # after the 00:20 slot
    late, slot = lc.minutes_late(at(2026, 11, 15, 0, 10), [(23, 20)])       # slot was yesterday evening
    assert late == 50 and slot.day == 14


def test_on_time_firing_sends_nothing(tmp_path):
    s = Sender()
    r = lc.check("work-loop", SLOTS, now=at(2026, 11, 14, 23, 31), state_path=str(tmp_path / "s.json"), sender=s)
    assert not r["late"] and not r["sent"] and s.sent == []


def test_late_firing_sends_once_and_never_twice_for_the_same_slot(tmp_path):
    st, s = str(tmp_path / "s.json"), Sender()
    r1 = lc.check("work-loop", SLOTS, now=at(2026, 11, 15, 2, 5), state_path=st, sender=s)
    r2 = lc.check("work-loop", SLOTS, now=at(2026, 11, 15, 2, 40), state_path=st, sender=s)
    assert r1["late"] and r1["sent"] and r1["late_min"] == 105 and "105 min late" in s.sent[0]
    assert r2["late"] and not r2["sent"] and "already announced" in r2["reason"] and len(s.sent) == 1


def test_different_tasks_and_different_slots_are_announced_separately(tmp_path):
    st, s = str(tmp_path / "s.json"), Sender()
    lc.check("brief", [(23, 5)], now=at(2026, 11, 15, 1, 0), state_path=st, sender=s)
    lc.check("work-loop", SLOTS, now=at(2026, 11, 15, 2, 5), state_path=st, sender=s)
    lc.check("work-loop", SLOTS, now=at(2026, 11, 16, 2, 5), state_path=st, sender=s)             # next night
    assert len(s.sent) == 3


def test_threshold_is_respected(tmp_path):
    s = Sender()
    assert not lc.check("x", [(23, 20)], threshold=30, now=at(2026, 11, 14, 23, 45), state_path=str(tmp_path / "a"), sender=s)["late"]
    assert lc.check("x", [(23, 20)], threshold=30, now=at(2026, 11, 14, 23, 55), state_path=str(tmp_path / "b"), sender=s)["late"]


def test_failed_send_is_recorded_so_it_is_not_retried_and_never_raises(tmp_path):
    st = str(tmp_path / "s.json")
    r = lc.check("x", SLOTS, now=at(2026, 11, 15, 2, 5), state_path=st, sender=Sender(ok=False))
    assert r["late"] and not r["sent"] and "unavailable" in r["reason"]
    boom = lambda text: (_ for _ in ()).throw(RuntimeError("network down"))                          # noqa: E731
    r2 = lc.check("y", SLOTS, now=at(2026, 11, 15, 2, 5), state_path=str(tmp_path / "t.json"), sender=boom)
    assert not r2["sent"] and "check failed" in r2["reason"]


def test_dry_run_sends_and_writes_nothing(tmp_path):
    st, s = str(tmp_path / "s.json"), Sender()
    r = lc.check("x", SLOTS, now=at(2026, 11, 15, 2, 5), state_path=st, sender=s, dry_run=True)
    assert r["late"] and not r["sent"] and s.sent == [] and not os.path.exists(st)


def test_corrupt_state_file_is_ignored(tmp_path):
    st = tmp_path / "s.json"
    st.write_text("{not json")
    s = Sender()
    assert lc.check("x", SLOTS, now=at(2026, 11, 15, 2, 5), state_path=str(st), sender=s)["sent"]
    assert json.loads(st.read_text())                                                               # rewritten as valid JSON


def test_old_state_entries_are_pruned(tmp_path):
    st = tmp_path / "s.json"
    st.write_text(json.dumps({"x|2026-10-01T23:20:00+02:00": "2026-10-02T01:00:00+02:00"}))
    lc.check("x", SLOTS, now=at(2026, 11, 15, 2, 5), state_path=str(st), sender=Sender())
    assert all("2026-10-01" not in k for k in json.loads(st.read_text()))


def test_cli_bad_slot_is_reported_not_raised(capsys):
    assert lc.main(["--name", "x", "--slot", "25:99", "--dry-run"]) == 0
    assert "bad slot" in capsys.readouterr().out


def test_env_reader_strips_quotes_and_comments(tmp_path):
    p = tmp_path / ".env"
    p.write_text('# c\nTELEGRAM_BOT_TOKEN="abc"\nTELEGRAM_CHAT_ID=123\n')
    assert lc.read_env(str(p)) == {"TELEGRAM_BOT_TOKEN": "abc", "TELEGRAM_CHAT_ID": "123"}
    assert lc.send_telegram("x", env_path=str(tmp_path / "missing.env")) is False                    # no credentials -> False, no network
