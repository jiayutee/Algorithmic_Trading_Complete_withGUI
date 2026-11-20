#!/usr/bin/env python3
"""Tell the owner, once per slot, when an overnight scheduled task starts long after its slot (usually: the Mac was asleep).

    python3 scripts/late_check.py --name work-loop --slot 23:20 --slot 00:20

Lateness = minutes since the most recent of the given slot times (Europe/Berlin by default). If it exceeds --threshold (60)
a single Telegram message is sent; the same slot is never announced twice (state file), so the 23:20 and 00:20 firings of one
night and repeated polls cannot spam. Best effort by design: any failure (no credentials, no network, unreadable state) is
reported on stdout and the exit code is still 0, because a lateness alert must never block the task that called it.

Limit: with several slots the "most recent" one is used, so a 23:20 firing that wakes at 00:30 reads as 10 minutes after the
00:20 slot and is not flagged. State lives in ~/.claude/scheduled-tasks (NOT the repo), so this is safe to call from the
report-only tasks that must leave the working tree untouched. Credentials come from the project .env (never printed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Callable, List, Optional
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_STATE = os.path.expanduser("~/.claude/scheduled-tasks/late_alert_state.json")
KEEP_STATE_DAYS = 14


def parse_slot(text: str) -> tuple:
    h, m = text.split(":")
    h, m = int(h), int(m)
    if not (0 <= h < 24 and 0 <= m < 60):
        raise ValueError(f"bad slot {text!r} (want HH:MM)")
    return h, m


def most_recent_slot(now: datetime, slots: List[tuple]) -> datetime:
    """The latest slot time that is <= now (today's, else yesterday's occurrence)."""
    best = None
    for h, m in slots:
        for back in (0, 1):
            cand = (now - timedelta(days=back)).replace(hour=h, minute=m, second=0, microsecond=0)
            if cand <= now and (best is None or cand > best):
                best = cand
    if best is None:
        raise ValueError("no slots given")
    return best


def minutes_late(now: datetime, slots: List[tuple]) -> tuple:
    slot = most_recent_slot(now, slots)
    return (now - slot).total_seconds() / 60.0, slot


def _load_state(path: str) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_state(path: str, state: dict, now: datetime) -> None:
    cutoff = (now - timedelta(days=KEEP_STATE_DAYS)).isoformat()
    state = {k: v for k, v in state.items() if v >= cutoff}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=1, sort_keys=True)


def read_env(path: str) -> dict:
    out = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


def send_telegram(text: str, env_path: str = os.path.join(ROOT, ".env")) -> bool:
    import requests
    env = read_env(env_path)
    token, chat = env.get("TELEGRAM_BOT_TOKEN"), env.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False
    r = requests.get(f"https://api.telegram.org/bot{token}/sendMessage", params={"chat_id": chat, "text": text}, timeout=10)
    return r.ok


def check(name: str, slots: List[tuple], threshold: float = 60.0, tz: str = "Europe/Berlin",
          state_path: str = DEFAULT_STATE, now: Optional[datetime] = None,
          sender: Callable[[str], bool] = send_telegram, dry_run: bool = False) -> dict:
    """Returns {"late_min", "slot", "late": bool, "sent": bool, "reason": str}. Never raises."""
    try:
        now = now or datetime.now(ZoneInfo(tz))
        late, slot = minutes_late(now, slots)
        result = {"late_min": round(late, 1), "slot": slot.strftime("%H:%M"), "late": late > threshold, "sent": False, "reason": ""}
        if late <= threshold:
            result["reason"] = "on time"
            return result
        key = f"{name}|{slot.isoformat()}"
        state = _load_state(state_path)
        if key in state:
            result["reason"] = "already announced for this slot"
            return result
        text = (f"⏰ {name} is running {int(late)} min late (slot {slot.strftime('%H:%M')}) — was the Mac asleep? "
                f"The overnight caffeinate job only holds on AC power.")
        if dry_run:
            result["reason"] = "dry run: " + text
            return result
        result["sent"] = bool(sender(text))
        result["reason"] = "sent" if result["sent"] else "telegram unavailable (not retried this slot)"
        state[key] = now.isoformat()                      # record even a failed send: one best-effort try per slot
        _save_state(state_path, state, now)
        return result
    except Exception as exc:  # noqa: BLE001 -- must never block the calling task
        return {"late_min": None, "slot": None, "late": False, "sent": False, "reason": f"check failed: {type(exc).__name__}: {exc}"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="task name shown in the message")
    ap.add_argument("--slot", action="append", required=True, help="HH:MM slot time (repeatable)")
    ap.add_argument("--threshold", type=float, default=60.0, help="minutes late before alerting (default 60)")
    ap.add_argument("--tz", default="Europe/Berlin")
    ap.add_argument("--state", default=DEFAULT_STATE)
    ap.add_argument("--dry-run", action="store_true", help="report only; send nothing, write nothing")
    args = ap.parse_args(argv)
    try:
        slots = [parse_slot(s) for s in args.slot]
    except ValueError as exc:
        print(json.dumps({"late": False, "sent": False, "reason": str(exc)}))
        return 0
    print(json.dumps(check(args.name, slots, args.threshold, args.tz, args.state, dry_run=args.dry_run)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
