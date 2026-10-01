# Proposal: overnight schedule resilience (Mac sleep + agenda race)

Status: **PROPOSAL, owner decides.** Nothing in this PR changes scheduled-task files, launchd plists or `pmset` settings.
Evidence: [dns-outage-followups-2026-09-28.md](../artifacts/dns-outage-followups-2026-09-28.md) and `pmset -g log`.

## Problem

The overnight Claude tasks (23:05 brief, 23:20 and 00:20 work-loop, 01:00 debrief) and the always-on paper runner only
make progress while the Mac is awake. With the lid closed the Mac goes into Deep Idle and wakes only briefly (DarkWake, 2–30 s,
every 10–17 min).

- **09-27 night, cycle missed.** Lid closed 09-27 16:54, next full wake 09-28 11:25. The Day-93 Daily Log row was
  created at 02:26 local, inside a DarkWake at 02:25:45, three hours late. By then both work-loop slots (23:20, 00:20)
  had passed, so no work ran. The paper runner stalled 00:07 → 03:33 for the same reason (see the artifact).
- **Race.** When the brief runs late, a work-loop firing can arrive while the row is still a stub (the case the 08-14 fix
  handles by skipping). If the brief is also late, both work-loop slots can skip, and nothing retries until the next night.
- **Tonight (09-28) worked** only because the Mac was fully awake at 20:41 and an app held a sleep-prevention
  assertion (`pmset -g`: `sleep 0 (sleep prevented by ChatGPT, Claude)`). That depends on which apps happen to be open.

## Options

| | Change | Fixes | Cost / risk |
|---|---|---|---|
| **a** | `sudo pmset repeat wakeorpoweron MTWRFSU 22:55:00` | Wakes the Mac before the brief | A scheduled wake does **not** keep it awake. With the lid closed on battery it may go back to sleep within seconds. Needs sudo (owner only). On its own it is not enough. |
| **b** | Keep the Mac awake for the window: a launchd job at 22:58 running `caffeinate -i -s -t 10800` (about 3 h, until ~02:00) | The whole 23:05–01:30 window runs at full speed, and so does the paper runner | Uses battery unless plugged in (`-s` only works on AC). With the lid closed, macOS forces clamshell sleep unless on AC with an external display, so the lid must stay open or the Mac must be on AC. No sudo needed. |
| **c** | The work-loop polls for the agenda for up to N minutes (e.g. 20 min, every 2 min) instead of skipping once when it finds the stub | The brief/work-loop race | Only helps when the Mac is awake. Costs a few extra Notion reads, and a firing may take longer. It is a text edit to the scheduled-task SKILL.md. |
| **d** | The morning brief reuses a same-date row that is already `Done` (reopens it and appends a new Agenda) instead of creating a duplicate or skipping | Duplicate or confused rows when a late brief lands after midnight | Needs a clear rule for which date a firing belongs to. It touches the brief's SKILL.md. |
| **e** | Alongside any of these: when a firing detects that it runs more than 60 min after its slot, it sends one Telegram "late by X min (Mac asleep?)" | Visibility | Cheap. It does not fix anything by itself. |

## Recommendation

**b + c + e.** Option b removes the cause, provided the Mac is on AC overnight (or the lid is left open). c closes the
remaining race cheaply, and e makes any recurrence visible the same night. Skip a unless b turns out to be impractical,
because a scheduled wake alone does not survive a closed lid on battery. d is only worth doing if duplicate rows show up again.

Sketch for b (for the owner to install, not installed here):

```xml
<!-- ~/Library/LaunchAgents/com.algotrader.overnight-awake.plist -->
<key>ProgramArguments</key><array><string>/usr/bin/caffeinate</string><string>-i</string><string>-s</string><string>-t</string><string>10800</string></array>
<key>StartCalendarInterval</key><dict><key>Hour</key><integer>22</integer><key>Minute</key><integer>58</integer></dict>
```

A launchd calendar job that falls due while the Mac sleeps runs at the next wake, so it pairs well with a if the lid
is left open.

## Not proposed

Do not move the schedule into daytime. The 19:30–23:00 and 02:00–16:00 exclusions in CLAUDE.md stay as they are.
