#!/bin/bash
# Local orchestrator runner — called by macOS launchd at 23:05, 23:20, 00:20, 01:15 Berlin.
# Overnight-only window: avoids token contention with CariGaji (02:00-16:00)
# and the owner's reserved manual-prompting window (19:30-23:00).
# Runs unattended via launchd (does NOT require the Claude Code app to be open,
# unlike claude.ai Routines -- which is why this exists instead of Routines).
# On wake, catches up all missed run slots since the Mac was last active.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
LAST_RUN_FILE="$PROJECT_DIR/logs/.last_run"
mkdir -p "$LOG_DIR"

# Load .env
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

# ── Compute all missed run slots since last run ─────────────────────────────
# Exact (hour, minute) slots in Berlin local time:
#   23:05 morning (report-only)
#   23:20 work-loop (codes)
#   00:20 work-loop (codes, safety-net retry)
#   01:15 evening (report-only)
# Two different run types share hour 23, so this walks in 5-minute steps
# and matches exact (hour, minute) pairs -- hourly granularity would conflate
# the 23:05 morning slot with the 23:20 work-loop slot.
MISSED=$(python3 - <<'PYEOF'
from datetime import datetime, timedelta, timezone
import os, sys

try:
    from zoneinfo import ZoneInfo
    tz = ZoneInfo("Europe/Berlin")
except Exception:
    from datetime import timezone
    tz = timezone(timedelta(hours=2))

SLOTS = {
    (23, 5): "morning",
    (23, 20): "work-loop",
    (0, 20): "work-loop",
    (1, 15): "evening",
}
now = datetime.now(tz)
last_run_file = sys.argv[1] if len(sys.argv) > 1 else ""

if last_run_file and os.path.exists(last_run_file):
    try:
        ts = float(open(last_run_file).read().strip())
        last = datetime.fromtimestamp(ts, tz=tz)
    except Exception:
        last = now - timedelta(hours=2.5)
else:
    last = now - timedelta(hours=2.5)

missed = []
check = now.replace(second=0, microsecond=0)
check = check - timedelta(minutes=check.minute % 5)  # snap to 5-min grid
while check > last:
    key = (check.hour, check.minute)
    if key in SLOTS and check > last and check <= now:
        missed.append((check.strftime("%Y-%m-%d %H:%M"), SLOTS[key], check.hour))
    check -= timedelta(minutes=5)

for dt, rt, h in sorted(missed, key=lambda x: x[0]):
    print(f"{dt}|{rt}|{h}")
PYEOF
)

if [ -z "$MISSED" ]; then
    echo "[$(date -u)] No missed slots — nothing to do." >> "$LOG_DIR/launchd.log"
    exit 0
fi

# ── Run each missed slot in order ───────────────────────────────────────────
DAYS=$(python3 -c "from datetime import date; print((date(2026,8,18)-date.today()).days)" 2>/dev/null || echo "?")
DAY_N=$(python3 -c "from datetime import date; print(max(1, 51-(date(2026,8,18)-date.today()).days+1))" 2>/dev/null || echo "?")

LAST_SUCCESS_EPOCH=""

while IFS='|' read -r SLOT_TIME RUN_TYPE SLOT_HOUR; do
    LOCAL_TIME=$(python3 -c "
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
    print(datetime.now(ZoneInfo('Europe/Berlin')).strftime('%I:%M%p'))
except: print('?')
" 2>/dev/null || echo "?")

    echo "[$(date -u)] Running missed slot: $SLOT_TIME ($RUN_TYPE)" | tee -a "$LOG_DIR/launchd.log"

    case "$RUN_TYPE" in
        morning)
            PROMPT="RUN_TYPE=morning. 23:05 Berlin morning brief (REPORT ONLY -- do not touch code) for Day ${DAY_N}/51 (${DAYS} days to launch 2026-08-18 (revised, was 2026-07-28)). Note: this is a catch-up run triggered at ${LOCAL_TIME}. Follow the Morning Brief procedure in .github/agents/orchestrator.agent.md: read yesterday Notion carry-forwards, plan today agenda from the Sprint Board backlog, create today Daily Log row, add new Sprint Board tasks only if not already covered, send Telegram morning brief. Do NOT spawn any subagent."
            ;;
        work-loop)
            PROMPT="RUN_TYPE=work-loop. Work-loop cycle for Day ${DAY_N}/51 (catch-up run at ${LOCAL_TIME}). Follow the Work Loop procedure in .github/agents/orchestrator.agent.md exactly: STEP 0.5 foreign-dirty-tree guard first, then read today's Daily Log agenda, then repeat pick/classify/execute/test-gate/commit/update-Notion for every pending item, then send ONE consolidated Telegram summary at the end."
            ;;
        evening)
            PROMPT="RUN_TYPE=evening. 01:15 Berlin EOD debrief (REPORT ONLY -- do not touch code) for Day ${DAY_N}/51. Note: catch-up run at ${LOCAL_TIME}. Follow Evening Debrief in .github/agents/orchestrator.agent.md: collect Sprint Board outcomes, update Issue Tracker for blockers, update Daily Log (Done Today, Blockers, Carry Forward, Commits via fixed 4-hour git log window, Status->Done), update Launch Roadmap %, update Agent Status Board, send Telegram EOD debrief, verify the Status write landed."
            ;;
        *)
            PROMPT="RUN_TYPE=progress. Missed ${SLOT_TIME} Berlin progress update for Day ${DAY_N}/51 (catch-up run at ${LOCAL_TIME}). Check Sprint Board statuses. Send Telegram: 'AlgoTrader ${SLOT_TIME} update (catch-up) | Day ${DAY_N}/51\nDone: N | In progress: N | Blocked: N\n<one line on status>'. Update Notion only if new info."
            ;;
    esac

    LOG_FILE="$LOG_DIR/orchestrator-$(date +%Y%m%d-%H%M)-${RUN_TYPE}.log"

    CLAUDE_BIN="${CLAUDE_BIN:-/Users/jiayutee/.local/bin/claude}"

    SESSION_LIMIT_HIT=false
    SLOT_OK=false
    for ATTEMPT in 1 2; do
        if [ "$ATTEMPT" -eq 2 ]; then
            echo "[$(date -u)] Retrying slot $SLOT_TIME ($RUN_TYPE) after failure..." | tee -a "$LOG_DIR/launchd.log"
            sleep 10
        fi

        cd "$PROJECT_DIR" && "$CLAUDE_BIN" \
            --append-system-prompt-file "$PROJECT_DIR/.github/agents/orchestrator.agent.md" \
            --print \
            --allowedTools "Bash,Read,Edit,Write,Agent,WebSearch,WebFetch" \
            --max-turns 80 \
            -p "$PROMPT" \
            >> "$LOG_FILE" 2>&1
        CLAUDE_EXIT=$?

        if grep -q "hit your session limit\|session limit.*resets\|usage limit" "$LOG_FILE" 2>/dev/null; then
            SESSION_LIMIT_HIT=true
            break
        fi

        if [ "$CLAUDE_EXIT" -eq 0 ] && ! grep -q "API Error" "$LOG_FILE"; then
            SLOT_OK=true
            break
        fi
    done

    if [ "$SLOT_OK" = true ]; then
        echo "[$(date -u)] Finished: $SLOT_TIME ($RUN_TYPE)" | tee -a "$LOG_DIR/launchd.log"
        LAST_SUCCESS_EPOCH=$(python3 -c "
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
    tz = ZoneInfo('Europe/Berlin')
except Exception:
    from datetime import timezone, timedelta
    tz = timezone(timedelta(hours=2))
dt = datetime.strptime('$SLOT_TIME', '%Y-%m-%d %H:%M').replace(tzinfo=tz)
print(dt.timestamp())
" 2>/dev/null)
    else
        if [ "$SESSION_LIMIT_HIT" = true ]; then
            echo "[$(date -u)] Session limit hit: $SLOT_TIME ($RUN_TYPE) — skipped retry (resets ~3:20am Berlin); morning self-healing check will recover" | tee -a "$LOG_DIR/launchd.log"
            if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
                curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
                    --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
                    --data-urlencode "text=AlgoTrader orchestrator: ${RUN_TYPE} slot ${SLOT_TIME} hit Claude session limit (resets ~3:20am Berlin). Skipped retry — morning run self-healing check will backfill." \
                    --data-urlencode "parse_mode=Markdown" > /dev/null
            fi
        else
            echo "[$(date -u)] FAILED after retry: $SLOT_TIME ($RUN_TYPE) — will not mark as done" | tee -a "$LOG_DIR/launchd.log"
            if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
                curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
                    --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
                    --data-urlencode "text=AlgoTrader orchestrator: ${RUN_TYPE} run for ${SLOT_TIME} failed twice (API error) — will retry next wake." \
                    --data-urlencode "parse_mode=Markdown" > /dev/null
            fi
        fi

        if [ "$RUN_TYPE" = "progress" ] || [ "$RUN_TYPE" = "work-loop" ]; then
            # A stuck work-loop or progress slot must never block morning/evening
            # slots bundled behind it in the same catch-up batch (see the Day 7
            # incident in orchestrator.agent.md). A second work-loop firing (00:20)
            # exists precisely as a safety net if the first fails -- don't let a
            # failure here eat the evening debrief.
            echo "[$(date -u)] Skipping failed $RUN_TYPE slot, continuing to next slot in batch." | tee -a "$LOG_DIR/launchd.log"
            sleep 5
            continue
        fi

        # morning/evening are load-bearing -- stop processing further slots
        # this pass and leave LAST_RUN_FILE untouched so this slot is retried
        # next wake.
        break
    fi

    sleep 5

done <<< "$MISSED"

if [ -n "$LAST_SUCCESS_EPOCH" ]; then
    python3 -c "open('$LAST_RUN_FILE','w').write('$LAST_SUCCESS_EPOCH')"
fi

ls -t "$LOG_DIR"/orchestrator-*.log 2>/dev/null | tail -n +31 | xargs rm -f
