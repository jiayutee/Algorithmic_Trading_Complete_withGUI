#!/bin/bash
# Local orchestrator runner — called by macOS launchd every 2 hours.
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
# Run slots in Berlin local time (24h hours): 6 8 10 12 14 16 18 20
MISSED=$(python3 - <<'PYEOF'
from datetime import datetime, timedelta, timezone
import os, sys

try:
    from zoneinfo import ZoneInfo
    tz = ZoneInfo("Europe/Berlin")
except Exception:
    from datetime import timezone
    tz = timezone(timedelta(hours=2))

SLOTS = [6, 8, 10, 12, 14, 16, 18, 20]
now = datetime.now(tz)
last_run_file = sys.argv[1] if len(sys.argv) > 1 else ""

# Read last run time
if last_run_file and os.path.exists(last_run_file):
    try:
        ts = float(open(last_run_file).read().strip())
        last = datetime.fromtimestamp(ts, tz=tz)
    except Exception:
        last = now - timedelta(hours=2.5)
else:
    # First ever run — just do the current slot
    last = now - timedelta(hours=2.5)

missed = []
# Walk backwards from now to find all slots after last_run
check = now.replace(minute=0, second=0, microsecond=0)
while check > last:
    if check.hour in SLOTS and check > last and check <= now:
        if check.hour == 6:
            run_type = "morning"
        elif check.hour == 20:
            run_type = "evening"
        else:
            run_type = "progress"
        missed.append((check.strftime("%Y-%m-%d %H:%M"), run_type, check.hour))
    check -= timedelta(hours=1)

# Output oldest first
for dt, rt, h in sorted(missed, key=lambda x: x[0]):
    print(f"{dt}|{rt}|{h}")
PYEOF
)

if [ -z "$MISSED" ]; then
    echo "[$(date -u)] No missed slots — nothing to do." >> "$LOG_DIR/launchd.log"
    exit 0
fi

# ── Run each missed slot in order ───────────────────────────────────────────
DAYS=$(python3 -c "from datetime import date; print((date(2026,7,28)-date.today()).days)" 2>/dev/null || echo "?")
DAY_N=$(python3 -c "from datetime import date; print(max(1, 30-(date(2026,7,28)-date.today()).days+1))" 2>/dev/null || echo "?")

# Timestamp (epoch) of the most recent slot that actually completed successfully.
# Only this gets persisted to LAST_RUN_FILE — a failed slot must stay eligible
# for retry on the next catch-up pass, so we never blindly advance past it.
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
            PROMPT="RUN_TYPE=morning. 6am Berlin morning brief for Day ${DAY_N}/30 (${DAYS} days to launch 2026-07-28). Note: this is a catch-up run triggered at ${LOCAL_TIME}. Follow the Morning Brief procedure: read yesterday Notion carry-forwards, plan today agenda, assign tasks to specialist agents, create today Daily Log row, add Sprint Board tasks, send Telegram morning brief, spawn specialist agents."
            ;;
        evening)
            PROMPT="RUN_TYPE=evening. 8pm Berlin EOD debrief for Day ${DAY_N}/30. Note: catch-up run at ${LOCAL_TIME}. Follow Evening Debrief: collect agent outcomes, update Issue Tracker for blockers, update Daily Log (Done Today, Blockers, Carry Forward, Commits, Status→Done), update Launch Roadmap %, update Agent Status Board, send Telegram EOD debrief."
            ;;
        *)
            PROMPT="RUN_TYPE=progress. Missed ${SLOT_TIME} Berlin progress update for Day ${DAY_N}/30 (catch-up run at ${LOCAL_TIME}). Check Sprint Board statuses. Send Telegram: '⏱ AlgoTrader ${SLOT_TIME} update (catch-up) | Day ${DAY_N}/30\n✅ Done: N | 🔄 In progress: N | 🔴 Blocked: N\n<one line on status>'. Update Notion only if new info."
            ;;
    esac

    LOG_FILE="$LOG_DIR/orchestrator-$(date +%Y%m%d-%H%M)-${RUN_TYPE}.log"

    CLAUDE_BIN="${CLAUDE_BIN:-/Users/jiayutee/.local/bin/claude}"

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

        if [ "$CLAUDE_EXIT" -eq 0 ] && ! grep -q "API Error" "$LOG_FILE"; then
            SLOT_OK=true
            break
        fi
    done

    if [ "$SLOT_OK" = true ]; then
        echo "[$(date -u)] Finished: $SLOT_TIME ($RUN_TYPE)" | tee -a "$LOG_DIR/launchd.log"

        # Track this slot as the new high-water mark for LAST_RUN_FILE.
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
        echo "[$(date -u)] FAILED after retry: $SLOT_TIME ($RUN_TYPE) — will not mark as done" | tee -a "$LOG_DIR/launchd.log"

        if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
            curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
                --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
                --data-urlencode "text=⚠️ AlgoTrader orchestrator: ${RUN_TYPE} run for ${SLOT_TIME} failed twice (API error) — will retry next wake." \
                --data-urlencode "parse_mode=Markdown" > /dev/null
        fi

        # Stop processing further (later) slots this pass — leave LAST_RUN_FILE
        # untouched so this slot (and anything after it) is retried next wake.
        break
    fi

    # Small gap between catch-up runs to avoid rate limits
    sleep 5

done <<< "$MISSED"

# ── Save last run timestamp (only advances past slots that actually succeeded) ──
if [ -n "$LAST_SUCCESS_EPOCH" ]; then
    python3 -c "open('$LAST_RUN_FILE','w').write('$LAST_SUCCESS_EPOCH')"
fi

# Keep only last 30 log files
ls -t "$LOG_DIR"/orchestrator-*.log 2>/dev/null | tail -n +31 | xargs rm -f
