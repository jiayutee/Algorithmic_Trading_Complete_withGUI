#!/bin/bash
# Local orchestrator runner — called by macOS launchd on schedule.
# Uses Claude Code CLI with your existing plan (no separate API billing).
# Telegram bot token read from .env in the project root.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# Load .env
if [ -f "$PROJECT_DIR/.env" ]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
fi

# Determine run type from current Berlin local time
HOUR=$(python3 -c "
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
    h = datetime.now(ZoneInfo('Europe/Berlin')).hour
except Exception:
    import time; h = (datetime.utcnow().hour + 2) % 24
print(h)
" 2>/dev/null || date +%H | sed 's/^0//')

case "$HOUR" in
    6)  RUN_TYPE="morning"  ;;
    20) RUN_TYPE="evening"  ;;
    *)  RUN_TYPE="progress" ;;
esac

DAYS=$(python3 -c "from datetime import date; print((date(2026,7,28)-date.today()).days)" 2>/dev/null || echo "?")
DAY_N=$(python3 -c "from datetime import date; print(max(1, 30-(date(2026,7,28)-date.today()).days+1))" 2>/dev/null || echo "?")
LOCAL_TIME=$(python3 -c "from datetime import datetime; from zoneinfo import ZoneInfo; print(datetime.now(ZoneInfo('Europe/Berlin')).strftime('%I:%M%p'))" 2>/dev/null || echo "?")

case "$RUN_TYPE" in
    morning)
        PROMPT="RUN_TYPE=morning. 6am Berlin morning brief for Day ${DAY_N}/30 (${DAYS} days to launch on 2026-07-28). Follow the Morning Brief procedure: read yesterday's Notion carry-forwards, plan today's agenda, assign tasks to specialist agents, create today's Daily Log row, add Sprint Board tasks, send Telegram morning brief, then spawn specialist agents."
        ;;
    evening)
        PROMPT="RUN_TYPE=evening. 8pm Berlin EOD debrief for Day ${DAY_N}/30. Follow the Evening Debrief procedure: collect agent outcomes from Sprint Board, update Issue Tracker for blockers, update today's Daily Log (Done Today, Blockers, Carry Forward, Commits, Status→Done), update Launch Roadmap checklist percentages, update Agent Status Board, send Telegram EOD debrief."
        ;;
    *)
        PROMPT="RUN_TYPE=progress. ${LOCAL_TIME} Berlin progress update for Day ${DAY_N}/30. Check Sprint Board task statuses (Done/In Progress/Blocked). Triage any newly blocked tasks. Send a concise Telegram progress message: '⏱ AlgoTrader ${LOCAL_TIME} | Day ${DAY_N}/30\n✅ Done: N | 🔄 In progress: N | 🔴 Blocked: N\n<one sentence on biggest risk or win>'. Update Notion Daily Log only if there is something new to report."
        ;;
esac

LOG_FILE="$LOG_DIR/orchestrator-$(date +%Y%m%d-%H%M).log"

echo "[$( date -u )] Starting orchestrator — RUN_TYPE=$RUN_TYPE" | tee "$LOG_FILE"

cd "$PROJECT_DIR" && claude \
    --agent orchestrator \
    --non-interactive \
    --max-turns 80 \
    -p "$PROMPT" \
    >> "$LOG_FILE" 2>&1

echo "[$( date -u )] Orchestrator finished — RUN_TYPE=$RUN_TYPE" | tee -a "$LOG_FILE"

# Keep only last 30 log files
ls -t "$LOG_DIR"/orchestrator-*.log 2>/dev/null | tail -n +31 | xargs rm -f
