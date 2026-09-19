#!/bin/bash
# Always-on PAPER execution service, run by launchd (scripts/launchd/com.algotrader.paper-execution.plist).
# Paper only: the service itself refuses anything but the simulated broker. Settings: scripts/execution.env.
# Logs: logs/execution.log. Status any time:  python -m core.execution.service status
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${EXECUTION_PYTHON:-$HOME/miniconda3/bin/python3}"
LOG="$REPO/logs/execution.log"
cd "$REPO" || exit 1
mkdir -p logs
stamp() { date "+%Y-%m-%d %H:%M:%S"; }
if [ ! -f core/execution/service.py ]; then
  # this checkout does not contain the service (e.g. an older branch): say so once a minute instead of crash-looping silently
  echo "[$(stamp)] core/execution is missing from the current branch ($(git rev-parse --abbrev-ref HEAD 2>/dev/null)); check out a branch that has it" >> "$LOG"
  sleep 60; exit 1
fi
# shellcheck disable=SC1091
. "$REPO/scripts/execution.env"
ARGS=(run --wait --symbols $SYMBOLS --strategy "$STRATEGY" --interval "$INTERVAL" --allocation "$ALLOCATION" --poll "$POLL_SECONDS")
[ -n "${TREND_OVERLAY:-}" ] && ARGS+=(--trend-overlay)
echo "[$(stamp)] START paper execution: $SYMBOLS | $STRATEGY | $INTERVAL | alloc $ALLOCATION" >> "$LOG"
exec "$PY" -m core.execution.service "${ARGS[@]}" >> "$LOG" 2>&1
