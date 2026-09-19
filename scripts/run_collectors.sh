#!/bin/bash
# Daily data collectors, run by launchd (scripts/launchd/*.plist). Plain Python: uses NO Claude tokens.
#   scripts/run_collectors.sh news     # headline history for the H3 news test (Phase 6.5 pre-registration)
#   scripts/run_collectors.sh kalshi   # Kalshi open-market snapshots + resolve outcomes of settled ones (Phase 9.3)
# Runs on the 3.9 conda base env like the rest of the orchestration. Appends to logs/collectors.log.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${COLLECTOR_PYTHON:-$HOME/miniconda3/bin/python3}"
LOG="$REPO/logs/collectors.log"
cd "$REPO" || exit 1
mkdir -p logs
stamp() { date "+%Y-%m-%d %H:%M:%S"; }
run() { echo "[$(stamp)] START $*" >> "$LOG"; "$@" >> "$LOG" 2>&1; rc=$?; echo "[$(stamp)] END rc=$rc $*" >> "$LOG"; return $rc; }
case "${1:-}" in
  news)   run "$PY" -m core.news_collector collect ;;
  kalshi) run "$PY" -m core.kalshi_collector collect; run "$PY" -m core.kalshi_collector resolve ;;
  *) echo "usage: $0 news|kalshi" >&2; exit 2 ;;
esac
