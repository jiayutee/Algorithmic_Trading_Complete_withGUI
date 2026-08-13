# AlgoTrader — CLAUDE.md

## Project
**Algorithmic Trading Complete with GUI** — sprint to ship by **2026-08-18** (revised launch date — original 2026-07-28 target pushed 3 weeks by owner decision on 2026-08-09, see Daily Log). 51-day sprint total.
Repo: `jiayutee/Algorithmic_Trading_Complete_withGUI`
Stack: Python 3.11, PyQt5, backtrader, OpenBB, Binance/Alpaca/KuCoin/SimulatedBroker, SQLite news store.

## Architecture
```
app.py                    # Entry point — QtWebEngineWidgets imported first
core/
  data_loader.py          # OpenBB first, Yahoo/Binance fallback
  news_pipeline.py        # DuckDuckGo → OpenBB → GDELT
  news_sources.py         # OpenBBNewsSource, RSSSource, GDELTSource
  backtesting.py          # backtrader engine, pyfolio reports
brokers/
  simulatedbroker.py      # Paper trading, order history, positions
  binancebroker.py        # Live Binance (paper flag)
  alpacabroker.py         # Alpaca live
strategies/
  macd_rsi_strategy.py
  ema_crossover_strategy.py
  stochastic_strategy.py
ui/main_window.py         # MainWindow — all PyQt5 widgets
scripts/
  orchestrator-local.sh   # launchd entry point (8 slots/day Berlin time)
  telegram-listener.py    # Two-way Telegram bot (polls every 3s)
.github/agents/           # Source-of-truth agent definitions (also mirrored to .claude/agents/)
```

## Sprint Context
- Launch: 2026-08-18 (revised, was 2026-07-28) | Day counter: `python3 -c "from datetime import date; print(51-(date(2026,8,18)-date.today()).days+1)"`
- Notion hub: https://app.notion.com/p/36ad2ab050d980439d4ce7d7d235c9af
- Daily Log DB ID: `00008c59-c054-4c67-97f8-9753a9a23163`
- Sprint Board DB ID: `91e3aa02-65de-40fb-8cb4-d297683bd67e`
- Issue Tracker DB ID: `e575e816-cab1-4d24-8f40-89b1d5ca8f27`

## Orchestrator
- Runs via Claude Code scheduled tasks (cron-based, defined under `~/.claude/scheduled-tasks/`) overnight only: 23:05 (morning brief), 23:20 + 00:20 (work-loop `algotrader-work-loop`: safety-first pass + safety-net retry), 01:00 (EOD debrief) — all Berlin local time
- Schedule deliberately avoids: CariGaji orchestrator (02:00-16:00) and the owner's
  reserved manual-prompting window (19:30-23:00) — both share the same Claude token pool
- Two-way Telegram bot (chat_id=51218456) — user can send instructions, orchestrator responds
- Uses `--append-system-prompt-file .github/agents/orchestrator.agent.md` (NOT `--agent`)
- Notion updated via REST API (curl) using `$NOTION_API_KEY` from `.env`
- CI status checked via `$GITHUB_PAT` from `.env`

## Environment
- Python: `~/miniconda3/bin/python3` (base env — NOT myenv, it OOM-kills)
- Claude CLI: `/Users/jiayutee/.local/bin/claude`
- `.env` is gitignored — contains all secrets (never commit)
- Run tests: `~/miniconda3/bin/python3 -m pytest --ignore=test_gui.py -v`

## Coding Rules
- **No `.env` in commits** — always check `git status` before committing
- Run `~/miniconda3/bin/python3 -m pytest --ignore=test_gui.py -q` after every code change
- Import `QtWebEngineWidgets` before `QApplication` in app.py (Qt ordering requirement)
- OpenBB tests use `pytest.importorskip` — they skip gracefully if openbb not installed
- GUI tests (`test_gui.py`) are excluded from CI (no Qt display on Ubuntu runner)
- Max 6 tasks per orchestrator day cycle to avoid context overload

## Subagent Routing
| File touched | Agent |
|---|---|
| core/data_loader.py, news_sources.py, news_pipeline.py | Data Pipeline Agent |
| strategies/*.py | Strategy Agent |
| brokers/*.py | Execution Broker Agent |
| core/backtesting.py, metrics | Backtest and Metrics Agent |
| ui/main_window.py, app.py | UI Agent |
| test_*.py, scripts/, smoke tests | QA Test Agent |
| .github/workflows/, CI, packaging | Reliability Release Agent |

## Definition of Done
1. Code committed with descriptive message
2. `pytest --ignore=test_gui.py` passes locally
3. Sprint Board row updated to Done
4. CI green on GitHub Actions
