# AlgoTrader — CLAUDE.md

## Project
**Algorithmic Trading Complete with GUI** — sprint to ship by **2026-08-18** (revised launch date — original 2026-07-28 target pushed 3 weeks by owner decision on 2026-08-09, see Daily Log). 51-day sprint total.
Repo: `jiayutee/Algorithmic_Trading_Complete_withGUI`
Stack: Python 3.11, PyQt5, backtrader, OpenBB, Binance/Alpaca/KuCoin/SimulatedBroker, SQLite news store.

## Architecture

> **GUI entrypoint decision (Phase 3.1, 2026-08-17):** PyQt5 (`app.py`) is the primary/canonical
> entrypoint. Dash (`dash_app/app.py`) is an optional web interface — feature-parity review found
> 5 critical gaps (dynamic P&L/account, live trading, simulation mode, broker switching, agent
> monitor) that block Dash from replacing PyQt5. Full checklist: `docs/PHASE_3_1_FEATURE_PARITY.md`.

```
app.py                    # Entry point — QtWebEngineWidgets imported first (PyQt5 primary)
dash_app/
  app.py                  # Dash web entry point — serve on http://127.0.0.1:8050
  layout.py               # Full dark-themed layout; mirrors PyQt5 color palette
  callbacks.py            # All Dash callbacks (chart load, live price, orders, backtest, news)
core/
  data_loader.py          # OpenBB first, Yahoo/Binance fallback
  news_pipeline.py        # DuckDuckGo → OpenBB → GDELT
  news_sources.py         # OpenBBNewsSource, RSSSource, GDELTSource
  backtester.py           # backtrader engine, pyfolio reports
  broker_manager.py       # Broker routing/switching
  feature_engineering.py  # ML feature matrix (technicals/news/macro/time), no-lookahead by construction
  ml_validation.py        # Walk-forward splits + purge gap; walk_forward_predict() = OOS predictions
  trade_rationale.py      # Structured "why" record attached to every order/signal
  experiment_log.py       # SQLite log of training/eval runs (params, metrics, git commit): python -m core.experiment_log list
  kalshi_data.py          # Read-only Kalshi public-API client (no auth, no order path; NOT a broker)
  kalshi_arbitrage.py     # Kalshi mispricing SIGNALS only (YES+NO<$1, exclusive-event sets), after assumed fees
  kalshi_collector.py     # Daily Kalshi snapshot collector + outcome resolver (python -m core.kalshi_collector collect|resolve|status)
  execution/              # PAPER-ONLY execution service: bar -> signal -> sizing -> risk -> order -> fill -> reconcile (journal.py, risk.py, signals.py, service.py, launcher.py, view.py)
  research_loop.py        # Autonomous research loop: evaluate candidates vs buy&hold, promote/retire, forward paper ledger
  trend_overlay.py        # Trend-filter rule (28-bar, weekly): drawdown reduction, NOT alpha (Phases 6.7-6.9)
  risk_sizing.py          # Volatility-targeting helpers (Phase 6.6; result: did not help)
  news_health.py          # Per-source circuit breaker so one rate-limited news source can't stall a refresh
brokers/
  simulatedbroker.py      # Paper trading, order history, positions
  binance_connector.py    # Live Binance (paper flag)
  alpaca_connector.py     # Alpaca live
  kucoin_connector.py     # KuCoin live
  ib_connector.py         # Interactive Brokers (Phase 4.1, not yet wired into broker_manager)
strategies/
  simple_strategies.py    # MACD/RSI, EMA crossover, Stochastic
  ml_strategies.py        # DEPRECATED LSTM (hidden from UI; see docstring for why) -- use gbm_strategy.py
  trend_filter_strategy.py # Trend Filter (28d) strategy + with_trend_overlay(cls) wrapper (UI checkbox 'Trend overlay')
  gbm_strategy.py         # LightGBM direction model, retrained walk-forward (Phase 6.2)
  FinRL_strategy.py
  TD3_strategy.py
  ddpg_strategy.py
ui/main_window.py         # MainWindow — all PyQt5 widgets (primary desktop UI)
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
- Python: `~/miniconda3/bin/python3` (base env, 3.9 — NOT myenv, it OOM-kills). Orchestrator, Telegram bot and the daily collectors run on this.
- Python 3.11 env (NautilusTrader, Phase 8.0): `scripts/setup_py311_env.sh` creates `~/.venvs/algotrader311` (needs `brew install libomp` for lightgbm). Full suite passes there (1074 passed). Use `~/.venvs/algotrader311/bin/python` for anything touching `nautilus_trader`. pandas-ta has no release for <3.12, so it is absent from both envs' required set.
- Claude CLI: `/Users/jiayutee/.local/bin/claude`
- `.env` is gitignored — contains all secrets (never commit)
- Run tests: `~/miniconda3/bin/python3 -m pytest --ignore=test_gui.py -v` (or the 3.11 env's python)
- News fetch budget: `NEWS_FETCH_DEADLINE_SECONDS` (default 6); sources still running when it expires are abandoned and
  skipped for a cool-down after repeated failures
- IBKR (Phase 4.1): opt-in. Set `IBKR_ENABLED=1` (+ optional `IBKR_HOST`/`IBKR_PORT` (default 7497 = TWS paper; Gateway paper 4002)/`IBKR_CLIENT_ID`) and run TWS/Gateway with API enabled; needs `pip install ib_insync`. Tests use a mocked `ib_insync`; a live connection is only for manual verification. Orders always go through the live-order guard.
- Daily data collectors (launchd, plain Python, no Claude tokens): `com.algotrader.collect-news` (07:20) and
  `com.algotrader.collect-kalshi` (07:05, 12:05, 17:05, local time) run `scripts/run_collectors.sh news|kalshi`, log to
  `logs/collectors.log`. Plist sources: `scripts/launchd/`; installed copies in `~/Library/LaunchAgents/`.
  Check: `launchctl list | grep collect`, `python -m core.news_collector status`, `python -m core.kalshi_collector status`.
  Stop: `launchctl bootout gui/$(id -u)/com.algotrader.collect-news` (same for `-kalshi`).
- Paper account (Simulator broker): durable SQLite file `training_ground/paper/paper_account.sqlite3` (override `PAPER_ACCOUNT_PATH`),
  shared by the desktop app and Dash, so it survives restarts. Real prices only: the old random-walk generator is opt-in
  (`SimulatedBroker(simulate_prices=True)`), a market order with no/stale price is REJECTED with a reason (not filled at $100),
  pending limit/stop orders re-check on every price update, and every open holding (not just the charted symbol) is marked
  to a real price (`core/paper_marking.py`). Reset: `SimulatedBroker(persist_path=...).reset(100000)`.
  Tests and agents that build `BrokerManager()`/`SimulatedBroker()` with no path keep the plain in-memory simulator.
- Alpha/beta: one formula (`Backtester._alpha_beta_core`, sample stats, annualization 252, risk-free 0 -- both attributes on
  `Backtester`); unavailable = `None` shown as "n/a" with a reason, never 0.
- Paper execution service (`core/execution/`): runs a rule-based strategy on the paper account automatically. Start it from the desktop
  "Start Paper" button, the Dash "Execution" tab, or `python -m core.execution.service run --symbols BTCUSDT --strategy "EMA Crossover" --interval 1h`;
  `status | halt "why" | resume | flatten` also work. PAPER ONLY (it refuses any broker but the strict SimulatedBroker). One runner at a time
  (lease in the journal `training_ground/paper/execution.sqlite3`, override `EXECUTION_DB_PATH`); one decision per completed bar so restarts
  cannot repeat an order; risk gate blocks/shrinks ENTRIES only (per-symbol/gross exposure, daily loss, drawdown, order count, stale data,
  `halt`) and never blocks exits; long/flat by default. Not a profit claim: no strategy tested so far shows an edge (Phases 6.5-6.9).
- Research loop: `python -m core.research_loop run|status` (also visible in the Dash "Research Loop" tab). Promote/retire
  rules are fixed in the module docstring; retired strategies never revive automatically
- Experiment log: `python -m core.experiment_log list|show|best|compare` (file: training_ground/results/experiments.sqlite3,
  override with `EXPERIMENT_LOG_PATH`)
- Train/evaluate the ML model: `~/miniconda3/bin/python3 training_ground/train_gbm.py --symbol BTCUSDT --days 1500 --interval 1d`
  (prints out-of-sample AUC with a confidence interval; read the VERDICT line before trusting any number)

## Live trading safety (Phase 11.2)
Every live connector's `submit_order` goes through `brokers/execution_guard.py` (inside the connector,
so nothing can route around it). **By default all live orders are blocked.** Env vars (read per order):
- `LIVE_TRADING_ENABLED=true` -- master switch (default off = kill switch ON)
- `LIVE_DRY_RUN=false` -- default true: orders are logged as "would submit", not sent. Real orders need BOTH settings
- `touch .kill_switch` -- stops all live trading instantly, no restart (delete the file to release)
- `MAX_ORDER_NOTIONAL_USD` (100), `MAX_SESSION_NOTIONAL_USD` (500 per broker), `MAX_ORDERS_PER_MINUTE` (6)
- An order whose value can't be determined (no price) is refused. Simulator and paper-mode connectors are unaffected.

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
| core/backtester.py, metrics | Backtest and Metrics Agent |
| ui/main_window.py, app.py, dash_app/*.py | UI Agent |
| test_*.py, scripts/, smoke tests | QA Test Agent |
| .github/workflows/, CI, packaging | Reliability Release Agent |

## Definition of Done
1. Code committed with descriptive message
2. `pytest --ignore=test_gui.py` passes locally
3. Sprint Board row updated to Done
4. CI green on GitHub Actions
