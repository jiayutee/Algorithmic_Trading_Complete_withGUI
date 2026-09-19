# AGENTS.md — rules for any AI coding tool working in this repo

This project is worked on by more than one tool (Claude Code, Codex, scheduled routines) and one human owner.
**Read `CLAUDE.md` first**: it is the detailed source of truth (architecture, environments, commands). This file is the short list
of things that go wrong when several tools share one folder. If the two ever disagree, `CLAUDE.md` wins; tell the owner.

## Never do these
- **Never place real orders or weaken the live-trading guard.** Live orders are blocked by default (`LIVE_TRADING_ENABLED` off,
  `LIVE_DRY_RUN` on, `.kill_switch` file). Do not change those defaults, set them in code, or route around `brokers/execution_guard.py`.
  The execution service (`core/execution/`) is **paper only** by design; keep it that way.
- **Never commit secrets.** `.env` and `config/settings.py` hold real keys and are git-ignored. Run `git status` before every commit.
- **Never push to `main` unless the owner explicitly asks in that message.** Use a branch and a pull request.
- **Never stop, restart or reconfigure the background jobs** (below) without asking, and never `flatten`/`halt` the paper account as a test.
- **Never claim a trading edge.** Nothing tested so far beats buy-and-hold on risk-adjusted return (Phases 6.5–6.9). Report results as they are.

## Several tools, one folder
- Do your work in **your own branch and, ideally, your own git worktree**: `git worktree add ../algotrader-<task> -b <tool>/<task>`.
  Do not switch branches in the main folder: background jobs run the code checked out there (`main`).
- Pull/fetch before starting; keep changes small; open a PR. Use **merge commits** (not squash) so pre-registration history survives.
- Stacked PRs do not retarget themselves when their base merges: retarget to `main` before merging. CI runs only on PRs targeting `main`.
- Git-ignored files (`.env`, `config/settings.py`) do not exist in new worktrees; copy `config/settings.py` in if a test needs `import app`.

## Handing work between tools
- Each slice of work leaves a trail: update the **handoff and plan** (`docs/CODEX_HANDOFF.md`, `docs/CONTINUATION_PLAN.md`, written by Codex; they may sit on PR #8 until it merges),
  the notebook, the Notion Sprint Board task, and the PR description. Say exactly what was tested and what remains. Never mark a whole phase done for a diagnostic-only change.
- Before starting, read the newest handoff and the open PRs (`gh pr list` or the GitHub page) so you do not redo or collide with someone else's slice.
- Untracked documentation copies in the main folder can block `git checkout` of the branch that contains them: compare with the PR, then remove the duplicate.

## Things running on this machine (launchd + scheduled routines)
| What | How to check | Notes |
|---|---|---|
| Paper execution agent `com.algotrader.paper-execution` | `python -m core.execution.service status` · `logs/execution.log` | Trades the paper account on its own. Settings: `scripts/execution.env`. Only one runner may hold the lease. |
| Data collectors `com.algotrader.collect-news`, `collect-kalshi` | `logs/collectors.log` | Append to SQLite files under `training_ground/`. |
| Telegram listener `com.algotrader.telegram-listener` | `logs/` | Two-way bot for the owner. |
| Overnight Claude routine (briefing, work loop, debrief) | `~/.claude/scheduled-tasks/` | May edit code and push overnight (Berlin time). Keep your branch names distinct. |

The paper account and execution journal are shared SQLite files in `training_ground/paper/` (`PAPER_ACCOUNT_PATH`, `EXECUTION_DB_PATH`).
Running `app.py`, the Dash app or the execution CLI touches the **same** account. For experiments, point those env vars at a temp path.

## Environment and tests
- Use `~/miniconda3/bin/python3` (Python 3.9). **Never** the `myenv` conda env (it runs out of memory). NautilusTrader needs `~/.venvs/algotrader311`
  (`scripts/setup_py311_env.sh`). Code must stay Python 3.9-compatible (`from __future__ import annotations` for `X | None` type hints).
- Run after every code change: `~/miniconda3/bin/python3 -m pytest --ignore=test_gui.py -q`.
  Desktop GUI tests: `QT_QPA_PLATFORM=offscreen ~/miniconda3/bin/python3 -m pytest test_gui.py -q` (not run in CI).
- Add tests with the change. Tests must not touch the real paper account, real journal, network keys or live brokers (use `tmp_path` and fakes).

## Research rules (this repo's house style)
- **Pre-register before you run.** Write the hypotheses, fixed protocol, multiple-comparison correction and pass/fail rule in `docs/PHASE_*_PREREGISTRATION.md`,
  commit it, *then* write and run the code. No tuning after seeing results; any change after a run is labelled and disclosed (see Phases 6.5–6.9, 9.3a).
- No lookahead: walk-forward validation with a purge gap; use only completed bars; features must be available at decision time.
- Report negative results plainly. Void and disclose a run that turns out to be a data artefact rather than quietly rerunning it.

## Known gotchas
- Yahoo's `yfinance` path gets HTTP 429; `core/yahoo_chart.py` is the fallback. Kalshi `/markets` needs `mve_filter=exclude` or a scan sees only empty combos.
- The paper broker is **strict**: unpriced or stale market orders are rejected. Feed prices with `update_price()`; do not invent them.
- `include_news=False` for anything that only needs OHLCV (news scraping is slow).
- `docs/algotrader_notebook.html` is the source of the published "AlgoTrader Notebook" artifact (edits are not visible on claude.ai until someone republishes it, and two tools editing it can overwrite each other). The Notion Sprint Board tracks tasks. Coordinate with the owner before editing either.
