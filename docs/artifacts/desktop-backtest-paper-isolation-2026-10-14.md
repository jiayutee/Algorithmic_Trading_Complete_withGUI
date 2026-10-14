# Desktop backtest no longer touches the paper account (2026-10-14)

Base: main `47a9ded`. Branch `claude/elastic-kalam-772fd4`. Files: `ui/main_window.py`, `tests/test_backtest_paper_isolation.py`.

## Problem
`MainWindow.run_backtest()` called `broker_manager.get_broker("Simulator")` and overwrote its `balance`,
`initial_balance`, `portfolio_value`, `positions`, `order_history`, `closed_positions`, `orders`, `market_fee` and
`limit_fee`. After the run it wrote the backtest's Final Value into the balance and set `self.current_broker` to it. In
`app.py` that Simulator is the durable paper account (`training_ground/paper/paper_account.sqlite3`), which Dash and
the always-on paper execution service also use.

The broker was not needed. `Backtester.run_backtest` runs on backtrader's own broker; it takes cash and fees as
arguments and only logs the `broker` argument.

## Reproduction (scratch account only)
`PAPER_ACCOUNT_PATH` pointed at a temp file; the script asserted the resolved path was inside that temp dir. The script
opened BTCUSDT 0.5 @ 50,000, built `MainWindow` offscreen (`QT_QPA_PLATFORM=offscreen`) with a 300-bar synthetic
series, then ran EMA Crossover with cash 10,000 and fees 0.5% / 0.25%. Plotting was stubbed because no chart was
loaded; without the stub the run stops at `plot_signals` after the broker was already overwritten.

| | positions | orders | balance | market_fee |
|---|---|---|---|---|
| Before, memory and file | BTC 0.5 | 1 | 74,975.00 | 0.001 |
| After backtest, **memory** | none | 0 | 9,388.99 (backtest Final Value) | **0.005** |
| After backtest, file | BTC 0.5 | 1 | 74,975.00 | 0.001 |
| Next paper order (buy 1 ETH @ 3,000), memory | BTC, ETH | 2 | **71,960.00** | 0.005 |
| Next paper order, **file** | BTC, ETH | 2 | 71,960.00 | **0.005** |

Findings:
1. **In memory: lost.** The desktop Account panel (`current_broker` was switched to the paper Simulator) showed the
   backtest's $9,388.99 with no positions, even when another broker was selected.
2. **In the file: positions/orders survive.** `_persist_locked` only runs inside `_operation()`, which reloads from
   sqlite first, so the next account operation restores them. `get_account_info()` only reloads when another
   connection has committed (`changed_externally`), so the wrong numbers stay on screen until then.
3. **In the file: the fees leak.** `_load_from_store_locked` does not reload `market_fee`/`limit_fee`, so the next real
   paper order was charged the backtest's 0.5% (71,960 instead of 71,972) and `_persist_locked` wrote 0.005 to the
   account's meta. That is a durable change.
4. Not exercised: `SimulatedBroker.reset()` with no argument uses the in-memory `initial_balance`, so a reset right
   after a backtest would also have persisted the backtest cash. No UI path calls `reset()` today.

The real account was checked read-only (`mode=ro`) on 2026-10-14: `market_fee` 0.001, `limit_fee` 0.0005,
`initial_balance` 100000. No leaked fee is visible there. The service process keeps its own broker object, so the
desktop clobber could not reach it except through the file.

## Fix
`run_backtest()` now calls `_run_backtest_logic()` (no broker). It no longer fetches or modifies any app broker, writes
no Final Value into a broker, and leaves `current_broker` alone, so the Account panel keeps the selected broker. The
unused `_run_backtest_logic_with_broker` wrapper was removed. The results (status bar, Sharpe/win-rate/maxDD/alpha/beta
labels, signals, equity curve, Research Lab) are unchanged: same Final Value (9,388.99) before and after.

Re-run of the reproduction after the fix: memory and file both unchanged after the backtest (BTC 0.5, 1 order,
74,975.00, fee 0.001), `current_broker` unchanged, and the next ETH order charged 0.1% (71,972.00).

## Tests
- `tests/test_backtest_paper_isolation.py` (2 cases: Account panel on no broker / on the paper Simulator). It asserts
  that the in-memory and sqlite state, `current_broker` and the Account label are unchanged after a backtest, and that
  the next paper order uses the account's own fee. **Fails on the old code (2 failed), passes with the fix.** It skips
  where PyQt5 is not installed, so it does not run on the CI runner.
- Full suite, base env: `~/miniconda3/bin/python3 -m pytest --ignore=test_gui.py -q` gave **1505 passed, 1 skipped**.
  The first full run had one failure in `tests/test_paper_account.py::test_two_processes_share_one_account` (this change
  does not touch that code). The file passed 3 of 3 runs alone, and the full re-run passed, so it looks like an
  intermittent failure. Not investigated further.
- Not run: the 3.11 env and `test_gui.py`.

## Status
Implemented and tested locally; PR open for review. Not merged, not deployed. The running desktop app keeps the old
behaviour until it is restarted on a checkout that includes this fix.
