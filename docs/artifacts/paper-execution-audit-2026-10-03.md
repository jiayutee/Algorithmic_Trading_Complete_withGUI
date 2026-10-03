# Paper execution status audit — 2026-10-03 (Day 98, report only)

CONTINUATION_PLAN item 4, first step: look at what the always-on paper execution service is actually doing
before any alerts or stop rules are designed. **Read-only.** No code changes, no halt/resume/flatten, no
writes to the paper account or the execution journal (both SQLite files were opened with `mode=ro`), no
launchd changes. Checked at 2026-10-03 ~23:30 CEST against the running checkout (main `79eaaa9`).

## Sources
- `launchctl list | grep paper-execution`
- `python -m core.execution.service status`
- `logs/execution.log` (3.4 MB, runs since 2026-09-19)
- `training_ground/paper/execution.sqlite3` (journal: `decisions`, `kv`), `training_ground/paper/paper_account.sqlite3`
- Code read: `core/execution/service.py` (`_baselines`, decision steps 5-6), `core/data_loader.py` (Yahoo fallback map)

## Current state
| Item | Value |
|---|---|
| launchd | `com.algotrader.paper-execution` loaded, PID 33125, last exit 0 |
| Runner | RUNNING, Trend Filter (28d), 1d bars, BTC/ETH/SOL/BNB, 15% each, poll 300 s |
| Lease | owner `248f94536787`, TTL 180 s, renewed every poll; heartbeat 23:26:34 (current) |
| Halted | no; `last_error` empty; reconciliation ok, no issues |
| Last decision bar | 2026-10-02 for all 4 symbols (state `up_to_date`, data age 0.89 bars) — next bar decides after 02:00 |
| Account | equity 101,675.39; cash 39,953.51; unrealised +1,735.37; realised 0.00 |
| Positions | BTC 0.184069 @ 81,491.19 · ETH 5.667591 @ 2,646.23 · SOL 134.440562 @ 111.54 · BNB 19.642929 @ 763.29 (all opened 2026-09-18 bar) |

## Decision history (journal)
- 60 decisions: 4 `submitted/filled` entries (2026-09-18 bar, risk "all entry checks passed") and 56 `hold`
  (signal LONG, already long). 15 decisions per symbol = bars 09-18 through 10-02, **no missing bar, no
  duplicate**, so the one-decision-per-bar guarantee held across restarts and the 10-02 clamshell sleep.
- **0 risk-gate blocks, 0 halts, 0 errors** recorded in the journal. No exits yet (no trend flip).

## Log review (last 7 days)
Errors in `execution.log` are all market-data fallbacks, none reached a decision:
- 129× `Binance historical failed: No data returned ... BTCUSDT` (with a matching `Failed to fetch batch` warning).
  Only BTCUSDT — the first symbol fetched each poll — and intermittently (4–35/day; 29 on 10-03). ETH/SOL/BNB
  load from Binance seconds later in the same poll. Each time the loader fell back to Yahoo and got 500 candles.
- 123× Yahoo chart read timeouts and ~50 `yf.download ... rate-limited; giving up` for BTC-USD (the fallback's own fallback).

## Findings
1. **Peak equity and day baseline only update when an order is about to be sent** (CONFIRMED in code).
   `_baselines()` is called only in step 6 (risk approval), after the `hold` early-return. With 56 holds and no
   orders since 09-18, the journal still has `peak_equity = 100000.0` and `day_baseline = 2026-09-19 / 100000`
   while equity is 101,675. Effect: the drawdown limit (15%) is measured from a stale, lower peak, and the
   daily-loss limit (3%) sees the day start as "now" at the moment of the first order of the day. Both make
   the entry gate *less* conservative than documented. Exits are unaffected (never blocked). Fix idea (not
   done): refresh baselines once per poll, before the per-symbol loop.
2. **BNBUSDT has no Yahoo fallback** (CONFIRMED). `_get_yahoo_crypto_historical` maps BTC/ETH/SOL/ADA only, so a
   Binance failure for BNB ends in `yf.download(BNBUSDT)` returning empty (seen 2026-09-21). Fix idea: add
   `BNBUSDT -> BNB-USD` (or derive `<BASE>-USD` generically).
3. **Intermittent Binance failure on the first symbol of each poll** (PLAUSIBLE cause unknown). The warning text
   is truncated at the URL, so the exception type is not logged. Harmless today (Yahoo fallback works for BTC)
   but it is the main source of log noise and makes the BTC bar depend on Yahoo. Next step: log the exception
   class/message; check whether it is a first-request/time-sync issue in the CCXT client.

## What the front ends already show
The Dash "Execution" tab and the desktop Execution tab both render `core.execution.view.execution_view` (the durable
journal): running/halted, heartbeat, per-symbol state/bar/data age, account, recent decisions with
rationale. There is **no alert path**: a stale heartbeat, a halt, a risk block or a reconciliation issue is only
visible if someone opens the tab or runs `status`.

## Inputs for the alert / stop-rule design (next step, not started)
- Alert on: heartbeat older than ~3 polls; `halted` set; `last_error` non-empty; reconciliation not ok; any
  `blocked` decision; a missing decision for a completed bar.
- Fix finding 1 before relying on drawdown/daily-loss limits as stop rules.
- Not a profit claim: the paper P&L (+1.7% since 09-18) is a 2-week, single-regime sample.
