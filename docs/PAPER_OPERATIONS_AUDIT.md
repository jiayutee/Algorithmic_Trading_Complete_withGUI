# Paper Operations Audit
**Date:** 2026-09-21  
**Branch:** orchestrator/day85-paper-ops-inspection  
**Scope:** CONTINUATION_PLAN item 4 — inspect the existing execution status surfaces before adding alerts, protective stops or multi-strategy allocation. Preserve: one runner, idempotency, entry-blocks-only risk gate, paper-only defaults.  
**Status:** INSPECTION slice. No profit claims. No strategy tested so far shows an edge (Phases 6.5-6.9).

---

## 1. What the existing status surfaces actually show

### 1.1 core/execution/view.py — the shared data source

`execution_view(journal, now, limit)` is the single read path; both UIs call it.  
File: `core/execution/view.py`.

**Return dict keys (as of this audit):**

| Key | Content |
|---|---|
| `headline` | One-line summary: RUNNING/STOPPED, strategy name, interval, symbols, `last heartbeat MM-DD HH:MM:SS`, halt reason if any. |
| `running` | `True` iff heartbeat is fresh (< 3×poll_seconds or 60s, whichever is larger) AND a valid lease holder exists. (view.py:46) |
| `halted` | `{"reason": ..., "at": ...}` or `None`. (journal.py:73) |
| `symbols` | Per-symbol dict from the last published `per_symbol` state: symbol, state, signal, position, bar, data_age_bars, detail. (view.py:57-58) |
| `decisions` | Last 40 decisions from the journal: when (formatted), symbol, action, status, detail, id. (view.py:59-60) |
| `issues` | Reconciliation issues, last_error, stale-heartbeat warning string. (view.py:48-52) |
| `account` | From the last heartbeat: equity, cash, realized_pnl, unrealized_pnl. (service.py:410-412) |
| `risk` | Snapshot of `RiskConfig` fields at last heartbeat. (service.py:419) |
| `hb_age_s` | Heartbeat age in seconds (float, rounded to 1 decimal), `None` if no heartbeat. **Added in this slice.** (view.py:62) |

**What was computed but NOT previously in the return dict:**  
`hb_age` (heartbeat age in seconds) was computed at view.py:44 but only used internally. UIs received a formatted timestamp (`MM-DD HH:MM:SS`) in the headline but not a machine-readable age. This made it impossible for a UI to colour the heartbeat red when stale without parsing the headline string or re-computing from raw data.

**Small improvement made in this slice:** `hb_age_s` is now added to all three return paths in `execution_view`. Tests: `tests/test_execution_view_audit.py`. See Section 5.

---

### 1.2 Desktop PyQt5 — ui/main_window.py

**Entry point:** `_setup_execution_tab()` at main_window.py:1014.  
**Refresh:** QTimer fires every 5000 ms → `_refresh_execution_view()` at main_window.py:1044.

**Displayed fields:**

| Widget | Content | File:line |
|---|---|---|
| `_exec_headline` (QLabel) | `v["headline"]` — one-line summary. | main_window.py:1018, 1051 |
| `_exec_issues` (QLabel, red) | `"\n".join(v["issues"])` | main_window.py:1027, 1052 |
| `_exec_table` (QTableWidget) | `v["decisions"]` — columns: When / Symbol / Action / Status / Detail | main_window.py:1031-1057 |
| Status bar | Action feedback messages (halted, resumed, closed, error). | main_window.py:1062-1081 |

**Buttons:** "Halt entries" (main_window.py:1022, 1059), "Resume" (1065), "Flatten all" (1071).  
"Start Paper" / "Stop" at main_window.py:400, 1877-1904 starts/stops the in-process service.

**What the desktop tab does NOT show:**  
- Per-symbol state table (`v["symbols"]` is fetched but not rendered in any widget on the execution tab)  
- Account balance / P&L (`v["account"]` is available but not displayed in this tab)  
- Heartbeat age in seconds (now available as `hb_age_s` but not yet wired to a widget)  
- Risk config snapshot  

---

### 1.3 Dash — dash_app/callbacks.py and dash_app/layout.py

**Callback:** `execution_tab` at callbacks.py:2464. Interval: 5000 ms (`exec-interval`).

**Displayed fields:**

| Output id | Content | File:line |
|---|---|---|
| `exec-headline` | `v["headline"]` | callbacks.py:2476, layout.py:1623 |
| `exec-issues` | `"\n".join(v["issues"])` | callbacks.py:2476, layout.py:1624 |
| `exec-symbols-table` | `v["symbols"]` — columns: Symbol / State / Signal / Position / Bar / Data age (bars) / Note | callbacks.py:2476, layout.py:1626-1629 |
| `exec-decisions-table` | `v["decisions"]` — columns: When / Symbol / Action / Status / Detail | callbacks.py:2476, layout.py:1631-1633 |
| `exec-message` | Action feedback from `_execution_action`. | callbacks.py:2473-2476 |

**Buttons:** Start paper trading / Stop / Halt entries / Resume / Flatten all (layout.py:1614-1618).

**What Dash does NOT show in the execution tab:**  
- Account balance / P&L (tracked by a separate `update_pnl_card` callback at callbacks.py:2509, driven by price-interval — not the execution interval)  
- Heartbeat age in seconds (now available as `hb_age_s` but not yet wired to the layout)  
- Risk config snapshot  

**Dash advantage over desktop:** the per-symbol state table (`v["symbols"]`) IS rendered in Dash; the desktop execution tab renders only the decisions table.

---

### 1.4 Journal schema — core/execution/journal.py

Two SQLite tables.

**`decisions`** (journal.py:30-38): one row per decision_id (UNIQUE). Columns:  
seq, decision_id, ts, symbol, bar_ts, strategy, signal, target_qty, current_qty, order_qty, side, price, action, status, order_id, filled_qty, fill_price, risk (JSON), rationale (JSON), error.

`action` values: `hold | submitted | blocked | error | submitting`  
`status` values (broker): `filled | rejected | pending | ...`

**`kv`** key/value store keys (current): `status` (heartbeat, per_symbol, account, risk config...); `halt`; `lease`; `day_baseline`; `peak_equity`.

Idempotency is enforced by the UNIQUE constraint on `decision_id` (journal.py:116-123). `record()` returns `False` (and writes nothing) if the id already exists. This is a database-level guarantee, not just an if-check.

---

## 2. Design proposals — what would be needed

These are PROPOSED designs only. Nothing in Section 2 is implemented. Each requires explicit owner approval before any implementation begins.

---

### 2.1 Alerts (Telegram / desktop notification)

**What is needed:**  
A non-blocking alert channel that fires when notable events are detected in the journal or view. The five event types already detectable from existing data:

| Event | Detection point | Already-stored data |
|---|---|---|
| Service halted | `v["halted"]` truthy | `halt.reason`, `halt.at` |
| Stale data detected | `issues` list contains "stale" | `data_age_bars` per symbol in `per_symbol` |
| Risk-gate entry blocked | `decisions[*].action == "blocked"` | `decisions[*].risk.reasons` |
| Runner death / heartbeat gap | `v["running"] is False` after being True; or `hb_age_s > threshold` | `status.heartbeat` |
| Fill (order executed) | `decisions[*].action == "submitted"` and `status == "filled"` | `decisions[*].fill_price`, `filled_qty` |

**Proposed design:**  
New module `core/execution/alerts.py` with a single callable:

```
check_and_send(journal, prev_state, now) -> new_state
```

- `prev_state` is a small dict stored in the kv table under key `"alert_state"` (or in memory per process for volatile state).  
- Detects transitions (e.g. `halted` was `None`, now it is not) rather than re-firing on every poll.  
- Telegram channel: `scripts/telegram-listener.py` already has a send path; alerts could reuse a `send_message(text)` helper exposed from that script.  
- Desktop: `QSystemTrayIcon.showMessage` in `ui/main_window.py` (opt-in; requires tray icon setup not currently present).

**Files and functions that would change:**  
- New: `core/execution/alerts.py` (AlertConfig dataclass, AlertState dataclass, `check_and_send`)  
- New: `tests/test_execution_alerts.py` (all tests with tmp_path journals — never live files)  
- `core/execution/service.py:_publish_status` (line 406) — call `check_and_send` after writing the kv status  
- `scripts/telegram-listener.py` — expose a `send_message(text)` helper callable from outside the bot  
- `ui/main_window.py` — optional tray-icon wiring (new method, no changes to existing tab widgets)

**Risks:**  
- Alert-on-every-poll would duplicate; transition detection is required.  
- A Telegram send that blocks the service loop must be on a separate thread with a timeout.  
- Alerts must NEVER block exits. The alert check runs after `_publish_status`, never inside `RiskGate.evaluate()`.  
- Adding `"alert_state"` to the kv table is backward-compatible; existing code ignoring unknown keys is safe.

**Tests needed:**  
- Halt alert fires exactly once on transition from `None` to a halt record, not on every subsequent poll.  
- Heartbeat-gap alert fires only after a configurable gap (not on the first missed tick).  
- Fill alert fires for each new fill, not for already-seen decision ids.  
- Alert failure (Telegram unreachable) is logged but does not stop or delay the service loop.  
- All tests use tmp_path journals; no live files.

**What is NOT decided — owner approval needed:**  
- Telegram vs. desktop notification vs. both; configurable per-event.  
- Alert thresholds: how many missed heartbeats before "runner death" fires (2× poll_seconds? 5×?).  
- Whether filled-order alerts should include fill price and estimated P&L impact.  
- Whether the alert module is co-located in `core/execution/` or kept in `scripts/`.

---

### 2.2 Protective paper stop rules

**What is needed:**  
Automatically close a position if it moves adversely beyond a threshold (e.g. 2% fixed stop-loss from entry price, or a trailing stop from position peak).

**Layer analysis:**  
A protective stop is an EXIT, not a blocked entry. The risk gate (`core/execution/risk.py`) is the entry-blocking layer; the design rule at risk.py:1-9 states it can only block/shrink entries. Protective stops do not belong there. The correct layer is the signal/pipeline step in `_process_symbol` (service.py:254-365) — specifically a new check before the strategy signal that forces `direction=0` (flat) if a stop condition is met.

**Proposed design (Option B — preferred):**  
Add a stop-check block inside `_process_symbol` at service.py before the strategy signal evaluation (step 3). If the current position's mark-to-market loss exceeds the threshold:

1. Build a unique `decision_id` that encodes the stop type: `f"{symbol}|stop|{bar_open.isoformat()}"`.
2. Check idempotency: `journal.has_decision(decision_id)`.
3. Record intent, submit the close order, update the journal — identical to the normal order path (steps 7-8 of the existing pipeline).
4. Return `{"state": "traded", "side": ..., ...}` to `_publish_status`.

The risk gate's `is_exit=True` path (risk.py:68) returns `RiskVerdict(True, qty, True, ["exit: ..."])` unconditionally, so a stop order bypasses all entry limits (daily loss, drawdown, halt, gross exposure) and can NEVER be blocked.

**Idempotency and single-runner-safe:**  
- One stop decision per bar per symbol: the idempotency gate (journal.py:116-123) prevents re-entry on restarts.  
- The stop decision_id includes the bar timestamp, so two ticks on the same bar produce one decision.

**Journal / schema implications:**  
- No schema change. The stop decision uses the existing `decisions` table with `action="submitted"` and `strategy="stop"` (or `f"{strategy_name}|stop"`).  
- `rationale` should record: stop type, entry price, current price, loss threshold, and the resulting decision_id.  
- No new kv keys required.

**Files and functions that would change:**  
- `core/execution/risk.py:RiskConfig` — add optional `stop_loss_pct: Optional[float] = None` and `trailing_stop_pct: Optional[float] = None` (both None = no stop, preserving current behaviour exactly)  
- `core/execution/service.py:_process_symbol` — add stop-check block (approximately 20 lines before the signal step)  
- `core/execution/service.py:ExecutionConfig` — no change; it embeds `RiskConfig`  
- `core/execution/launcher.py` — no change; RiskConfig is passable already  
- New: `tests/test_execution_stops.py` (tmp_path only)

**Tests needed:**  
- Stop fires and closes the position when loss exceeds threshold.  
- Stop does NOT fire when the position is within threshold.  
- Stop fires even when daily-loss, drawdown and halt flags are active (exits can never be blocked).  
- Restarting the service on the same bar does not re-submit the stop (idempotency).  
- Stop decision appears in `journal.decisions()` with `strategy` containing "stop" and `action == "submitted"`.  
- A single bar produces at most one stop decision per symbol.  
- Trailing stop updates the reference price correctly across multiple bars.

**What is NOT decided — owner approval needed:**  
- Stop type: fixed loss from entry vs. trailing from position peak vs. both.  
- Whether the stop replaces the strategy's own exit (i.e. signal exit would also fire on the same bar) or is additive.  
- Entry-price tracking: the broker position has `avg_price` (simulatedbroker.py) which is sufficient, but cost-basis after partial fills must be validated.  
- Whether paper stop fills should be labelled distinctly in the Execution tab decisions table.

---

### 2.3 Multi-strategy allocation

**What is needed:**  
Running more than one strategy simultaneously on the same paper account (e.g. EMA Crossover on BTC, Trend Filter on ETH) with separate allocation percentages per strategy.

**Interaction with one-runner / one-lease:**  
The existing guarantee is: exactly one process holds the lease and trades the account at any time (journal.py:77-107). This prevents double-trading and race conditions.

**Scenario analysis:**

*Scenario A — current model:* One runner, one strategy, multiple symbols. Already works. The gross-exposure limit prevents total over-allocation.

*Scenario B — one runner, one signal per symbol:* The runner's `_process_symbol` loop is per-symbol. If `ExecutionConfig` holds a `signals: Dict[str, BacktraderReplaySignal]` instead of a single signal, each symbol runs its own strategy class. The journal's decision_id already includes the signal name (`service.py:268`), so mixed-strategy decisions coexist without key collision. This is the minimal change needed.

*Scenario C — multiple runners, one per strategy:* The lease prevents this entirely. A second runner blocks at `acquire_lease`. This scenario would require removing or relaxing the single-runner guarantee, which breaks idempotency and is OUT OF SCOPE without a new design.

**Exposure-limit interaction (Scenario B):**  
- `max_position_pct` is per-symbol: a 25% cap on BTCUSDT prevents any single decision from exceeding that, regardless of which strategy generated it.  
- `max_gross_pct` sums `|qty * price|` across the entire broker position book (service.py:249-251), already accounting for all open positions from any strategy.  
- **New risk:** if two different strategy signals call for entries on the same symbol at the same bar, the second call hits the idempotency gate (same decision_id only if the strategy name is identical — so different names produce two distinct decisions). The per-symbol and gross exposure caps prevent combined over-allocation, but two entries on the same bar could each be approved up to `max_position_pct`, doubling the symbol notional.  

**Journal key conflicts between strategies on the same symbol:**  
The decision_id format is `f"{symbol}|{interval}|{bar_open.isoformat()}|{signal.name}"` (service.py:268). Two strategies with different `signal.name` values do not conflict. However:  
- `_publish_status` stores `per_symbol` as a flat dict keyed by symbol alone (service.py:211-213). Two strategies on the same symbol overwrite each other's `per_symbol` entry. For Scenario B, `per_symbol` must be keyed by `(symbol, strategy_name)` or `f"{symbol}|{strategy_name}"`.  
- The `v["symbols"]` table in both UIs is built from this dict (view.py:57-58): without the schema change, two-strategy-per-symbol rows collapse to one. **A UI schema change is required.**

**Files and functions that would change (Scenario B only):**  
- `core/execution/service.py`: `ExecutionConfig.signal` → `signals: Dict[str, BacktraderReplaySignal]`; `_process_symbol(sym, now)` receives the per-symbol signal; `_publish_status` writes per-symbol state keyed by `f"{symbol}|{strategy_name}"`.  
- `core/execution/launcher.py:build_service` — accepts a `signals` dict or list of `(symbol, signal)` pairs.  
- `core/execution/view.py` — `symbols` list must handle the new composite key.  
- `core/execution/journal.py` — no schema change (decision_id already includes strategy name).  
- `ui/main_window.py` and `dash_app/layout.py` — the per-symbol table needs a strategy-name column.  
- New tests in `tests/test_execution_service.py` and `tests/test_execution_ui.py`.

**What is NOT decided — owner approval needed:**  
- Whether Scenario B (one runner, per-symbol signals) or Scenario C (multi-runner) is the intended model.  
- How to configure per-strategy `allocation_pct` when strategies trade different symbols vs. the same symbol.  
- Whether cross-signal hedging (strategy A long BTC, strategy B short BTC) should be allowed or blocked.  
- Whether the launchd always-on runner (`scripts/execution.env`) should support multi-strategy config, and what its format would be.  
- How to handle a signal whose `strategy.name` changes between restarts (existing journal rows have the old name).

---

## 3. What is implemented vs. proposed

| Capability | Implemented | Location |
|---|---|---|
| Execution status in PyQt5 headline | YES | main_window.py:1044-1057 |
| Execution status in Dash tab | YES | callbacks.py:2464-2476 |
| Per-symbol state in Dash (not desktop) | YES | layout.py:1626-1629 |
| Per-symbol state in PyQt5 execution tab | NO | v["symbols"] fetched but not rendered |
| Account balance in execution tab (either UI) | NO | v["account"] available, not rendered in exec tab |
| Heartbeat age in seconds in view output | YES (added this slice) | view.py:62 |
| Risk-gate blocks entries, never exits | YES | risk.py:68 |
| Single runner / lease | YES | journal.py:77-107 |
| Idempotency per bar | YES | journal.py:114-123 |
| Halt/resume persistent flag | YES | journal.py:66-74 |
| Alerts (Telegram / desktop) | NO | — |
| Protective paper stop rules | NO | — |
| Multi-strategy allocation | NO | — |

---

## 4. Commands run against live state

Only read-only (the live launchd paper runner is running; no writes were made):

```
python -m core.execution.service status
```

(No halt, resume, flatten, start, stop, kickstart, bootout or sqlite3-write command was run. The live `training_ground/paper/paper_account.sqlite3` and `training_ground/paper/execution.sqlite3` files were not opened for writing.)

---

## 5. Code change made in this slice

**File changed:** `core/execution/view.py`  
**Change:** Added `hb_age_s` (heartbeat age in seconds, rounded to 1 decimal place, `None` if no heartbeat) to all three return paths in `execution_view()`. This field was already computed (`hb_age` at view.py:44) but not surfaced in the dict. Making it explicit allows UIs to display "heartbeat 42s ago" or colour the status red when stale without re-parsing the headline string.

This is a purely additive, read-only change. No behaviour changes; no live-file access.

**Tests added:** `tests/test_execution_view_audit.py` — four tests covering all return paths (normal with heartbeat, normal without heartbeat, never-run fast-path, broken journal), all using tmp_path journals or env-var isolation — never the live files.
