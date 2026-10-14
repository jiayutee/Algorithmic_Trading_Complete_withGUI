# Design slice 1: safety and clarity (2026-10-13)

Source: owner-requested design review of both front ends (canvas "AlgoTrader Design Review", board 4 roadmap,
"Slice 1 · safety & clarity"). Branch `design/slice1-safety-clarity`, base main `47a9ded`.

## What changed
| Area | Desktop (`ui/main_window.py`) | Web (`dash_app/`) |
|---|---|---|
| Account mode | Header pill: PAPER / DRY-RUN / LIVE / LIVE BLOCKED / NOT CONNECTED, refreshed every 1 s | Same pill (always PAPER: Dash trades the Simulator only), every 2 s |
| Kill switch | Header button: one click engages, release asks first | Same; release uses a confirm dialog |
| Price age | "Last SYMBOL: price" + "Price 12s ago" (amber > 60 s, red "stale" > 300 s) in Order Entry | Same line in Order Entry (crypto: WebSocket tick age; equities: last REST fetch) |
| Order review | BUY/SELL open a review (price used, notional, fee, cash after, account, warnings). Problems such as qty 0, no price or an unconnected broker block the order | Same review inline under Buy/Sell with Confirm / Back to edit |
| Skip review | "Review before sending" (default on) can be turned off for the paper account only; orders with warnings are always reviewed | Same toggle |
| Positions | New Positions tab (first): symbol, side, qty, avg, last, market value, unrealized P&L with ▲/▼ and %, Close button (reviewed). The right-panel text box became a compact table | Positions tab is now a table with the same columns and a Close button |
| Type | UI font is a sans stack (IBM Plex Sans → SF Pro → Helvetica Neue → Segoe UI); numbers in tables use a mono font; minimum text size 12 px; disabled text #484f58 → #6e7681 | Page font is the sans stack (was SF Mono everywhere); `assets/theme.css` adds a `.num` mono/tabular class; muted labels and status bar 11 → 12 px |
| Header | Glyph-only buttons now have words ("▶ Play", "⏸ Pause", "Reset zoom"); brand glyph removed | Header wraps on narrow windows (it used to push controls off-screen); brand glyph and "\|" separator removed |

Shared logic lives in `core/trading_status.py` (pure, no Qt/Dash): `account_mode`, `price_freshness`, `preview_order`,
`position_rows`, `engage_kill_switch` / `release_kill_switch` (they create/remove the file `brokers/execution_guard.py`
already watches, so the switch holds across the desktop app, Dash and scripts). `core/live_price_service.py` now
records when each cached price arrived (`get_price_age`).

## Behavior fixes found while doing this
- **Desktop broker selector vs order destination.** Once any order or backtest had set `current_broker`, changing the
  Broker selector did not change where orders went, and the Account box showed the selector's name with the old
  broker's balance. The selector now switches the broker, and an unconfigured broker shows "not connected" and blocks orders.
- **Desktop streaming price.** In "Realtime Stream" mode a market order filled at the last *loaded* bar's close, not the
  latest tick. It now uses the latest streamed price.
- **Dash broker creation race.** `_get_broker()` had no lock, so concurrent first-load callbacks could each create a
  SimulatedBroker on the durable account. During the first browser run, the P&L calendar and P&L card callbacks returned 500 on page load
  (traceback not captured). A lock now makes creation happen once. A clean first load afterwards: 16 callback requests,
  none failed. Root cause of the earlier 500 is plausible, not proven.

## Verification
- `tests/test_trading_status.py`: 27 tests (mode for every broker state incl. kill-switch file round trip, freshness
  thresholds, preview math and every warning/blocking path, Dash layout ids, positions table Close ids, review body,
  price fallback for non-charted positions, header view with the kill switch, live-service price age).
- Full suite (base env): **1530 passed, 1 skipped**.
- Desktop: offscreen `MainWindow` smoke with an in-memory BrokerManager. Checked: pill, price age, review text, fill,
  positions rows, stale-price warning still reviewed with the toggle off, qty 0 blocked, Close review + fill, kill switch
  on/off, Alpaca (unconfigured) pill "NOT CONNECTED" and blocked order.
- Web: ran the worktree's Dash app on port 8061 against a **scratch** paper account and **scratch** kill-switch file
  (real account and repo `.kill_switch` untouched). Loaded AAPL, Buy 5 → review (price 341.06, 9 s old, notional
  $1,705.30, fee $1.71, PAPER) → Confirm → filled; Positions table showed the row; Close → "SELL 5 AAPL" review →
  Confirm → filled, "No active positions". Kill switch engaged (scratch file written) and released (file removed).

## Not done / limits
- Not verified on a real display: the desktop smoke ran offscreen; nobody has looked at the desktop window yet.
- The browser's native confirm dialog for releasing the kill switch was stubbed in automation, not clicked by hand.
- Dash has no broker selector, so its pill is always PAPER (the parity gap in `PHASE_3_1_FEATURE_PARITY.md` remains).
- IBM Plex fonts are not bundled; the stacks fall back to system fonts.
- Slice 2 (design tokens file, four workspaces, watchlist, risk panel, journal, bracket orders) is not started.
- Seen but not changed: desktop **Backtest resets the shared Simulator account in memory** (`run_backtest` clears
  positions/orders of `broker_manager.get_broker("Simulator")`, which is the durable paper account). It needs its own fix.
- Seen but not changed: after a fill the Dash P&L card showed +$0.00 while the account fell by the fee.
