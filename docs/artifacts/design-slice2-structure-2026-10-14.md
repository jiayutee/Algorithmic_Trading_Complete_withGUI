# Design slice 2: structure (2026-10-14)

Source: owner request "merge and proceed with notion and slice 2" (2026-10-13) and roadmap board 4 of the design review
canvas. Branch `design/slice2-structure`, **stacked on** `design/slice1-safety-clarity` (PR #39, not merged at the time
of writing because the merge was blocked by the session's permission check, so the owner merges it).

## What changed
| Area | Desktop (`ui/main_window.py`) | Web (`dash_app/`) |
|---|---|---|
| Design tokens | Global stylesheet is now `design_tokens.qt_stylesheet()` (was a 40-line hard-coded sheet) | `design_tokens.css_root()` is injected as CSS variables; `assets/theme.css` uses only `var(--…)` |
| Workspaces | Trade / Research / Automate / Markets buttons in the header. Each shows only its bottom tabs and its actions: Backtest + backtest parameters + results in Research; Simulate / Play / Pause / Start Paper in Automate | Same four as a segmented control. Tabs outside the workspace are hidden; Backtest Results card only in Research; Risk card outside Research |
| Watchlist | Left panel: symbol, last, change; click a row to chart it; Add / Remove | Right column: same table; click a symbol to chart it (sets the symbol and presses Load Chart); × removes; Enter or Add adds |
| Risk panel | Right panel: gross exposure, largest position, day P&L vs start, drawdown from peak, runner orders today, each against its limit (amber at 75%, red "at limit") | Same rows in a card, refreshed every 2 s and after each order |

Shared code: `core/design_tokens.py` (colors by role, font stacks, type/space/radius scales; `core/chart_builder.THEME`
and `core/trading_status` colors now read from it), `core/watchlist.py` (shared JSON list at
`training_ground/paper/watchlist.json`, git-ignored, override `WATCHLIST_PATH`; background `QuoteBook`, crypto 10 s /
others 60 s cache), `core/risk_view.py` (read-only; limits come from what the runner published, else `RiskConfig`
defaults; never creates the execution journal), `core/ui_options.WORKSPACES` / `TAB_WORKSPACE` (one map for both apps).

Also fixed: since slice 1 made Positions the first tab, the desktop's "switch to Orders after an order" went to
Positions (`setCurrentIndex(0)`); it now finds the Orders tab by name.

## Dropped from the slice, with reason
- **Trade journal tab.** The design review said rationale was "recorded but never shown". That was wrong: both apps
  already show it in the Orders "Why" column, and the runner's decisions are in the Execution tab. A separate tab would
  duplicate them, so it was not built.
- **Bracket orders and time in force.** These change how the shared paper account fills orders, which is a broker change, not a
  design change. They need their own slice with broker tests.

## Verification
- `tests/test_design_slice2.py`: 16 tests. Generated Qt and CSS use only token colors; chart/mode colors come from the tokens;
  `theme.css` has no stray colors. Watchlist store round trip and `WATCHLIST_PATH`. Quote book caching, last-good price kept on
  error, age. Binance and error parsing. Risk snapshot with and without a journal, reading runner limits, day
  baseline, peak, order count, halt, and confirming the journal file is not created. Workspace map. Dash ids, workspace view, watchlist and
  risk bodies.
- Full suite (base env): **1546 passed, 1 skipped**.
- Live quotes: `python -m core.watchlist` returned all six defaults (BTC/ETH/SOL/BNB from Binance, SPY/AAPL from
  Yahoo, delayed).
- Desktop, offscreen: each workspace shows the expected tabs/buttons/panels; `_show_tab("Orders")` switches to Trade
  and Orders; watchlist filled with live quotes; add/remove persisted to a scratch file; risk rows after a 0.5 BTC buy
  (25.0% / 25.0% "at limit" for one symbol). **Not yet looked at on a real display.**
- Web, in a browser at 1400×900 against a scratch paper account, journal, kill switch and watchlist: workspace switch shows
  the right tabs (Trade: Positions/PnL Calendar/Orders; Research: Equity Curve/Research Lab/Research Loop; Automate:
  Agent Monitor/Execution; Markets: Market Context/News & Earnings/Options Chain), Backtest card only in Research;
  clicking ETHUSDT in the watchlist charted it (live price 0 s); MSFT added and removed; no console errors, no server
  tracebacks, no failed callbacks; a 4 px horizontal page scroll (bootstrap row gutters) was removed.

## Not done / limits
- Many inline colors remain inside individual desktop widgets and Dash components (134 + 48 before this slice). The global
  sheet, chart theme, mode colors and the web CSS now come from tokens; per-widget colors are next.
- The web layout is still not usable at phone width (fixed bootstrap columns), and the workspace choice is not remembered between
  sessions.
- Watchlist change % for non-crypto is the last daily close vs the previous one from Yahoo (delayed, not a live quote).
- The risk panel shows the runner's limits; manual orders are not gated by them (the panel says so).
