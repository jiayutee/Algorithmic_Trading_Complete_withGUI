# Phase 3.1 — Feature Parity Review: PyQt5 vs Dash

**Date:** 2026-08-17 (Day 52/51, sprint day before launch)
**Reviewed by:** Reliability Release Agent (autonomous overnight cycle)
**Scope:** `ui/main_window.py` (1811 lines) vs `dash_app/layout.py` + `dash_app/callbacks.py` + `dash_app/app.py`

---

## Decision

**PyQt5 is retained as the primary/canonical entrypoint. Dash is designated as the optional web
interface.** Dash must not become the sole entrypoint yet because five material gaps block it from
replacing PyQt5: (1) account balance and P&L are static placeholders — the Phase 4 callback that
wires `account-balance` and `pnl-value` to `SimulatedBroker` has not been implemented;
(2) no broker-selection dropdown — Dash is permanently bound to `SimulatedBroker` with no path to
Alpaca, Binance, or Interactive Brokers; (3) no "Go Live" trigger for live-trading sessions;
(4) step-through simulation mode (play/pause/candle-by-candle portfolio replay) is entirely absent;
(5) the Agent Monitor tab (start/stop `Supervisor`, per-agent status table, LLM summary line) does
not exist in Dash. Until these five items are closed, `app.py` / PyQt5 remains the production
entry point and `dash_app/app.py` serves as an additional read-only analysis interface.

See the **Gaps** section below for the full list. See **Recommendation** for the sequenced plan to
close the gaps before designating Dash as sole entrypoint.

---

## Feature Comparison Table

| # | Feature | PyQt5 | Dash | Notes |
|---|---------|-------|------|-------|
| **Top Bar** | | | | |
| 1 | Brand label "◈ AlgoTrader" | Present | Present | Identical wording and color |
| 2 | Symbol dropdown | Present (9 items: BTCUSDT, ETHUSDT, SOLUSDT, ADAUSDT, AAPL, TSLA, GOLD, SPY, QQQ) | Partial (7 items: missing ADAUSDT, GOLD) | Minor gap |
| 3 | Interval dropdown (1d/1h/15m/5m/1m) | Present | Present | Identical options |
| 4 | Days input (lookback period, default 365) | Present | **Missing** | Dash hardcodes 365 days in every data fetch |
| 5 | Data-source selector (Historical/Live/Realtime Stream/FinRL-Yahoo) | Present | **Missing** | Dash always uses Historical mode |
| 6 | Strategy dropdown | Present (5 items: None, MACD/RSI, EMA Crossover, Stochastic, LSTM Predictor, TD3 Strategy) | Partial (4 items: None, MACD/RSI, EMA Crossover, Stochastic — missing LSTM Predictor, TD3 Strategy) | ML strategies not mapped in `_STRATEGY_CLASS_MAP` |
| 7 | Broker dropdown (Simulator/Alpaca/IB/Binance) | Present | **Missing** | Dash has no broker switcher; always uses `SimulatedBroker` |
| 8 | Load button | Present ("Load") | Present ("Load Chart") | Functionally equivalent |
| 9 | Backtest button | Present (top bar) | Present ("Run Backtest" inside Backtest Results card) | Repositioned but functionally equivalent |
| 10 | Simulate button | Present | **Missing** | No simulation mode in Dash |
| 11 | Play / Pause buttons for simulation | Present | **Missing** | No step-through replay |
| 12 | "Go Live" button | Present | **Missing** | No live-trading trigger in Dash |
| 13 | Reset zoom button (↺) | Present | **Missing** | No chart zoom reset control |
| **Left Panel** | | | | |
| 14 | Initial cash input (Parameters group) | Present | Partial | Dash has `bt-cash-input` inside the Backtest Results card only; no standalone "Cash" parameter |
| 15 | Market Fee % input | Present | **Missing** | No fee configuration in Dash |
| 16 | Limit Fee % input | Present | **Missing** | No fee configuration in Dash |
| 17 | Order type (Market/Limit/Stop) | Present | Present | Identical options; conditional price input correctly toggled in both |
| 18 | Quantity input | Present | Present | Equivalent |
| 19 | Limit/Stop price input (conditional) | Present | Present | Both hide/show based on order type |
| 20 | BUY button | Present | Present | Identical styling (green) |
| 21 | SELL button | Present | Present | Identical styling (red) |
| **Chart Panel** | | | | |
| 22 | Plotly candlestick chart | Present (QWebEngineView) | Present (dcc.Graph) | Dash renders natively in browser; no temp-file workaround needed |
| 23 | Technical indicator overlays (MA20/MA50/MA200, EMA, MACD, RSI, Stochastic K/D) | Present (computed in `calculate_technical_indicators`) | Partial (shared `build_candlestick_figure` handles some overlays; `show_ma` flag exists; RSI/MACD sub-panels not exposed to Dash user) | Dash chart has no toggle for individual indicators |
| 24 | Signal markers on chart (buy/sell triangles) | Present | Present | Both use `overlay_signals()` from `core.chart_builder` |
| 25 | Live price badge ("🟢 Live" / "🟡 Near real-time") | Present (status bar message) | Present (dedicated `live-badge` div above chart) | Dash implementation is visually more prominent |
| 26 | Real-time crypto price via WebSocket | Present (`data_loader.start_realtime_stream`) | Present (`LivePriceService` via daemon thread) | Dash uses a scatter-trace Patch approach; PyQt5 rebuilds full OHLCV streaming candles |
| 27 | Realtime Stream candlestick mode | Present (full OHLCV candle streaming, `process_realtime_data`) | **Missing** | Dash ticks price into a single scatter trace, not OHLCV candle stream |
| 28 | Near real-time equity polling | Present | Present | Both throttle REST calls |
| **Right Panel / Metrics** | | | | |
| 29 | Account balance display | Present (dynamic, updates every 5s from broker) | Partial (static "$100,000.00" placeholder; `account-balance` id exists but no callback updates it — Phase 4 not implemented) | **Critical gap** |
| 30 | Account refresh button | Present | **Missing** | No manual refresh trigger |
| 31 | P&L display | Present (dynamic, color-coded green/red) | Partial (static "$0.00"; `pnl-value` id exists but Phase 4 callback not implemented) | **Critical gap** |
| 32 | Backtest Results: Sharpe | Present | Present | Equivalent |
| 33 | Backtest Results: Win Rate | Present | Present | Equivalent |
| 34 | Backtest Results: Max Drawdown | Present | Present | Equivalent |
| 35 | Backtest Results: Alpha | **Missing** (right panel shows only 3 metrics) | **Present** | Dash shows more than PyQt5 here |
| 36 | Backtest Results: Beta | **Missing** (right panel) | **Present** | Dash shows more than PyQt5 here |
| 37 | Backtest status text feedback | Present (status bar) | Present (`bt-status` div in card) | Equivalent |
| 38 | Positions display | Present (QTextEdit, symbol/qty/avg price/PnL text) | Present (styled HTML rows in `positions-content` div) | Dash implementation is visually cleaner |
| **Bottom Tabs** | | | | |
| 39 | Orders / Trade-blotter tab | Present (7 cols: Time/Symbol/Side/Type/Qty/Fill Price/Status; color-coded) | Present (identical 7 columns, identical color rules in DataTable) | Full parity |
| 40 | Orders: Clear button | Present | **Missing** | No clear/reset action in Dash orders tab |
| 41 | Orders: row count / filled count summary | Present (`_orders_status_label`) | Present (`orders-status` div) | Equivalent |
| 42 | PnL Calendar tab (42-cell month grid) | Present | Present | Full parity — same Monday-first layout, same color rules |
| 43 | PnL Calendar: prev/next/today navigation | Present | Present | Equivalent |
| 44 | PnL Calendar: month total | Present | Present | Equivalent |
| 45 | News tab | Present (Time/Headline/Source/Sentiment/Score columns) | Partial (shows headline + source + time as rich HTML rows; **missing sentiment label and confidence score columns**) | Dash news uses `fetch_news_items()` which does not expose sentiment scores |
| 46 | News: manual refresh button | Present | Present | Equivalent |
| 47 | News: auto-refresh timer (every 5 min) | Present (`_news_auto_timer`) | **Missing** | Dash refreshes only on button click or chart load |
| 48 | News: clickable headline links | **Missing** (plain QTableWidgetItem) | **Present** (html.A anchor with target=_blank) | Dash is better here |
| 49 | Earnings Calendar tab | **Missing** | **Present** | New feature in Dash not in PyQt5 |
| 50 | Equity Curve tab | **Missing** (only in StatisticsWindow via matplotlib) | **Present** (dcc.Graph line chart, populated after backtest) | New feature in Dash |
| 51 | Agent Monitor tab (start/stop Supervisor, per-agent table, LLM summary) | **Present** | **Missing** | No agent monitoring in Dash |
| 52 | ⚠ Deps tab (missing optional packages warning) | Present (conditional) | **Missing** | Minor: informational only |
| **Windows / Dialogs** | | | | |
| 53 | StatisticsWindow (separate matplotlib/pyfolio detailed stats window) | Present | **Missing** | No equivalent detailed stats popup in Dash |
| **Behaviors / State** | | | | |
| 54 | Step-through simulation mode (play/pause/250ms timer, portfolio tracking) | Present | **Missing** | Major UX feature absent from Dash |
| 55 | Off-thread data loading (`DataLoadWorker` QThread) | Present | Present (Dash callback runs in server worker thread) | Equivalent in effect |
| 56 | Off-thread news fetch (`NewsWorker` QThread) | Present | Present (callback) | Equivalent |
| 57 | Broker monitoring timer (every 5s) | Present | **Missing** | Dash has no periodic account refresh |
| 58 | Window close / cleanup (os._exit, stop threads) | Present | N/A (web server lifecycle) | Not applicable to web app |
| 59 | Screen-aware window sizing | Present | N/A | Browser handles sizing |
| 60 | Status bar | Present (QStatusBar) | Present (styled Div at bottom) | Equivalent |

---

## Gaps

The following PyQt5 features are Missing or Partial in the Dash implementation. Items marked
**CRITICAL** block the cutover decision; items marked **MEDIUM** are significant UX regressions;
items marked **MINOR** are low-impact and can be deferred post-launch.

### CRITICAL (block sole-entrypoint designation)

1. **Account balance not dynamically updated** — `account-balance` div is a static "$100,000.00"
   placeholder. Phase 4 comment in callbacks.py acknowledges this is unimplemented. Users placing
   orders via Dash have no way to see their running balance change.

2. **P&L not dynamically updated** — `pnl-value` div is a static "$0.00". Same Phase 4 gap as
   above. After a buy/sell order is filled, the P&L display stays at zero.

3. **No broker selection** — Dash is permanently bound to `SimulatedBroker`. There is no dropdown
   or configuration path to route orders to Alpaca, Binance, or Interactive Brokers.

4. **No "Go Live" trigger** — No button or mechanism to start live-trading a real broker session
   with a selected strategy, as `start_trading()` provides in PyQt5.

5. **Simulation mode entirely absent** — Step-through candle-by-candle replay with play/pause
   controls, portfolio state tracking, and signal detection at each step is not in Dash at all.
   This was a prominent feature in PyQt5 (Simulate / ▶ / ⏸ buttons, `simulation_timer`).

### MEDIUM (significant regressions, should be tracked as issues)

6. **Agent Monitor tab missing** — No UI to start/stop the `Supervisor`, see per-agent status
   (ok/warning/error), or view the LLM summary line. Reduces operational observability.

7. **News sentiment label and confidence score missing** — Dash news panel shows headline, source,
   and time but does not surface the sentiment label (positive/negative/neutral) or confidence
   score that PyQt5 displays. `fetch_news_items()` returns `NewsItem` objects without sentiment;
   the PyQt5 path goes through `NewsPipeline.fetch_news_dataframe()` which runs
   `SentimentAnalyzer`.

8. **Days input absent** — All data loads hardcode 365 days. Users cannot adjust the lookback
   window (e.g., 30d for intraday, 730d for longer backtests).

9. **Data-source selector absent** — No option to switch to Live, Realtime Stream, or FinRL-Yahoo
   mode.

10. **OHLCV realtime streaming candles absent** — Dash adds a price-tick scatter point every
    1 500 ms instead of reconstructing live OHLCV candles from a WebSocket feed. The PyQt5
    Realtime Stream mode builds proper candlesticks from the stream.

11. **StatisticsWindow / pyfolio report absent** — No detailed statistics popup in Dash.

### MINOR (low impact, can be deferred post-launch)

12. **Market Fee % / Limit Fee % inputs missing** — Backtest runs with default fee rates; users
    cannot customize.

13. **Missing symbols: ADAUSDT, GOLD** — Dash symbol list has 7 items vs 9 in PyQt5.

14. **Missing strategies: LSTM Predictor, TD3 Strategy** — Not in Dash `_STRATEGY_CLASS_MAP`.

15. **No Reset zoom button** — No ↺ zoom-reset control.

16. **No Clear orders button** — Orders tab cannot be cleared without restarting the server.

17. **News auto-refresh timer absent** — PyQt5 polls every 5 minutes; Dash only refreshes on
    button click or chart load.

18. **No account refresh button** — No ↻ Refresh trigger for account info.

19. **⚠ Deps tab absent** — Missing optional-packages warning panel.

---

## Recommendation

**Do not designate Dash as the sole entrypoint before the following are closed:**

| Priority | Gap | Suggested Sprint Task |
|----------|-----|-----------------------|
| P1 | Account balance + P&L update (Phase 4 callback) | Wire `account-balance` and `pnl-value` to `SimulatedBroker.get_account_info()` on order-status change and a `dcc.Interval` (5 s) |
| P1 | News sentiment scores | Switch `_build_news_content` to use `NewsPipeline.fetch_news_dataframe()` + `SentimentAnalyzer`; surface label + score in the news rows |
| P2 | Broker dropdown + Go Live | Add broker selector to Dash topbar; add a "Go Live" button; route to `BrokerManager` |
| P2 | Agent Monitor tab | Port `_setup_agent_monitor_tab()` to Dash (Supervisor start/stop via Interval + Store) |
| P2 | Simulation mode | Port play/pause step-through to Dash (dcc.Interval-driven, stored state in dcc.Store) |
| P3 | Days input / source selector | Add controls to topbar; wire to `load_chart` callback |
| P3 | LSTM Predictor + TD3 in strategy map | Extend `_STRATEGY_CLASS_MAP` and strategy dropdown |

Once P1 and P2 items are complete, Dash may be re-evaluated as the sole/primary entrypoint.
PyQt5 can then become an optional "offline / no-browser" fallback rather than the canonical entry.

**What Dash already does better than PyQt5** (do not regress these):
- Alpha + Beta metrics in Backtest Results panel (PyQt5 right panel has only 3 metrics)
- Dedicated Equity Curve tab (PyQt5 only shows this via separate matplotlib StatisticsWindow)
- Earnings Calendar table (entirely absent from PyQt5)
- Clickable headline links in news panel
- No temp-file Plotly rendering hack (PyQt5 must write chart to disk for QWebEngineView)
- Browser-accessible without local desktop installation
