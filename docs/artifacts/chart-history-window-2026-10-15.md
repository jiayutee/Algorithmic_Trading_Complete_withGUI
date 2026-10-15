# Minute candles with Days = 365: why they did not load, and the Binance-style fix — 2026-10-15

## What was wrong (measured)
- **Crypto, 1m, 365 days = 525,600 candles.** Binance returns 1,000 per request at about 0.9 s each, fetched one after another:
  **~8 minutes** before anything could be drawn, then a figure of roughly **40 MB** of JSON (a 100,000-candle figure measured 7.9 MB).
  Nothing was cached, so every load, refresh and backtest paid this again.
- **Stocks, any intraday interval:** the request was sent to OpenBB first for a range the provider cannot serve, failed, then fell
  back to Yahoo with a flat 7-day cap for *every* intraday interval, although Yahoo keeps ~60 days of 5-30 minute and ~2 years of
  hourly candles. So 1h silently showed 7 days. About 15 s per load on the rate-limited path.

## How Binance's chart does it, and what we now do
Binance draws only the latest few hundred candles at once and fetches older ones as you scroll left. Same here:

| Piece | Behaviour |
|---|---|
| First paint (crypto) | The latest ~1,500 candles (whole days, so 1m = 2 days / 2,880 bars). Status line says what was shown and that older candles load on scroll. Interval 1d is untouched (365 days = 365 bars). |
| Scroll left (Dash) | When the view reaches the left 15% of the loaded range, older candles are fetched and *prepended with a Patch* (only the candle arrays change; zoom, live tick and overlays untouched). A fast drag far to the left is filled in one go (view + 30% margin, up to 8,000 bars per event). Stops at the Days limit or 20,000 bars, and says why. |
| ◀ Older (desktop) | The desktop chart is a re-rendered HTML page with no JS bridge, so scroll-back is a button: up to 3,000 older candles per click, in a background thread. |
| Disk cache | Closed candles are cached (`training_ground/datasets/klines_cache.sqlite3`, git-ignored) with exact coverage tracking. A range is downloaded once; later requests read the file; a refresh fetches only candles that closed since. The forming candle is always fetched fresh and never stored. |
| Parallel pages | Missing ranges are fetched by 4 threads (stateless REST, honours Retry-After on 429/418). Measured: 20,000 candles in 3.5 s vs 10.2 s sequential; re-read from disk 0.45 s. A full uncached year of 1m candles (backtest): ~1.5 min once, instant after. |
| Stocks | `plan_window` clamps to the source's real limits (1m 7 d, 2-30m 59 d, 1h 729 d) *before* asking OpenBB, with a message when it cut; Yahoo's flat 7-day cap is now per interval (env `YAHOO_INTRADAY_MAX_DAYS` still works as an extra ceiling). |

Code: `core/chart_window.py` (pure rules), `core/kline_cache.py` (cache + fetcher), `DataLoader._get_binance_historical` /
`get_older_candles`, `dash_app/callbacks.py` (`older_candles_patch`), `ui/main_window.py` (`OlderCandlesWorker`, ◀ Older).
The old CCXT loop is kept as the fallback if the cached path fails or the interval is not a Binance REST interval.

## Measured in the running Dash (BTCUSDT, 1m, Days = 365)
- First paint: under 6 s including a cold cache; 2,880 candles; status explains the window.
- Scroll left, one step: +1,000 candles in 0.8-1.1 s (fetch + render). Five steps in a row: 2,880 → 7,880 candles, all in order,
  both traces (candles + live tick) intact. A 3-day jump to the left: 5,241 candles in 0.86 s, view fully covered.
- No server errors.

## Tests
`tests/test_chart_window.py` (17), `tests/test_kline_cache.py` (10), `tests/test_data_loader_chart_window.py` (16),
`tests/test_dash_scrollback.py` (14), `test_gui.py` +4 (66 total, offscreen). The repo-wide offline fixture in `conftest.py` points every
test at a private temp cache and makes the HTTP fetcher fail, so no test touches the real cache or the network.

## Limits
- Dash scroll-back is verified in a browser with simulated drags (`Plotly.relayout`), not a human dragging with a mouse.
- The desktop button is checked offscreen only, not on a real display.
- 20,000 candles is the cap for one chart; zooming far out on 1m still shows candles that are too thin to read. Auto-switching to a
  coarser interval when zoomed out (what exchanges do) is not built.
- Stocks have no scroll-back: the source simply does not keep more intraday history (use 1d for long ranges).
- A backtest on 365 days of 1m crypto still has to download the year once (~1.5 min); it is cached afterwards.
- The always-on paper runner uses the same loader, so after its next restart it also reads daily candles from the cache and
  fetches only the new ones (fewer Binance calls). I did not restart it.
