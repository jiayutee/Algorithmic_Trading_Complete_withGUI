# News provider failure classification — evidence (2026-09-20)

Branch `orchestrator/day85-news-failure-visibility`. Status: **implemented and tested locally; PR open, not merged, not deployed.**

## What changed
Every network news adapter now returns a classified outcome next to its items (`BaseNewsSource.fetch_classified()`), and the
health registry / `NewsPipeline.source_status()` / `scripts/smoke_news.py` report it as the source's `status`:

| status | meaning | counts as failure (circuit breaker) |
|---|---|---|
| `ok` | items delivered | no |
| `ok_empty` | provider answered normally, no matches | no |
| `rate_limited` | HTTP 429 | yes |
| `auth_failed` | HTTP 401/403, or key missing/rejected | yes |
| `parse_error` | payload could not be parsed | yes |
| `timeout` | request timeout, or the pipeline's shared fetch deadline expired | yes |
| `error` | any other failure (connection error, ...) | yes |

`fetch()` keeps its old return type (a list); third-party adapters that only define `fetch()` are wrapped by a default
`fetch_classified()`. Diagnostics carry only the class (and an HTTP status at most): no credentials, request URLs, article bodies
or raw exception text. A pre-existing `AttributeError` in `EventRegistrySource` (`articles.results` present but a list) was fixed.
Smoke report: `ok_empty` counts as healthy, every failure class marks the probe `degraded`.

## Tests
- New `tests/test_news_failure_classification.py` (offline mocks only, no network, no store writes): each class for each adapter.
- Two existing assertions changed to the new vocabulary: `empty` -> `ok_empty` (`test_news_diagnostics.py`); a raw "429" reason string
  -> `last_status == "error"` (`test_news_hardening.py`, the raw text is no longer stored).
- Full suite: see the PR description for the final count and latest-head CI.

## Live probe (2026-09-20, ~12 s) — [artifact](news-source-smoke-2026-09-20.json)
Run once against the real configured sources (BTCUSDT, AAPL; no sentiment scoring, no store writes; base 215e62b, dirty tree).
BTCUSDT: `duckduckgo` ok (5), `openbb_news` ok (5); `brave`, `rss`, `gdelt` = `timeout` at the 6 s shared budget.
AAPL: `duckduckgo` ok (5), `openbb_news` ok (5); `brave`, `rss`, `gdelt` = `cooldown` (circuit opened by the first probe).
NewsAPI and EventRegistry are unconfigured. Both probes `degraded`, exit 0.

## Limits — read before relying on this
- **The probe still cannot say WHY brave/rss/gdelt are slow.** When the shared 6 s deadline expires first, the pipeline records
  `timeout` and abandons the worker, so the adapter's own classification (429 vs auth vs slow) is discarded. Answering that needs a
  longer/per-source budget in an experiment, not more diagnostics code. Not done.
- `OpenBBNewsSource` classification is best-effort from exception message text (the library hides the HTTP status).
- `DuckDuckGoSource`: a CAPTCHA/interstitial page parses leniently and is reported `ok_empty`, not `parse_error`.
- DuckDuckGo delivered items today although the 2026-09-19 probe reported none: provider behaviour is intermittent, one probe proves nothing.
- Nothing here validates news relevance, freshness, or sentiment; no tradable news signal is established. Plan items 2 and 3 remain open.
