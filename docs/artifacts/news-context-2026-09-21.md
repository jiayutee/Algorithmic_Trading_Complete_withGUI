# Market Context — implementation evidence, 2026-09-21

Source: owner-provided MEXC screenshot; base `215e62b`; branch `codex/news-event-timeline`.
Notion: https://app.notion.com/p/3e2d2ab050d98108bc0ddaa22221c02d

## What is implemented
Dash → Market Context. Load candles with the existing controls, open the tab and click Refresh context. The panel snapshots those candles, fetches news under the existing source deadline, and shows publication markers, event selection, filters and a highlighted observation window. Selecting an event reports a descriptive change from a prior completed candle close to the last loaded close, with candle-open timestamps and a forming-candle caveat. This is not an event study, execution return or causal attribution. Candle duration is inferred from median spacing; sparse/irregular data may make the comparison approximate.

Expandable cards show source text/link, event category, asset relevance, possible mechanism, contrary factors and checks to watch. Conditional bullish/bearish cases are supported only for narrow explicit direct-asset completed-event wording (security incidents and earnings/guidance); uncertain, indirect, negated, rumored and future claims remain unclear. Actual asset impact remains unknown. Rules inspect supplied headlines/summaries; they do not verify the source or generate a calibrated forecast.

The new context fetch reuses source routing/deadline/prefilter and avoids sentiment inference and news-store writes. Refresh is explicit and disabled while running. Source health is visible. Duplicate headlines, future and invalid publication dates are excluded. No fixture data is used in production.

## Validation
- Interpretation tests: 35 passed, including negation, rumor, indirect relevance and future timestamps.
- Combined new core/view tests plus existing Dash tests: 238 passed.
- Browser verification on isolated localhost with explicitly labeled synthetic ETH candles/news: refresh loads markers/cards, bearish filter selects the security story, selection highlights the chart, expanded interpretation displays evidence and limitations.
- Real chart-builder JSON regression covers Plotly 6 base64-encoded OHLC arrays; browser verified candles and marker clicks with that serialization.
- Final local suite: 1,326 passed, 1 skipped, 14 warnings in 200.25 seconds. PR CI status is recorded on GitHub and Notion.

## Remaining scope
Desktop integration, a connected upcoming macro calendar, grounded LLM interpretations, provider coverage repair and calibrated impact evaluation remain separate. Scheduled earnings remain in News & Earnings. No predictive price arrow, paid subscription, order path or background service configuration was added. PR16's provider-failure changes are independent.

The repository notebook source is updated; the separately published claude.ai artifact is not automatically republished.
