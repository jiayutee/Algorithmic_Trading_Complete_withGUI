# AI research (Groq) live activation smoke — 2026-09-23

Unattended orchestrator work-loop, Day 88 (23:20 firing). Base: main `d00c66b`. Base Python env 3.9.
The key was read from `.env` into the process environment only; it is not written to this file, the logs, or git.

## What was run
One real event from the local news store (read-only): id 22914, openbb:TheStreet, 2026-09-21T23:47Z,
"Bitcoin surges to $87,000 on Yom Kippur", read against BTCUSDT. It went through the real Market Context path:
`core.news_context.build_snapshot` → `ai_research_for_event` → `core.ai_research.research_event` → `dash_app.news_context.event_card`.
Only `requests.post` was wrapped to record HTTP status and latency. No sentiment calls, no store writes.

## Results
| Call | Model | HTTP | Latency | Outcome |
|---|---|---|---|---|
| 1 (as configured) | `llama-3.3-70b-versatile` (code default; no `GROQ_MODEL` in .env) | 404 | 0.20 s | **Failed.** `research_event` returned `None`; one warning logged; deterministic card still rendered. |
| diagnostic | GET `/openai/v1/models` | 200 | — | Key valid. 11 models served; `llama-3.3-70b-versatile` **not among them** (retired). Chat-capable: `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`. |
| 2 (one verification) | `openai/gpt-oss-120b` via `GROQ_MODEL` | 200 | 1.92 s | **Succeeded.** Valid JSON note, AI block and deterministic card both rendered. |

No retries beyond this single verification call.

Returned note (call 2):
- conditional_bias: **bullish**, confidence 0.80
- reasoning: "The headline states "Bitcoin surges to $87,000" and the summary notes it "surged toward $87,000," indicating a clear upward price movement for BTCUSDT. This positive price action suggests a bullish bias for the asset."
- contrary_view: "The surge could be short-lived or driven by hype, and price may reverse quickly, undermining the bullish implication."
- corroboration_needed: current BTCUSDT price and recent trend; trading volume during the surge; broader market sentiment and related news.

Deterministic reading shown beside it (unchanged): event_category `unclassified`, relevance `direct`, conditional_bias `unknown`, method `deterministic-rules-v1`.

## Code change in this PR
`core/ai_research.py`: default model `llama-3.3-70b-versatile` → `openai/gpt-oss-120b` (the only change; `GROQ_MODEL` still overrides). CLAUDE.md env note updated to match.
Without this change the feature is keyed but still inert: every click silently returns nothing.

## Observations and limits
- **Quality caveat:** the note calls a report of a *past* price move "bullish" at 0.80 confidence. That describes what already happened, not a conditional case for what comes next. The deterministic path says `unknown` here. One sample is not an evaluation. The prompt may need to separate "reports a past move" from "gives a reason for a future move". Not changed tonight.
- Failure diagnosis was only possible out-of-band: `research_event` logs `404 Client Error` without Groq's response body (the body would name the missing model). If the model gets retired again, the only visible symptom is that no AI block appears.
- Accuracy/calibration of AI notes remains unevaluated. Free-tier rate limits were not exercised. The Dash button was not clicked in a browser; rendering was checked through `event_card`'s component tree.
- Status: implemented and live-smoked on a task branch; **not merged, not deployed.** The running checkout still has the old default until the owner merges.
