# AI research on Market Context — implementation evidence, 2026-09-22

Base: `215e62b` (main) + `codex/news-event-timeline` (PR #21); branch `feat/ai-research-groq`.

## What is implemented
`core/ai_research.py`: given one event's headline/summary, the selected symbol, the deterministic
event category from `core/news_interpretation.py`, and the existing sentiment-pipeline label from
`core/sentiment.py`, calls Groq's free-tier hosted-LLM chat API and returns a structured research
note (conditional bias, confidence, reasoning, contrary view, corroboration checklist) or `None`.
The model is instructed to reason only from the supplied text and to say "unclear" rather than
invent facts; the function itself never raises -- a missing `GROQ_API_KEY`, `AI_RESEARCH_ENABLED=false`,
network error, timeout (12s) or malformed JSON all return `None`.

Dash wiring (`dash_app/news_context.py`): a "Get AI research on selected event" button fetches the
note for the currently selected event only, on click -- not automatically on every refresh or
selection change, so cost/latency is bounded to what the reader opens. The result is stored keyed
by event id + symbol so switching events or symbols does not show stale AI output. A clear fallback
message is shown when unavailable, and the deterministic interpretation card above it is unaffected
either way (see screenshot: `Physician Leaders – Home | AAPL`, "AI research unavailable... Deterministic
interpretation above is unaffected").

`core/sentiment.py` was reviewed before building on it: FinBERT (optional) -> DeepSeek LLM (opt-in,
chunked, per-row fallback on partial miscounts) -> keyword rule-based fallback, with tests for each
path. Judged solid enough to feed its label into the Groq prompt as one input, not ground truth.

## Validation
- `tests/test_ai_research.py` (new, 8 tests): no key, disabled flag, empty headline, successful
  parse, invalid-bias clamping, confidence clamping, missing-reasoning rejection, network failure,
  malformed JSON -- all mocked, no live network calls in CI.
- `tests/test_news_context.py` (+4 tests): `ai_research_for_event` argument passthrough, event-card
  placeholder and populated rendering.
- Focused: `pytest tests/test_ai_research.py tests/test_news_context.py tests/test_news_interpretation.py -q`
  -- 54 passed.
- Full: `pytest --ignore=test_gui.py -q` -- see PR CI status (recorded on GitHub).
- Browser verified on an isolated local Dash instance (port 8077, separate from any already-running
  instance) with a real AAPL chart load and live news fetch: clicking "Get AI research on selected
  event" with no `GROQ_API_KEY` configured shows the intended unavailable message, both above the
  event list and inside the selected card, and the deterministic reading is untouched. The success
  path (parsed Groq response) is covered by the mocked unit tests above, since no `GROQ_API_KEY` is
  configured in this environment's `.env` yet.

## Remaining scope / limits
No evaluation of the AI note's own accuracy or calibration was done or is claimed -- it is presented
identically to the deterministic path, as a hypothesis to verify, not a forecast or trading signal.
`GROQ_API_KEY` is not set in `.env`; the feature is inert (falls back silently) until the owner adds
a free key from console.groq.com. Desktop (PyQt5) integration is out of scope, matching PR #21's
Dash-first scope. No order path, persistence, or background scheduling was added.
