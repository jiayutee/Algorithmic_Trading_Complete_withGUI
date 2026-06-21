# Repo cleanup and news-fix plan

This document proposes a safe, focused cleanup pass for the repository and a prioritized plan to diagnose and fix news-source failures.

## Proposed layout (simple, non-breaking)
- code/
  - core/ (existing core logic)
  - ui/ (existing UI)
  - brokers/
  - strategies/
  - scripts/
- docs/
- tests/
- data/ (small test data and input fixtures)
- artifacts/ (large generated artifacts, models, logs)
- experiments/ (training_ground, tmp_ddpg, trained_models)
- tools/ (dev and maintenance scripts)

Notes: This is a logical grouping recommendation — do not move files automatically. Start by moving small, obvious files and updating import paths in a controlled follow-up.

## Main cleanup actions (safe, prioritized)
- Remove or consolidate duplicate example config files (keep `config/settings.py`, archive `*.example` into `docs/` or `config/examples/`).
- Add a top-level `tests/` index and move or link existing `test_*.py` files there.
- Create `data/` for small CSV fixtures used by tests (do not move large training artifacts).
- Move large artifacts (trained_models, results, tmp_ddpg) into `artifacts/` or keep them out of the main repo and document them in `.gitignore`.
- Add a small `CONTRIBUTING.md` or `docs/development.md` describing environment setup and optional dependencies (torch/transformers) and how to run the smoke harness.
- Add a `Makefile` or `scripts/` entries for common checks: `make lint`, `make test`, `make smoke-news`.

## News-source failure diagnosis
Likely causes for intermittent or consistent news-source failures:
- Empty placeholder MCP DuckDuckGo/Brave source present in `news_sources` configuration causing no results when used in the normal pipeline.
- External API/network dependence (rate limiting, site layout changes, HTTP failures) causing scrapers to return empty or malformed results.
- Missing keys / environment configuration for paid/private sources (API keys not provided) leading to silent failures.
- Duplicate or mis-ordered sources: placeholder or low-quality sources placed earlier in the pipeline may suppress downstream healthy sources.

## Prioritized fix plan for news sources
1. Immediate (safe)
   - Remove the placeholder MCP DDG/Brave source from the *normal* pipeline path and keep it only in a `dev`/`experimental` namespace.
   - Add explicit health checks and status metadata to each configured source (eg. `enabled`, `last_success`, `last_error`, `is_placeholder`).
   - Document which sources require API keys and fail fast with clear logs when keys are missing.
2. Short term
   - Implement a source ordering policy: prefer configured reliable sources first, experimental/placeholder sources last.
   - Add retries and basic backoff for network requests and surface HTTP error details in logs.
   - Add a lightweight smoke harness script in `scripts/` that runs each news source locally and reports success/failure counts.
3. Verification
   - Use the smoke harness to verify Brave/DDG behavior locally (headless, user-agent, and proxy if needed).
   - Run the harness in CI or locally before deploying to catch regressions.
4. Medium term
   - Add source-specific parsers with unit tests for common site changes.
   - Consider caching raw HTML and parsing outputs to speed debugging and allow replays.

## How to verify locally (smoke steps)
- Activate the repo virtualenv and install optional dependencies only if needed for model inference.
- Run `python scripts/test_live_search.py` (or the proposed `scripts/smoke_news.py`) to exercise each source and collect a small report.
- For FinBERT/transformers problems, ensure the optional dependencies are documented and that missing stack falls back to rule-based logic.

## Risk notes and next steps
- Do not attempt a large, automated file move in a single change — do small, documented moves and run tests after each.
- Make optional ML stack (torch/transformers) explicit in `requirements-optional.txt` and document runtime implications.
- After the above safe changes, schedule a follow-up to tidy imports and entrypoints to reflect any file moves.


Prepared as a concise plan to be iterated with the team. If you want, I can create the smoke harness script and add the `enabled/health` metadata to `news_sources` next.
