"""Optional AI-generated research note for a single news event.

Best-effort only: this augments the deterministic reading in
`core.news_interpretation` with a free hosted LLM (Groq). It never blocks or
replaces the deterministic path -- any missing key, network error, timeout,
or malformed response returns ``None`` and the caller falls back to the
rule-based interpretation alone.

The model is told to reason ONLY from the supplied headline/summary text and
the existing sentiment-pipeline label (see core/sentiment.py), and to say
"unclear" rather than invent facts not present in that text. Output is a
hypothesis for the reader to check, never a forecast, trading signal, or
verified causal claim -- consistent with core/news_interpretation.py's
`conditional_bias_basis` / `limitations` fields.
"""
from __future__ import annotations

import json
import os
from typing import Any

import requests

from core.logger import logger

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_TIMEOUT_S = 12
# gpt-oss models spend part of this budget on hidden reasoning; at 400 tokens 5 of 8 live calls failed with
# HTTP 400 'max completion tokens reached before generating a valid document', at 1500 all 6 succeeded.
_MAX_TOKENS = 1500
_ALLOWED_BIAS = {"bullish", "bearish", "mixed", "unclear"}

_SYSTEM_PROMPT = (
    "You are a cautious financial-news research assistant. You are given one "
    "headline/summary, the asset it is being read against, an existing "
    "deterministic event category, and an existing sentiment-pipeline label "
    "(which may be wrong -- treat it as one input, not ground truth). "
    "Reason ONLY from the supplied text. Do not assume facts, dates, prices, "
    "or outcomes that are not stated. If the text is too vague, macro, or "
    "indirect to support a directional read for the named asset, say so -- "
    "do not force a bullish/bearish call. Respond with ONLY a JSON object: "
    '{"conditional_bias": "bullish|bearish|mixed|unclear", '
    '"confidence": 0.0-1.0, "reasoning": "2-3 sentences, cites only the '
    'supplied text", "contrary_view": "1-2 sentences on how this could be '
    'wrong", "corroboration_needed": ["short items to verify before acting"]}'
)


def _enabled() -> bool:
    if os.getenv("AI_RESEARCH_ENABLED", "").strip().lower() in ("0", "false", "no"):
        return False
    return bool(os.getenv("GROQ_API_KEY"))


def research_event(
    *,
    headline: str,
    summary: str,
    symbol: str,
    event_category: str,
    sentiment_label: str,
    api_key: str | None = None,
    model: str | None = None,
) -> dict[str, Any] | None:
    """Return a best-effort AI research note, or None if unavailable/unusable.

    Never raises: any failure (missing key, network, timeout, malformed
    response) is logged and swallowed so the caller can fall back to the
    deterministic interpretation alone.
    """
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    if os.getenv("AI_RESEARCH_ENABLED", "").strip().lower() in ("0", "false", "no"):
        return None

    model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")  # llama-3.3-70b-versatile retired by Groq (HTTP 404, 2026-09-23)
    headline = (headline or "").strip()
    summary = (summary or "").strip()
    if not headline:
        return None

    user_content = (
        f"Asset: {symbol}\n"
        f"Deterministic event category (rule-based, may be 'unclassified'): {event_category}\n"
        f"Sentiment-pipeline label (existing model, treat as one signal): {sentiment_label}\n"
        f"Headline: {headline}\n"
        f"Summary: {summary[:800] if summary else '(none supplied)'}"
    )

    try:
        resp = requests.post(
            _GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.0,
                "max_tokens": _MAX_TOKENS,
            },
            timeout=_TIMEOUT_S,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        row = json.loads(content)
    except Exception as exc:  # pragma: no cover - network/API failures
        logger.warning("Groq AI research call failed, skipping: %s", exc)
        return None

    if not isinstance(row, dict):
        return None

    bias = str(row.get("conditional_bias", "unclear")).strip().lower()
    if bias not in _ALLOWED_BIAS:
        bias = "unclear"
    try:
        confidence = float(row.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    reasoning = str(row.get("reasoning") or "").strip()
    if not reasoning:
        return None
    contrary_view = str(row.get("contrary_view") or "").strip()
    corroboration = row.get("corroboration_needed")
    if not isinstance(corroboration, list):
        corroboration = []
    corroboration = [str(item).strip() for item in corroboration if str(item).strip()][:5]

    return {
        "method": f"ai-research-groq-{model}",
        "conditional_bias": bias,
        "confidence": confidence,
        "reasoning": reasoning,
        "contrary_view": contrary_view or "Not supplied by the model.",
        "corroboration_needed": corroboration,
        "limitations": (
            "Hosted LLM hypothesis generated from the supplied headline/summary text only. "
            "Not independently verified, not a forecast, not a trading signal. "
            "May hallucinate despite instructions; corroborate before acting."
        ),
    }


# --------------------------------------------------------------------- batch readings for the chart
# One short label per headline so Market Context can colour events the rule-based reading leaves "unknown". A labelling
# aid, not a forecast: nothing here has been shown to predict price (Phase 13.1 found the rule labels untestable; these
# have not been tested at all).
import hashlib
import time

READING_CHUNK = 10                 # headlines per request: keeps each answer short and the free-tier token use low
READING_MAX_TOKENS = 4000          # reasoning models spend part of this on hidden thinking (see _MAX_TOKENS)
READING_TTL_S = 6 * 3600           # the same headline is not re-read on every refresh
READING_MIN_CONFIDENCE = 0.6       # a model reading colours an event only at or above this; fixed in advance, not tuned
_reading_cache: dict = {}

_READING_PROMPT = (
    "You label news headlines for a market-context chart of ONE asset. For each numbered headline decide what the "
    "reported event implies for that asset's price over the next few days IF the report is true, using only the text "
    "shown. Answer bullish, bearish, mixed or unclear. Rules: (1) A headline that only reports that price already moved "
    "(surges, drops, rallies, slides) describes the past, not a reason: answer unclear unless it also states a new cause "
    "with a clear implication. (2) Macro news or news about another asset is unclear unless the text links it to this "
    "asset. (3) Do not assume facts that are not in the text. (4) unclear is the right answer whenever the direction is "
    "not established. Respond with ONLY a JSON object: "
    '{"results": [{"id": 1, "bias": "bullish|bearish|mixed|unclear", "confidence": 0.0-1.0, "reason": "max 20 words"}]} '
    "with exactly one entry per headline; id must equal the headline's number."
)


def readings_enabled() -> bool:
    return _enabled()


def _reading_key(symbol: str, headline: str) -> tuple:
    return (symbol, hashlib.sha1(" ".join((headline or "").lower().split()).encode()).hexdigest())


def _parse_reading(row: dict, model: str) -> dict | None:
    bias = str(row.get("bias", "")).strip().lower()
    reason = str(row.get("reason") or "").strip()
    if bias not in _ALLOWED_BIAS or not reason:
        return None
    try:
        confidence = float(row.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {"bias": bias, "confidence": max(0.0, min(1.0, confidence)), "reason": reason[:240],
            "method": f"model-reading-groq-{model}"}


def read_events(items: list, symbol: str, *, api_key: str | None = None, model: str | None = None,
                deadline_s: float = 30.0) -> dict:
    """Model readings for ``items`` ([{'id','headline','summary','category'}]) -> {id: reading}; never raises.

    Missing key / disabled flag / failures give fewer (or no) entries; the caller keeps the rule-based reading for those.
    A rate limit (HTTP 429) stops further requests for this call instead of hammering the free tier."""
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if not api_key or os.getenv("AI_RESEARCH_ENABLED", "").strip().lower() in ("0", "false", "no"):
        return {}
    model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    now, started = time.time(), time.monotonic()
    out, todo = {}, []
    for it in items:
        cached = _reading_cache.get((model,) + _reading_key(symbol, it["headline"]))
        if cached and now - cached[0] < READING_TTL_S:
            out[it["id"]] = cached[1]
        elif (it.get("headline") or "").strip():
            todo.append(it)
    for start in range(0, len(todo), READING_CHUNK):
        if time.monotonic() - started > deadline_s:
            break
        chunk = todo[start:start + READING_CHUNK]
        lines = "\n".join(
            f"{i + 1}. {it['headline'].strip()}" + (f" -- {it['summary'].strip()[:200]}" if (it.get("summary") or "").strip() else "")
            for i, it in enumerate(chunk))
        try:
            resp = requests.post(
                _GROQ_URL, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "temperature": 0.0, "max_tokens": READING_MAX_TOKENS,
                      "response_format": {"type": "json_object"},
                      "messages": [{"role": "system", "content": _READING_PROMPT},
                                   {"role": "user", "content": f"Asset: {symbol}\n\nHeadlines:\n{lines}"}]},
                timeout=max(5.0, min(25.0, deadline_s - (time.monotonic() - started))))
            if resp.status_code == 429:
                logger.warning("Groq rate limit hit during model readings; skipping the rest")
                break
            resp.raise_for_status()
            rows = json.loads(resp.json()["choices"][0]["message"]["content"])["results"]
        except Exception as exc:  # noqa: BLE001 -- one bad chunk must not lose the rest
            logger.warning("Groq model-reading call failed, skipping this chunk: %s", exc)
            continue
        if not isinstance(rows, list):
            continue
        seen = set()
        for row in rows:
            try:
                idx = int(row["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if not 1 <= idx <= len(chunk) or idx in seen or not isinstance(row, dict):
                continue
            seen.add(idx)
            reading = _parse_reading(row, model)
            if reading:
                item = chunk[idx - 1]
                out[item["id"]] = reading
                _reading_cache[(model,) + _reading_key(symbol, item["headline"])] = (now, reading)
    return out
