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

    model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
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
                "max_tokens": 400,
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
