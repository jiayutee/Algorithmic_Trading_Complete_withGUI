from contextlib import nullcontext
from typing import Any
from .base import BaseRuntimeAgent, AgentResult, RuntimeContext
from . import tools


def _extract_headline(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("headline") or item.get("title") or item.get("summary") or item.get("content") or "")
    return str(getattr(item, "headline", None) or getattr(item, "title", None) or getattr(item, "summary", None) or getattr(item, "content", None) or item or "")


def _normalize_tag(text: str) -> str:
    cleaned = (text or "").strip().upper()
    if "BEAR" in cleaned:
        return "[BEARISH]"
    if "BULL" in cleaned:
        return "[BULLISH]"
    if "NEUT" in cleaned:
        return "[NEUTRAL]"
    if cleaned in {"[BULLISH]", "[BEARISH]", "[NEUTRAL]"}:
        return cleaned
    return "[NEUTRAL]"


def _classify_headlines(shared_llm: Any, headlines: list[str]) -> list[str]:
    if not shared_llm or not headlines:
        return []

    client = getattr(shared_llm, "client", None)
    model = getattr(shared_llm, "model", None)
    lock = getattr(shared_llm, "lock", None)
    if client is None or not model:
        return []

    prompt_lines = [f"{idx + 1}. {headline[:180]}" for idx, headline in enumerate(headlines) if headline]
    if not prompt_lines:
        return []

    messages = [
        {
            "role": "system",
            "content": "You classify financial news. Output only one tag per line from [BULLISH], [BEARISH], [NEUTRAL]. No prose. No punctuation.",
        },
        {
            "role": "user",
            "content": "Classify each headline with exactly one tag per line.\n" + "\n".join(prompt_lines),
        },
    ]

    ctx_manager = lock if lock is not None else nullcontext()
    try:
        with ctx_manager:
            resp = client.chat(model, messages, params={"temperature": 0, "stream": False})
    except Exception:
        return []

    if not isinstance(resp, dict) or not resp.get("ok"):
        return []

    response = resp.get("response")
    raw_text = ""
    if isinstance(response, str):
        raw_text = response
    elif isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            msg = first.get("message") or first
            if isinstance(msg, dict):
                raw_text = str(msg.get("content") or msg.get("text") or "")
            else:
                raw_text = str(msg)
        else:
            raw_text = str(response.get("text") or "")
    else:
        raw_text = str(response)

    tags = [_normalize_tag(line) for line in raw_text.splitlines() if line.strip()]
    while len(tags) < len(prompt_lines):
        tags.append("[NEUTRAL]")
    return tags[: len(prompt_lines)]


class NewsAgent(BaseRuntimeAgent):
    def __init__(self):
        super().__init__("news")

    def run_once(self, ctx: RuntimeContext) -> AgentResult:
        try:
            res = tools.fetch_recent_news(limit=5)
            if not res.get("ok"):
                return AgentResult(self.name, "warning", "news pipeline unavailable", {"raw": res})
            news = res.get("news")
            shared_llm = ctx.tools.get("shared_llm")
            top_headlines = [_extract_headline(item) for item in (news or [])[:3]]
            sentiment_tags = _classify_headlines(shared_llm, top_headlines)

            writer = ctx.tools.get("sql_writer")
            store = tools.store_news(news, writer=writer)
            summary = f"fetched={len(news)} stored_ok={store.get('ok', False)} sentiments={','.join(sentiment_tags) if sentiment_tags else 'n/a'}"
            return AgentResult(self.name, "ok", summary, {"news_count": len(news), "sentiment_tags": sentiment_tags})
        except Exception as e:
            return AgentResult(self.name, "error", f"exception: {e}")
