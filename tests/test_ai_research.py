from unittest.mock import MagicMock, patch

from core.ai_research import research_event


def _mock_groq_response(payload):
    import json
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": json.dumps(payload)}}]}
    return resp


def test_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    result = research_event(headline="Company beats earnings", summary="", symbol="AAPL",
                             event_category="earnings", sentiment_label="positive")
    assert result is None


def test_returns_none_when_disabled(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("AI_RESEARCH_ENABLED", "false")
    result = research_event(headline="Company beats earnings", summary="", symbol="AAPL",
                             event_category="earnings", sentiment_label="positive")
    assert result is None


def test_returns_none_for_empty_headline(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    result = research_event(headline="", summary="", symbol="AAPL",
                             event_category="unclassified", sentiment_label="unknown")
    assert result is None


def test_successful_call_returns_structured_note(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    payload = {
        "conditional_bias": "bullish",
        "confidence": 0.62,
        "reasoning": "The headline states a confirmed earnings beat for the named company.",
        "contrary_view": "The beat may already be priced in.",
        "corroboration_needed": ["Check guidance", "Check margins"],
    }
    with patch("core.ai_research.requests.post", return_value=_mock_groq_response(payload)) as mock_post:
        result = research_event(headline="Acme Corp beats earnings estimates", summary="Q3 results",
                                 symbol="ACME", event_category="earnings", sentiment_label="positive")
    assert mock_post.called
    assert result["conditional_bias"] == "bullish"
    assert result["confidence"] == 0.62
    assert "confirmed earnings beat" in result["reasoning"]
    assert result["corroboration_needed"] == ["Check guidance", "Check margins"]
    assert result["method"].startswith("ai-research-groq-")
    assert "not independently verified" in result["limitations"].lower()


def test_invalid_bias_falls_back_to_unclear(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    payload = {"conditional_bias": "definitely up", "confidence": 5, "reasoning": "Some reasoning text here."}
    with patch("core.ai_research.requests.post", return_value=_mock_groq_response(payload)):
        result = research_event(headline="Something happened", summary="", symbol="AAPL",
                                 event_category="unclassified", sentiment_label="unknown")
    assert result["conditional_bias"] == "unclear"
    assert result["confidence"] == 1.0  # clamped


def test_missing_reasoning_returns_none(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    payload = {"conditional_bias": "bullish", "confidence": 0.5}
    with patch("core.ai_research.requests.post", return_value=_mock_groq_response(payload)):
        result = research_event(headline="Something happened", summary="", symbol="AAPL",
                                 event_category="unclassified", sentiment_label="unknown")
    assert result is None


def test_network_failure_returns_none(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    with patch("core.ai_research.requests.post", side_effect=Exception("network error")):
        result = research_event(headline="Something happened", summary="", symbol="AAPL",
                                 event_category="unclassified", sentiment_label="unknown")
    assert result is None


def test_malformed_json_returns_none(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": "not json"}}]}
    with patch("core.ai_research.requests.post", return_value=resp):
        result = research_event(headline="Something happened", summary="", symbol="AAPL",
                                 event_category="unclassified", sentiment_label="unknown")
    assert result is None


def test_token_limit_leaves_room_for_a_reasoning_models_hidden_thinking(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    payload = {"conditional_bias": "unclear", "confidence": 0.3, "reasoning": "Too vague to call a direction."}
    with patch("core.ai_research.requests.post", return_value=_mock_groq_response(payload)) as mock_post:
        research_event(headline="Something happened", summary="", symbol="AAPL", event_category="unclassified", sentiment_label="unknown")
    assert mock_post.call_args.kwargs["json"]["max_tokens"] >= 1000
    assert mock_post.call_args.kwargs["json"]["model"] == "openai/gpt-oss-120b"


# ------------------------------------------------------------------ batch model readings
import json as _json

from core import ai_research as ar


def _batch_response(rows):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": _json.dumps({"results": rows})}}]}
    return resp


def _items(n):
    return [{"id": f"e{i}", "headline": f"Headline number {i}", "summary": ""} for i in range(n)]


def test_read_events_needs_a_key_and_respects_the_off_switch(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert ar.read_events(_items(2), "BTCUSDT") == {}
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("AI_RESEARCH_ENABLED", "false")
    with patch("core.ai_research.requests.post") as post:
        assert ar.read_events(_items(2), "BTCUSDT") == {}
        post.assert_not_called()


def test_read_events_maps_rows_by_id_clamps_and_drops_bad_rows(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    ar._reading_cache.clear()
    rows = [{"id": 1, "bias": "bullish", "confidence": 0.9, "reason": "ETF inflows"},
            {"id": 2, "bias": "sideways", "confidence": 0.9, "reason": "invalid bias"},       # dropped
            {"id": 3, "bias": "bearish", "confidence": 7, "reason": "clamped"},
            {"id": 3, "bias": "bullish", "confidence": 0.5, "reason": "duplicate id ignored"},
            {"id": 99, "bias": "bullish", "confidence": 0.5, "reason": "out of range"},
            {"id": "x", "bias": "bullish", "confidence": 0.5, "reason": "bad id"}]
    with patch("core.ai_research.requests.post", return_value=_batch_response(rows)):
        out = ar.read_events(_items(3), "BTCUSDT")
    assert set(out) == {"e0", "e2"}
    assert out["e0"]["bias"] == "bullish" and out["e2"]["bias"] == "bearish" and out["e2"]["confidence"] == 1.0
    assert out["e0"]["method"].startswith("model-reading-groq-")


def test_read_events_chunks_caches_and_survives_a_failed_chunk(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    ar._reading_cache.clear()
    calls = []

    def fake_post(url, **kw):
        calls.append(kw["json"]["messages"][1]["content"])
        if len(calls) == 1:
            raise RuntimeError("network down")                       # first chunk fails
        return _batch_response([{"id": i, "bias": "bullish", "confidence": 0.8, "reason": "r"} for i in range(1, 11)])

    with patch("core.ai_research.requests.post", side_effect=fake_post):
        out = ar.read_events(_items(15), "BTCUSDT")                  # 2 chunks: 10 + 5
    assert len(calls) == 2 and set(out) == {f"e{i}" for i in range(10, 15)}      # chunk 1 lost, chunk 2 kept
    with patch("core.ai_research.requests.post") as post:
        again = ar.read_events(_items(15)[10:], "BTCUSDT")                    # served from the cache
        post.assert_not_called()
    assert set(again) == set(out)


def test_read_events_stops_on_a_rate_limit(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    ar._reading_cache.clear()
    limited = MagicMock(status_code=429)
    with patch("core.ai_research.requests.post", return_value=limited) as post:
        assert ar.read_events(_items(25), "BTCUSDT") == {}
    assert post.call_count == 1                                       # did not hammer the free tier
