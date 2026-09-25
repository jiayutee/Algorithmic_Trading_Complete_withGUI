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
