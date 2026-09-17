"""Tests for core/sentiment.py, including the DeepSeek LLM sentiment path
added for Phase 6.4 and a regression guard against re-hardcoding the UI's
sentiment analyzer to the weak rule-based fallback."""

import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.sentiment import SentimentAnalyzer, SentimentResult


def test_rule_based_fallback_still_works():
    analyzer = SentimentAnalyzer(force_rule_based=True)
    result = analyzer.analyze_one("Company beats earnings estimates, stock surges")
    assert result.label == "positive"
    assert result.model_name == "rule-based-headline-v1"


def test_rule_based_neutral_on_no_keyword_hits():
    analyzer = SentimentAnalyzer(force_rule_based=True)
    result = analyzer.analyze_one("Company releases quarterly report")
    assert result.label == "neutral"


def _mock_deepseek_response(rows):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": json.dumps({"results": rows})}}]
    }
    return resp


def test_llm_path_used_when_api_key_present_and_not_forced(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    analyzer = SentimentAnalyzer()
    assert analyzer.force_rule_based is False

    mock_rows = [{"positive": 0.8, "negative": 0.05, "neutral": 0.15, "label": "positive", "confidence": 0.8}]
    with patch("core.sentiment.requests.post", return_value=_mock_deepseek_response(mock_rows)) as mock_post:
        results = analyzer.analyze_many(["Company beats earnings estimates"])

    assert mock_post.called
    assert len(results) == 1
    assert results[0].label == "positive"
    assert results[0].model_name == "deepseek-chat"


def test_llm_failure_falls_back_to_rule_based(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    analyzer = SentimentAnalyzer()

    with patch("core.sentiment.requests.post", side_effect=Exception("network error")):
        results = analyzer.analyze_many(["Company beats earnings estimates, stock surges"])

    # transformers isn't installed in the test environment, so this must land
    # on the rule-based analyzer, not crash and not silently return nothing.
    assert len(results) == 1
    assert results[0].model_name == "rule-based-headline-v1"
    assert results[0].label == "positive"


def test_llm_malformed_response_falls_back(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    analyzer = SentimentAnalyzer()

    # Wrong number of rows for the number of input texts.
    with patch("core.sentiment.requests.post", return_value=_mock_deepseek_response([])):
        results = analyzer.analyze_many(["Headline one", "Headline two"])

    assert len(results) == 2
    assert all(r.model_name == "rule-based-headline-v1" for r in results)


def test_force_rule_based_skips_llm_even_with_api_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    analyzer = SentimentAnalyzer(force_rule_based=True)

    with patch("core.sentiment.requests.post") as mock_post:
        analyzer.analyze_many(["Company beats earnings estimates"])

    mock_post.assert_not_called()


def test_no_api_key_skips_llm_and_falls_back_cleanly(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    analyzer = SentimentAnalyzer()

    with patch("core.sentiment.requests.post") as mock_post:
        results = analyzer.analyze_many(["Company beats earnings estimates"])

    mock_post.assert_not_called()
    assert results[0].model_name == "rule-based-headline-v1"


def test_ui_does_not_hardcode_rule_based_sentiment():
    """Regression guard for the Phase 6.4 bug: ui/main_window.py must not
    override NewsPipeline's sentiment_analyzer with force_rule_based=True,
    which silently disabled the real (FinBERT/LLM) sentiment path regardless
    of what was installed or configured."""
    source = Path(__file__).parent.joinpath("ui", "main_window.py").read_text()
    assert "force_rule_based=True" not in source
