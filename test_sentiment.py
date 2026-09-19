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

    mock_rows = [{"id": 1, "positive": 0.8, "negative": 0.05, "neutral": 0.15, "label": "positive", "confidence": 0.8}]
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


def test_llm_empty_response_falls_back_for_all(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    analyzer = SentimentAnalyzer()

    with patch("core.sentiment.requests.post", return_value=_mock_deepseek_response([])):
        results = analyzer.analyze_many(["Headline one", "Headline two"])

    assert len(results) == 2
    assert all(r.model_name == "rule-based-headline-v1" for r in results)


def test_llm_partial_miscount_only_falls_back_for_missing_ids(monkeypatch):
    """Regression test for the observed bug: DeepSeek returned 81 rows for
    79 inputs (a miscount), which used to discard the entire batch's real
    sentiment and fall back to rule-based for everything. Matching by id
    should now keep the correctly-matched rows and only backfill the ones
    the model dropped or duplicated."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    analyzer = SentimentAnalyzer()

    # 3 inputs; model returns id=1 twice (a miscount) and never returns id=3.
    mock_rows = [
        {"id": 1, "positive": 0.9, "negative": 0.05, "neutral": 0.05, "label": "positive", "confidence": 0.9},
        {"id": 1, "positive": 0.1, "negative": 0.1, "neutral": 0.8, "label": "neutral", "confidence": 0.8},
        {"id": 2, "positive": 0.05, "negative": 0.9, "neutral": 0.05, "label": "negative", "confidence": 0.9},
    ]
    with patch("core.sentiment.requests.post", return_value=_mock_deepseek_response(mock_rows)):
        results = analyzer.analyze_many(["Beats estimates", "Regulatory probe launched", "Company beats earnings"])

    assert len(results) == 3
    # id=1: first occurrence wins, real LLM result kept.
    assert results[0].model_name == "deepseek-chat" and results[0].label == "positive"
    # id=2: matched correctly.
    assert results[1].model_name == "deepseek-chat" and results[1].label == "negative"
    # id=3: never returned by the model -> falls back to rule-based for just this row.
    assert results[2].model_name == "rule-based-headline-v1"
    assert results[2].label == "positive"  # "beats" is a positive keyword


def test_llm_chunks_large_batches(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    analyzer = SentimentAnalyzer()

    texts = [f"Headline number {i}" for i in range(45)]  # 3 chunks at size 20

    def make_response(*args, **kwargs):
        prompt = kwargs["json"]["messages"][1]["content"]
        n = len(prompt.strip().split("\n"))
        rows = [
            {"id": i + 1, "positive": 0.1, "negative": 0.1, "neutral": 0.8, "label": "neutral", "confidence": 0.8}
            for i in range(n)
        ]
        return _mock_deepseek_response(rows)

    with patch("core.sentiment.requests.post", side_effect=make_response) as mock_post:
        results = analyzer.analyze_many(texts)

    assert len(results) == 45
    assert all(r.model_name == "deepseek-chat" for r in results)
    assert mock_post.call_count == 3  # ceil(45/20)


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


def test_rule_based_scores_are_a_distribution_and_confidence_grows_with_evidence():
    from core.sentiment import SentimentAnalyzer
    a = SentimentAnalyzer(force_rule_based=True)
    pos = sorted(a.POSITIVE_WORDS)[:3]
    neg = sorted(a.NEGATIVE_WORDS)[:1]
    cases = {"none": "the market was open today", "one": f"{pos[0]} noted", "three": " ".join(pos),
             "tie": f"{pos[0]} but {neg[0]}"}
    r = {k: a._analyze_rule_based(t) for k, t in cases.items()}
    for x in r.values():
        assert x.positive + x.negative + x.neutral == pytest.approx(1.0)        # a real distribution (was up to 1.5)
        assert 0.0 <= x.confidence <= 1.0
    assert r["one"].label == r["three"].label == "positive"
    assert r["one"].confidence < r["three"].confidence < 1.0                    # more agreeing keywords, more confidence
    assert r["one"].confidence < 0.5                                            # a single keyword was reported as 100%
    assert r["tie"].label == "neutral"
    assert r["none"].label == "neutral"
