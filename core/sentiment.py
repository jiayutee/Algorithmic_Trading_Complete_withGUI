from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Iterable

from core.logger import logger

# Optional heavy ML stack. We avoid importing these at module import time
# so the application can start when the optional FinBERT stack isn't installed.
torch = None
AutoModelForSequenceClassification = None
AutoTokenizer = None


@dataclass
class SentimentResult:
    positive: float
    negative: float
    neutral: float
    label: str
    confidence: float
    model_name: str


class SentimentAnalyzer:
    """Headline sentiment analyzer with FinBERT when available and a rule-based fallback."""

    POSITIVE_WORDS = {
        "beat",
        "beats",
        "bullish",
        "growth",
        "improve",
        "improves",
        "launch",
        "outperform",
        "profit",
        "raise",
        "raises",
        "record",
        "strong",
        "surge",
        "upgrade",
    }
    NEGATIVE_WORDS = {
        "bearish",
        "cut",
        "cuts",
        "decline",
        "downgrade",
        "fall",
        "lawsuit",
        "loss",
        "probe",
        "risk",
        "slump",
        "weak",
        "warn",
        "warning",
        "miss",
        "misses",
    }

    def __init__(self, model_name: str | None = None, force_rule_based: bool = False):
        self.model_name = model_name or os.getenv("NEWS_SENTIMENT_MODEL", "ProsusAI/finbert")
        self.force_rule_based = force_rule_based
        self._tokenizer = None
        self._model = None
        self._loaded_model_name = None

    def _load_model(self) -> bool:
        if self.force_rule_based:
            return False

        # Try to import the optional heavy dependencies lazily. If they're
        # not available, fall back cleanly to the rule-based analyzer.
        try:
            # local imports to avoid failing at app startup
            import importlib

            torch_mod = importlib.import_module("torch")
            transformers_mod = importlib.import_module("transformers")
            AutoTokenizer_local = getattr(transformers_mod, "AutoTokenizer")
            AutoModel_local = getattr(transformers_mod, "AutoModelForSequenceClassification")
        except Exception:
            logger.warning("Optional model stack not installed; using rule-based sentiment fallback.")
            return False

        if self._model is not None and self._loaded_model_name == self.model_name:
            return True

        try:
            self._tokenizer = AutoTokenizer_local.from_pretrained(self.model_name)
            self._model = AutoModel_local.from_pretrained(self.model_name)
            self._torch = torch_mod
            self._loaded_model_name = self.model_name
            logger.info("Loaded sentiment model %s", self.model_name)
            return True
        except Exception as exc:  # pragma: no cover - network/model download failures
            logger.warning("Could not load sentiment model %s: %s", self.model_name, exc)
            self._tokenizer = None
            self._model = None
            self._torch = None
            self._loaded_model_name = None
            return False

    def analyze_many(self, texts: Iterable[str]) -> list[SentimentResult]:
        texts = [text or "" for text in texts]
        if not texts:
            return []

        if self._load_model():
            return self._analyze_with_model(texts)
        return [self._analyze_rule_based(text) for text in texts]

    def analyze_one(self, text: str) -> SentimentResult:
        results = self.analyze_many([text])
        return results[0] if results else self._analyze_rule_based(text)

    def _analyze_with_model(self, texts: list[str]) -> list[SentimentResult]:
        assert self._tokenizer is not None
        assert self._model is not None
        assert getattr(self, "_torch", None) is not None

        torch_mod = self._torch
        inputs = self._tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch_mod.no_grad():
            outputs = self._model(**inputs)

        probabilities = torch_mod.nn.functional.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
        id2label = {int(key): value.lower() for key, value in getattr(self._model.config, "id2label", {}).items()}
        default_order = ["negative", "neutral", "positive"]

        results: list[SentimentResult] = []
        for row in probabilities:
            values = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            for index, score in enumerate(row):
                label = id2label.get(index, default_order[index] if index < len(default_order) else f"label_{index}")
                if label not in values:
                    continue
                values[label] = float(score)

            label = max(values, key=values.get)
            confidence = values[label]
            results.append(
                SentimentResult(
                    positive=values["positive"],
                    negative=values["negative"],
                    neutral=values["neutral"],
                    label=label,
                    confidence=confidence,
                    model_name=self._loaded_model_name or self.model_name,
                )
            )
        return results

    def _analyze_rule_based(self, text: str) -> SentimentResult:
        words = [token.lower() for token in re.findall(r"[A-Za-z']+", text)]
        positive_hits = sum(1 for word in words if word in self.POSITIVE_WORDS)
        negative_hits = sum(1 for word in words if word in self.NEGATIVE_WORDS)
        total_hits = positive_hits + negative_hits

        if total_hits == 0:
            return SentimentResult(
                positive=0.1,
                negative=0.1,
                neutral=0.8,
                label="neutral",
                confidence=0.8,
                model_name="rule-based-headline-v1",
            )

        positive = positive_hits / total_hits
        negative = negative_hits / total_hits
        neutral = max(0.0, 1.0 - (positive + negative) / 2.0)

        if positive > negative:
            label = "positive"
            confidence = positive
        elif negative > positive:
            label = "negative"
            confidence = negative
        else:
            label = "neutral"
            confidence = neutral

        return SentimentResult(
            positive=float(positive),
            negative=float(negative),
            neutral=float(neutral),
            label=label,
            confidence=float(confidence),
            model_name="rule-based-headline-v1",
        )
