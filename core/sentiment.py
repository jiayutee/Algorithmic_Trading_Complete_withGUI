from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Iterable

import requests

from core.logger import logger

# Optional heavy ML stack. We avoid importing these at module import time
# so the application can start when the optional FinBERT stack isn't installed.
torch = None
AutoModelForSequenceClassification = None
AutoTokenizer = None

_DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
# Kept small deliberately: larger batches (tried up to ~79 headlines in one
# call) made the model occasionally miscount entries in its own response
# (observed: 81 rows for 79 inputs), which used to blow away the whole
# batch's LLM result. Matching by "id" below makes that self-correcting even
# within a chunk, but a smaller chunk also just makes miscounts rarer.
_LLM_CHUNK_SIZE = 20
_LLM_SYSTEM_PROMPT = (
    "You are a financial headline sentiment classifier. Each headline is "
    "numbered. For each one, score how positive, negative, and neutral it is "
    "for the mentioned company/asset's near-term stock price (three floats "
    "summing to 1.0), and give a one-word label (positive/negative/neutral) "
    "with a confidence 0-1. Respond with ONLY a JSON object: "
    '{"results": [{"id": 1, "positive": 0.0, "negative": 0.0, "neutral": 0.0, '
    '"label": "...", "confidence": 0.0}, ...]} -- exactly one entry per '
    "headline, and \"id\" must equal that headline's number from the input."
)


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
        self._deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

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

        if not self.force_rule_based and self._deepseek_api_key:
            return self._analyze_with_llm_batched(texts)

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

    def _analyze_with_llm_batched(self, texts: list[str]) -> list[SentimentResult]:
        """Runs the LLM sentiment path in fixed-size chunks so a single
        chunk's failure (network error, or the model miscounting its own
        response) only degrades that chunk to rule-based, not the whole
        batch."""
        results: list[SentimentResult] = []
        for start in range(0, len(texts), _LLM_CHUNK_SIZE):
            chunk = texts[start:start + _LLM_CHUNK_SIZE]
            chunk_results = self._analyze_with_llm(chunk)
            if chunk_results is None:
                chunk_results = [self._analyze_rule_based(text) for text in chunk]
            results.extend(chunk_results)
        return results

    def _analyze_with_llm(self, texts: list[str]) -> list[SentimentResult] | None:
        """Hosted LLM sentiment path (DeepSeek) for one chunk. Returns None
        only when the request/response itself is unusable (network error,
        unparsable JSON, no "results" list) -- the caller then falls back to
        rule-based for the whole chunk. Individual rows are matched by their
        "id" field rather than by list position, so a model miscount (seen
        in practice: 81 rows returned for 79 inputs) degrades to a per-row
        rule-based fallback for just the missing/duplicate ids instead of
        discarding the entire chunk's real sentiment scores."""
        prompt = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(texts))
        try:
            resp = requests.post(
                _DEEPSEEK_URL,
                headers={
                    "Authorization": f"Bearer {self._deepseek_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.0,
                },
                timeout=20,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            rows = json.loads(content)["results"]
        except Exception as exc:  # pragma: no cover - network/API failures
            logger.warning("DeepSeek sentiment call failed, falling back: %s", exc)
            return None

        if not isinstance(rows, list):
            logger.warning("DeepSeek sentiment response had no 'results' list, falling back.")
            return None

        by_id: dict[int, SentimentResult] = {}
        for row in rows:
            try:
                row_id = int(row["id"])
                if not (1 <= row_id <= len(texts)):
                    continue
                result = SentimentResult(
                    positive=float(row["positive"]),
                    negative=float(row["negative"]),
                    neutral=float(row["neutral"]),
                    label=str(row["label"]).lower(),
                    confidence=float(row["confidence"]),
                    model_name="deepseek-chat",
                )
            except (KeyError, TypeError, ValueError):
                continue
            by_id.setdefault(row_id, result)  # first occurrence wins on duplicate ids

        missing = len(texts) - len(by_id)
        if missing:
            logger.warning(
                "DeepSeek sentiment returned %d/%d matched rows for this chunk; "
                "filling the rest with rule-based sentiment.",
                len(by_id), len(texts),
            )

        return [
            by_id.get(i + 1) or self._analyze_rule_based(text)
            for i, text in enumerate(texts)
        ]

    _NEUTRAL_PRIOR = 2.0     # pseudo-hits of 'neutral' evidence added to every headline

    def _analyze_rule_based(self, text: str) -> SentimentResult:
        words = [token.lower() for token in re.findall(r"[A-Za-z']+", text)]
        positive_hits = sum(1 for word in words if word in self.POSITIVE_WORDS)
        negative_hits = sum(1 for word in words if word in self.NEGATIVE_WORDS)
        total_hits = positive_hits + negative_hits

        # A proper probability distribution: the three scores sum to 1, and a neutral "prior" worth _NEUTRAL_PRIOR keyword
        # hits keeps confidence honest -- one matching word is weak evidence (33%), several agreeing words are stronger.
        # (It used to report 100% confidence from a single word, with scores summing to 1.5.) Still keyword matching, not a
        # calibrated model: read confidence as "how much keyword evidence", not as a probability of being right.
        if total_hits == 0:                       # no keyword evidence at all: lean neutral, but never "certain"
            return SentimentResult(positive=0.1, negative=0.1, neutral=0.8, label="neutral", confidence=0.8,
                                   model_name="rule-based-headline-v1")
        denom = total_hits + self._NEUTRAL_PRIOR
        positive = positive_hits / denom
        negative = negative_hits / denom
        neutral = self._NEUTRAL_PRIOR / denom

        if positive_hits > negative_hits:
            label, confidence = "positive", positive
        elif negative_hits > positive_hits:
            label, confidence = "negative", negative
        else:
            label, confidence = "neutral", neutral

        return SentimentResult(
            positive=float(positive),
            negative=float(negative),
            neutral=float(neutral),
            label=label,
            confidence=float(confidence),
            model_name="rule-based-headline-v1",
        )
