"""Local, evidence-linked reading aids, not forecasts or verified causal analysis.

Only supplied headline/summary text is inspected. No model, network, persistence,
order path, or instruction execution is involved. Direction is deliberately left
unknown for macro news, ambiguous wording, and unestablished asset relevance.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from core.instruments import GROUPS
from core.news_sources import NewsItem

METHOD = "deterministic-rules-v1"

# Category recognition is a reading aid; it does not establish that an event occurred.
_RULES = (
    ("security", r"\b(hack(?:ed|ing)?|exploit(?:ed)?|breach|stolen|theft)\b",
     "If the reported security incident affects this asset or its infrastructure, losses or reduced trust could pressure demand.",
     "The incident may concern another entity, be contained, or already be reflected in price.",
     ["Official incident confirmation and affected assets", "Loss size, containment and recovery"]),
    ("monetary_policy", r"\b(federal reserve|fed|central bank|interest rates?|rate cuts?|rate hikes?|fomc)\b",
     "If policy differs from market expectations, changes in funding costs and liquidity could alter risk appetite; the direction for this asset is not established here.",
     "An expected decision may have little effect, while guidance or growth concerns can offset the headline.",
     ["Official decision and guidance", "Difference from consensus and subsequent market reaction"]),
    ("macro_data", r"\b(inflation|cpi|jobs report|payrolls|retail sales|gdp|unemployment)\b",
     "An economic-data surprise could change growth or policy expectations. A rise in the reported number alone does not establish bullish or bearish asset impact.",
     "Revisions, consensus expectations and the policy response can reverse a simple headline reading.",
     ["Release versus consensus and prior revisions", "Rates, currency and asset reaction"]),
    ("regulation", r"\b(regulat\w*|legislation|bill|sec|lawsuit|approval|ban(?:ned)?|reserve)\b",
     "If adopted and applicable, the reported policy could change market access, compliance costs or demand. A proposal is not an implemented rule.",
     "Implementation, legal challenges and scope may differ from the headline; the news may concern a different asset.",
     ["Primary legal text and current procedural stage", "Affected assets, effective date and implementation"]),
    ("earnings", r"\b(earnings|revenue|profit|guidance|quarterly results)\b",
     "If results or guidance change expected cash flows for the named company, valuation could adjust; comparison with expectations matters more than headline tone.",
     "Margins, one-off items, guidance or an already-priced expectation can outweigh a headline beat or miss.",
     ["Reported results versus consensus", "Margins, cash flow and forward guidance"]),
    ("network_activity", r"\b(upgrade|mainnet|fork|staking|network activity|outage)\b",
     "If the reported network change affects this asset, reliability, usage or token supply could change. Price impact depends on adoption and token economics.",
     "Technical activity need not create token demand, and implementation may be delayed.",
     ["Official network status and deployment", "Usage, fees and token-supply implications"]),
    ("market_flows", r"\b(etf|inflows?|outflows?|liquidations?|institutional|treasury)\b",
     "If flows are confirmed and material relative to liquidity, buying or selling pressure could change. Headline flow direction is not a forecast.",
     "Hedging, offsetting flows and prior positioning can absorb or reverse the apparent effect.",
     ["Confirmed net flows and measurement window", "Liquidity and offsetting positioning"]),
)


def _mentions(text: str, term: str) -> bool:
    return bool(term) and re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", text, re.I) is not None


def _aliases(symbol: str) -> list[str]:
    symbol = symbol.upper().strip()
    aliases = [symbol]
    for entries in GROUPS.values():
        for ticker, name in entries:
            if ticker == symbol:
                aliases.append(name.split(" (")[0])
    for suffix in ("USDT", "USDC", "USD"):
        if symbol.endswith(suffix) and len(symbol) > len(suffix):
            aliases.append(symbol[:-len(suffix)])
            break
    if symbol in ("ETH", "ETHUSDT", "ETHUSD", "ETHUSDC"):
        aliases.extend(["Ethereum", "Ether"])
    if symbol in ("BTC", "BTCUSDT", "BTCUSD", "BTCUSDC"):
        aliases.append("Bitcoin")
    return aliases



def _conditional_bias(headline: str, direct: bool, category: str) -> str:
    """Conservative reading case, never an assertion that a reported event is true.

    Require the asset mention and completed-event wording in the headline itself.
    Summary keywords cannot create a directional case. Broad ambiguity suppression
    favors unknown over accidentally treating a proposed/denied event as completed.
    """
    if not direct or category not in ("security", "earnings"):
        return "unknown"
    if re.search(r"\b(no|not|never|denies?|denied|without|avoid(?:s|ed)?|prevent(?:s|ed)?|unconfirmed|alleged|rumou?r\w*|could|may|might|would|will|expects?|expected|forecast\w*|propos\w*|plans?|planned|upcoming|tomorrow|if|risk|warning|fake|false)\b|n['’]t\b|\?", headline, re.I):
        return "unknown"
    positive = category == "earnings" and bool(re.search(
        r"\b(?:raises?|raised|hikes?|hiked) (?:its )?(?:full.year |annual |quarterly )?guidance\b|"
        r"\bguidance (?:raised|increased)\b|\b(?:earnings|revenue|profit) (?:beat|beats|exceeded) (?:expectations|estimates|consensus)\b", headline, re.I))
    negative = (category == "security" and bool(re.search(
        r"\b(?:was |is |has been )?hacked\b|\b(?:confirmed|confirms) (?:a |the )?(?:hack|exploit|breach)\b|\b(?:funds|tokens|assets) (?:were |are )?stolen\b", headline, re.I))) or (
        category == "earnings" and bool(re.search(
            r"\b(?:cuts?|cut|lowers?|lowered|reduced) (?:its )?(?:full.year |annual |quarterly )?guidance\b|"
            r"\bguidance (?:cut|lowered|reduced)\b|\b(?:earnings|revenue|profit) (?:miss|misses|missed) (?:expectations|estimates|consensus)\b", headline, re.I)))
    if positive and negative:
        return "mixed"
    return "bullish" if positive else "bearish" if negative else "unknown"


def interpret_news(item: NewsItem, symbol: str) -> dict[str, Any]:
    """Return JSON-safe, conditional interpretation of the supplied news item.

    Sentiment is reported with its existing provenance, never promoted to a price
    forecast. Provider-assigned tickers alone do not prove direct relevance.
    """
    headline = str(item.headline or "").strip()
    summary = str(item.summary or "").strip()
    text = headline + " " + summary
    category = "unclassified"
    mechanism = "The supplied headline and summary do not establish a supported mechanism for this asset. Read the source and seek corroboration."
    counterargument = "Headline sentiment alone does not establish an asset-price effect."
    watch = ["Primary source and fuller context", "Direct relevance to the selected asset"]
    matched = []
    for name, pattern, explanation, counter, checks in _RULES:
        if re.search(pattern, text, re.I):
            matched.append(name)
            if category == "unclassified":
                category, mechanism, counterargument, watch = name, explanation, counter, checks
    # Avoid false direct matches from ordinary short words (e.g. V, OP, or LINK).
    aliases = _aliases(symbol)
    direct = any(_mentions(text, alias) for alias in aliases if len(alias) >= 4 and not alias.isupper())
    direct = direct or any(re.search(r"\$" + re.escape(alias) + r"\b", text, re.I) for alias in aliases if alias)
    direct = direct or any(re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text) for alias in aliases if len(alias) >= 2 and alias.isupper())
    crypto_entries = GROUPS.get("Crypto (Binance, USDT pairs)", [])
    crypto_selected = symbol.upper() in {ticker for ticker, _ in crypto_entries}
    other_crypto = any(_mentions(text, name) for ticker, name in crypto_entries if ticker != symbol.upper() and len(name) >= 4)
    relevance = "direct" if direct else ("macro" if category in ("monetary_policy", "macro_data") else
                                          "indirect_crypto" if crypto_selected and other_crypto else "unestablished")
    if relevance == "indirect_crypto":
        mechanism += " This report concerns another crypto asset; spillover to " + symbol + " is possible but unestablished."

    if relevance == "unestablished":
        mechanism += " The supplied text does not establish direct relevance to " + (symbol or "the selected asset") + "."
    headline_direct = any(_mentions(headline, alias) for alias in aliases if len(alias) >= 4 and not alias.isupper())
    headline_direct = headline_direct or any(re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", headline) for alias in aliases if len(alias) >= 2 and alias.isupper())
    conditional_bias = _conditional_bias(headline, headline_direct, category)
    published = item.datetime_utc
    if isinstance(published, datetime):
        published = published.replace(tzinfo=timezone.utc) if published.tzinfo is None else published
        if published > datetime.now(timezone.utc):
            conditional_bias = "unknown"
    sentiment = item.sentiment or {}
    label = str(sentiment.get("label") or "unknown").lower()
    if label not in ("positive", "negative", "neutral"):
        label = "unknown"
    return {
        "version": 1,
        "method": METHOD,
        "symbol": symbol,
        "event_category": category,
        "matched_categories": matched,
        "headline_tone": {"label": label, "model": str(sentiment.get("model_name") or "unavailable")},
        "asset_impact": "unknown",
        "conditional_bias": conditional_bias,
        "conditional_bias_basis": "Conditional reading of explicit completed-event wording in the supplied headline, assuming the claim is true; not a forecast or verified event.",
        "relevance": relevance,
        "mechanism": mechanism,
        "counterargument": counterargument,
        "what_to_watch": list(watch),
        "evidence": {"headline": headline, "excerpt": summary[:600] if summary else headline[:600],
                     "source": str(item.source or "Unknown source"), "url": str(item.url or "")},
        "limitations": "Rule-based hypotheses from supplied text; source claims are not independently verified. No calibrated forecast, trading signal or proof of price causation.",
    }
