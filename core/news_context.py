"""Bounded, read-only news snapshots and observed price context for the UI."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import math
import base64
import binascii
import re
import numpy as np
import pandas as pd

from core.news_interpretation import interpret_news
from core.ai_research import research_event


# Web-search sources return pages, not dated articles: the item time is just the fetch time, so they would
# sit at "now" on the timeline and cannot be placed against the candles.
UNDATED_SOURCES = {'duckduckgo'}

# Evergreen explainers, reference/price pages and promotions carry no information about a price move.
_EVERGREEN = re.compile(
    r"\b(what(?:\s+\w+){0,3}?\s+(?:is|are|was)\b(?!\s+(?:driving|behind|causing|pushing|moving|next|happening|going|up|down)\b)|"
    r"how(?:\s+\w+){0,3}?\s+works?\b|how (?:to|does|do)\b|explained\b|beginner'?s?|basics\b|for beginners|guide to|a guide\b|"
    r"tutorial|101\b|wikipedia|price and chart|price chart|live chart|"
    r"best (crypto|trading|broker|exchange)|top \d+ .*(broker|platform|exchange)|broker(-| )?(vergleich|comparison)|"
    r"plattform|review\b.*\b(2\d{3})\b)", re.I)

# Kept on the timeline: about the selected asset, or market-wide (rates, inflation...). Everything else is counted, not shown.
_RELEVANT = {'direct', 'macro'}


def is_evergreen(headline):
    return bool(_EVERGREEN.search(headline or ''))


def build_snapshot(items, symbol, source_status=(), now=None):
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    now = now.tz_localize('UTC') if now.tzinfo is None else now.tz_convert('UTC')
    events, seen = [], set()
    hidden = {'undated': 0, 'evergreen': 0, 'off_topic': 0}
    for item in items:
        timestamp = pd.to_datetime(item.datetime_utc, utc=True, errors='coerce')
        if pd.isna(timestamp) or timestamp > now:  # published news is never an upcoming calendar
            continue
        headline = str(item.headline or '').strip()
        key = ' '.join(headline.lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        if str(item.source or '').lower() in UNDATED_SOURCES:
            hidden['undated'] += 1
            continue
        if is_evergreen(headline):
            hidden['evergreen'] += 1
            continue
        interpretation = interpret_news(item, symbol)
        if interpretation['relevance'] not in _RELEVANT:
            hidden['off_topic'] += 1
            continue
        events.append({'id': hashlib.sha256((timestamp.isoformat()+key).encode()).hexdigest()[:20],
                       'time': timestamp.isoformat(), 'headline': headline,
                       'source': item.source or 'Unknown source', 'url': item.url,
                       'interpretation': interpretation})
    events.sort(key=lambda e: e['time'], reverse=True)
    return {'symbol': symbol, 'as_of': now.isoformat(), 'events': events[:40],
            'sources': list(source_status), 'hidden': hidden, 'error': None}


def fetch_snapshot(symbol, pipeline=None):
    """Reuse app routing, deadline and prefilter without sentiment calls or DB writes."""
    from core.news_pipeline import get_default_news_pipeline, _query_variants
    pipeline = pipeline or get_default_news_pipeline()
    items = pipeline._fetch_all_sources(_query_variants(symbol)[0], 40, ticker_query=symbol)
    items = pipeline._prefilter(items, symbol, None)
    return build_snapshot(items, symbol, pipeline.source_status())


def _chart_array(value):
    # Plotly 6 transports numeric arrays as typed base64 buffers in Dash State.
    if isinstance(value, dict) and 'bdata' in value:
        dtype = np.dtype(value.get('dtype', 'f8'))
        if dtype.kind not in 'fiu' or dtype.itemsize > 8:
            raise ValueError('unsupported chart dtype')
        return np.frombuffer(base64.b64decode(value['bdata'], validate=True), dtype=dtype).tolist()
    return value


def candles_from_figure(figure):
    """Read the loaded chart only; no second market-data request or live-tick trace."""
    for trace in (figure or {}).get('data', []):
        if trace.get('type') == 'candlestick':
            try:
                df = pd.DataFrame({k: _chart_array(trace[k]) for k in ('open', 'high', 'low', 'close')},
                                  index=pd.to_datetime(trace['x'], utc=True, errors='coerce'))
                df = df.apply(pd.to_numeric, errors='coerce').dropna()
                return df[~df.index.isna()].sort_index()
            except (KeyError, ValueError, TypeError, binascii.Error):
                break
    return pd.DataFrame(columns=['open', 'high', 'low', 'close'])


def candle_step(candles):
    diffs = candles.index.to_series().diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    return diffs.median() if not diffs.empty else pd.Timedelta(days=1)


def observed_move(candles, event_time):
    """Descriptive close-to-close comparison; not causal inference or a trade return."""
    timestamp = pd.to_datetime(event_time, utc=True, errors='coerce')
    if candles.empty or pd.isna(timestamp):
        return None
    # Use the prior bar, not the publication bar's close (unavailable at publication).
    step = candle_step(candles)
    before = candles[candles.index + step <= timestamp]
    after = candles[candles.index + step > timestamp]
    if before.empty or after.empty or timestamp >= candles.index[-1] + step:
        return None
    baseline, latest = float(before['close'].iloc[-1]), float(after['close'].iloc[-1])
    if baseline <= 0 or not all(map(math.isfinite, (baseline, latest))):
        return None
    return {'percent': (latest / baseline - 1) * 100,
            'start': before.index[-1].isoformat(), 'end': after.index[-1].isoformat()}


def ai_research_for_event(event, symbol):
    """Best-effort AI research note for one already-built event (see
    core.ai_research.research_event); returns None if unavailable/unusable."""
    if not event:
        return None
    interpretation = event.get('interpretation') or {}
    return research_event(
        headline=event.get('headline', ''),
        summary=(interpretation.get('evidence') or {}).get('excerpt', ''),
        symbol=symbol,
        event_category=interpretation.get('event_category', 'unclassified'),
        sentiment_label=(interpretation.get('headline_tone') or {}).get('label', 'unknown'),
    )
