"""Feature matrix for ML strategies (Phase 6.0).

Builds one DataFrame of model inputs per (symbol, timestamp) from:

  1. technical indicators   (core/ta_engine.py + a few pure-pandas ones)
  2. news sentiment         (columns core/news_pipeline.py merges into the price frame)
  3. macro / market context (VIX, 10y yield, dollar index -- aligned with an explicit publication lag)
  4. time features          (cyclical day-of-week / month / hour)

THE ONE RULE: the feature at row *t* may only use information that was known
at *t*. Everything here is a trailing rolling / ewm / shift(+n) computation, or
an as-of join with a publication lag. ``test_feature_engineering.py`` proves it
by truncation: features computed on ``df[:k]`` must equal the first ``k`` rows
of features computed on the full ``df``.

Labels are the opposite (they look forward), so they live in a separate
function, :func:`make_target`, and are never part of the feature matrix.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd

from core.logger import logger
from core.ta_engine import TAEngine

FEATURE_SCHEMA_VERSION = 1

# Columns core/data_loader.py + core/news_pipeline.py add to the price frame.
NEWS_SOURCE_COLUMNS = [
    "sentiment_balance",     # mean(positive - negative) of news in the bar, [-1, 1]
    "sentiment_magnitude",
    "sentiment_confidence",
    "news_count",
    "impact_score",
    "news_flow_ratio",
]

DEFAULT_MACRO_TICKERS = {"vix": "^VIX", "us10y": "^TNX", "dollar": "DX-Y.NYB"}


def _ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise to lower-case float OHLCV on a sorted, de-duplicated DatetimeIndex."""
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    missing = {"open", "high", "low", "close"} - set(out.columns)
    if missing:
        raise ValueError(f"OHLC columns missing from frame: {sorted(missing)}")
    if "volume" not in out.columns:
        out["volume"] = 0.0
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    return out


def technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trailing technical features. Scale-free (returns / ratios / z-scores), so one
    model can be trained across symbols with different price levels."""
    d = _ohlcv(df)
    close, high, low, volume = d["close"], d["high"], d["low"], d["volume"]
    cap = d.rename(columns=str.capitalize)      # TAEngine expects Close/High/Low
    f = pd.DataFrame(index=d.index)

    ret_1 = close.pct_change()
    for n in (1, 5, 10, 20):
        f[f"ret_{n}"] = close.pct_change(n)
    f["vol_10"] = ret_1.rolling(10).std()
    f["vol_20"] = ret_1.rolling(20).std()

    f["rsi_14"] = TAEngine.calculate_rsi(cap, window=14)
    macd = TAEngine.calculate_macd(cap)
    f["macd_hist_pct"] = macd["histogram"] / close
    f["macd_line_pct"] = macd["macd_line"] / close
    ema_fast = TAEngine.calculate_ema(cap, window=12)
    ema_slow = TAEngine.calculate_ema(cap, window=26)
    f["ema_ratio"] = ema_fast / ema_slow - 1.0
    stoch = TAEngine.calculate_stochastic(cap)
    f["stoch_k"] = stoch["percent_k"]
    f["stoch_d"] = stoch["percent_d"]

    prev_close = close.shift(1)
    true_range = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    f["atr_pct_14"] = true_range.rolling(14).mean() / close
    f["range_pct"] = (high - low) / close

    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    f["bb_pos"] = (close - ma20) / (2.0 * std20.replace(0, np.nan))
    f["dist_ma50"] = close / close.rolling(50).mean() - 1.0

    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    f["volume_z"] = ((volume - vol_mean) / vol_std.replace(0, np.nan)).where(vol_std > 0, 0.0)
    return f.replace([np.inf, -np.inf], np.nan)


def time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclical calendar features. Depend only on the timestamp itself."""
    idx = pd.DatetimeIndex(index)
    f = pd.DataFrame(index=idx)
    f["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7.0)
    f["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7.0)
    f["month_sin"] = np.sin(2 * np.pi * (idx.month - 1) / 12.0)
    f["month_cos"] = np.cos(2 * np.pi * (idx.month - 1) / 12.0)
    if (idx.normalize() != idx).any():          # intraday bars only
        f["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24.0)
        f["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24.0)
    return f


def news_features(df: pd.DataFrame, half_life_bars: float = 3.0) -> pd.DataFrame:
    """Clean sentiment feature set from the columns the news pipeline merges into the price frame.

    Adds an exponentially-decayed sentiment (the diffusion kernel: a shock's
    influence halves every ``half_life_bars``) and a trailing news-flow count.
    Missing source columns are treated as "no news" (0), so the schema is stable.
    """
    d = df.copy()
    d.columns = [str(c) for c in d.columns]
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index)
    d = d[~d.index.duplicated(keep="last")].sort_index()
    f = pd.DataFrame(index=d.index)
    for col in NEWS_SOURCE_COLUMNS:
        f[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0) if col in d.columns else 0.0
    f["sentiment_ewm"] = f["sentiment_balance"].ewm(halflife=half_life_bars, adjust=False).mean()
    f["news_count_5"] = f["news_count"].rolling(5, min_periods=1).sum()
    return f


def align_macro(
    index: pd.DatetimeIndex,
    macro: pd.DataFrame,
    publication_lag: Union[str, pd.Timedelta] = "0D",
    change_bars: int = 5,
) -> pd.DataFrame:
    """As-of join of macro series onto *index* with an explicit publication lag.

    ``macro`` is indexed by observation time, one column per series. Each
    observation becomes usable only at ``obs_time + publication_lag`` -- e.g.
    CPI for a month is published weeks later, so joining it on its reference
    date would leak the future. Market-price series (VIX, yields) closing at
    the same instant as a daily bar can use lag "0D"; for intraday bars use
    "1D" so a day's close is not visible to that day's earlier bars.

    Returns, per series, its level and its ``change_bars``-observation change
    (computed on the raw series before joining, so it is causal too).
    """
    idx = pd.DatetimeIndex(index)
    if macro is None or macro.empty:
        return pd.DataFrame(index=idx)
    m = macro.copy()
    m.index = pd.to_datetime(m.index)
    m = m[~m.index.duplicated(keep="last")].sort_index()
    for col in list(m.columns):
        m[f"{col}_chg{change_bars}"] = m[col].pct_change(change_bars)
    m.index = m.index + pd.Timedelta(publication_lag)      # effective (usable-from) time
    m = m.sort_index()
    left = pd.DataFrame({"_t": idx}).sort_values("_t")
    joined = pd.merge_asof(left, m.reset_index(names="_eff"), left_on="_t", right_on="_eff", direction="backward")
    joined = joined.drop(columns=["_eff"]).set_index("_t")
    joined.index.name = idx.name
    return joined.reindex(idx)


def fetch_macro_series(start, end, tickers: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Best-effort download of market-context series (VIX, 10y yield, dollar index) via yfinance.

    Returns a frame of closes indexed by date (empty on any failure -- callers
    treat that as "no macro features"). CPI/employment need a FRED key and are
    not fetched here; :func:`align_macro` handles them once supplied.
    """
    tickers = tickers or DEFAULT_MACRO_TICKERS
    try:
        import yfinance as yf
        cols = {}
        for name, ticker in tickers.items():
            hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
            if hist is not None and not hist.empty:
                s = hist["Close"].copy()
                s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
                cols[name] = s
        return pd.DataFrame(cols).sort_index() if cols else pd.DataFrame()
    except Exception as exc:  # noqa: BLE001 -- macro is optional context, never fatal
        logger.warning("fetch_macro_series failed (%s); continuing without macro features", exc)
        return pd.DataFrame()


def build_features(
    df: pd.DataFrame,
    *,
    include_news: bool = False,
    macro: Optional[pd.DataFrame] = None,
    macro_lag: Union[str, pd.Timedelta] = "0D",
    news_half_life_bars: float = 3.0,
) -> pd.DataFrame:
    """Feature matrix for ONE symbol, indexed by timestamp (see :func:`build_feature_matrix`)."""
    parts = [technical_features(df)]
    parts.append(time_features(parts[0].index))
    if include_news:
        parts.append(news_features(df, news_half_life_bars).reindex(parts[0].index))
    if macro is not None and not macro.empty:
        parts.append(align_macro(parts[0].index, macro, publication_lag=macro_lag))
    return pd.concat(parts, axis=1)


def build_feature_matrix(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    symbol: str = "ASSET",
    **kwargs,
) -> pd.DataFrame:
    """Unified feature matrix indexed by (symbol, timestamp).

    ``data`` is a single OHLCV frame (labelled with ``symbol``) or a dict
    ``{symbol: frame}``. Keyword arguments are forwarded to :func:`build_features`.
    """
    frames = data if isinstance(data, dict) else {symbol: data}
    pieces = []
    for sym, frame in frames.items():
        feats = build_features(frame, **kwargs)
        feats.index = pd.MultiIndex.from_arrays([[sym] * len(feats), feats.index], names=["symbol", "timestamp"])
        pieces.append(feats)
    return pd.concat(pieces) if pieces else pd.DataFrame()


def make_target(df: pd.DataFrame, horizon: int = 1, kind: str = "direction") -> pd.Series:
    """Label for supervised learning. **Looks forward** -- never use it as a feature.

    ``kind="return"``: close[t+horizon] / close[t] - 1.
    ``kind="direction"``: 1.0 if that return is > 0 else 0.0.
    The last ``horizon`` rows have no label (NaN).
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    close = _ohlcv(df)["close"]
    fwd = close.shift(-horizon) / close - 1.0
    if kind == "return":
        return fwd.rename(f"fwd_ret_{horizon}")
    if kind != "direction":
        raise ValueError("kind must be 'direction' or 'return'")
    return (fwd > 0).astype(float).where(fwd.notna()).rename(f"up_{horizon}")


def clean_xy(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Drop warm-up rows (NaN features) and unlabelled tail rows, keeping X and y aligned."""
    joined = X.copy()
    joined["__y__"] = y.reindex(X.index)          # X and y must share an index type
    joined = joined.replace([np.inf, -np.inf], np.nan).dropna()
    return joined.drop(columns="__y__"), joined["__y__"]
