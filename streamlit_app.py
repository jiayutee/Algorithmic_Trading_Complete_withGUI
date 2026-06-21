from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import os
import time

from core.data_loader import DataLoader
from core.logger import logger
from core.ta_engine import TAEngine
from core.strategy_manager import StrategyManager
from core.news_pipeline import get_default_news_pipeline


@st.cache_resource
def get_runtime_supervisor():
    try:
        from core.runtime.supervisor import Supervisor

        return Supervisor()
    except Exception:
        return None


st.set_page_config(page_title="Algorithmic Trading Terminal", layout="wide")


@st.cache_resource
def get_data_loader() -> DataLoader:
    return DataLoader()


@st.cache_resource
def get_strategy_manager() -> StrategyManager:
    return StrategyManager()


@st.cache_resource
def get_news_pipeline():
    return get_default_news_pipeline()


def add_indicators(df: pd.DataFrame, ma_window: int | None, ema_window: int | None) -> pd.DataFrame:
    frame = df.copy()
    if ma_window:
        frame[f"MA_{ma_window}"] = frame["Close"].rolling(window=ma_window, min_periods=1).mean()
    if ema_window:
        try:
            frame[f"EMA_{ema_window}"] = TAEngine.calculate_ema(frame, window=ema_window)
        except Exception:
            frame[f"EMA_{ema_window}"] = frame["Close"].ewm(span=ema_window, adjust=False).mean()

    # MACD / RSI
    try:
        macd = TAEngine.calculate_macd(frame)
        frame["MACD"] = macd["macd_line"]
        frame["MACD_Signal"] = macd["signal_line"]
        frame["MACD_Hist"] = macd["histogram"]
    except Exception:
        frame["MACD"] = 0
        frame["MACD_Signal"] = 0
        frame["MACD_Hist"] = 0

    try:
        frame["RSI"] = TAEngine.calculate_rsi(frame)
    except Exception:
        frame["RSI"] = frame["Close"].diff().apply(lambda x: x if x>0 else 0).rolling(window=14).mean()

    return frame


def build_chart(df: pd.DataFrame, symbol: str, indicators: list[str]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=symbol,
        )
    )

    # overlay indicators
    if any(col.startswith("MA_") for col in df.columns) and "MA" in indicators:
        for col in sorted([c for c in df.columns if c.startswith("MA_")]):
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(width=1)))

    if any(col.startswith("EMA_") for col in df.columns) and "EMA" in indicators:
        for col in sorted([c for c in df.columns if c.startswith("EMA_")]):
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(width=1, dash="dash")))

    fig.update_layout(
        title=f"{symbol} Price Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark",
        height=700,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


st.title("Algorithmic Trading Terminal")
st.caption("Browser version of the desktop GUI")

with st.sidebar:
    st.header("Controls")
    symbol = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "AAPL", "TSLA", "SPY", "QQQ"], index=0)
    source = st.selectbox("Data Source", ["Historical", "Live", "FinRL-Yahoo"], index=0)
    interval = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d"], index=4)
    days = st.slider("Days", min_value=5, max_value=365, value=60, step=5)

    # Warn user about Yahoo intraday data limits (warning only; do not modify `days`)
    minute_intervals = {"1m", "2m", "5m", "15m", "30m", "60m"}
    try:
        max_intraday_days = int(os.getenv("YAHOO_INTRADAY_MAX_DAYS", "7"))
    except Exception:
        max_intraday_days = 7

    if interval in minute_intervals and days > max_intraday_days:
        st.warning(
            f"Yahoo limits intraday historical data to {max_intraday_days} days for the selected interval. "
            f"Data will be capped to {max_intraday_days} days when loading."
        )

    st.markdown("---")
    st.subheader("Indicators")
    show_ma = st.checkbox("MA", value=True)
    ma_window = st.number_input("MA Window", min_value=2, max_value=500, value=20, step=1)
    show_ema = st.checkbox("EMA", value=True)
    ema_window = st.number_input("EMA Window", min_value=2, max_value=500, value=20, step=1)
    show_macd = st.checkbox("MACD", value=True)
    show_rsi = st.checkbox("RSI", value=True)

    st.markdown("---")
    st.subheader("Strategy & Backtest")
    strategy_manager = get_strategy_manager()
    available = strategy_manager.get_available_strategies()
    selected_strategy = st.selectbox("Select Strategy", available)
    initial_cash = st.number_input("Initial Cash", min_value=1000.0, value=100000.0, step=1000.0)
    run_backtest = st.button("Run Backtest")

    st.markdown("---")
    st.subheader("News")
    news_limit = st.number_input("News Items", min_value=1, max_value=200, value=25, step=1)
    load_button = st.button("Load Data")

data_loader = get_data_loader()
news_pipeline = get_news_pipeline()

if load_button or "loaded_df" not in st.session_state:
    try:
        with st.spinner("Loading market data..."):
            df = data_loader.load_data(symbol=symbol, source=source, days=days, interval=interval)
        st.session_state["loaded_df"] = df
        st.session_state["loaded_symbol"] = symbol
        st.session_state["loaded_source"] = source
        logger.info("Loaded browser dashboard data for %s", symbol)
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

df = st.session_state.get("loaded_df")
if df is None or df.empty:
    st.warning("No data loaded. Choose symbol and press 'Load Data'.")
    st.stop()

# Ensure datetime index
if not isinstance(df.index, pd.DatetimeIndex):
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
        df = df.set_index("Datetime")
    else:
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

# Add indicators
indicators = []
if show_ma:
    indicators.append("MA")
if show_ema:
    indicators.append("EMA")
if show_macd:
    indicators.append("MACD")
if show_rsi:
    indicators.append("RSI")

df = add_indicators(df, ma_window if show_ma else None, ema_window if show_ema else None)

col1, col2, col3 = st.columns(3)
col1.metric("Rows", len(df))
col2.metric("Close", f"{float(df['Close'].iloc[-1]):.2f}" if "Close" in df.columns and not df.empty else "n/a")
col3.metric("Symbol", st.session_state.get("loaded_symbol", symbol))

st.plotly_chart(build_chart(df, st.session_state.get("loaded_symbol", symbol), indicators), use_container_width=True)

if show_macd:
    st.subheader("MACD")
    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(x=df.index, y=df.get("MACD", []), name="MACD"))
    macd_fig.add_trace(go.Scatter(x=df.index, y=df.get("MACD_Signal", []), name="Signal"))
    macd_fig.add_trace(go.Bar(x=df.index, y=df.get("MACD_Hist", []), name="Histogram"))
    macd_fig.update_layout(template="plotly_dark", height=250, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(macd_fig, use_container_width=True)

if show_rsi:
    st.subheader("RSI")
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=df.index, y=df.get("RSI", []), name="RSI", line=dict(color="orange")))
    rsi_fig.update_layout(template="plotly_dark", height=200, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(rsi_fig, use_container_width=True)

st.subheader("Latest Data")
st.dataframe(df.tail(20), use_container_width=True)

# Backtest execution
if run_backtest:
    try:
        with st.spinner("Running backtest..."):
            wrapper = strategy_manager.get_strategy(selected_strategy)
            if wrapper is None:
                st.error("Selected strategy unavailable")
            else:
                results = strategy_manager.run_backtest(wrapper, data=df, cash=float(initial_cash))
                if results is None:
                    st.error("Backtest returned no results")
                elif "error" in results:
                    st.error(f"Backtest error: {results.get('error')}")
                else:
                    st.success("Backtest complete")
                    summary = results.get("summary") or {}
                    st.subheader("Backtest Summary")
                    for k, v in summary.items():
                        st.write(f"**{k}**: {v}")

                    # Show PnL curve
                    cum = results.get("total_asset_value") or []
                    if cum:
                        pnl_fig = go.Figure()
                        pnl_fig.add_trace(go.Scatter(y=cum, x=list(range(len(cum))), name="Portfolio Value"))
                        pnl_fig.update_layout(template="plotly_dark", height=350)
                        st.plotly_chart(pnl_fig, use_container_width=True)
    except Exception as exc:  # pragma: no cover - surface errors to user
        st.error(f"Backtest failed: {exc}")

# News panel
st.subheader("News Panel")
try:
    with st.spinner("Fetching news..."):
        news_df = news_pipeline.fetch_news_dataframe(symbol=symbol, limit=int(news_limit))

    if news_df is None or news_df.empty:
        st.info("No news items found for this symbol or no news sources configured.")
    else:
        st.write(f"Showing {len(news_df)} news items")
        feed_df = news_df.copy()
        if "datetime" in feed_df.columns:
            feed_df["datetime"] = pd.to_datetime(feed_df["datetime"], utc=True, errors="coerce")
            feed_df = feed_df.dropna(subset=["datetime"]).sort_values("datetime", ascending=False)
        else:
            feed_df = feed_df.head(50)

        for _, row in feed_df.head(50).iterrows():
            source_name = str(row.get("source", "news")).strip() or "news"
            headline = str(row.get("headline", "")).strip()
            snippet = str(row.get("summary") or row.get("content") or "").strip()
            if not snippet:
                snippet = str(row.get("description") or "").strip()
            dt_value = row.get("datetime")
            dt_label = dt_value.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(dt_value) else ""
            title_line = f"**[{source_name}] {headline or 'Untitled'}**"
            body = title_line if not snippet else f"{title_line}\n\n{snippet}"
            st.markdown(body)
            if dt_label:
                st.caption(dt_label)

        # Merge news features into prices if possible
        merged = news_pipeline.merge_features_into_prices(df.copy(), news_df, interval=interval)
        if merged is not None and not merged.empty:
            st.subheader("Merged News Features (sample)")
            st.dataframe(merged[[c for c in merged.columns if c.startswith("news_") or c in ["impact_score", "news_count", "news_flow_ratio"]]].tail(20))

except Exception as exc:
    logger.exception("News fetch failed")
    st.error(f"News pipeline failed: {exc}")


# Optional runtime agents UI (enabled via environment variable)
if os.getenv("ENABLE_RUNTIME_AGENTS", "0") == "1":
    sup = get_runtime_supervisor()
    if sup is None:
        st.info("Runtime supervisor unavailable (missing dependencies)")
    else:
        # start background loop (Supervisor.start is idempotent)
        try:
            sup.start()
        except Exception:
            pass

        st.subheader("Runtime Agents Status")
        cols = st.columns([1, 1, 1, 1])

        auto = st.checkbox("Auto-refresh status", value=False)
        interval = st.number_input("Refresh interval (s)", min_value=1, max_value=60, value=5, step=1)
        if st.button("Refresh status"):
            st.session_state["runtime_last_poll"] = time.time()
            st.experimental_rerun()

        # auto-refresh via simple time-based rerun (non-blocking)
        last = st.session_state.get("runtime_last_poll", 0.0)
        if auto and (time.time() - float(last) > float(interval)):
            st.session_state["runtime_last_poll"] = time.time()
            st.experimental_rerun()

        try:
            snap = sup.status() or {}
        except Exception:
            snap = {}

        agent_names = ["portfolio", "news", "price", "stats"]
        for c, name in zip(cols, agent_names):
            info = snap.get(name) or {}
            latest = info.get("latest")
            hist_len = info.get("history_len", 0)
            with c:
                st.markdown(f"**{name.capitalize()}**")
                if latest is None:
                    st.write("no data")
                else:
                    status = getattr(latest, "status", None) or getattr(latest, "state", None) or "unknown"
                    msg = getattr(latest, "message", None) or getattr(latest, "summary", None) or ""
                    st.write(status)
                    if msg:
                        st.caption(msg)
                st.caption(f"history={hist_len}")

