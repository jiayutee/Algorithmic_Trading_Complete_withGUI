Algorithmic Trading Terminal
A comprehensive Python-based algorithmic trading platform with real-time data visualization, technical analysis, and automated trading strategies.

🚀 Features
📊 Advanced Charting
Interactive Candlestick Charts with Plotly visualization

Multiple Timeframes: 1m, 5m, 15m, 30m, 1h, 1d

Technical Indicators:

Moving Averages (MA20, MA50, MA200)

Exponential Moving Averages (EMA12, EMA26)

MACD with Signal Line

Relative Strength Index (RSI)

Stochastic Oscillator (%K, %D)

Toggle Indicators with interactive buttons

Real-time Data Streaming with live updates

🤖 Trading Strategies
MACD/RSI Strategy - Momentum-based trading

EMA Crossover - Trend-following strategy

Stochastic Oscillator - Overbought/oversold signals

LSTM Predictor - Machine learning price prediction

FinRL Integration - Deep reinforcement learning

DDPG Strategy - Advanced AI trading

🔌 Multi-Broker Support
Simulator - Paper trading with realistic simulation

Alpaca - Commission-free API trading

Interactive Brokers - Professional trading platform

Binance - Cryptocurrency trading

📈 Data Sources
Historical Data - Yahoo Finance integration

Live Data - Real-time market data

Realtime Stream - WebSocket streaming

FinRL-Yahoo - Enhanced financial datasets

🗞 News & Sentiment
Multi-source news ingestion with NewsAPI, GDELT, EventRegistry, and RSS feeds

Headline sentiment scoring with FinBERT when available and a deterministic fallback when offline

Canonical news/event features merged into price data for strategy use

Event classification for earnings, guidance, M&A, analyst actions, macro, regulatory, product, litigation, and dividend headlines

### Configure the news sources
Store your secret keys in `.env` for local development, or add them as Codespaces secrets. `app.py` calls `load_dotenv()`, so the values are loaded before `config/settings.py` reads them with `os.getenv()`.

Set these environment variables before running the app:

```bash
export NEWSAPI_API_KEY="your_newsapi_key"
export EVENTREGISTRY_API_KEY="your_eventregistry_key"   # optional
export RSS_FEEDS="https://example.com/feed.xml,https://example.org/rss"
export NEWS_SENTIMENT_MODEL="ProsusAI/finbert"          # optional
```

`NEWSAPI_API_KEY` is required for NewsAPI, `EVENTREGISTRY_API_KEY` is optional, and `RSS_FEEDS` can contain one or more public feed URLs separated by commas.

### Free MCPs already available in `.vscode/mcp.json`
- `duckduckgo-search` for discovery and lightweight search.
- `firecrawl/firecrawl-mcp-server` for extraction and structured scraping.
- `io.github.tavily-ai/tavily-mcp` for search and page retrieval.

Those MCPs are useful when you want to discover article URLs, scrape article pages, or enrich event coverage before feeding it into the local pipeline.

🎮 Simulation & Backtesting
Historical Simulation - Walk-forward analysis

Portfolio Tracking - Real-time P&L monitoring

Strategy Optimization - Parameter tuning

Performance Metrics - Sharpe ratio, drawdown, etc.


