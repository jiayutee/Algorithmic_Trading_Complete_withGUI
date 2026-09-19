"""Catalogue of instruments offered in both front ends' dropdowns, and rules for what each kind can do.

Every entry here was CHECKED against real data on 2026-09-19 (crypto through the app's own loader / Binance; the rest against
Yahoo Finance's chart endpoint: ticker exists with a live price history). TONUSDT was tried and dropped (not served).
The dropdowns are not a limit: the desktop box also accepts any typed symbol, and Dash has a "custom symbol" field. Anything
containing "USDT" is loaded from Binance; everything else from Yahoo Finance (stocks, ETFs, indices ^GSPC, futures GC=F, FX EURUSD=X).

Kinds and what they support:
  crypto / stock / etf  -> chart, backtest AND the paper execution service
  index / future / fx   -> chart and backtest only (you cannot buy an index; futures and FX need contract/margin models the
                           paper broker does not have), so the paper execution service refuses them with an explanation.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

GROUPS: Dict[str, List[Tuple[str, str]]] = {
    "Crypto (Binance, USDT pairs)": [
        ("BTCUSDT", "Bitcoin"),
        ("ETHUSDT", "Ethereum"),
        ("BNBUSDT", "BNB"),
        ("SOLUSDT", "Solana"),
        ("XRPUSDT", "XRP"),
        ("ADAUSDT", "Cardano"),
        ("DOGEUSDT", "Dogecoin"),
        ("AVAXUSDT", "Avalanche"),
        ("LINKUSDT", "Chainlink"),
        ("DOTUSDT", "Polkadot"),
        ("LTCUSDT", "Litecoin"),
        ("TRXUSDT", "TRON"),
        ("AAVEUSDT", "Aave"),
        ("APTUSDT", "Aptos"),
        ("ARBUSDT", "Arbitrum"),
        ("ATOMUSDT", "Cosmos"),
        ("BCHUSDT", "Bitcoin Cash"),
        ("ETCUSDT", "Ethereum Classic"),
        ("FILUSDT", "Filecoin"),
        ("INJUSDT", "Injective"),
        ("NEARUSDT", "NEAR Protocol"),
        ("OPUSDT", "Optimism"),
        ("PEPEUSDT", "Pepe"),
        ("SHIBUSDT", "Shiba Inu"),
        ("SUIUSDT", "Sui"),
        ("UNIUSDT", "Uniswap"),
        ("XLMUSDT", "Stellar"),
    ],
    "US stocks": [
        ("AAPL", "Apple"),
        ("MSFT", "Microsoft"),
        ("NVDA", "NVIDIA"),
        ("GOOGL", "Alphabet (Google)"),
        ("AMZN", "Amazon"),
        ("META", "Meta Platforms"),
        ("TSLA", "Tesla"),
        ("AVGO", "Broadcom"),
        ("JPM", "JPMorgan Chase"),
        ("V", "Visa"),
        ("UNH", "UnitedHealth"),
        ("XOM", "Exxon Mobil"),
        ("WMT", "Walmart"),
        ("LLY", "Eli Lilly"),
        ("COST", "Costco"),
        ("NFLX", "Netflix"),
        ("AMD", "AMD"),
        ("BRK-B", "Berkshire Hathaway B"),
        ("ORCL", "Oracle"),
        ("CRM", "Salesforce"),
        ("INTC", "Intel"),
        ("DIS", "Disney"),
        ("BA", "Boeing"),
        ("PYPL", "PayPal"),
        ("COIN", "Coinbase"),
        ("MSTR", "MicroStrategy"),
        ("GOLD", "Barrick Gold (the stock, not gold)"),
    ],
    "ETFs": [
        ("SPY", "S&P 500 ETF"),
        ("QQQ", "Nasdaq-100 ETF"),
        ("IWM", "Russell 2000 ETF"),
        ("DIA", "Dow Jones ETF"),
        ("VTI", "Total US Market ETF"),
        ("GLD", "Gold ETF"),
        ("SLV", "Silver ETF"),
        ("TLT", "20+ Year Treasury ETF"),
        ("XLK", "Technology Sector ETF"),
        ("XLF", "Financials Sector ETF"),
        ("XLE", "Energy Sector ETF"),
        ("ARKK", "ARK Innovation ETF"),
        ("EEM", "Emerging Markets ETF"),
        ("EFA", "Developed ex-US ETF"),
    ],
    "Commodities (futures)": [
        ("GC=F", "Gold futures"),
        ("SI=F", "Silver futures"),
        ("CL=F", "Crude oil (WTI) futures"),
        ("NG=F", "Natural gas futures"),
        ("HG=F", "Copper futures"),
    ],
    "Indices": [
        ("^GSPC", "S&P 500 index"),
        ("^IXIC", "Nasdaq Composite index"),
        ("^DJI", "Dow Jones index"),
        ("^VIX", "VIX volatility index"),
    ],
    "Forex": [
        ("EURUSD=X", "EUR/USD"),
        ("GBPUSD=X", "GBP/USD"),
        ("USDJPY=X", "USD/JPY"),
        ("AUDUSD=X", "AUD/USD"),
    ],
}

_KIND_BY_GROUP = {"Crypto (Binance, USDT pairs)": "crypto", "US stocks": "stock", "ETFs": "etf",
                  "Commodities (futures)": "future", "Indices": "index", "Forex": "fx"}
NAMES: Dict[str, str] = {s: n for items in GROUPS.values() for s, n in items}
_GROUP_OF: Dict[str, str] = {s: g for g, items in GROUPS.items() for s, _ in items}
PAPER_TRADABLE_KINDS = ("crypto", "stock", "etf")


def all_symbols() -> List[str]:
    return [s for items in GROUPS.values() for s, _ in items]


def name_of(symbol: str) -> str:
    return NAMES.get(symbol.upper(), "")


def label_of(symbol: str) -> str:
    """"AAPL -- Apple": what the dropdowns show (search matches either part)."""
    n = name_of(symbol)
    return f"{symbol} — {n}" if n else symbol


def kind_of(symbol: str) -> str:
    """crypto | stock | etf | index | future | fx -- also for symbols that are NOT in the catalogue (typed by the user)."""
    s = symbol.strip().upper()
    if s in _GROUP_OF:
        return _KIND_BY_GROUP[_GROUP_OF[s]]
    if "USDT" in s:
        return "crypto"
    if s.startswith("^"):
        return "index"
    if s.endswith("=F"):
        return "future"
    if s.endswith("=X"):
        return "fx"
    return "stock"                      # cannot tell a typed stock from a typed ETF without a lookup; both behave the same


def paper_tradable(symbol: str) -> bool:
    return kind_of(symbol) in PAPER_TRADABLE_KINDS


def paper_refusal(symbol: str) -> str:
    """Plain-words reason the paper execution service will not trade symbol (empty string if it will)."""
    k = kind_of(symbol)
    if k in PAPER_TRADABLE_KINDS:
        return ""
    why = {"index": "an index cannot be bought directly (use an ETF such as SPY or QQQ)",
           "future": "futures need contract and margin handling that the paper broker does not model",
           "fx": "FX needs lot sizes, leverage and rollover that the paper broker does not model"}[k]
    return f"{symbol} is chart/backtest-only: {why}"
