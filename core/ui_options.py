"""Single source of truth for the choices both front ends offer (desktop PyQt5 app and Dash web view).

Both UIs import from here so they cannot drift apart (they had: the desktop offered ADAUSDT/GOLD, Dash did not).
Change a list here and both UIs change; tests/test_ui_parity.py fails if either stops using it.
"""
from __future__ import annotations

from core.instruments import all_symbols, label_of

SYMBOLS = all_symbols()          # 81 verified instruments (core/instruments.py); the desktop box also accepts any typed symbol
SYMBOL_LABELS = {s: label_of(s) for s in SYMBOLS}
INTERVALS = ["1d", "1h", "15m", "5m", "1m"]
DEFAULT_SYMBOL_DASH = "AAPL"
DEFAULT_DAYS = 365
DEFAULT_CASH = 100_000
DEFAULT_MARKET_FEE_PCT = 0.1      # percent, as typed in the UI
DEFAULT_LIMIT_FEE_PCT = 0.05

TREND_OVERLAY_LABEL = "Trend overlay"
TREND_OVERLAY_TIP = (
    "Only hold positions while the trailing 28-bar return is positive, otherwise sit in cash.\n"
    "Evidence (Phases 6.7-6.9): reduced the worst drawdown in 3 tests; did NOT show higher return.\n"
    "Also stops the strategy from shorting. Validated on daily bars, crypto only."
)

# ---------------------------------------------------------------------------------------------------- workspaces
# Both front ends group their tabs into four workspaces (design slice 2). A tab not listed here shows in "Trade".
WORKSPACES = ["Trade", "Research", "Automate", "Markets"]
WORKSPACE_TIPS = {
    "Trade": "Act on a market now: chart, order ticket, positions, orders, risk",
    "Research": "Test an idea before trusting it: backtest, equity curve, Research Lab, Research Loop",
    "Automate": "Let a rule trade and watch it: paper runner, replay, agents",
    "Markets": "Understand what is moving: news, market context, options",
}
TAB_WORKSPACE = {
    "Positions": "Trade", "Orders": "Trade", "PnL Calendar": "Trade",
    "Equity Curve": "Research", "Research Lab": "Research", "Research Loop": "Research",
    "Execution": "Automate", "Agent Monitor": "Automate",
    "News": "Markets", "Market Context": "Markets", "News & Earnings": "Markets", "Options Chain": "Markets",
}


def workspace_for(tab_title: str) -> str:
    """Workspace a tab belongs to (by its visible title; '&&' is Qt's escaped '&')."""
    return TAB_WORKSPACE.get(str(tab_title).replace("&&", "&").strip(), "Trade")
