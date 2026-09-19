"""Single source of truth for the choices both front ends offer (desktop PyQt5 app and Dash web view).

Both UIs import from here so they cannot drift apart (they had: the desktop offered ADAUSDT/GOLD, Dash did not).
Change a list here and both UIs change; tests/test_ui_parity.py fails if either stops using it.
"""
from __future__ import annotations

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "AAPL", "TSLA", "GOLD", "SPY", "QQQ"]
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
