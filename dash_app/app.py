"""
dash_app/app.py

Entry point for the AlgoTrader Dash web application.

Run standalone:
    python dash_app/app.py

Serves on http://127.0.0.1:8050 by default (localhost only).

IMPORTANT: This file must be run from the project root so that ``core/``
and ``brokers/`` are on the Python path:

    cd /path/to/Algorithmic_Trading_Complete_withGUI
    python dash_app/app.py
"""

from __future__ import annotations

import os
import sys

# Ensure the project root is on sys.path so sibling packages (core/, brokers/,
# strategies/, …) are importable when this file is run directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import dash
import dash_bootstrap_components as dbc

from dash_app.layout import build_layout
from dash_app.callbacks import register_callbacks

# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    # DBC dark theme provides a sensible base; our custom inline CSS (layout.py)
    # overrides colors to match the PyQt5 palette exactly.
    external_stylesheets=[dbc.themes.DARKLY],
    title="AlgoTrader",
    # Suppress callback exceptions so placeholder IDs from future phases don't
    # raise errors in Phase 1.1.
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

app.layout = build_layout()

register_callbacks(app)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    host = os.getenv("ALGOTRADER_HOST", "127.0.0.1")
    port = int(os.getenv("ALGOTRADER_PORT", "8050"))
    debug = os.getenv("ALGOTRADER_DEBUG", "false").lower() == "true"

    print(f"AlgoTrader Dash app starting on http://{host}:{port}/")
    app.run(host=host, port=port, debug=debug)
