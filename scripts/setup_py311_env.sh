#!/usr/bin/env bash
# Create the Python 3.11 environment (needed for NautilusTrader, Phase 8.0) WITHOUT touching the Python 3.9
# conda base env that the orchestrator / Telegram bot / collectors run on.
#
#   scripts/setup_py311_env.sh            # creates ~/.venvs/algotrader311
#   ~/.venvs/algotrader311/bin/python -m pytest --ignore=test_gui.py -q
#
# Notes (learned the hard way, 2026-09-19):
#  * lightgbm needs the OpenMP runtime on macOS:  brew install libomp   (conda bundled it; a plain venv does not)
#  * pandas-ta has no release for Python < 3.12 -> skipped here (optional dependency, code guards the import)
#  * yfinance must stay < 0.2.60 (see requirements.txt), and openbb pulls a newer one, so it is re-pinned last
set -euo pipefail
VENV="${VENV:-$HOME/.venvs/algotrader311}"
PY311="${PY311:-$HOME/.pyenv/versions/3.11.9/bin/python}"
command -v uv >/dev/null || { echo "uv not found (brew install uv)"; exit 1; }
[ -x "$PY311" ] || { echo "Python 3.11 not found at $PY311 (pyenv install 3.11.9)"; exit 1; }
[ -e /opt/homebrew/opt/libomp ] || { echo "Missing libomp: run 'brew install libomp' first"; exit 1; }

uv venv "$VENV" --python "$PY311"
uv pip install --python "$VENV/bin/python" \
  nautilus_trader \
  yfinance pandas numpy scikit-learn lightgbm ta backtrader requests python-dotenv newsapi-python beautifulsoup4 \
  matplotlib plotly scipy statsmodels joblib pytest ccxt websocket-client websockets openai \
  "dash>=2.16,<3" "dash-bootstrap-components>=1.5,<2" \
  PyQt5 PyQtWebEngine PyQtChart alpaca-py python-binance ib-insync \
  openbb openbb-yfinance
uv pip install --python "$VENV/bin/python" "yfinance<0.2.60"
"$VENV/bin/python" -c "import nautilus_trader, lightgbm, yfinance; print('nautilus', nautilus_trader.__version__, '| lightgbm', lightgbm.__version__, '| yfinance', yfinance.__version__)"
