"""Regression: a desktop backtest must never touch the durable paper account.

``MainWindow.run_backtest`` used to borrow ``broker_manager.get_broker("Simulator")`` -- in the app that is the
persistent paper account shared with Dash and the paper execution service -- and overwrite its balance, positions,
orders and fees, then make it the Account panel's broker showing the backtest's Final Value. The positions/orders
came back on the next account operation (it reloads from sqlite), but the backtest fee settings did not: they were
charged on the next real paper order and written to the account file.

Runs offscreen; skipped where PyQt5 is not installed (the CI runner).
"""
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PyQt5.QtWebEngineWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PyQt5 import QtWebEngineWidgets  # noqa: F401  (must be imported before QApplication)
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _file_state(path):
    c = sqlite3.connect(path)
    try:
        return {"positions": c.execute("SELECT symbol, qty, avg_price FROM positions ORDER BY symbol").fetchall(),
                "orders": c.execute("SELECT COUNT(*) FROM orders").fetchone()[0],
                "meta": dict(c.execute("SELECT key, value FROM meta"))}
    finally:
        c.close()


def _mem_state(b):
    return {"positions": {s: (p.qty, p.avg_price) for s, p in b.positions.items()}, "orders": len(b.order_history),
            "balance": b.balance, "initial_balance": b.initial_balance,
            "market_fee": b.market_fee, "limit_fee": b.limit_fee}


@pytest.mark.parametrize("panel_on_paper", [False, True])
def test_backtest_leaves_paper_account_and_account_panel_alone(qapp, tmp_path, monkeypatch, panel_on_paper):
    from unittest.mock import MagicMock
    from core.broker_manager import BrokerManager
    from core.strategy_manager import StrategyManager
    from ui.main_window import MainWindow

    acct = str(tmp_path / "paper_account.sqlite3")
    monkeypatch.setenv("PAPER_ACCOUNT_PATH", acct)
    monkeypatch.setenv("EXECUTION_DB_PATH", str(tmp_path / "exec.sqlite3"))
    bm = BrokerManager(paper_account_path=acct)
    sim = bm.get_broker("Simulator")
    sim.update_price("BTCUSDT", 50000.0)
    sim.submit_order("BTCUSDT", 0.5, "buy")
    mem_before, file_before = _mem_state(sim), _file_state(acct)

    win = MainWindow(data_loader=MagicMock(), strategy_manager=StrategyManager(), broker_manager=bm, missing_deps=[])
    try:
        win.broker_combo.setCurrentText("Simulator")
        if panel_on_paper:
            win.current_broker = sim
            win.refresh_account_info()
        panel_before, label_before = win.current_broker, win.account_label.text()

        idx = pd.date_range("2024-01-01", periods=300, freq="D")
        close = 100 + np.cumsum(np.sin(np.arange(300) / 7.0) * 2)
        win.df = pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1000.0},
                              index=idx)
        win.plot_signals = lambda *_: None            # no chart loaded in this test
        win._show_equity_curve = lambda *_: None
        win.strategy_combo.setCurrentText("EMA Crossover")
        win.cash_input.setText("10000")
        win.market_fee_input.setText("0.5")           # differs from the account's 0.1% on purpose
        win.limit_fee_input.setText("0.25")

        win.run_backtest()

        assert win.statusBar().currentMessage().startswith("Backtest complete"), win.statusBar().currentMessage()
        assert _mem_state(sim) == mem_before
        assert _file_state(acct) == file_before
        assert win.current_broker is panel_before     # Account panel keeps the selected broker
        if panel_on_paper:
            assert "$74,975.00" in win.account_label.text() and win.account_label.text() == label_before

        # The next real paper order is charged the account's own fee, not the backtest's.
        sim.update_price("ETHUSDT", 3000.0)
        sim.submit_order("ETHUSDT", 1, "buy")
        assert sim.market_fee == mem_before["market_fee"]
        assert sim.balance == pytest.approx(mem_before["balance"] - 3000.0 * (1 + mem_before["market_fee"]))
        assert _file_state(acct)["meta"]["market_fee"] == file_before["meta"]["market_fee"]
    finally:
        win.destroy()
