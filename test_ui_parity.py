"""Guards against the desktop app and the Dash web view drifting apart.

Shared choices live in core/ui_options.py and core/strategy_manager.py; the desktop half of these checks is in test_gui.py
(it needs Qt). Known, documented gaps between the two are listed in docs/PHASE_3_1_FEATURE_PARITY.md.
"""
import pytest

from core import ui_options
from core.strategy_manager import backtrader_strategies, build_strategy_registry


def _walk(node):
    yield node
    kids = getattr(node, "children", None)
    if kids is None:
        return
    for k in (kids if isinstance(kids, (list, tuple)) else [kids]):
        if hasattr(k, "children") or hasattr(k, "id"):
            yield from _walk(k)


def _by_id(layout):
    return {n.id: n for n in _walk(layout) if getattr(n, "id", None)}


@pytest.fixture(scope="module")
def dash_module():
    import dash_app.app as m
    return m


def test_dash_dropdowns_use_the_shared_lists(dash_module):
    ids = _by_id(dash_module.app.layout)
    assert [o["value"] for o in ids["symbol-dropdown"].options] == ui_options.SYMBOLS
    assert [o["value"] for o in ids["interval-dropdown"].options] == ui_options.INTERVALS
    assert [o["value"] for o in ids["strategy-dropdown"].options] == ["None"] + list(backtrader_strategies())


def test_dash_has_the_controls_the_desktop_app_has(dash_module):
    ids = _by_id(dash_module.app.layout)
    for needed in ("days-input", "bt-cash-input", "bt-mkt-fee-input", "bt-lim-fee-input", "trend-overlay-check"):
        assert needed in ids, f"Dash is missing {needed}"
    assert ids["days-input"].value == ui_options.DEFAULT_DAYS
    assert ids["bt-cash-input"].value == ui_options.DEFAULT_CASH
    assert ids["bt-mkt-fee-input"].value == ui_options.DEFAULT_MARKET_FEE_PCT
    assert ids["bt-lim-fee-input"].value == ui_options.DEFAULT_LIMIT_FEE_PCT


def test_dash_backtest_uses_the_chart_interval_days_and_fees(dash_module):
    states = {}
    for key, cb in dash_module.app.callback_map.items():
        if "bt-sharpe" in key:
            states = {s["id"] for s in cb["state"]}
    assert {"interval-dropdown", "days-input", "bt-mkt-fee-input", "bt-lim-fee-input", "trend-overlay-check"} <= states
    load_states = {s["id"] for k, cb in dash_module.app.callback_map.items() if "main-chart" in k and "chart-status" in k for s in cb["state"]}
    assert "days-input" in load_states


def test_days_and_fee_helpers():
    from dash_app.callbacks import _days_or_default, _fee_fraction
    assert _days_or_default(30) == 30 and _days_or_default("90") == 90
    assert _days_or_default(None) == ui_options.DEFAULT_DAYS and _days_or_default(0) == ui_options.DEFAULT_DAYS
    assert _days_or_default(99999) == ui_options.DEFAULT_DAYS and _days_or_default("x") == ui_options.DEFAULT_DAYS
    assert _fee_fraction(0.1, 0.5) == pytest.approx(0.001)
    assert _fee_fraction(None, 0.05) == pytest.approx(0.0005)
    assert _fee_fraction(-3, 0.05) == 0.0 and _fee_fraction(50, 0.05) == pytest.approx(0.10)


def test_every_backtrader_strategy_is_in_the_shared_registry_and_dash_can_run_it(dash_module):
    reg = build_strategy_registry()
    assert set(backtrader_strategies()) <= set(reg)
    for name in backtrader_strategies():
        assert name in [o["value"] for o in _by_id(dash_module.app.layout)["strategy-dropdown"].options]


def test_research_loop_view_is_shared_between_the_uis():
    from dash_app.callbacks import _research_loop_view
    from core.research_loop import research_loop_view
    assert _research_loop_view().keys() == research_loop_view().keys()
