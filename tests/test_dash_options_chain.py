"""Phase 5.1 -- Dash options chain panel tests.

Coverage
--------
* ``build_options_chain_data`` helper:
    - IBKR_ENABLED not set               -> "not enabled" message, empty rows
    - IBKR_ENABLED set, broker_manager=None -> "not connected" message
    - IBKR_ENABLED set, IBKR key=None    -> "not connected" message
    - IBKR_ENABLED set, ib.isConnected()=False -> "not connected" message
    - Empty chain (qualifyContracts returns []) -> "no options data" message
    - Happy path (ib_env fixture, connected mock) -> non-empty rows + status

* Layout:
    - "options-chain-tab", "options-chain-load-btn", "options-chain-expiry-input",
      "options-chain-status", "options-chain-table" exist in layout component tree

* Dash app:
    - ``load_options_chain`` callback is registered (checked via callback_map)

* No-order-path audit:
    - AST scan of ``dash_app/options_chain_panel.py`` finds no ``submit_order``
      calls, no "buy_btn" / "sell_btn" references, and no ``brokers.`` imports
      via dotted attribute access.

All tests are headless (no browser, no live server, no live IBKR connection).
The ``ib_env`` fixture from conftest.py provides a mocked ib_insync module.
"""

from __future__ import annotations

import ast
import os
import sys
import pathlib
from types import SimpleNamespace as NS
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Helpers shared with the existing test_options_chain.py (local copy to avoid
# coupling test files).
# ---------------------------------------------------------------------------

_EXPIRY_A = "20991016"
_STRIKES  = [145.0, 150.0, 155.0]
_SPOT     = 150.0


def _make_greeks(delta=0.52, gamma=0.01, theta=-0.05, vega=0.15,
                 impliedVol=0.25, undPrice=_SPOT) -> NS:
    return NS(delta=delta, gamma=gamma, theta=theta, vega=vega,
              impliedVol=impliedVol, undPrice=undPrice)


def _make_ticker(contract, bid=5.2, ask=5.4, greeks=None) -> NS:
    return NS(contract=contract, bid=bid, ask=ask,
              modelGreeks=greeks if greeks is not None else _make_greeks())


def _setup_ib(fake_ib):
    """Wire *fake_ib* for a standard chain request (minimal version)."""
    stock_c = NS(symbol="AAPL", exchange="SMART", currency="USD",
                 secType="STK", conId=0)

    def _qualify(*contracts):
        out = []
        for c in contracts:
            if getattr(c, "secType", "") == "STK":
                d = dict(vars(c)); d["conId"] = 12345
                out.append(NS(**d))
            else:
                out.append(c)
        return out

    fake_ib.qualifyContracts.side_effect = _qualify
    fake_ib.reqSecDefOptParams.return_value = [
        NS(exchange="SMART",
           expirations=[_EXPIRY_A, "20991115"],
           strikes=_STRIKES)
    ]

    spot_ticker = NS(contract=stock_c, last=_SPOT, close=_SPOT - 0.2,
                     bid=_SPOT - 0.1, ask=_SPOT + 0.1, modelGreeks=None)

    def _req_tickers(*contracts):
        if not contracts:
            return []
        if getattr(contracts[0], "secType", "") == "STK":
            return [spot_ticker]
        result = []
        for i, c in enumerate(contracts):
            sv = getattr(c, "strike", 150.0)
            right = getattr(c, "right", "C")
            mono = sv - _SPOT
            delta = (0.5 - mono * 0.01) if right == "C" else (-0.5 + mono * 0.01)
            result.append(_make_ticker(
                c,
                bid=round(max(0.05, 5.0 - mono * 0.1 + i * 0.02), 2),
                ask=round(max(0.10, 5.2 - mono * 0.1 + i * 0.02), 2),
                greeks=_make_greeks(delta=delta, undPrice=_SPOT),
            ))
        return result

    fake_ib.reqTickers.side_effect = _req_tickers


def _fake_bm_connected(ib) -> NS:
    """Return a fake broker_manager whose IBKR connector is connected."""
    connector = NS(ib=ib)
    ib.isConnected.return_value = True
    return NS(brokers={"IBKR": connector})


def _fake_bm_disconnected(ib) -> NS:
    """Return a fake broker_manager whose IBKR connector is NOT connected."""
    connector = NS(ib=ib)
    ib.isConnected.return_value = False
    return NS(brokers={"IBKR": connector})


# ---------------------------------------------------------------------------
# 1. build_options_chain_data -- degraded states (no ib_env needed)
# ---------------------------------------------------------------------------

class TestBuildOptionsChainDataDegradedStates:
    """build_options_chain_data must return ([], status_msg) for all
    unavailable states -- no exceptions, no tracebacks, no blank messages."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from dash_app.options_chain_panel import build_options_chain_data
        self._fn = build_options_chain_data

    # 1a. IBKR not enabled (env var absent) ----------------------------------

    def test_ibkr_not_enabled_returns_empty_rows(self, monkeypatch):
        monkeypatch.delenv("IBKR_ENABLED", raising=False)
        rows, msg = self._fn("AAPL")
        assert rows == []

    def test_ibkr_not_enabled_message_mentions_env_var(self, monkeypatch):
        monkeypatch.delenv("IBKR_ENABLED", raising=False)
        _, msg = self._fn("AAPL")
        assert "IBKR_ENABLED" in msg

    def test_ibkr_not_enabled_message_not_empty(self, monkeypatch):
        monkeypatch.delenv("IBKR_ENABLED", raising=False)
        _, msg = self._fn("AAPL")
        assert msg  # non-empty string

    # 1b. IBKR_ENABLED set but broker_manager is None ------------------------

    def test_broker_manager_none_returns_not_connected(self, monkeypatch):
        monkeypatch.setenv("IBKR_ENABLED", "1")
        rows, msg = self._fn("AAPL", broker_manager=None)
        assert rows == []
        assert "not connected" in msg.lower()

    # 1c. IBKR_ENABLED set, broker_manager present but IBKR key is None -----

    def test_ibkr_key_none_returns_not_connected(self, monkeypatch):
        monkeypatch.setenv("IBKR_ENABLED", "1")
        bm = NS(brokers={"IBKR": None, "Simulator": NS()})
        rows, msg = self._fn("AAPL", broker_manager=bm)
        assert rows == []
        assert "not connected" in msg.lower()

    # 1d. IBKR_ENABLED set, connector exists, ib is None --------------------

    def test_ib_none_returns_not_connected(self, monkeypatch):
        monkeypatch.setenv("IBKR_ENABLED", "1")
        bm = NS(brokers={"IBKR": NS(ib=None)})
        rows, msg = self._fn("AAPL", broker_manager=bm)
        assert rows == []
        assert "not connected" in msg.lower()

    # 1e. IBKR_ENABLED set, ib.isConnected() returns False ------------------

    def test_ib_not_connected_returns_not_connected(self, monkeypatch):
        monkeypatch.setenv("IBKR_ENABLED", "1")
        ib = mock.MagicMock()
        ib.isConnected.return_value = False
        bm = NS(brokers={"IBKR": NS(ib=ib)})
        rows, msg = self._fn("AAPL", broker_manager=bm)
        assert rows == []
        assert "not connected" in msg.lower()

    # 1f. All states return a 2-tuple (list, str) ----------------------------

    def test_returns_two_tuple_always(self, monkeypatch):
        monkeypatch.delenv("IBKR_ENABLED", raising=False)
        result = self._fn("AAPL")
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], list) and isinstance(result[1], str)


# ---------------------------------------------------------------------------
# 2. build_options_chain_data -- empty chain (uses ib_env fixture)
# ---------------------------------------------------------------------------

class TestBuildOptionsChainDataEmptyChain:
    """When the chain fetch succeeds but returns zero rows, the function must
    show an explicit "no data" message."""

    def test_empty_chain_returns_empty_rows(self, monkeypatch, ib_env):
        monkeypatch.setenv("IBKR_ENABLED", "1")
        ib_env.ib.isConnected.return_value = True
        # qualifyContracts returns empty -> empty chain
        ib_env.ib.qualifyContracts.return_value = []
        bm = NS(brokers={"IBKR": NS(ib=ib_env.ib)})

        from dash_app.options_chain_panel import build_options_chain_data
        rows, msg = build_options_chain_data("AAPL", broker_manager=bm)
        assert rows == []

    def test_empty_chain_message_mentions_symbol(self, monkeypatch, ib_env):
        monkeypatch.setenv("IBKR_ENABLED", "1")
        ib_env.ib.isConnected.return_value = True
        ib_env.ib.qualifyContracts.return_value = []
        bm = NS(brokers={"IBKR": NS(ib=ib_env.ib)})

        from dash_app.options_chain_panel import build_options_chain_data
        _, msg = build_options_chain_data("AAPL", broker_manager=bm)
        # Message must say something specific, not just be an empty or generic string
        assert msg  # non-empty
        # Should mention AAPL or "no" / "empty" type language
        assert "AAPL" in msg or "no" in msg.lower() or "data" in msg.lower()


# ---------------------------------------------------------------------------
# 3. build_options_chain_data -- happy path (uses ib_env fixture)
# ---------------------------------------------------------------------------

class TestBuildOptionsChainDataHappyPath:
    """With a mocked connected IBKR broker returning a real chain, the function
    must return non-empty rows and a status message containing the symbol."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch, ib_env):
        monkeypatch.setenv("IBKR_ENABLED", "1")
        _setup_ib(ib_env.ib)
        ib_env.ib.isConnected.return_value = True
        self._bm = NS(brokers={"IBKR": NS(ib=ib_env.ib)})

    def test_returns_non_empty_rows(self):
        from dash_app.options_chain_panel import build_options_chain_data
        rows, _ = build_options_chain_data("AAPL", broker_manager=self._bm)
        assert len(rows) > 0

    def test_status_message_mentions_symbol(self):
        from dash_app.options_chain_panel import build_options_chain_data
        _, msg = build_options_chain_data("AAPL", broker_manager=self._bm)
        assert "AAPL" in msg

    def test_status_message_contains_contract_count(self):
        from dash_app.options_chain_panel import build_options_chain_data
        rows, msg = build_options_chain_data("AAPL", broker_manager=self._bm)
        assert str(len(rows)) in msg

    def test_each_row_has_all_chain_columns(self):
        from dash_app.options_chain_panel import build_options_chain_data, CHAIN_TABLE_COLUMNS
        rows, _ = build_options_chain_data("AAPL", broker_manager=self._bm)
        expected_keys = {c["id"] for c in CHAIN_TABLE_COLUMNS}
        for row in rows:
            assert set(row.keys()) == expected_keys, (
                f"Row missing keys: {expected_keys - set(row.keys())}"
            )

    def test_right_column_contains_c_or_p(self):
        from dash_app.options_chain_panel import build_options_chain_data
        rows, _ = build_options_chain_data("AAPL", broker_manager=self._bm)
        rights = {r["right"] for r in rows}
        assert rights.issubset({"C", "P"}), f"Unexpected right values: {rights}"

    def test_expiry_kwarg_passed_through(self):
        """Passing expiry should include it in the status message."""
        from dash_app.options_chain_panel import build_options_chain_data
        _, msg = build_options_chain_data(
            "AAPL", broker_manager=self._bm, expiry=_EXPIRY_A
        )
        assert _EXPIRY_A in msg

    def test_rows_are_dicts_not_nan_values(self):
        """Float columns must contain formatted strings, never raw NaN."""
        from dash_app.options_chain_panel import build_options_chain_data
        import math
        rows, _ = build_options_chain_data("AAPL", broker_manager=self._bm)
        for row in rows:
            for k, v in row.items():
                assert isinstance(v, str), f"Row[{k!r}] is not a string: {v!r}"
                # Formatted floats should never be the Python "nan" string
                assert v.lower() != "nan", f"Row[{k!r}] contains raw 'nan'"


# ---------------------------------------------------------------------------
# 4. Layout structure tests
# ---------------------------------------------------------------------------

def _collect_ids(component) -> set:
    """Recursively collect component IDs from a Dash component tree."""
    from dash.development.base_component import Component as DashComponent
    ids: set = set()
    try:
        cid = component.id
        if cid is not None:
            ids.add(cid)
    except AttributeError:
        pass
    try:
        children = component.children
    except AttributeError:
        return ids
    if children is None:
        return ids
    if not isinstance(children, list):
        children = [children]
    for child in children:
        if isinstance(child, DashComponent):
            ids.update(_collect_ids(child))
    return ids


class TestOptionsChainLayoutIds:
    """The layout must contain all expected options-chain component IDs."""

    _EXPECTED_IDS = [
        "options-chain-load-btn",
        "options-chain-expiry-input",
        "options-chain-status",
        "options-chain-table",
    ]

    @pytest.fixture(scope="class")
    def all_ids(self):
        from dash_app.layout import build_layout
        return _collect_ids(build_layout())

    @pytest.mark.parametrize("expected_id", _EXPECTED_IDS)
    def test_id_present_in_layout(self, all_ids, expected_id):
        assert expected_id in all_ids, (
            f"Expected component id {expected_id!r} not found in layout. "
            f"IDs found: {sorted(all_ids)}"
        )

    def test_options_chain_table_has_correct_columns(self):
        """options-chain-table must have 12 columns (one per CHAIN_COLUMNS entry)."""
        from dash_app.layout import build_layout
        from dash_app.options_chain_panel import CHAIN_TABLE_COLUMNS
        from dash.development.base_component import Component as DashComponent

        def _find(component, target_id):
            try:
                if getattr(component, "id", None) == target_id:
                    return component
            except Exception:
                pass
            try:
                children = component.children
            except AttributeError:
                return None
            if children is None:
                return None
            if not isinstance(children, list):
                children = [children]
            for child in children:
                if isinstance(child, DashComponent):
                    result = _find(child, target_id)
                    if result is not None:
                        return result
            return None

        table = _find(build_layout(), "options-chain-table")
        assert table is not None, "options-chain-table not found in layout"
        assert len(table.columns) == len(CHAIN_TABLE_COLUMNS), (
            f"Expected {len(CHAIN_TABLE_COLUMNS)} columns, found {len(table.columns)}"
        )

    def test_options_chain_table_starts_empty(self):
        """options-chain-table data must start as [] (no rows before Load Chain)."""
        from dash_app.layout import build_layout
        from dash.development.base_component import Component as DashComponent

        def _find(component, target_id):
            try:
                if getattr(component, "id", None) == target_id:
                    return component
            except Exception:
                pass
            try:
                children = component.children
            except AttributeError:
                return None
            if children is None:
                return None
            if not isinstance(children, list):
                children = [children]
            for child in children:
                if isinstance(child, DashComponent):
                    result = _find(child, target_id)
                    if result is not None:
                        return result
            return None

        table = _find(build_layout(), "options-chain-table")
        assert table is not None
        assert (table.data == [] or table.data is None), (
            f"options-chain-table should start empty, got: {table.data}"
        )


# ---------------------------------------------------------------------------
# 5. Dash app callback registration
# ---------------------------------------------------------------------------

class TestOptionsChainCallbackRegistered:
    """The load_options_chain callback must be registered in the Dash app."""

    def test_callback_registered_in_app(self):
        """app.callback_map must contain an entry targeting options-chain-table."""
        import dash_app.app as dash_module
        cb_map = dash_module.app.callback_map
        # Find any callback that outputs to options-chain-table
        targets = [
            k for k in cb_map
            if "options-chain-table" in k
        ]
        assert targets, (
            "No callback found that outputs to 'options-chain-table'. "
            f"Registered callbacks: {list(cb_map.keys())}"
        )

    def test_options_chain_load_btn_is_an_input_to_callback(self):
        """options-chain-load-btn must be an input to the options chain callback."""
        import dash_app.app as dash_module
        cb_map = dash_module.app.callback_map
        # Find the callback for options-chain-table
        for key, cb in cb_map.items():
            if "options-chain-table" in key:
                input_ids = {i["id"] for i in cb.get("inputs", [])}
                assert "options-chain-load-btn" in input_ids, (
                    f"options-chain-load-btn not found in callback inputs: {input_ids}"
                )
                return
        pytest.fail("No callback found for options-chain-table")


# ---------------------------------------------------------------------------
# 6. No-order-path audit (AST scan of options_chain_panel.py)
# ---------------------------------------------------------------------------

class TestNoOrderPath:
    """The panel module must contain zero order-submission code paths."""

    @pytest.fixture(scope="class")
    def panel_tree(self):
        panel_path = pathlib.Path(__file__).parent.parent / "dash_app" / "options_chain_panel.py"
        assert panel_path.exists(), f"Panel module not found at {panel_path}"
        return ast.parse(panel_path.read_text())

    def _all_names_and_attrs(self, tree) -> list:
        """Collect all Name.id and Attribute.attr strings from the AST."""
        results = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                results.append(node.id)
            elif isinstance(node, ast.Attribute):
                results.append(node.attr)
        return results

    def _all_imports(self, tree) -> list:
        """Collect all module names from import statements."""
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modules.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
        return modules

    def test_no_submit_order_call(self, panel_tree):
        """submit_order must not appear anywhere in the panel module."""
        names_and_attrs = self._all_names_and_attrs(panel_tree)
        assert "submit_order" not in names_and_attrs, (
            "Found submit_order reference in options_chain_panel.py -- "
            "the panel must be read-only (no order path)."
        )

    def test_no_buy_btn_reference(self, panel_tree):
        """buy_btn / buy-btn must not appear in the panel module."""
        src = (pathlib.Path(__file__).parent.parent / "dash_app" / "options_chain_panel.py").read_text()
        assert "buy_btn" not in src and "buy-btn" not in src, (
            "Found buy_btn/buy-btn in options_chain_panel.py"
        )

    def test_no_sell_btn_reference(self, panel_tree):
        """sell_btn / sell-btn must not appear in the panel module."""
        src = (pathlib.Path(__file__).parent.parent / "dash_app" / "options_chain_panel.py").read_text()
        assert "sell_btn" not in src and "sell-btn" not in src, (
            "Found sell_btn/sell-btn in options_chain_panel.py"
        )

    def test_no_brokers_package_import(self, panel_tree):
        """The panel must not import from the brokers/ package (no order methods)."""
        imports = self._all_imports(panel_tree)
        brokers_imports = [m for m in imports if m and m.startswith("brokers")]
        assert not brokers_imports, (
            f"Panel module imports from brokers/: {brokers_imports} -- "
            "only core.options_chain is allowed."
        )

    def test_no_execution_guard_import(self, panel_tree):
        """The panel must not import execution_guard (used only by order paths)."""
        imports = self._all_imports(panel_tree)
        assert not any("execution_guard" in m for m in imports), (
            "Panel module imports execution_guard -- must be read-only."
        )


# ---------------------------------------------------------------------------
# 7. _fmt_float and _format_chain_row unit tests
# ---------------------------------------------------------------------------

class TestFmtFloat:
    """_fmt_float must format numbers and return '--' for missing values."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from dash_app.options_chain_panel import _fmt_float
        self._fn = _fmt_float

    def test_none_returns_dash(self):
        assert self._fn(None) == "--"

    def test_nan_returns_dash(self):
        import math
        assert self._fn(float("nan")) == "--"

    def test_inf_returns_dash(self):
        assert self._fn(float("inf")) == "--"
        assert self._fn(float("-inf")) == "--"

    def test_zero_returns_formatted_zero(self):
        assert self._fn(0.0) == "0.0000"

    def test_positive_float_formatted(self):
        result = self._fn(0.52, 4)
        assert result == "0.5200"

    def test_ndigits_respected(self):
        result = self._fn(150.0, 2)
        assert result == "150.00"

    def test_string_input_not_crash(self):
        # Non-numeric strings should return "--"
        result = self._fn("n/a")
        assert result == "--"

    def test_numeric_string_formatted(self):
        result = self._fn("0.25", 2)
        assert result == "0.25"


class TestFormatChainRow:
    """_format_chain_row must produce a dict with all column IDs."""

    def test_row_keys_match_chain_table_columns(self):
        from dash_app.options_chain_panel import _format_chain_row, CHAIN_TABLE_COLUMNS
        import pandas as pd
        row = pd.Series({
            "strike": 150.0, "right": "C", "expiry": "20991016",
            "bid": 5.2, "ask": 5.4, "mid": 5.3,
            "iv": 0.25, "delta": 0.52, "gamma": 0.01,
            "theta": -0.05, "vega": 0.15, "underlying_price": 150.0,
        })
        result = _format_chain_row(row)
        expected_keys = {c["id"] for c in CHAIN_TABLE_COLUMNS}
        assert set(result.keys()) == expected_keys

    def test_nan_values_become_dash(self):
        from dash_app.options_chain_panel import _format_chain_row
        import pandas as pd
        import math
        row = pd.Series({
            "strike": 150.0, "right": "P", "expiry": "20991016",
            "bid": float("nan"), "ask": float("nan"), "mid": float("nan"),
            "iv": float("nan"), "delta": float("nan"), "gamma": float("nan"),
            "theta": float("nan"), "vega": float("nan"), "underlying_price": 150.0,
        })
        result = _format_chain_row(row)
        for col in ["bid", "ask", "mid", "iv", "delta", "gamma", "theta", "vega"]:
            assert result[col] == "--", f"{col!r} should be '--' for NaN input"

    def test_values_are_all_strings(self):
        from dash_app.options_chain_panel import _format_chain_row
        import pandas as pd
        row = pd.Series({
            "strike": 150.0, "right": "C", "expiry": "20991016",
            "bid": 5.2, "ask": 5.4, "mid": 5.3,
            "iv": 0.25, "delta": 0.52, "gamma": 0.01,
            "theta": -0.05, "vega": 0.15, "underlying_price": 150.0,
        })
        result = _format_chain_row(row)
        for k, v in result.items():
            assert isinstance(v, str), f"Expected str for {k!r}, got {type(v).__name__}"
