"""
test_dash_app.py

Headless, network-free test coverage for the AlgoTrader Dash web app:
  - Layout structure: build_layout() produces the expected component IDs.
  - toggle_price_input logic: _price_input_style_and_placeholder() pure helper.
  - submit_order logic: _validate_and_submit_order() pure helper.
  - App import sanity: dash_app.app imports cleanly without starting a server.

Style convention follows test_dash_live_price.py — no Selenium, no dash.testing
browser driver, no live server, no network calls.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_ids(component) -> set:
    """Recursively walk a Dash component tree and return the set of all IDs.

    Handles the three forms of ``.children``:
      - None          (leaf node with no children)
      - str / number  (text content — no ID)
      - Component     (single child)
      - list          (multiple children, possibly mixed with strings)

    Important: components such as ``dcc.Graph``, ``dcc.Store``,
    ``dcc.Interval``, ``dcc.Dropdown``, and ``dcc.Input`` do NOT have a
    ``children`` prop (it's not in their ``_prop_names``).  Accessing
    ``.children`` on them raises ``AttributeError``.  This walker handles that
    by catching the ``AttributeError`` when recursing, and by using
    ``isinstance(child, DashComponent)`` (not ``hasattr(child, "children")``)
    to filter out plain strings/numbers from child lists.
    """
    from dash.development.base_component import Component as DashComponent

    ids: set = set()

    # Try to get this component's own ID.  Not all Dash components expose
    # ``id`` (e.g. plain html.Div with no id= kwarg), so guard with try/except.
    try:
        cid = component.id
        if cid is not None:
            ids.add(cid)
    except AttributeError:
        pass

    # Recurse into children.  Raises AttributeError for dcc.* components that
    # have no ``children`` prop — treat those as leaf nodes.
    try:
        children = component.children
    except AttributeError:
        return ids

    if children is None:
        return ids

    # Normalise to a list so the loop below handles both single-child and
    # multi-child cases uniformly.
    if not isinstance(children, list):
        children = [children]

    for child in children:
        # Skip plain strings / numbers that are not Dash components.
        if not isinstance(child, DashComponent):
            continue
        ids.update(_collect_ids(child))

    return ids


# ---------------------------------------------------------------------------
# 1. Layout structure
# ---------------------------------------------------------------------------

class TestLayoutStructure:
    """build_layout() must produce a component tree containing all expected IDs."""

    @pytest.fixture(scope="class")
    def all_ids(self):
        from dash_app.layout import build_layout
        layout = build_layout()
        return _collect_ids(layout)

    # IDs that must be present -----------------------------------------------

    _EXPECTED_IDS = [
        # Order entry panel
        "order-qty-input",
        "order-type-dropdown",
        "order-price-input",
        "order-price-wrapper",
        "buy-btn",
        "sell-btn",
        "order-status",
        # Chart area
        "main-chart",
        "live-badge",
        # Top bar
        "symbol-dropdown",
        "interval-dropdown",
        "strategy-dropdown",
        "load-btn",
        # Stores & interval
        "ohlcv-store",
        "signals-store",
        "active-symbol-store",
        "price-interval",
        # Metrics panel
        "account-balance",
        "pnl-value",
        "bt-sharpe",
        "bt-winrate",
        "bt-maxdd",
        "chart-status",
        # Status bar
        "status-bar",
    ]

    @pytest.mark.parametrize("expected_id", _EXPECTED_IDS)
    def test_expected_id_present(self, all_ids, expected_id):
        assert expected_id in all_ids, (
            f"Expected component id {expected_id!r} not found in layout. "
            f"IDs found: {sorted(all_ids)}"
        )

    def test_build_layout_returns_without_raising(self):
        """build_layout() must complete without throwing."""
        from dash_app.layout import build_layout
        layout = build_layout()
        assert layout is not None

    def test_layout_root_has_children(self):
        """Top-level Div must have at least 4 children (stores, topbar, content, statusbar)."""
        from dash_app.layout import build_layout
        layout = build_layout()
        children = layout.children
        if not isinstance(children, list):
            children = [children]
        assert len(children) >= 4

    def test_price_interval_starts_disabled(self, all_ids):
        """price-interval must be disabled on page load (enabled after chart loads)."""
        from dash_app.layout import build_layout

        def _find_interval(component):
            """Find the dcc.Interval component with id 'price-interval'."""
            try:
                if getattr(component, "id", None) == "price-interval":
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
                result = _find_interval(child)
                if result is not None:
                    return result
            return None

        layout = build_layout()
        interval = _find_interval(layout)
        assert interval is not None, "price-interval component not found"
        assert interval.disabled is True, "price-interval should start disabled"

    def test_order_price_wrapper_starts_hidden(self, all_ids):
        """order-price-wrapper must have display:none initially (shown for Limit/Stop)."""
        from dash_app.layout import build_layout

        def _find_by_id(component, target_id):
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
                result = _find_by_id(child, target_id)
                if result is not None:
                    return result
            return None

        layout = build_layout()
        wrapper = _find_by_id(layout, "order-price-wrapper")
        assert wrapper is not None
        assert wrapper.style.get("display") == "none", (
            "order-price-wrapper should start hidden (display: none)"
        )


# ---------------------------------------------------------------------------
# 2. toggle_price_input logic
# ---------------------------------------------------------------------------

class TestTogglePriceInput:
    """_price_input_style_and_placeholder must return correct style + placeholder."""

    @pytest.fixture(autouse=True)
    def _import_helper(self):
        from dash_app.callbacks import _price_input_style_and_placeholder
        self._helper = _price_input_style_and_placeholder

    def test_market_returns_hidden_style(self):
        style, _ = self._helper("market")
        assert style.get("display") == "none"

    def test_market_returns_price_placeholder(self):
        _, placeholder = self._helper("market")
        assert placeholder == "Price"

    def test_limit_returns_block_style(self):
        style, _ = self._helper("limit")
        assert style.get("display") == "block"

    def test_limit_returns_correct_placeholder(self):
        _, placeholder = self._helper("limit")
        assert placeholder == "Limit Price"

    def test_stop_returns_block_style(self):
        style, _ = self._helper("stop")
        assert style.get("display") == "block"

    def test_stop_returns_correct_placeholder(self):
        _, placeholder = self._helper("stop")
        assert placeholder == "Stop Price"

    def test_returns_two_element_tuple(self):
        result = self._helper("market")
        assert len(result) == 2

    def test_unknown_type_defaults_to_hidden(self):
        """Any unrecognised order type falls back to the Market behaviour."""
        style, placeholder = self._helper("unknown_type")
        assert style.get("display") == "none"

    @pytest.mark.parametrize("order_type,expected_display", [
        ("market", "none"),
        ("limit",  "block"),
        ("stop",   "block"),
    ])
    def test_display_values_parametrized(self, order_type, expected_display):
        style, _ = self._helper(order_type)
        assert style.get("display") == expected_display


# ---------------------------------------------------------------------------
# 3. submit_order logic
# ---------------------------------------------------------------------------

class TestValidateAndSubmitOrder:
    """_validate_and_submit_order covers all validation branches + broker call."""

    @pytest.fixture(autouse=True)
    def _import_helpers(self):
        from dash_app.callbacks import _validate_and_submit_order
        from core.chart_builder import THEME
        self._fn = _validate_and_submit_order
        self._red = THEME["red"]
        self._green = THEME["green"]
        self._orange = THEME["orange"]

    # 3a. Validation failures — no broker call expected ----------------------

    def test_no_symbol_returns_error(self):
        broker = MagicMock()
        text, style = self._fn(broker, "buy", 1.0, "market", None, None)
        assert "chart" in text.lower() or "load" in text.lower() or "symbol" in text.lower()
        assert style["color"] == self._red
        broker.submit_order.assert_not_called()

    def test_empty_symbol_returns_error(self):
        broker = MagicMock()
        text, style = self._fn(broker, "buy", 1.0, "market", None, "")
        assert style["color"] == self._red
        broker.submit_order.assert_not_called()

    def test_zero_qty_returns_error(self):
        broker = MagicMock()
        text, style = self._fn(broker, "buy", 0, "market", None, "AAPL")
        assert "qty" in text.lower() or "0" in text
        assert style["color"] == self._red
        broker.submit_order.assert_not_called()

    def test_negative_qty_returns_error(self):
        broker = MagicMock()
        text, style = self._fn(broker, "buy", -5.0, "market", None, "AAPL")
        assert style["color"] == self._red
        broker.submit_order.assert_not_called()

    def test_none_qty_returns_error(self):
        broker = MagicMock()
        text, style = self._fn(broker, "buy", None, "market", None, "AAPL")
        assert style["color"] == self._red
        broker.submit_order.assert_not_called()

    def test_limit_order_with_no_price_returns_error(self):
        broker = MagicMock()
        text, style = self._fn(broker, "buy", 1.0, "limit", None, "AAPL")
        assert "limit" in text.lower() or "price" in text.lower()
        assert style["color"] == self._red
        broker.submit_order.assert_not_called()

    def test_limit_order_with_zero_price_returns_error(self):
        broker = MagicMock()
        text, style = self._fn(broker, "buy", 1.0, "limit", 0.0, "AAPL")
        assert style["color"] == self._red
        broker.submit_order.assert_not_called()

    def test_stop_order_with_no_price_returns_error(self):
        broker = MagicMock()
        text, style = self._fn(broker, "buy", 1.0, "stop", None, "AAPL")
        assert "stop" in text.lower() or "price" in text.lower()
        assert style["color"] == self._red
        broker.submit_order.assert_not_called()

    def test_stop_order_with_zero_price_returns_error(self):
        broker = MagicMock()
        text, style = self._fn(broker, "sell", 1.0, "stop", 0.0, "BTCUSDT")
        assert style["color"] == self._red
        broker.submit_order.assert_not_called()

    # 3b. Valid market orders — use real SimulatedBroker ----------------------

    def test_valid_market_buy_calls_broker_and_returns_ok(self):
        """Market buy on a real SimulatedBroker must fill immediately."""
        from brokers.simulatedbroker import SimulatedBroker
        broker = SimulatedBroker()
        try:
            text, style = self._fn(broker, "buy", 1.0, "market", None, "AAPL")
            # Market orders on SimulatedBroker fill immediately.
            assert "filled" in text.lower() or "BUY" in text, (
                f"Expected filled status, got: {text!r}"
            )
            assert style["color"] == self._green
        finally:
            broker.close()

    def test_valid_market_buy_text_contains_side_and_symbol(self):
        from brokers.simulatedbroker import SimulatedBroker
        broker = SimulatedBroker()
        try:
            text, _ = self._fn(broker, "buy", 2.5, "market", None, "TSLA")
            assert "BUY" in text
            assert "TSLA" in text
        finally:
            broker.close()

    def test_valid_market_sell_fills_or_creates_short(self):
        """Market sell on a fresh broker creates a short position (fills immediately)."""
        from brokers.simulatedbroker import SimulatedBroker
        broker = SimulatedBroker()
        try:
            text, style = self._fn(broker, "sell", 1.0, "market", None, "AAPL")
            # SimulatedBroker allows shorting — should fill.
            assert "filled" in text.lower() or "SELL" in text, (
                f"Expected filled status for sell, got: {text!r}"
            )
            assert style["color"] == self._green
        finally:
            broker.close()

    def test_valid_market_sell_text_contains_sell(self):
        from brokers.simulatedbroker import SimulatedBroker
        broker = SimulatedBroker()
        try:
            text, _ = self._fn(broker, "sell", 1.0, "market", None, "AAPL")
            assert "SELL" in text
        finally:
            broker.close()

    def test_broker_submit_order_called_with_correct_args_for_buy(self):
        """Verify the broker receives the right keyword arguments for a market buy."""
        broker = MagicMock()
        from brokers.simulatedbroker import Order, OrderStatus
        # Construct a realistic filled order for the mock to return.
        mock_order = MagicMock()
        mock_order.status.value = "filled"
        mock_order.filled_avg_price = 150.0
        broker.submit_order.return_value = mock_order

        self._fn(broker, "buy", 3.0, "market", None, "AAPL")

        broker.submit_order.assert_called_once_with(
            symbol="AAPL",
            qty=3.0,
            side="buy",
            order_type="market",
            limit_price=None,
            stop_price=None,
        )

    def test_limit_order_passes_limit_price_to_broker(self):
        """Limit orders must pass limit_price to broker.submit_order."""
        broker = MagicMock()
        mock_order = MagicMock()
        mock_order.status.value = "filled"
        mock_order.filled_avg_price = 145.0
        broker.submit_order.return_value = mock_order

        self._fn(broker, "buy", 1.0, "limit", 145.0, "AAPL")

        _, kwargs = broker.submit_order.call_args
        assert kwargs.get("limit_price") == 145.0
        assert kwargs.get("stop_price") is None

    def test_stop_order_passes_stop_price_to_broker(self):
        """Stop orders must pass stop_price to broker.submit_order."""
        broker = MagicMock()
        mock_order = MagicMock()
        mock_order.status.value = "pending"
        mock_order.filled_avg_price = 0.0
        broker.submit_order.return_value = mock_order

        self._fn(broker, "sell", 1.0, "stop", 140.0, "AAPL")

        _, kwargs = broker.submit_order.call_args
        assert kwargs.get("stop_price") == 140.0
        assert kwargs.get("limit_price") is None

    # 3c. Non-filled order statuses ------------------------------------------

    def test_pending_order_returns_orange(self):
        broker = MagicMock()
        mock_order = MagicMock()
        mock_order.status.value = "pending"
        mock_order.filled_avg_price = 0.0
        broker.submit_order.return_value = mock_order

        text, style = self._fn(broker, "buy", 1.0, "limit", 200.0, "AAPL")
        assert "pending" in text.lower()
        assert style["color"] == self._orange

    def test_rejected_order_returns_red(self):
        broker = MagicMock()
        mock_order = MagicMock()
        mock_order.status.value = "rejected"
        mock_order.filled_avg_price = 0.0
        broker.submit_order.return_value = mock_order

        text, style = self._fn(broker, "buy", 1.0, "market", None, "AAPL")
        assert "rejected" in text.lower()
        assert style["color"] == self._red

    # 3d. Broker exception — must be caught, not propagated ------------------

    def test_broker_exception_returns_error_message(self):
        """If broker.submit_order raises, the exception must be caught and
        reported as an error string — it must NOT propagate to the caller."""
        broker = MagicMock()
        broker.submit_order.side_effect = RuntimeError("WebSocket disconnected")

        text, style = self._fn(broker, "buy", 1.0, "market", None, "AAPL")

        # Must not raise; instead return an error message.
        assert "error" in text.lower() or "Order error" in text, (
            f"Expected an error message, got: {text!r}"
        )
        assert style["color"] == self._red

    def test_broker_exception_does_not_propagate(self):
        """Calling _validate_and_submit_order must never raise to the caller."""
        broker = MagicMock()
        broker.submit_order.side_effect = Exception("Unexpected crash")

        # Must not raise.
        result = self._fn(broker, "sell", 1.0, "market", None, "BTCUSDT")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_return_value_is_always_two_tuple(self):
        """All code paths must return a (text, style_dict) 2-tuple."""
        from brokers.simulatedbroker import SimulatedBroker
        broker = SimulatedBroker()
        try:
            for args in [
                ("buy",  1.0, "market", None,   "AAPL"),
                ("sell", 1.0, "market", None,   "AAPL"),
                ("buy",  1.0, "limit",  150.0,  "AAPL"),
                ("buy",  0,   "market", None,   "AAPL"),  # validation error
                ("buy",  1.0, "market", None,   None),    # no symbol
            ]:
                side, qty, ot, price, sym = args
                result = self._fn(broker, side, qty, ot, price, sym)
                assert isinstance(result, tuple) and len(result) == 2, (
                    f"Expected 2-tuple for args={args}, got {result!r}"
                )
        finally:
            broker.close()


# ---------------------------------------------------------------------------
# 4. App import sanity
# ---------------------------------------------------------------------------

class TestDashAppImport:
    """Importing dash_app.app must not start a server or raise errors."""

    def test_app_import_succeeds(self):
        """dash_app.app imports without raising."""
        import dash_app.app  # noqa: F401 — import side-effect is the test

    def test_app_object_is_dash_instance(self):
        """The module-level `app` variable must be a Dash instance."""
        import dash
        import dash_app.app as dash_module
        assert isinstance(dash_module.app, dash.Dash)

    def test_app_has_layout(self):
        """app.layout must be set (non-None) after module import."""
        import dash_app.app as dash_module
        assert dash_module.app.layout is not None

    def test_callbacks_registered(self):
        """After import, app.callback_map must contain at least the four
        Phase-1 callbacks (load_chart, update_live_price, toggle_price_input,
        submit_order_callback)."""
        import dash_app.app as dash_module
        # Dash stores callbacks in app.callback_map (dict keyed by output ID string).
        callback_map = dash_module.app.callback_map
        assert len(callback_map) >= 4, (
            f"Expected at least 4 registered callbacks, found {len(callback_map)}: "
            f"{list(callback_map.keys())}"
        )
