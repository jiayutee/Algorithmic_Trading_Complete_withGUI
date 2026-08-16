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
        "pnl-calendar-store",       # Phase 1.4: holds displayed year/month
        # Metrics panel
        "account-balance",
        "pnl-value",
        "bt-sharpe",
        "bt-winrate",
        "bt-maxdd",
        "chart-status",
        # Backtest controls + results (Phase 1.5)
        "bt-cash-input",
        "bt-run-btn",
        "bt-alpha",
        "bt-beta",
        "bt-status",
        # Status bar
        "status-bar",
        # Bottom tabs panel (Phase 1.4)
        "bottom-tabs",
        "positions-content",
        "pnl-prev-btn",
        "pnl-next-btn",
        "pnl-today-btn",
        "pnl-calendar-title",
        "pnl-calendar-total",
        "pnl-calendar-grid",
        # Equity Curve tab (Phase 1.5)
        "equity-curve-chart",
        # Orders tab (Phase 1.6)
        "orders-table",
        "orders-status",
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
        """After import, app.callback_map must contain at least nine callbacks:
        the four Phase-1 callbacks (load_chart, update_live_price,
        toggle_price_input, submit_order_callback) plus three Phase-1.4 callbacks
        (update_calendar_store, update_pnl_calendar_display, update_positions)
        plus one Phase-1.5 callback (run_backtest_callback) plus one Phase-1.6
        callback (update_orders_table)."""
        import dash_app.app as dash_module
        # Dash stores callbacks in app.callback_map (dict keyed by output ID string).
        callback_map = dash_module.app.callback_map
        assert len(callback_map) >= 9, (
            f"Expected at least 9 registered callbacks, found {len(callback_map)}: "
            f"{list(callback_map.keys())}"
        )


# ---------------------------------------------------------------------------
# 5. Positions + PnL Calendar helpers (Phase 1.4)
# ---------------------------------------------------------------------------

class TestPositionsHelpers:
    """_build_positions_content covers no-broker, empty, and populated states."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from dash_app.callbacks import _build_positions_content
        self._fn = _build_positions_content

    def test_no_broker_returns_list(self):
        result = self._fn(None)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_no_broker_shows_no_positions_text(self):
        from dash.development.base_component import Component as DashComponent
        result = self._fn(None)
        # Flatten all text content from the component tree
        def _text(c):
            parts = []
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, DashComponent):
                try:
                    ch = c.children
                except AttributeError:
                    ch = None
                if ch is not None:
                    if not isinstance(ch, list):
                        ch = [ch]
                    for item in ch:
                        parts.extend(_text(item))
            elif isinstance(c, list):
                for item in c:
                    parts.extend(_text(item))
            return parts
        text_content = " ".join(_text(result)).lower()
        assert "no active positions" in text_content

    def test_empty_positions_dict_returns_no_positions_message(self):
        from unittest.mock import MagicMock
        broker = MagicMock()
        broker.positions = {}
        result = self._fn(broker)
        assert isinstance(result, list)

    def test_nonzero_position_yields_row(self):
        """A broker with one open position must produce at least one row."""
        from unittest.mock import MagicMock
        broker = MagicMock()
        pos = MagicMock()
        pos.qty = 2.0
        pos.avg_price = 150.0
        pos.pnl = 25.0
        broker.positions = {"AAPL": pos}
        result = self._fn(broker)
        # Should have at least one Div (the position row), not the empty message
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_zero_qty_position_is_excluded(self):
        """Positions with qty == 0 must be skipped (flat positions)."""
        from unittest.mock import MagicMock
        from dash.development.base_component import Component as DashComponent
        broker = MagicMock()
        pos = MagicMock()
        pos.qty = 0
        broker.positions = {"AAPL": pos}
        result = self._fn(broker)
        # With all positions at zero qty, should fall back to "no active positions"
        def _has_text(components, needle):
            for c in (components if isinstance(components, list) else [components]):
                if isinstance(c, str) and needle in c.lower():
                    return True
                if isinstance(c, DashComponent):
                    try:
                        ch = c.children
                        if _has_text(ch if isinstance(ch, list) else [ch], needle):
                            return True
                    except AttributeError:
                        pass
            return False
        assert _has_text(result, "no active")

    def test_real_broker_with_order_shows_position(self):
        """After a real market buy, the broker has a non-zero position."""
        from brokers.simulatedbroker import SimulatedBroker
        from dash_app.callbacks import _validate_and_submit_order
        broker = SimulatedBroker()
        try:
            _validate_and_submit_order(broker, "buy", 1.0, "market", None, "AAPL")
            result = self._fn(broker)
            assert isinstance(result, list)
            # Should have at least one non-empty-message item (the AAPL row)
            assert len(result) >= 1
        finally:
            broker.close()


class TestPnLCalendarHelpers:
    """_build_pnl_calendar_grid covers shape, styling, and edge-cases."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from dash_app.callbacks import _build_pnl_calendar_grid
        self._fn = _build_pnl_calendar_grid

    def test_returns_list_with_one_outer_div(self):
        result = self._fn(2026, 8, {})
        assert isinstance(result, list)
        assert len(result) == 1

    def test_outer_div_has_42_children(self):
        """The CSS-grid container must have exactly 42 day cells."""
        result = self._fn(2026, 8, {})
        outer_div = result[0]
        children = outer_div.children
        assert len(children) == 42

    def test_positive_pnl_cell_uses_green_bg(self):
        """A day with positive PnL must use the dark-green background."""
        import datetime
        from dash_app.callbacks import _CAL_GREEN_BG
        by_day = {datetime.date(2026, 8, 5): 200.0}
        result = self._fn(2026, 8, by_day)
        outer_div = result[0]
        cells = outer_div.children
        # Aug 5 2026 is a Wednesday (col index 2 in week starting Mon Jul 27)
        # Find cell for Aug 5: it should be among the in-month cells
        green_cells = [c for c in cells if c.style.get("backgroundColor") == _CAL_GREEN_BG]
        assert len(green_cells) >= 1

    def test_negative_pnl_cell_uses_red_bg(self):
        """A day with negative PnL must use the dark-red background."""
        import datetime
        from dash_app.callbacks import _CAL_RED_BG
        by_day = {datetime.date(2026, 8, 6): -75.5}
        result = self._fn(2026, 8, by_day)
        outer_div = result[0]
        cells = outer_div.children
        red_cells = [c for c in cells if c.style.get("backgroundColor") == _CAL_RED_BG]
        assert len(red_cells) >= 1

    def test_out_of_month_cells_use_dark_bg(self):
        """Filler cells outside the target month must use THEME[bg_dark]."""
        from core.chart_builder import THEME
        result = self._fn(2026, 8, {})
        outer_div = result[0]
        cells = outer_div.children
        # August 2026 starts Saturday → Mon-Fri (Jul 27-31) are out-of-month
        dim_cells = [c for c in cells if c.style.get("backgroundColor") == THEME["bg_dark"]]
        assert len(dim_cells) >= 5  # at least 5 out-of-month fillers (Mon-Fri Jul 27-31)

    def test_grid_style_is_7_column_css_grid(self):
        """The outer container must use a 7-column CSS grid."""
        result = self._fn(2026, 8, {})
        outer_div = result[0]
        assert outer_div.style.get("display") == "grid"
        assert "repeat(7, 1fr)" in outer_div.style.get("gridTemplateColumns", "")

    def test_calendar_store_default_is_current_month(self):
        """pnl-calendar-store in the layout must initialise to today's month."""
        import datetime
        from dash_app.layout import build_layout
        from dash.development.base_component import Component as DashComponent

        def _find_store(comp):
            try:
                if getattr(comp, "id", None) == "pnl-calendar-store":
                    return comp
            except Exception:
                pass
            try:
                children = comp.children
            except AttributeError:
                return None
            if children is None:
                return None
            if not isinstance(children, list):
                children = [children]
            for child in children:
                if isinstance(child, DashComponent):
                    result = _find_store(child)
                    if result is not None:
                        return result
            return None

        layout = build_layout()
        store = _find_store(layout)
        assert store is not None, "pnl-calendar-store not found in layout"
        today = datetime.date.today()
        assert store.data["year"] == today.year
        assert store.data["month"] == today.month


# ---------------------------------------------------------------------------
# 6. Backtest helpers (Phase 1.5)
# ---------------------------------------------------------------------------

class TestExtractBacktestMetrics:
    """_extract_backtest_metrics covers all branches of the metrics extractor."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from dash_app.callbacks import _extract_backtest_metrics
        self._fn = _extract_backtest_metrics

    # 6a. Degenerate inputs ---------------------------------------------------

    def test_empty_dict_returns_na_tuple(self):
        """Empty results dict → all display strings are 'N/A'."""
        result = self._fn({})
        # Empty dict is falsy → first guard branch
        sharpe, winrate, maxdd, alpha, beta, status = result
        assert sharpe == "N/A"
        assert winrate == "N/A"
        assert maxdd == "N/A"
        assert alpha == "N/A"
        assert beta == "N/A"

    def test_none_like_returns_na(self):
        """None-equivalent (falsy) results → all display strings are 'N/A'."""
        # {} is the normal empty case; also verify None if passed accidentally
        result = self._fn(None)
        assert result[0] == "N/A"

    def test_error_dict_returns_na_and_error_message(self):
        results = {"error": "Cerebro crashed"}
        sharpe, winrate, maxdd, alpha, beta, status = self._fn(results)
        assert sharpe == "N/A"
        assert "Cerebro crashed" in status

    def test_returns_six_element_tuple(self):
        result = self._fn({"sharpe": 1.5, "max_drawdown": 10.0, "win_rate": 55.0,
                           "alpha": 0.02, "beta": 0.8})
        assert isinstance(result, tuple)
        assert len(result) == 6

    # 6b. Shorthand top-level keys (no summary sub-dict) ---------------------

    def test_sharpe_formatted_to_two_decimal_places(self):
        result = self._fn({"sharpe": 1.23456, "max_drawdown": 0, "win_rate": 0,
                           "alpha": 0, "beta": 0})
        assert result[0] == "1.23"

    def test_maxdd_formatted_with_percent_suffix(self):
        result = self._fn({"sharpe": 0, "max_drawdown": 12.5, "win_rate": 0,
                           "alpha": 0, "beta": 0})
        assert result[2] == "12.50%"

    def test_winrate_float_formatted_as_percent_string(self):
        """win_rate as a float should be formatted 'XX.YY%'."""
        result = self._fn({"sharpe": 0, "max_drawdown": 0, "win_rate": 62.5,
                           "alpha": 0, "beta": 0})
        assert result[1] == "62.50%"

    def test_winrate_string_passed_through_unchanged(self):
        """win_rate already as a string must not be double-formatted."""
        result = self._fn({"sharpe": 0, "max_drawdown": 0, "win_rate": "55.00%",
                           "alpha": 0, "beta": 0})
        assert result[1] == "55.00%"

    def test_alpha_formatted_to_four_decimal_places(self):
        result = self._fn({"sharpe": 0, "max_drawdown": 0, "win_rate": 0,
                           "alpha": 0.123456, "beta": 0})
        assert result[3] == "0.1235"

    def test_beta_formatted_to_four_decimal_places(self):
        result = self._fn({"sharpe": 0, "max_drawdown": 0, "win_rate": 0,
                           "alpha": 0, "beta": 1.234567})
        assert result[4] == "1.2346"

    # 6c. Summary sub-dict takes priority over shorthand keys ----------------

    def test_summary_dict_sharpe_preferred_over_top_level(self):
        """summary['Sharpe Ratio'] must shadow top-level 'sharpe' key."""
        results = {
            "sharpe": 0.0,
            "max_drawdown": 0, "win_rate": 0, "alpha": 0, "beta": 0,
            "summary": {"Sharpe Ratio": 2.5, "Max Drawdown (%)": 0,
                        "Win Rate": "0.00%", "Alpha": 0, "Beta": 0,
                        "Final Value": 110000, "P&L": 10000},
        }
        assert self._fn(results)[0] == "2.50"

    def test_status_message_contains_final_and_pnl(self):
        """status_msg must mention the final portfolio value and P&L."""
        results = {
            "sharpe": 1.0, "max_drawdown": 5.0, "win_rate": 50.0,
            "alpha": 0.01, "beta": 0.9,
            "summary": {
                "Sharpe Ratio": 1.0, "Max Drawdown (%)": 5.0,
                "Win Rate": "50.00%", "Alpha": 0.01, "Beta": 0.9,
                "Final Value": 105000.0, "P&L": 5000.0,
            },
        }
        status = self._fn(results)[5]
        assert "105,000" in status or "105000" in status
        assert "5,000" in status or "5000" in status

    def test_status_message_contains_complete_keyword(self):
        results = {
            "sharpe": 1.0, "max_drawdown": 5.0, "win_rate": 50.0,
            "alpha": 0.01, "beta": 0.9,
        }
        status = self._fn(results)[5]
        assert "complete" in status.lower() or "Backtest" in status


class TestBuildEquityCurveFigure:
    """_build_equity_curve_figure must return a valid plotly Figure."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from dash_app.callbacks import _build_equity_curve_figure
        self._fn = _build_equity_curve_figure

    def test_empty_list_returns_figure(self):
        """Empty asset-value list must still return a Figure (no crash)."""
        import plotly.graph_objects as go
        fig = self._fn([])
        assert isinstance(fig, go.Figure)

    def test_none_returns_figure(self):
        import plotly.graph_objects as go
        fig = self._fn(None)
        assert isinstance(fig, go.Figure)

    def test_non_empty_list_returns_figure_with_trace(self):
        """Non-empty value list must produce a Figure with at least one trace."""
        import plotly.graph_objects as go
        values = [100000, 101000, 102500, 101800, 103000]
        fig = self._fn(values)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_trace_y_values_match_input(self):
        """The Scatter trace y-values must equal the input list."""
        values = [100000.0, 110000.0, 95000.0]
        fig = self._fn(values)
        assert list(fig.data[0].y) == values

    def test_figure_uses_dark_theme_paper_bgcolor(self):
        """paper_bgcolor must match THEME['bg_dark'] for visual consistency."""
        from core.chart_builder import THEME
        fig = self._fn([1, 2, 3])
        assert fig.layout.paper_bgcolor == THEME["bg_dark"]

    def test_figure_height_is_200(self):
        """Equity curve must be 200 px tall to fit the bottom-tabs panel."""
        fig = self._fn([1, 2, 3])
        assert fig.layout.height == 200


# ---------------------------------------------------------------------------
# 7. Orders / trade-blotter tab (Phase 1.6)
# ---------------------------------------------------------------------------

class TestOrdersTableHelper:
    """_build_orders_table_data covers no-broker, empty, and populated states."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from dash_app.callbacks import _build_orders_table_data
        self._fn = _build_orders_table_data

    # 7a. Degenerate inputs --------------------------------------------------

    def test_no_broker_returns_two_tuple(self):
        data, status = self._fn(None)
        assert isinstance(data, list)
        assert isinstance(status, str)

    def test_no_broker_returns_empty_data(self):
        data, _ = self._fn(None)
        assert data == []

    def test_no_broker_status_text(self):
        _, status = self._fn(None)
        assert "none yet" in status.lower() or "orders" in status.lower()

    def test_broker_without_order_history_attr_returns_empty(self):
        from unittest.mock import MagicMock
        broker = MagicMock(spec=[])  # no 'order_history' attribute
        data, status = self._fn(broker)
        assert data == []
        assert "none yet" in status.lower() or "orders" in status.lower()

    def test_empty_order_history_returns_empty_data(self):
        from unittest.mock import MagicMock
        broker = MagicMock()
        broker.order_history = []
        data, status = self._fn(broker)
        assert data == []

    # 7b. Populated order history --------------------------------------------

    def test_single_filled_buy_order_produces_one_row(self):
        """A broker with one filled buy produces exactly one data row."""
        from unittest.mock import MagicMock
        import time

        broker = MagicMock()
        order = MagicMock()
        order.created_at = time.time()
        order.symbol = "AAPL"
        order.side.value = "buy"
        order.order_type.value = "market"
        order.status.value = "filled"
        order.filled_qty = 1.0
        order.filled_avg_price = 150.0
        broker.order_history = [order]

        data, status = self._fn(broker)
        assert len(data) == 1

    def test_row_keys_match_datatable_columns(self):
        """Every row dict must contain exactly the 7 column IDs."""
        from unittest.mock import MagicMock
        import time

        broker = MagicMock()
        order = MagicMock()
        order.created_at = time.time()
        order.symbol = "BTCUSDT"
        order.side.value = "buy"
        order.order_type.value = "market"
        order.status.value = "filled"
        order.filled_qty = 0.5
        order.filled_avg_price = 45000.0
        broker.order_history = [order]

        data, _ = self._fn(broker)
        expected_keys = {"time", "symbol", "side", "type", "qty", "fill_price", "status"}
        assert set(data[0].keys()) == expected_keys

    def test_side_is_uppercased(self):
        """PyQt5 displays side as uppercase (BUY / SELL)."""
        from unittest.mock import MagicMock
        import time

        broker = MagicMock()
        order = MagicMock()
        order.created_at = time.time()
        order.symbol = "AAPL"
        order.side.value = "buy"
        order.order_type.value = "market"
        order.status.value = "filled"
        order.filled_qty = 1.0
        order.filled_avg_price = 150.0
        broker.order_history = [order]

        data, _ = self._fn(broker)
        assert data[0]["side"] == "BUY"

    def test_sell_side_is_uppercased(self):
        from unittest.mock import MagicMock
        import time

        broker = MagicMock()
        order = MagicMock()
        order.created_at = time.time()
        order.symbol = "AAPL"
        order.side.value = "sell"
        order.order_type.value = "market"
        order.status.value = "filled"
        order.filled_qty = 1.0
        order.filled_avg_price = 150.0
        broker.order_history = [order]

        data, _ = self._fn(broker)
        assert data[0]["side"] == "SELL"

    def test_type_is_capitalized(self):
        """Order type must be title-cased (Market / Limit / Stop)."""
        from unittest.mock import MagicMock
        import time

        broker = MagicMock()
        order = MagicMock()
        order.created_at = time.time()
        order.symbol = "AAPL"
        order.side.value = "buy"
        order.order_type.value = "limit"
        order.status.value = "pending"
        order.filled_qty = 0.0
        order.filled_avg_price = 0.0
        broker.order_history = [order]

        data, _ = self._fn(broker)
        assert data[0]["type"] == "Limit"

    def test_status_is_capitalized(self):
        """Status must be title-cased (Filled / Pending / Rejected / Canceled)."""
        from unittest.mock import MagicMock
        import time

        broker = MagicMock()
        order = MagicMock()
        order.created_at = time.time()
        order.symbol = "AAPL"
        order.side.value = "buy"
        order.order_type.value = "market"
        order.status.value = "filled"
        order.filled_qty = 1.0
        order.filled_avg_price = 150.0
        broker.order_history = [order]

        data, _ = self._fn(broker)
        assert data[0]["status"] == "Filled"

    def test_fill_price_formatted_with_dollar_sign(self):
        """Filled orders must show price as '$NNN.NNNN'."""
        from unittest.mock import MagicMock
        import time

        broker = MagicMock()
        order = MagicMock()
        order.created_at = time.time()
        order.symbol = "AAPL"
        order.side.value = "buy"
        order.order_type.value = "market"
        order.status.value = "filled"
        order.filled_qty = 1.0
        order.filled_avg_price = 150.25
        broker.order_history = [order]

        data, _ = self._fn(broker)
        assert data[0]["fill_price"].startswith("$")
        assert "150.2500" in data[0]["fill_price"]

    def test_zero_fill_price_shows_dash(self):
        """Unfilled orders (fill_price == 0 / falsy) must show '—'."""
        from unittest.mock import MagicMock
        import time

        broker = MagicMock()
        order = MagicMock()
        order.created_at = time.time()
        order.symbol = "AAPL"
        order.side.value = "buy"
        order.order_type.value = "limit"
        order.status.value = "pending"
        order.filled_qty = 0.0
        order.filled_avg_price = 0.0  # not yet filled
        broker.order_history = [order]

        data, _ = self._fn(broker)
        assert data[0]["fill_price"] == "—"

    def test_symbol_preserved_as_is(self):
        from unittest.mock import MagicMock
        import time

        broker = MagicMock()
        order = MagicMock()
        order.created_at = time.time()
        order.symbol = "BTCUSDT"
        order.side.value = "buy"
        order.order_type.value = "market"
        order.status.value = "filled"
        order.filled_qty = 0.001
        order.filled_avg_price = 45000.0
        broker.order_history = [order]

        data, _ = self._fn(broker)
        assert data[0]["symbol"] == "BTCUSDT"

    def test_multiple_orders_produce_correct_count(self):
        from unittest.mock import MagicMock
        import time

        broker = MagicMock()
        orders = []
        for i in range(5):
            o = MagicMock()
            o.created_at = time.time()
            o.symbol = "AAPL"
            o.side.value = "buy"
            o.order_type.value = "market"
            o.status.value = "filled"
            o.filled_qty = float(i + 1)
            o.filled_avg_price = 150.0
            orders.append(o)
        broker.order_history = orders

        data, status = self._fn(broker)
        assert len(data) == 5

    def test_status_text_contains_total_count(self):
        """status_text must mention the total order count."""
        from unittest.mock import MagicMock
        import time

        broker = MagicMock()
        order = MagicMock()
        order.created_at = time.time()
        order.symbol = "AAPL"
        order.side.value = "buy"
        order.order_type.value = "market"
        order.status.value = "filled"
        order.filled_qty = 1.0
        order.filled_avg_price = 150.0
        broker.order_history = [order]

        _, status = self._fn(broker)
        assert "1" in status

    def test_status_text_shows_filled_count(self):
        """status_text must distinguish total vs filled count."""
        from unittest.mock import MagicMock
        import time

        broker = MagicMock()
        # 2 orders: 1 filled, 1 pending
        orders = []
        for status_val in ("filled", "pending"):
            o = MagicMock()
            o.created_at = time.time()
            o.symbol = "AAPL"
            o.side.value = "buy"
            o.order_type.value = "market"
            o.status.value = status_val
            o.filled_qty = 1.0
            o.filled_avg_price = 150.0 if status_val == "filled" else 0.0
            orders.append(o)
        broker.order_history = orders

        _, status_text = self._fn(broker)
        # "Orders: 2 total, 1 filled"
        assert "2" in status_text
        assert "1" in status_text
        assert "filled" in status_text.lower()

    # 7c. Real SimulatedBroker integration ----------------------------------

    def test_real_broker_after_buy_shows_one_row(self):
        """After a real market buy via SimulatedBroker, the blotter has 1 row."""
        from brokers.simulatedbroker import SimulatedBroker
        from dash_app.callbacks import _validate_and_submit_order
        broker = SimulatedBroker()
        try:
            _validate_and_submit_order(broker, "buy", 1.0, "market", None, "AAPL")
            data, status = self._fn(broker)
            assert len(data) >= 1
            # The row must reference AAPL
            assert any(row["symbol"] == "AAPL" for row in data)
        finally:
            broker.close()

    def test_real_broker_filled_order_status_text(self):
        """After a filled buy, status_text must say '1 total, 1 filled'."""
        from brokers.simulatedbroker import SimulatedBroker
        from dash_app.callbacks import _validate_and_submit_order
        broker = SimulatedBroker()
        try:
            _validate_and_submit_order(broker, "buy", 1.0, "market", None, "AAPL")
            _, status = self._fn(broker)
            assert "total" in status.lower()
            assert "filled" in status.lower()
        finally:
            broker.close()


class TestOrdersTabLayout:
    """Layout-level tests for the Orders tab component IDs and initial state."""

    @pytest.fixture(scope="class")
    def layout(self):
        from dash_app.layout import build_layout
        return build_layout()

    def test_orders_table_in_layout(self, layout):
        """orders-table must appear in the layout component tree."""
        from dash_app.layout import build_layout
        all_ids = _collect_ids(build_layout())
        assert "orders-table" in all_ids

    def test_orders_status_in_layout(self, layout):
        """orders-status must appear in the layout component tree."""
        from dash_app.layout import build_layout
        all_ids = _collect_ids(build_layout())
        assert "orders-status" in all_ids

    def test_orders_table_has_seven_columns(self, layout):
        """orders-table must declare exactly 7 columns matching the PyQt5 blotter."""
        from dash_app.layout import build_layout
        from dash.development.base_component import Component as DashComponent

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
                if isinstance(child, DashComponent):
                    result = _find_by_id(child, target_id)
                    if result is not None:
                        return result
            return None

        layout = build_layout()
        table = _find_by_id(layout, "orders-table")
        assert table is not None, "orders-table not found in layout"
        assert len(table.columns) == 7, (
            f"Expected 7 columns, found {len(table.columns)}: {table.columns}"
        )

    def test_orders_table_column_ids_match_pyqt5(self, layout):
        """Column IDs must match the 7 PyQt5 _orders_table columns."""
        from dash_app.layout import build_layout
        from dash.development.base_component import Component as DashComponent

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
                if isinstance(child, DashComponent):
                    result = _find_by_id(child, target_id)
                    if result is not None:
                        return result
            return None

        layout = build_layout()
        table = _find_by_id(layout, "orders-table")
        assert table is not None
        col_ids = [c["id"] for c in table.columns]
        expected = ["time", "symbol", "side", "type", "qty", "fill_price", "status"]
        assert col_ids == expected, f"Expected {expected}, got {col_ids}"

    def test_orders_table_starts_with_empty_data(self, layout):
        """orders-table must start with data=[] (no rows before any order)."""
        from dash_app.layout import build_layout
        from dash.development.base_component import Component as DashComponent

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
                if isinstance(child, DashComponent):
                    result = _find_by_id(child, target_id)
                    if result is not None:
                        return result
            return None

        layout = build_layout()
        table = _find_by_id(layout, "orders-table")
        assert table is not None
        assert table.data == [] or table.data is None, (
            f"orders-table should start empty, found: {table.data}"
        )


# ---------------------------------------------------------------------------
# 8. News & Earnings panel (Phase 1.7)
# ---------------------------------------------------------------------------

class TestNewsEarningsLayout:
    """Layout-level tests for the News & Earnings tab component IDs."""

    _NEW_IDS = [
        "news-refresh-btn",
        "news-content",
        "earnings-table",
        "earnings-status",
    ]

    @pytest.fixture(scope="class")
    def all_ids(self):
        from dash_app.layout import build_layout
        return _collect_ids(build_layout())

    @pytest.mark.parametrize("expected_id", _NEW_IDS)
    def test_news_earnings_ids_present(self, all_ids, expected_id):
        """Every new Phase-1.7 component ID must appear in the layout tree."""
        assert expected_id in all_ids, (
            f"Expected id {expected_id!r} not found. IDs: {sorted(all_ids)}"
        )

    def test_earnings_table_has_five_columns(self):
        """earnings-table must have exactly 5 columns."""
        from dash_app.layout import build_layout
        from dash.development.base_component import Component as DashComponent

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
                if isinstance(child, DashComponent):
                    result = _find_by_id(child, target_id)
                    if result is not None:
                        return result
            return None

        layout = build_layout()
        table = _find_by_id(layout, "earnings-table")
        assert table is not None, "earnings-table not found in layout"
        assert len(table.columns) == 5, (
            f"Expected 5 columns, found {len(table.columns)}: {table.columns}"
        )

    def test_earnings_table_column_ids(self):
        """earnings-table column IDs must match the defined schema."""
        from dash_app.layout import build_layout
        from dash.development.base_component import Component as DashComponent

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
                if isinstance(child, DashComponent):
                    result = _find_by_id(child, target_id)
                    if result is not None:
                        return result
            return None

        layout = build_layout()
        table = _find_by_id(layout, "earnings-table")
        assert table is not None
        col_ids = [c["id"] for c in table.columns]
        expected = ["date", "eps_estimate", "eps_actual", "revenue_estimate", "revenue_actual"]
        assert col_ids == expected, f"Expected {expected}, got {col_ids}"

    def test_earnings_table_starts_empty(self):
        """earnings-table must start with data=[] before any refresh."""
        from dash_app.layout import build_layout
        from dash.development.base_component import Component as DashComponent

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
                if isinstance(child, DashComponent):
                    result = _find_by_id(child, target_id)
                    if result is not None:
                        return result
            return None

        layout = build_layout()
        table = _find_by_id(layout, "earnings-table")
        assert table is not None
        assert table.data == [] or table.data is None

    def test_news_refresh_btn_starts_with_zero_clicks(self):
        """news-refresh-btn must start with n_clicks=0."""
        from dash_app.layout import build_layout
        from dash.development.base_component import Component as DashComponent

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
                if isinstance(child, DashComponent):
                    result = _find_by_id(child, target_id)
                    if result is not None:
                        return result
            return None

        layout = build_layout()
        btn = _find_by_id(layout, "news-refresh-btn")
        assert btn is not None, "news-refresh-btn not found"
        assert btn.n_clicks == 0


class TestBuildNewsContent:
    """_build_news_content covers no-symbol, no-items, items, and exception paths."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from dash_app.callbacks import _build_news_content
        self._fn = _build_news_content

    def test_no_symbol_returns_list(self):
        result = self._fn(None)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_no_symbol_returns_prompt_message(self):
        from dash.development.base_component import Component as DashComponent
        result = self._fn(None)
        def _flatten_text(c):
            parts = []
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, DashComponent):
                try:
                    ch = c.children
                except AttributeError:
                    ch = None
                if ch is not None:
                    if not isinstance(ch, list):
                        ch = [ch]
                    for item in ch:
                        parts.extend(_flatten_text(item))
            elif isinstance(c, list):
                for item in c:
                    parts.extend(_flatten_text(item))
            return parts
        text = " ".join(_flatten_text(result)).lower()
        assert "select" in text or "refresh" in text or "symbol" in text

    def test_empty_symbol_string_returns_prompt(self):
        result = self._fn("")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_no_news_items_returns_no_news_message(self):
        """When pipeline returns an empty list, show 'No news found' message."""
        from unittest.mock import MagicMock, patch
        mock_pipeline = MagicMock()
        mock_pipeline.fetch_news_items.return_value = []
        with patch("core.news_pipeline.get_default_news_pipeline", return_value=mock_pipeline):
            result = self._fn("AAPL")
        assert isinstance(result, list)
        assert len(result) >= 1
        # Verify some "no news" text is present
        from dash.development.base_component import Component as DashComponent
        def _flatten_text(c):
            parts = []
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, DashComponent):
                try:
                    ch = c.children
                except AttributeError:
                    ch = None
                if ch is not None:
                    if not isinstance(ch, list):
                        ch = [ch]
                    for item in ch:
                        parts.extend(_flatten_text(item))
            elif isinstance(c, list):
                for item in c:
                    parts.extend(_flatten_text(item))
            return parts
        text = " ".join(_flatten_text(result)).lower()
        assert "no news" in text or "not found" in text or "no " in text

    def test_news_items_returned_as_list_of_divs(self):
        """When pipeline returns items, _build_news_content returns one Div per item."""
        from unittest.mock import MagicMock, patch
        import datetime

        mock_item = MagicMock()
        mock_item.headline = "AAPL hits all-time high"
        mock_item.url = "https://example.com/news/aapl"
        mock_item.source = "Reuters"
        mock_item.datetime_utc = datetime.datetime(2026, 8, 15, 10, 30)

        mock_pipeline = MagicMock()
        mock_pipeline.fetch_news_items.return_value = [mock_item, mock_item]

        with patch("core.news_pipeline.get_default_news_pipeline", return_value=mock_pipeline):
            result = self._fn("AAPL")

        assert isinstance(result, list)
        assert len(result) == 2

    def test_item_with_url_produces_anchor(self):
        """News items with a URL must include a clickable link in the output."""
        from unittest.mock import MagicMock, patch
        from dash import html as dash_html
        import datetime

        mock_item = MagicMock()
        mock_item.headline = "Apple reports record revenue"
        mock_item.url = "https://example.com/apple-revenue"
        mock_item.source = "Bloomberg"
        mock_item.datetime_utc = datetime.datetime(2026, 8, 14, 9, 0)

        mock_pipeline = MagicMock()
        mock_pipeline.fetch_news_items.return_value = [mock_item]

        with patch("core.news_pipeline.get_default_news_pipeline", return_value=mock_pipeline):
            result = self._fn("AAPL")

        assert len(result) == 1
        # The row Div's children should contain an html.A link
        from dash.development.base_component import Component as DashComponent
        def _has_anchor(comp):
            if isinstance(comp, dash_html.A):
                return True
            try:
                ch = comp.children
            except AttributeError:
                return False
            if ch is None:
                return False
            if not isinstance(ch, list):
                ch = [ch]
            return any(_has_anchor(c) for c in ch if isinstance(c, DashComponent))
        assert _has_anchor(result[0]), "Expected an html.A anchor in the news row"

    def test_item_without_url_produces_no_anchor(self):
        """News items with no URL must not include an anchor element."""
        from unittest.mock import MagicMock, patch
        from dash import html as dash_html
        import datetime

        mock_item = MagicMock()
        mock_item.headline = "Market update"
        mock_item.url = ""
        mock_item.source = "Internal"
        mock_item.datetime_utc = datetime.datetime(2026, 8, 14, 9, 0)

        mock_pipeline = MagicMock()
        mock_pipeline.fetch_news_items.return_value = [mock_item]

        with patch("core.news_pipeline.get_default_news_pipeline", return_value=mock_pipeline):
            result = self._fn("AAPL")

        assert len(result) == 1
        from dash.development.base_component import Component as DashComponent
        def _has_anchor(comp):
            if isinstance(comp, dash_html.A):
                return True
            try:
                ch = comp.children
            except AttributeError:
                return False
            if ch is None:
                return False
            if not isinstance(ch, list):
                ch = [ch]
            return any(_has_anchor(c) for c in ch if isinstance(c, DashComponent))
        assert not _has_anchor(result[0]), "Expected no anchor for item without URL"

    def test_fetch_exception_returns_error_message(self):
        """If the news pipeline raises, an error message must be returned, not an exception."""
        from unittest.mock import MagicMock, patch

        mock_pipeline = MagicMock()
        mock_pipeline.fetch_news_items.side_effect = RuntimeError("Network timeout")

        with patch("core.news_pipeline.get_default_news_pipeline", return_value=mock_pipeline):
            result = self._fn("AAPL")

        assert isinstance(result, list)
        assert len(result) >= 1
        # Result must mention "error" somewhere
        from dash.development.base_component import Component as DashComponent
        def _flatten_text(c):
            parts = []
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, DashComponent):
                try:
                    ch = c.children
                except AttributeError:
                    ch = None
                if ch is not None:
                    if not isinstance(ch, list):
                        ch = [ch]
                    for item in ch:
                        parts.extend(_flatten_text(item))
            elif isinstance(c, list):
                for item in c:
                    parts.extend(_flatten_text(item))
            return parts
        text = " ".join(_flatten_text(result)).lower()
        assert "error" in text

    def test_fetch_exception_does_not_propagate(self):
        """_build_news_content must never raise to the caller."""
        from unittest.mock import MagicMock, patch

        mock_pipeline = MagicMock()
        mock_pipeline.fetch_news_items.side_effect = Exception("Unexpected crash")

        with patch("core.news_pipeline.get_default_news_pipeline", return_value=mock_pipeline):
            result = self._fn("AAPL")  # must not raise

        assert isinstance(result, list)

    def test_timestamp_none_renders_dash(self):
        """Items with datetime_utc=None must not crash — use '—' as fallback."""
        from unittest.mock import MagicMock, patch

        mock_item = MagicMock()
        mock_item.headline = "Breaking news"
        mock_item.url = ""
        mock_item.source = "Test"
        mock_item.datetime_utc = None

        mock_pipeline = MagicMock()
        mock_pipeline.fetch_news_items.return_value = [mock_item]

        with patch("core.news_pipeline.get_default_news_pipeline", return_value=mock_pipeline):
            result = self._fn("AAPL")

        assert isinstance(result, list)
        assert len(result) == 1  # still one row, not a crash


class TestBuildEarningsTableData:
    """_build_earnings_table_data covers no-symbol, crypto, empty, populated, error paths."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from dash_app.callbacks import _build_earnings_table_data
        self._fn = _build_earnings_table_data

    # 8a. Degenerate inputs --------------------------------------------------

    def test_no_symbol_returns_two_tuple(self):
        data, status = self._fn(None)
        assert isinstance(data, list)
        assert isinstance(status, str)

    def test_no_symbol_returns_empty_data(self):
        data, _ = self._fn(None)
        assert data == []

    def test_empty_string_symbol_returns_empty_data(self):
        data, _ = self._fn("")
        assert data == []

    # 8b. Crypto symbols — fast-path, no DataLoader call needed ----------------

    def test_crypto_symbol_returns_empty_data(self):
        """Crypto symbols (contains 'USDT') must return empty data immediately."""
        data, status = self._fn("BTCUSDT")
        assert data == []

    def test_crypto_symbol_status_mentions_crypto(self):
        _, status = self._fn("ETHUSDT")
        assert "crypto" in status.lower()

    def test_crypto_symbol_no_dataloader_called(self):
        """DataLoader must never be instantiated for crypto symbols."""
        from unittest.mock import patch, MagicMock
        with patch("core.data_loader.DataLoader") as mock_cls:
            self._fn("SOLUSDT")
            mock_cls.assert_not_called()

    # 8c. Empty earnings list ------------------------------------------------

    def test_empty_earnings_returns_empty_data(self):
        from unittest.mock import MagicMock, patch
        mock_loader = MagicMock()
        mock_loader.get_earnings_calendar.return_value = []
        with patch("core.data_loader.DataLoader", return_value=mock_loader):
            data, status = self._fn("AAPL")
        assert data == []

    def test_empty_earnings_status_mentions_symbol(self):
        from unittest.mock import MagicMock, patch
        mock_loader = MagicMock()
        mock_loader.get_earnings_calendar.return_value = []
        with patch("core.data_loader.DataLoader", return_value=mock_loader):
            _, status = self._fn("AAPL")
        assert "AAPL" in status

    # 8d. Populated earnings -------------------------------------------------

    def test_single_entry_produces_one_row(self):
        from unittest.mock import MagicMock, patch
        mock_loader = MagicMock()
        mock_loader.get_earnings_calendar.return_value = [
            {"date": "2026-10-28", "eps_estimate": 1.43, "eps_actual": 1.52,
             "revenue_estimate": 94_500_000_000.0, "revenue_actual": 96_100_000_000.0}
        ]
        with patch("core.data_loader.DataLoader", return_value=mock_loader):
            data, status = self._fn("AAPL")
        assert len(data) == 1

    def test_row_keys_match_datatable_columns(self):
        """Row dicts must contain exactly the 5 earnings-table column IDs."""
        from unittest.mock import MagicMock, patch
        mock_loader = MagicMock()
        mock_loader.get_earnings_calendar.return_value = [
            {"date": "2026-10-28", "eps_estimate": 1.43, "eps_actual": None,
             "revenue_estimate": None, "revenue_actual": None}
        ]
        with patch("core.data_loader.DataLoader", return_value=mock_loader):
            data, _ = self._fn("AAPL")
        expected_keys = {"date", "eps_estimate", "eps_actual", "revenue_estimate", "revenue_actual"}
        assert set(data[0].keys()) == expected_keys

    def test_none_eps_shows_dash(self):
        """None EPS estimate/actual values must render as '—'."""
        from unittest.mock import MagicMock, patch
        mock_loader = MagicMock()
        mock_loader.get_earnings_calendar.return_value = [
            {"date": "2026-10-28", "eps_estimate": None, "eps_actual": None,
             "revenue_estimate": None, "revenue_actual": None}
        ]
        with patch("core.data_loader.DataLoader", return_value=mock_loader):
            data, _ = self._fn("AAPL")
        assert data[0]["eps_estimate"] == "—"
        assert data[0]["eps_actual"] == "—"

    def test_none_revenue_shows_dash(self):
        """None revenue values must render as '—'."""
        from unittest.mock import MagicMock, patch
        mock_loader = MagicMock()
        mock_loader.get_earnings_calendar.return_value = [
            {"date": "2026-10-28", "eps_estimate": 1.0, "eps_actual": 1.1,
             "revenue_estimate": None, "revenue_actual": None}
        ]
        with patch("core.data_loader.DataLoader", return_value=mock_loader):
            data, _ = self._fn("AAPL")
        assert data[0]["revenue_estimate"] == "—"
        assert data[0]["revenue_actual"] == "—"

    def test_eps_formatted_to_four_decimal_places(self):
        from unittest.mock import MagicMock, patch
        mock_loader = MagicMock()
        mock_loader.get_earnings_calendar.return_value = [
            {"date": "2026-10-28", "eps_estimate": 1.4321, "eps_actual": 1.5678,
             "revenue_estimate": None, "revenue_actual": None}
        ]
        with patch("core.data_loader.DataLoader", return_value=mock_loader):
            data, _ = self._fn("AAPL")
        assert data[0]["eps_estimate"] == "1.4321"
        assert data[0]["eps_actual"] == "1.5678"

    def test_revenue_converted_to_millions(self):
        """Revenue values must be divided by 1,000,000 and shown with one decimal."""
        from unittest.mock import MagicMock, patch
        mock_loader = MagicMock()
        mock_loader.get_earnings_calendar.return_value = [
            {"date": "2026-10-28", "eps_estimate": None, "eps_actual": None,
             "revenue_estimate": 94_500_000_000.0, "revenue_actual": 96_000_000_000.0}
        ]
        with patch("core.data_loader.DataLoader", return_value=mock_loader):
            data, _ = self._fn("AAPL")
        assert data[0]["revenue_estimate"] == "94500.0"
        assert data[0]["revenue_actual"] == "96000.0"

    def test_multiple_entries_produce_correct_count(self):
        from unittest.mock import MagicMock, patch
        mock_loader = MagicMock()
        mock_loader.get_earnings_calendar.return_value = [
            {"date": f"2026-{m:02d}-28", "eps_estimate": 1.0, "eps_actual": None,
             "revenue_estimate": None, "revenue_actual": None}
            for m in range(1, 5)
        ]
        with patch("core.data_loader.DataLoader", return_value=mock_loader):
            data, status = self._fn("AAPL")
        assert len(data) == 4
        assert "4" in status

    def test_status_text_mentions_count_and_symbol(self):
        from unittest.mock import MagicMock, patch
        mock_loader = MagicMock()
        mock_loader.get_earnings_calendar.return_value = [
            {"date": "2026-10-28", "eps_estimate": 1.0, "eps_actual": None,
             "revenue_estimate": None, "revenue_actual": None}
        ]
        with patch("core.data_loader.DataLoader", return_value=mock_loader):
            _, status = self._fn("TSLA")
        assert "TSLA" in status
        assert "1" in status

    # 8e. Exception handling -------------------------------------------------

    def test_dataloader_exception_returns_empty_data(self):
        """If DataLoader raises, the function must return empty data, not propagate."""
        from unittest.mock import patch, MagicMock
        mock_loader = MagicMock()
        mock_loader.get_earnings_calendar.side_effect = RuntimeError("API down")
        with patch("core.data_loader.DataLoader", return_value=mock_loader):
            data, status = self._fn("AAPL")
        assert data == []
        assert "error" in status.lower()

    def test_dataloader_exception_does_not_propagate(self):
        """_build_earnings_table_data must never raise to the caller."""
        from unittest.mock import patch, MagicMock
        mock_loader = MagicMock()
        mock_loader.get_earnings_calendar.side_effect = Exception("Unexpected")
        with patch("core.data_loader.DataLoader", return_value=mock_loader):
            result = self._fn("AAPL")  # must not raise
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_return_value_is_always_two_tuple(self):
        """All code paths must return a (list, str) 2-tuple."""
        from unittest.mock import patch, MagicMock
        test_cases = [
            (None, None, None),
            ("", None, None),
            ("BTCUSDT", None, None),
            ("AAPL", [], None),
            ("AAPL", [{"date": "2026-10-01", "eps_estimate": 1.0, "eps_actual": None,
                       "revenue_estimate": None, "revenue_actual": None}], None),
        ]
        for symbol, return_val, exc in test_cases:
            mock_loader = MagicMock()
            if exc:
                mock_loader.get_earnings_calendar.side_effect = exc
            else:
                mock_loader.get_earnings_calendar.return_value = return_val or []
            with patch("core.data_loader.DataLoader", return_value=mock_loader):
                result = self._fn(symbol)
            assert isinstance(result, tuple) and len(result) == 2, (
                f"Expected 2-tuple for symbol={symbol!r}, got {result!r}"
            )


class TestCallbackCountPhase17:
    """After Phase 1.7, app must have at least 10 registered callbacks."""

    def test_callback_count_at_least_ten(self):
        """app.callback_map must have >= 10 entries after Phase-1.7 registration."""
        import dash_app.app as dash_module
        callback_map = dash_module.app.callback_map
        assert len(callback_map) >= 10, (
            f"Expected at least 10 registered callbacks, found {len(callback_map)}: "
            f"{list(callback_map.keys())}"
        )
