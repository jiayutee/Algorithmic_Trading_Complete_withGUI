"""Design slice 2: design tokens, watchlist store + quote book, risk snapshot, workspace mapping."""

import re
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from core import design_tokens as dt
from core import risk_view, watchlist
from brokers.simulatedbroker import SimulatedBroker


# ------------------------------------------------------------------------------------------------ design tokens

_HEX = re.compile(r"#[0-9a-fA-F]{6}\b")


def test_generated_stylesheets_use_only_token_colors():
    palette = dt.palette()
    for sheet in (dt.qt_stylesheet(), dt.css_root()):
        stray = {h.lower() for h in _HEX.findall(sheet)} - palette
        assert not stray, f"colors not in core/design_tokens.COLOR: {stray}"


def test_chart_theme_and_mode_colors_come_from_tokens():
    from core.chart_builder import THEME
    from core.trading_status import FRESHNESS_COLORS, MODE_COLORS
    palette = dt.palette()
    assert {v.lower() for v in THEME.values()} <= palette
    assert {c.lower() for trio in MODE_COLORS.values() for c in trio} <= palette
    assert {c.lower() for c in FRESHNESS_COLORS.values()} <= palette


def test_css_root_has_variables_and_size_floor():
    css = dt.css_root()
    assert "--color-surface-base: #0d1117;" in css
    assert "--font-ui: 'IBM Plex Sans'" in css
    assert min(dt.SIZE.values()) >= 12


def test_dash_theme_css_only_uses_variables_or_token_colors():
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dash_app", "assets", "theme.css")
    css = open(path, encoding="utf-8").read()
    stray = {h.lower() for h in _HEX.findall(css)} - dt.palette()
    assert not stray


# ---------------------------------------------------------------------------------------------------- watchlist

def test_watchlist_store_round_trip(tmp_path):
    path = str(tmp_path / "wl.json")
    assert watchlist.load(path) == watchlist.DEFAULT_SYMBOLS
    assert watchlist.save([" btcusdt", "AAPL", "aapl", ""], path) == ["BTCUSDT", "AAPL"]
    assert watchlist.add("msft", path) == ["BTCUSDT", "AAPL", "MSFT"]
    assert watchlist.add("MSFT", path) == ["BTCUSDT", "AAPL", "MSFT"]
    assert watchlist.remove("aapl", path) == ["BTCUSDT", "MSFT"]
    assert watchlist.load(path) == ["BTCUSDT", "MSFT"]


def test_watchlist_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("WATCHLIST_PATH", str(tmp_path / "x.json"))
    watchlist.add("NVDA")
    assert "NVDA" in watchlist.load()


def test_quote_book_caches_keeps_last_good_and_reports_age():
    calls = []
    clock = [1000.0]
    answers = {"BTCUSDT": [watchlist.Quote("BTCUSDT", 100.0, 99.0, 1.01, 1000.0, "binance"),
                           watchlist.Quote("BTCUSDT", error="HTTP 429")]}

    def fetch(sym):
        calls.append(sym)
        return answers[sym].pop(0)

    book = watchlist.QuoteBook(fetch=fetch, clock=lambda: clock[0])
    book.refresh_now(["BTCUSDT"])
    clock[0] += 5
    assert book.refresh_async(["BTCUSDT"]) == 0                 # still fresh (crypto ttl 10 s)
    row = book.snapshot(["BTCUSDT", "AAPL"])
    assert row[0]["last"] == 100.0 and row[0]["age_s"] == 5
    assert row[1]["last"] is None and row[1]["error"] is None   # not fetched yet
    clock[0] += 10
    book.refresh_now(["BTCUSDT"])                                # fails, keeps last price with the error noted
    row = book.snapshot(["BTCUSDT"])[0]
    assert row["last"] == 100.0 and row["error"] == "HTTP 429"
    book.close()


def test_fetch_quote_crypto_and_yahoo_paths():
    class Resp:
        def __init__(self, status, payload):
            self.status_code, self._p = status, payload

        def json(self):
            return self._p

    sess = SimpleNamespace(get=lambda url, **k: Resp(200, {"lastPrice": "83492.05", "prevClosePrice": "82700",
                                                             "priceChangePercent": "0.95"}))
    q = watchlist.fetch_quote("BTCUSDT", session=sess)
    assert q.last == 83492.05 and q.change_pct == 0.95 and q.error is None
    bad = watchlist.fetch_quote("BTCUSDT", session=SimpleNamespace(get=lambda url, **k: Resp(451, {})))
    assert bad.last is None and bad.error == "HTTP 451"


def test_formatters():
    assert watchlist.format_price(None) == "—"
    assert watchlist.format_price(83492.05) == "83,492.05"
    assert watchlist.format_price(1.0912) == "1.0912"
    assert watchlist.format_change(1.234) == "▲ +1.23%"
    assert watchlist.format_change(-0.5) == "▼ −0.50%"


# ------------------------------------------------------------------------------------------------ risk snapshot

def _broker_with_position():
    b = SimulatedBroker(initial_balance=100_000)
    b.update_price("BTCUSDT", 50_000.0)
    b.submit_order("BTCUSDT", 0.5, "buy", "market")            # 25,000 notional
    return b


def test_risk_snapshot_without_journal(tmp_path):
    snap = risk_view.risk_snapshot(_broker_with_position(), journal_path=str(tmp_path / "missing.sqlite3"))
    assert not (tmp_path / "missing.sqlite3").exists()          # looking never creates the journal
    rows = {r["label"]: r for r in snap["rows"]}
    gross = rows["Gross exposure"]
    assert gross["value"] == pytest.approx(25_000 / snap["equity"], rel=1e-3)
    assert gross["state"] == "ok" and gross["limit_text"] == "75.0%"
    top = rows["Largest position (BTCUSDT)"]
    assert top["used"] == pytest.approx(gross["value"] / 0.25, rel=1e-3) and top["state"] == "breach"
    assert rows["Day P&L vs start"]["value_text"] == "n/a"
    assert rows["Runner orders today"]["value_text"] == "n/a"
    assert "defaults" in snap["limits_source"] and "never run" in snap["note"]


def test_risk_snapshot_reads_runner_journal(tmp_path):
    from core.execution.journal import ExecutionJournal
    path = str(tmp_path / "exec.sqlite3")
    j = ExecutionJournal(path)
    now = time.time()
    day = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d")
    broker = _broker_with_position()
    eq = broker.get_account_info()["portfolio_value"]
    j.set("day_baseline", {"day": day, "equity": eq / 0.98})        # 2% down today
    j.set("peak_equity", eq / 0.90)                                  # 10% below peak
    j.set("status", {"risk": {"max_gross_pct": 0.5, "max_position_pct": 0.3, "max_daily_loss_pct": 0.03,
                              "max_drawdown_pct": 0.15, "max_orders_per_day": 4}})
    for i in range(3):
        j.record(f"d{i}", symbol="BTCUSDT", action="submitted")
    j.halt("testing")
    j.close()
    snap = risk_view.risk_snapshot(broker, journal_path=path, now=now)
    rows = {r["label"]: r for r in snap["rows"]}
    assert snap["limits_source"] == "paper runner settings"
    assert rows["Gross exposure"]["limit_text"] == "50.0%"
    assert rows["Day P&L vs start"]["value"] == pytest.approx(-0.02, abs=1e-6)
    assert rows["Day P&L vs start"]["state"] == "ok"                 # 2/3 of the limit used
    assert rows["Drawdown from peak"]["used"] == pytest.approx(0.10 / 0.15, rel=1e-4)
    assert rows["Runner orders today"]["value_text"] == "3" and rows["Runner orders today"]["state"] == "warn"
    assert snap["halted"]["reason"] == "testing"


def test_risk_snapshot_no_broker_never_raises(tmp_path):
    snap = risk_view.risk_snapshot(None, journal_path=str(tmp_path / "none.sqlite3"))
    assert snap["equity"] is None
    assert [r["label"] for r in snap["rows"]] == ["Runner orders today"]


def test_workspace_mapping_covers_both_apps_tabs():
    from core import ui_options
    assert ui_options.WORKSPACES == ["Trade", "Research", "Automate", "Markets"]
    assert ui_options.workspace_for("Research Lab") == "Research"
    assert ui_options.workspace_for("News && Earnings") == "Markets"
    assert ui_options.workspace_for("⚠ Deps") == "Trade"
    assert set(ui_options.TAB_WORKSPACE.values()) <= set(ui_options.WORKSPACES)


# ------------------------------------------------------------------------------------------- Dash wiring (slice 2)

def _ids(component, out=None):
    from dash.development.base_component import Component
    out = [] if out is None else out
    if isinstance(component, (list, tuple)):
        for c in component:
            _ids(c, out)
    elif isinstance(component, Component):
        if getattr(component, "id", None) is not None:
            out.append(component.id)
        _ids(getattr(component, "children", None), out)
    return out


def test_dash_every_bottom_tab_has_a_header_id_and_workspace():
    pytest.importorskip("dash")
    from dash_app.layout import bottom_tab_specs, build_layout
    specs = bottom_tab_specs()
    ids = _ids(build_layout())
    assert specs and all(hid in ids for hid, *_ in specs)
    assert {ws for *_, ws in specs} == {"Trade", "Research", "Automate", "Markets"}
    for needed in ("workspace-select", "watchlist-body", "watchlist-input", "watchlist-add-btn", "watchlist-interval",
                   "watchlist-version", "risk-panel", "risk-panel-body", "backtest-panel"):
        assert needed in ids, needed


def test_dash_workspace_view_hides_other_tabs_and_panels():
    pytest.importorskip("dash")
    from dash_app.callbacks import _workspace_view
    from dash_app.layout import bottom_tab_specs
    specs = bottom_tab_specs()
    styles, value, backtest, risk = _workspace_view("Research", "positions-tab", specs)
    shown = [label for (_, label, _, _), st in zip(specs, styles) if st.get("display") != "none"]
    assert set(shown) == {"Equity Curve", "Research Lab", "Research Loop"}
    assert value == "equity-curve-tab"                        # the current tab was hidden, so the first visible one
    assert backtest.get("display") != "none" and risk.get("display") == "none"
    styles, value, backtest, risk = _workspace_view("Trade", "orders-tab", specs)
    assert value == "orders-tab" and backtest.get("display") == "none" and risk.get("display") != "none"


def test_dash_watchlist_and_risk_bodies():
    pytest.importorskip("dash")
    from dash_app.callbacks import _risk_body, _watchlist_body
    rows = [{"symbol": "BTCUSDT", "last": 83492.05, "change_pct": 1.2, "age_s": 1, "source": "binance", "error": None},
            {"symbol": "AAPL", "last": None, "change_pct": None, "age_s": None, "source": "", "error": "no data"}]
    ids = _ids(_watchlist_body(rows))
    assert {"type": "watch-symbol", "symbol": "BTCUSDT"} in ids and {"type": "watch-remove", "symbol": "AAPL"} in ids
    assert "Empty" in str(_watchlist_body([])[0].children)
    snap = risk_view.risk_snapshot(_broker_with_position(), journal_path="/nonexistent/exec.sqlite3")
    text = str(_risk_body(snap))
    assert "Gross exposure" in text and "at limit" in text and "manual orders are not gated" in text
