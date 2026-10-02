from datetime import datetime, timedelta, timezone

import pytest
import requests

from core import kalshi_collector as kc
from core.kalshi_data import parse_market
from test_kalshi import FakeResp, FakeSession, KalshiClient, raw

NOW = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


def iso(h):
    return (NOW + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")


def mk(ticker, close_h=5, bid=0.40, ask=0.45, vol="500.00", **kw):
    return raw(ticker, yes_bid=bid, yes_ask=ask, volume_fp=vol, close_time=iso(close_h), **kw)


def client(routes):
    return KalshiClient(session=FakeSession(routes), min_interval=0, sleep=lambda s: None)


def test_collect_keeps_only_liquid_two_sided_near_close(tmp_path):
    db = str(tmp_path / "k.db")
    good, thin, one_sided, wide, far = (mk("GOOD"), mk("THIN", vol="5.00"), mk("ONE", bid=0.0, ask=0.99),
                                         mk("WIDE", bid=0.2, ask=0.6), mk("FAR", close_h=500))
    c = client({"/markets": FakeResp(payload={"markets": [good, thin, one_sided, wide, far], "cursor": ""})})
    out = kc.collect(c, db, now=NOW)
    assert out == {"scanned": 5, "stored": 1}
    assert kc.status(db)["markets"] == 1


def test_collect_is_idempotent_per_timestamp(tmp_path):
    db = str(tmp_path / "k.db")
    routes = lambda: {"/markets": FakeResp(payload={"markets": [mk("A")], "cursor": ""})}
    kc.collect(client(routes()), db, now=NOW)
    kc.collect(client(routes()), db, now=NOW)                    # same ts -> no duplicate row
    kc.collect(client(routes()), db, now=NOW + timedelta(hours=1))
    assert kc.status(db)["snapshots"] == 2


def test_resolve_labels_settled_and_retries_unsettled(tmp_path):
    db = str(tmp_path / "k.db")
    kc.collect(client({"/markets": FakeResp(payload={"markets": [mk("A", close_h=1), mk("B", close_h=1), mk("C", close_h=48)], "cursor": ""})}), db, now=NOW)
    later = NOW + timedelta(hours=3)
    c = client({"/markets/A": FakeResp(payload={"market": raw("A", result="yes", status="finalized")}),
                "/markets/B": FakeResp(payload={"market": raw("B", result="", status="closed")})})
    assert kc.resolve(c, db, now=later) == {"resolved": 1, "pending": 1, "errors": 0, "skipped": 0}   # C not past close: untouched
    s = kc.status(db)
    assert s["resolved_markets"] == 1 and s["labelled_snapshots"] == 1
    # already-resolved markets are not looked up again
    c2 = client({"/markets/B": FakeResp(payload={"market": raw("B", result="no", status="finalized")})})
    assert kc.resolve(c2, db, now=later)["resolved"] == 1


def test_resolve_survives_api_errors(tmp_path):
    db = str(tmp_path / "k.db")
    kc.collect(client({"/markets": FakeResp(payload={"markets": [mk("A", close_h=1)], "cursor": ""})}), db, now=NOW)
    c = KalshiClient(session=FakeSession({"/markets/A": FakeResp(404)}), min_interval=0, sleep=lambda s: None)
    assert kc.resolve(c, db, now=NOW + timedelta(hours=2))["errors"] == 1


def test_status_on_empty_db(tmp_path):
    assert kc.status(str(tmp_path / "e.db"))["snapshots"] == 0


def _main_resolve(monkeypatch, tmp_path, capsys, route):
    db = str(tmp_path / "k.db")
    kc.collect(client({"/markets": FakeResp(payload={"markets": [mk("A", close_h=-1)], "cursor": ""})}), db,
               now=NOW - timedelta(hours=2))
    monkeypatch.setattr(kc, "KalshiClient", lambda: client({"/markets/A": route}))
    rc = kc.main(["resolve", "--db", db])
    return rc, capsys.readouterr().out


def test_main_resolve_exits_nonzero_with_warning_on_lookup_errors(monkeypatch, tmp_path, capsys):
    rc, out = _main_resolve(monkeypatch, tmp_path, capsys, FakeResp(404))
    assert rc == kc.RESOLVE_ERRORS_RC == 3
    assert "WARN kalshi resolve: 1 lookup error" in out


def test_main_resolve_pending_only_is_not_an_error(monkeypatch, tmp_path, capsys):
    rc, out = _main_resolve(monkeypatch, tmp_path, capsys, FakeResp(payload={"market": raw("A", result="", status="closed")}))
    assert rc == 0 and "'pending': 1" in out and "WARN" not in out


def test_main_resolve_clean_run_exits_zero(monkeypatch, tmp_path, capsys):
    rc, out = _main_resolve(monkeypatch, tmp_path, capsys, FakeResp(payload={"market": raw("A", result="yes", status="finalized")}))
    assert rc == 0 and "'resolved': 1" in out and "WARN" not in out


class _NoNetwork(FakeSession):
    """Every lookup in ``down`` raises a connection error, as when DNS cannot resolve the host."""
    def __init__(self, routes, down):
        super().__init__(routes)
        self.down = set(down)

    def get(self, url, params=None, timeout=None):
        path = url.split("/trade-api/v2")[1]
        if path in self.down:
            self.calls.append((path, params))
            raise requests.ConnectionError("Failed to resolve 'api.elections.kalshi.com'")
        return super().get(url, params, timeout)


def _db_with(tmp_path, tickers):
    db = str(tmp_path / "k.db")
    kc.collect(client({"/markets": FakeResp(payload={"markets": [mk(t, close_h=1) for t in tickers], "cursor": ""})}),
               db, now=NOW)
    return db


def test_resolve_stops_early_when_network_is_down(tmp_path):
    tickers = ["A", "B", "C", "D", "E", "F"]
    db = _db_with(tmp_path, tickers)
    s = _NoNetwork({}, down=[f"/markets/{t}" for t in tickers])
    c = KalshiClient(session=s, min_interval=0, sleep=lambda s: None)
    out = kc.resolve(c, db, now=NOW + timedelta(hours=2))
    assert out == {"resolved": 0, "pending": 0, "errors": 3, "skipped": 3}
    assert len({p for p, _ in s.calls}) == 3                    # the other three markets were never requested
    assert kc.status(db)["resolved_markets"] == 0               # nothing written; all six stay unresolved


def test_resolve_http_errors_do_not_trigger_early_stop(tmp_path):
    tickers = ["A", "B", "C", "D", "E"]
    db = _db_with(tmp_path, tickers)
    c = KalshiClient(session=FakeSession({f"/markets/{t}": FakeResp(404) for t in tickers}),
                     min_interval=0, sleep=lambda s: None)
    assert kc.resolve(c, db, now=NOW + timedelta(hours=2)) == {"resolved": 0, "pending": 0, "errors": 5, "skipped": 0}


def test_resolve_streak_resets_after_a_successful_lookup(tmp_path):
    db = _db_with(tmp_path, ["A", "B", "C", "D", "E"])
    settled = FakeResp(payload={"market": raw("C", result="yes", status="finalized")})
    s = _NoNetwork({"/markets/C": settled}, down=["/markets/A", "/markets/B", "/markets/D", "/markets/E"])
    c = KalshiClient(session=s, min_interval=0, sleep=lambda s: None)
    out = kc.resolve(c, db, now=NOW + timedelta(hours=2))
    assert out == {"resolved": 1, "pending": 0, "errors": 4, "skipped": 0}   # never 3 connection failures in a row


def test_main_resolve_early_stop_keeps_rc3_and_says_why(monkeypatch, tmp_path, capsys):
    db = _db_with(tmp_path, ["A", "B", "C", "D"])
    down = _NoNetwork({}, down=[f"/markets/{t}" for t in "ABCD"])
    monkeypatch.setattr(kc, "KalshiClient", lambda: KalshiClient(session=down, min_interval=0, sleep=lambda s: None))
    rc = kc.main(["resolve", "--db", db])
    out = capsys.readouterr().out
    assert rc == kc.RESOLVE_ERRORS_RC
    assert "WARN kalshi resolve: 3 lookup error(s), stopped early (1 not tried: no connection)" in out
