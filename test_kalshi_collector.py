from datetime import datetime, timedelta, timezone

import pytest

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
    assert kc.resolve(c, db, now=later) == {"resolved": 1, "pending": 1, "errors": 0}   # C not past close: untouched
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
