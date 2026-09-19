"""Kalshi data client + arbitrage signals, against mocked HTTP only (no network, no credentials)."""
import pytest

from core.kalshi_arbitrage import Signal, scan, scan_event, scan_market, taker_fee
from core.kalshi_data import Event, KalshiClient, KalshiError, Market, parse_market

pytestmark = pytest.mark.filterwarnings("ignore")


def raw(ticker="M-1", event="E-1", status="active", yes_bid=0.40, yes_ask=0.45, no_bid=0.55, no_ask=0.60, **kw):
    d = {"ticker": ticker, "event_ticker": event, "status": status, "title": "t",
         "yes_bid_dollars": f"{yes_bid:.4f}", "yes_ask_dollars": f"{yes_ask:.4f}",
         "no_bid_dollars": f"{no_bid:.4f}", "no_ask_dollars": f"{no_ask:.4f}",
         "yes_bid_size_fp": "100.00", "yes_ask_size_fp": "100.00", "volume_fp": "10.00",
         "open_interest_fp": "5.00", "last_price_dollars": "0.4200"}
    d.update(kw)
    return d


class FakeResp:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code, self._p, self.headers, self.text = status, payload, headers or {}, "x"

    def json(self):
        if self._p is None:
            raise ValueError("no json")
        return self._p


class FakeSession:
    def __init__(self, routes):
        self.routes, self.calls = routes, []          # routes: path -> response or list of responses

    def get(self, url, params=None, timeout=None):
        path = url.split("/trade-api/v2")[1]
        self.calls.append((path, params))
        r = self.routes[path]
        if isinstance(r, list):
            r = r.pop(0)
        return r


def client(routes, **kw):
    return KalshiClient(session=FakeSession(routes), min_interval=0, sleep=lambda s: None, **kw)


# ---------------------------------------------------------------- parsing

def test_parse_market_dollars_and_result():
    m = parse_market(raw(result="yes", status="finalized"))
    assert m.yes_ask == 0.45 and m.no_ask == 0.60 and m.result is True and m.volume == 10.0
    assert parse_market(raw(result="no")).result is False
    assert parse_market(raw(result="")).result is None


def test_parse_market_tolerates_missing_and_garbage_prices():
    m = parse_market({"ticker": "X", "yes_ask_dollars": "n/a"})
    assert m.yes_ask is None and m.yes_ask_quote is None and m.mid is None


def test_empty_book_ask_is_not_a_quote():
    m = parse_market(raw(yes_ask=0.0, yes_ask_size_fp="0.00"))
    assert m.yes_ask_quote is None


# ---------------------------------------------------------------- client

def test_orderbook_parsing_sorted_best_first_and_implied_asks():
    c = client({"/markets/M-1/orderbook": FakeResp(payload={"orderbook_fp": {
        "yes_dollars": [["0.30", "10"], ["0.40", "5"]], "no_dollars": [["0.50", "7"], ["0.55", "0"]]}})})
    ob = c.get_orderbook("M-1")
    assert ob.best_yes_bid.price == 0.40                       # sorted, best first
    assert [q.size for q in ob.no_bids] == [7.0]               # zero-size level dropped
    assert ob.implied_yes_ask.price == 0.50 and ob.implied_no_ask.price == 0.60
    assert ob.depth("yes") == 15.0


def test_iter_markets_follows_cursor_and_respects_max_items():
    c = client({"/markets": [FakeResp(payload={"markets": [raw("A"), raw("B")], "cursor": "c1"}),
                             FakeResp(payload={"markets": [raw("C"), raw("D")], "cursor": ""})]})
    assert [m.ticker for m in c.iter_markets(max_items=3)] == ["A", "B", "C"]
    c2 = client({"/markets": [FakeResp(payload={"markets": [raw("A")], "cursor": "c1"}),
                              FakeResp(payload={"markets": [raw("B")], "cursor": ""})]})
    assert len(list(c2.iter_markets(max_items=10))) == 2       # stops when the cursor runs out


def test_retries_on_429_then_succeeds_and_gives_up_after_max():
    c = client({"/markets/A": [FakeResp(429), FakeResp(200, {"market": raw("A")})]})
    assert c.get_market("A").ticker == "A"
    c2 = client({"/markets/A": [FakeResp(503)] * 3}, max_retries=3)
    with pytest.raises(KalshiError):
        c2.get_market("A")


def test_client_error_paths():
    with pytest.raises(KalshiError):
        client({"/markets/A": FakeResp(404)}).get_market("A")
    with pytest.raises(KalshiError):
        client({"/markets/A": FakeResp(200, None)}).get_market("A")
    with pytest.raises(KalshiError):
        client({"/markets/A": FakeResp(200, {"oops": 1})}).get_market("A")


def test_multivariate_combos_excluded_by_default():
    s = FakeSession({"/markets": [FakeResp(payload={"markets": []}), FakeResp(payload={"markets": []})]})
    c = KalshiClient(session=s, min_interval=0, sleep=lambda x: None)
    c.list_markets(); c.list_markets(include_multivariate=True)
    assert s.calls[0][1]["mve_filter"] == "exclude" and "mve_filter" not in s.calls[1][1]


def test_none_params_not_sent():
    s = FakeSession({"/markets": FakeResp(payload={"markets": []})})
    KalshiClient(session=s, min_interval=0, sleep=lambda x: None).list_markets(status="open")
    assert "cursor" not in s.calls[0][1] and "event_ticker" not in s.calls[0][1]


def test_client_has_no_write_methods():
    """Phase 9.0 contract: read-only."""
    assert not [n for n in dir(KalshiClient) if any(w in n.lower() for w in ("order", "buy", "sell", "submit", "cancel"))
                and n != "get_orderbook"]


def test_pacing_sleeps_between_calls():
    sleeps, t = [], [0.0]
    c = KalshiClient(session=FakeSession({"/markets/A": FakeResp(payload={"market": raw("A")})}),
                     min_interval=0.5, sleep=lambda s: sleeps.append(s), clock=lambda: t[0])
    c.get_market("A"); c.get_market("A")
    assert sleeps and sleeps[-1] == pytest.approx(0.5)


# ---------------------------------------------------------------- arbitrage

def test_fee_formula():
    assert taker_fee(0.5) == 0.02                              # 0.07*0.25 = 0.0175 -> 0.02
    assert taker_fee(0.01) == 0.01                             # rounds up, never to zero
    assert taker_fee(0.0) == 0.0 and taker_fee(1.0) == 0.0


def test_within_market_arb_detected_and_normal_book_ignored():
    normal = parse_market(raw(yes_ask=0.45, no_ask=0.56))      # 1.01 + fees -> no arb
    assert scan_market(normal) is None
    arb = parse_market(raw(yes_ask=0.40, no_ask=0.50, yes_bid_size_fp="50.00", yes_ask_size_fp="50.00"))
    s = scan_market(arb)
    assert s and s.kind == "within_market" and s.edge_per_set == pytest.approx(1 - (0.90 + 0.02 + 0.02))
    assert s.max_sets == 50 and s.edge_pct > 0


def test_within_market_needs_edge_after_fees_and_size_and_active():
    thin = parse_market(raw(yes_ask=0.47, no_ask=0.50))         # 0.97 - fees(0.04) < 0.01 edge
    assert scan_market(thin) is None
    small = parse_market(raw(yes_ask=0.30, no_ask=0.30, yes_bid_size_fp="1.00", yes_ask_size_fp="1.00"))
    assert scan_market(small) is None
    closed = parse_market(raw(status="closed", yes_ask=0.30, no_ask=0.30))
    assert scan_market(closed) is None


def ev(markets, me=True):
    return Event("E-1", "t", me, "c", markets)


def m(t, yes_ask, no_ask, size="100.00"):
    return parse_market(raw(t, yes_ask=yes_ask, no_ask=no_ask, yes_bid_size_fp=size, yes_ask_size_fp=size))


def test_event_buy_all_yes_flagged_needs_exhaustive():
    sigs = scan_event(ev([m("A", 0.20, 0.85), m("B", 0.20, 0.85), m("C", 0.20, 0.85)]))
    yes = [s for s in sigs if s.kind == "event_buy_all_yes"]
    assert len(yes) == 1 and yes[0].needs_exhaustive and yes[0].confidence < 0.6
    assert yes[0].edge_per_set == pytest.approx(1 - (0.60 + 3 * taker_fee(0.20)))


def test_event_buy_all_no_signal_when_underpriced():
    sigs = scan_event(ev([m("A", 0.70, 0.20), m("B", 0.70, 0.20), m("C", 0.70, 0.20)]))
    no = [s for s in sigs if s.kind == "event_buy_all_no"]
    assert len(no) == 1 and not no[0].needs_exhaustive
    assert no[0].edge_per_set == pytest.approx(2 - (0.60 + 3 * taker_fee(0.20)))


def test_event_fair_pricing_gives_no_signal():
    assert scan_event(ev([m("A", 0.35, 0.68), m("B", 0.35, 0.68), m("C", 0.32, 0.68)])) == []


def test_event_skips_non_exclusive_incomplete_and_single():
    cheap = [m("A", 0.20, 0.20), m("B", 0.20, 0.20)]
    assert scan_event(ev(cheap, me=False)) == []
    assert scan_event(ev(cheap[:1])) == []
    unpriced = parse_market(raw("C", yes_ask=0.0, yes_ask_size_fp="0.00", no_ask=0.0))
    assert scan_event(ev([m("A", 0.2, 0.2), unpriced])) == []   # one dead leg -> set incomplete


def test_event_size_is_thinnest_leg():
    sigs = scan_event(ev([m("A", 0.20, 0.85, "100.00"), m("B", 0.20, 0.85, "8.00"), m("C", 0.20, 0.85, "50.00")]))
    assert sigs and sigs[0].max_sets == 8


def test_scan_end_to_end_skips_broken_event_and_sorts():
    routes = {
        "/markets": FakeResp(payload={"markets": [raw("A", "E-1"), raw("B", "E-2"), raw("C", "E-1")], "cursor": ""}),
        "/events/E-1": FakeResp(500), "/events/E-2": FakeResp(payload={"event": {
            "event_ticker": "E-2", "title": "t", "mutually_exclusive": True},
            "markets": [raw("X", "E-2", yes_ask=0.2, no_ask=0.85), raw("Y", "E-2", yes_ask=0.2, no_ask=0.85),
                        raw("Z", "E-2", yes_ask=0.2, no_ask=0.85)]}),
    }
    sigs = scan(client(routes, max_retries=1))
    assert [s.market_id for s in sigs] == ["E-2"] and sigs[0].kind == "event_buy_all_yes"
