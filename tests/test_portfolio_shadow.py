import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from core import portfolio_shadow as ps
from core.portfolio_rules import HISTORY_NEEDED
from training_ground import experiments_14_0 as ex


def make_closes(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.035, (n, len(ps.UNIVERSE))), axis=0)), index=idx, columns=ps.UNIVERSE)


class Feed:
    """A growing set of completed daily closes, like days passing."""
    def __init__(self, closes, upto):
        self.closes, self.upto = closes, upto

    def __call__(self):
        return self.closes.iloc[: self.upto]


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / "shadow.sqlite3")


def test_first_run_sets_the_start_day_and_records_nothing_to_conclude(path):
    c = make_closes()
    led = ps.ShadowLedger(path, Feed(c, HISTORY_NEEDED + 3))
    out = led.update()
    assert out["processed"] == 0 and out["started"] == str(c.index[HISTORY_NEEDED + 2].date())
    st = led.status()
    assert st["days"] == 0 and all(st["stats"][r]["nav"] == 1.0 for r in ps.RULES)
    assert st["forward"]["verdict"].startswith("NO CONCLUSION YET")


def test_incremental_ledger_reproduces_the_phase_14_0_simulation_exactly(path):
    c = make_closes(330, seed=3)
    h = HISTORY_NEEDED
    feed = Feed(c, h + 1)                                             # start day = close index 112, as simulate's first decision
    led = ps.ShadowLedger(path, feed)
    led.update()
    feed.upto = len(c)
    led.update()
    returns = c.pct_change().dropna()
    ser = led.series()
    for rule in ps.RULES:
        ref = ex.simulate(returns, rule)
        got = ser[rule].iloc[1:]
        assert len(got) == len(ref) and np.allclose(got.to_numpy(), ref.to_numpy(), atol=1e-12), rule


def test_missing_days_are_backfilled_identically_to_daily_updates(tmp_path):
    c = make_closes(260, seed=1)
    a = ps.ShadowLedger(str(tmp_path / "a.sqlite3"), Feed(c, 125))
    a.update()
    for upto in range(126, 261):                                       # updated every single day
        a.closes_fn.upto = upto
        a.update()
    b = ps.ShadowLedger(str(tmp_path / "b.sqlite3"), Feed(c, 125))
    b.update()
    b.closes_fn.upto = 260                                             # the Mac slept for 135 days, then one update
    assert b.update()["processed"] == 135
    for rule in ps.RULES:
        assert np.allclose(a.series()[rule].to_numpy(), b.series()[rule].to_numpy(), atol=1e-12)
    assert a.status()["targets"] == b.status()["targets"]


def test_update_is_idempotent(path):
    c = make_closes()
    led = ps.ShadowLedger(path, Feed(c, 150))
    led.update()
    led.closes_fn.upto = 170
    first = led.update()
    again = led.update()
    assert first["processed"] == 20 and again["processed"] == 0
    with sqlite3.connect(path) as con:
        assert con.execute("SELECT COUNT(*) FROM daily WHERE rule='C2'").fetchone()[0] == 21       # start day + 20


def test_past_rows_never_change_when_later_data_arrives(path):
    c = make_closes()
    led = ps.ShadowLedger(path, Feed(c, 140))
    led.update()
    led.closes_fn.upto = 160
    led.update()
    before = led.series()["C2"].copy()
    led.closes_fn.upto = 200
    led.update()
    assert np.allclose(led.series()["C2"].iloc[: len(before)].to_numpy(), before.to_numpy(), atol=0)


def test_a_missing_close_refuses_and_writes_nothing(path):
    c = make_closes()
    led = ps.ShadowLedger(path, Feed(c, 140))
    led.update()
    bad = c.iloc[:150].copy()
    bad.iloc[-1, 2] = np.nan                                           # one coin has no close for the latest completed day
    led.closes_fn = lambda: bad
    with pytest.raises(ps.ShadowError, match="invent"):
        led.update()
    assert led.status()["days"] == 0                                    # nothing partial was recorded


def test_missing_coin_or_too_little_history_is_an_error(path):
    c = make_closes()
    with pytest.raises(ps.ShadowError, match="missing coins"):
        ps.ShadowLedger(path, lambda: c.drop(columns=["LTCUSDT"])).update()
    with pytest.raises(ps.ShadowError, match="days of closes"):
        ps.ShadowLedger(path, lambda: c.iloc[:50]).update()


def test_ledger_holds_nothing_until_the_first_rebalance_and_charges_fees_on_turnover(path):
    c = make_closes(160, seed=2)
    led = ps.ShadowLedger(path, Feed(c, HISTORY_NEEDED + 1))
    led.update()
    led.closes_fn.upto = HISTORY_NEEDED + 3
    led.update()
    with sqlite3.connect(path) as con:
        rows = con.execute("SELECT day, ret, cost, turnover, held FROM daily WHERE rule='EW' ORDER BY day").fetchall()
    assert rows[0][2] == 0 and sum(json.loads(rows[0][4]).values()) == 0              # start day: cash
    assert rows[1][3] == pytest.approx(1.0) and rows[1][2] == pytest.approx(ps.FEE)   # first day: buys the whole basket, pays 0.1%
    assert rows[2][3] == 0 and rows[2][2] == 0                                          # no rebalance, no fee


def test_forward_verdict_waits_for_enough_days_and_applies_the_pre_registered_criteria():
    idx = pd.date_range("2025-01-01", periods=ps.CONCLUSION_DAYS + 1, freq="D")
    good = {"EW": pd.Series(np.r_[0.0, np.tile([0.03, -0.03], 90)], index=idx),
            "TS28": pd.Series(np.r_[0.0, np.tile([0.01, -0.012], 90)], index=idx),
            "C2": pd.Series(np.r_[0.0, np.tile([0.012, -0.004], 90)], index=idx)}
    assert ps.ShadowLedger.forward_verdict(good, 50)["verdict"].startswith("NO CONCLUSION YET")
    v = ps.ShadowLedger.forward_verdict(good, ps.CONCLUSION_DAYS)
    assert v["verdict"] == "FORWARD-CONSISTENT" and v["stat_ready"] is False and "more days" in v["note"]
    bad = dict(good, C2=good["EW"] * 0.5 - 0.002)
    assert ps.ShadowLedger.forward_verdict(bad, ps.CONCLUSION_DAYS)["verdict"] == "NOT CONFIRMED"


def test_cli_update_and_status_print_and_failures_exit_3(path, capsys, monkeypatch):
    c = make_closes()
    monkeypatch.setattr(ps, "closes_from_binance", lambda *a, **k: c.iloc[:140])
    assert ps.main(["update", "--path", path]) == 0
    out = capsys.readouterr().out
    assert "Shadow ledger" in out and "pre-registered pick" in out and "NO CONCLUSION YET" in out and "No orders" in out
    monkeypatch.setattr(ps, "closes_from_binance", lambda *a, **k: c.iloc[:30])
    assert ps.main(["update", "--path", path]) == 3
    assert "WARN shadow ledger not updated" in capsys.readouterr().out
    assert ps.main(["status", "--path", path + ".none"]) == 0
    assert "not started" in capsys.readouterr().out


def test_binance_loader_excludes_today_and_uses_the_cache(monkeypatch):
    import core.kline_cache as kc
    seen = []

    def fake_get(symbol, interval, start_ms, end_ms, **kw):
        seen.append((symbol, interval, start_ms, end_ms))
        idx = pd.date_range(pd.to_datetime(start_ms, unit="ms"), pd.to_datetime(end_ms, unit="ms"), freq="D", inclusive="left")
        return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 2.0, "Volume": 1.0}, index=idx)

    monkeypatch.setattr(kc, "get_klines", fake_get)
    out = ps.closes_from_binance(now=pd.Timestamp("2026-11-16 15:00", tz="UTC").to_pydatetime(), days=10)
    assert list(out.columns) == ps.UNIVERSE and out.index[-1] == pd.Timestamp("2026-11-15")      # 11-16 is still forming
    assert all(i == "1d" and e == int(pd.Timestamp("2026-11-16").value // 10**6) for _, i, _, e in seen)
