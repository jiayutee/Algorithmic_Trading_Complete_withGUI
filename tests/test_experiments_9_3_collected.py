import sqlite3
from contextlib import closing

import numpy as np
import pytest

from core import kalshi_collector as kc
from training_ground import experiments_9_3_collected as ec


def make_db(tmp_path, snaps, outcomes):
    """snaps: (ts, ticker, event, close_time, bid, ask); outcomes: {ticker: 0/1}."""
    db = str(tmp_path / "k.db")
    with closing(kc._connect(db)) as con:
        for ts, ticker, event, close, bid, ask in snaps:
            con.execute("INSERT INTO snapshots VALUES (?,?,?,?,?,?,?,?,?)", (ts, ticker, event, close, bid, ask, None, 500.0, 0.0))
        for ticker, res in outcomes.items():
            con.execute("INSERT INTO outcomes VALUES (?,?,?)", (ticker, res, "2026-09-21T00:00:00Z"))
        con.commit()
    return db


CLOSE = "2026-09-20T12:00:00Z"


def test_uses_last_snapshot_at_least_two_hours_before_close_never_a_later_one(tmp_path):
    db = make_db(tmp_path, [("2026-09-19T12:00:00Z", "A", "EV", CLOSE, 0.20, 0.25),     # 24h before close
                            ("2026-09-20T08:00:00Z", "A", "EV", CLOSE, 0.30, 0.35),     # 4h before: the one to use
                            ("2026-09-20T11:00:00Z", "A", "EV", CLOSE, 0.90, 0.95)],    # 1h before: must NOT be used
                 {"A": 1})
    df = ec.load_collected(db)
    assert len(df) == 1
    assert df.loc[0, "yes_ask"] == 0.35 and df.loc[0, "lead_h"] == pytest.approx(4.0)
    assert df.loc[0, "outcome"] == 1


def test_unresolved_and_too_late_markets_are_excluded_and_counted(tmp_path):
    db = make_db(tmp_path, [("2026-09-20T08:00:00Z", "OPEN", "E1", CLOSE, 0.2, 0.25),      # no outcome yet
                            ("2026-09-20T11:00:00Z", "LATE", "E2", CLOSE, 0.2, 0.25),      # only snapshot is 1h before close
                            ("2026-09-20T06:00:00Z", "OK", "E3", CLOSE, 0.2, 0.25)],
                 {"LATE": 0, "OK": 1})
    df = ec.load_collected(db)
    assert list(df["ticker"]) == ["OK"]
    assert df.attrs["resolved_markets"] == 2                     # OK and LATE resolved; LATE had no usable snapshot


def test_missing_event_becomes_its_own_cluster(tmp_path):
    db = make_db(tmp_path, [("2026-09-20T06:00:00Z", "X", None, CLOSE, 0.2, 0.25)], {"X": 0})
    assert list(ec.load_collected(db)["event"]) == ["X"]


def test_loading_is_read_only(tmp_path):
    db = make_db(tmp_path, [("2026-09-20T06:00:00Z", "A", "E", CLOSE, 0.2, 0.25)], {"A": 1})
    ec.load_collected(db)
    with closing(sqlite3.connect(db)) as con:
        assert con.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0] == 1


def test_empty_database_evaluates_without_crashing(tmp_path):
    db = make_db(tmp_path, [], {})
    res = ec.evaluate(ec.load_collected(db), n_boot=50)
    assert res["n_markets"] == 0 and res["lead_h"] is None
    assert all(h["underpowered"] and h["verdict"].startswith("UNDERPOWERED") for h in res["hypotheses"])
    assert "no usable snapshots" in res["protocol_deviations"][0]
    assert "UNDERPOWERED" in ec._fmt(res)


def test_small_sample_is_underpowered_not_no_evidence_or_a_finding(tmp_path):
    n = 40
    snaps = [(f"2026-09-20T0{i % 9}:00:00Z", f"T{i}", f"E{i}", CLOSE, 0.03, 0.05) for i in range(n)]
    db = make_db(tmp_path, snaps, {f"T{i}": 0 for i in range(n)})   # zero winners: looks like a strong longshot effect
    res = ec.evaluate(ec.load_collected(db), n_boot=200)
    L = res["hypotheses"][0]
    assert L["n_markets"] == n and L["underpowered"]
    assert L["verdict"].startswith("UNDERPOWERED")
    assert not L["criteria"]["FINDING"]                          # < 100 markets can never be a finding, however clean


def test_poisson_binomial_matches_binomial_and_sums_to_one():
    pmf = ec.poisson_binomial_pmf([0.1] * 5)
    assert pmf.sum() == pytest.approx(1.0)
    assert pmf[0] == pytest.approx(0.9 ** 5) and pmf[5] == pytest.approx(0.1 ** 5)


def test_calibrated_tail_directions():
    assert ec.calibrated_tail([0.05] * 76, 0, "negative") == pytest.approx(0.95 ** 76)         # P(no winners | fair)
    assert ec.calibrated_tail([0.98] * 38, 38, "positive") == pytest.approx(0.98 ** 38)        # P(all win | fair)
    assert ec.calibrated_tail([0.05] * 76, 76, "negative") == pytest.approx(1.0)
    assert 0 <= ec.calibrated_tail(np.linspace(0.01, 0.09, 30), 3, "positive") <= 1
