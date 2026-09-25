import sqlite3
import numpy as np
import pandas as pd

from training_ground import experiments_13_1 as ex


def _closes(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, n))), index=idx)


def _reads(closes, planted, n_days=120, seed=1):
    """One read per chosen day; planted=True makes the read follow the actual next-3-day direction."""
    rng = np.random.default_rng(seed)
    days = closes.index[10:10 + n_days]
    fwd = np.log(closes.shift(-3) / closes)
    reads = [int(np.sign(fwd.loc[d])) if planted else int(rng.choice([-1, 1])) for d in days]
    return pd.DataFrame({"symbol": "BTCUSDT", "date": [d.strftime("%Y-%m-%d") for d in days], "read": reads,
                         "n_bull": 1, "n_bear": 0})


def test_a_planted_signal_is_supported():
    c = _closes()
    res = ex.evaluate(_reads(c, planted=True), {"BTCUSDT": c})
    assert res["verdict"].startswith("SUPPORTED") and all(res["checks"].values())


def test_random_labels_are_not_supported():
    c = _closes()
    res = ex.evaluate(_reads(c, planted=False), {"BTCUSDT": c})
    assert not res["verdict"].startswith("SUPPORTED")


def test_too_few_reads_is_inconclusive_not_a_verdict():
    c = _closes()
    res = ex.evaluate(_reads(c, planted=True, n_days=8), {"BTCUSDT": c})
    assert res["verdict"].startswith("INCONCLUSIVE") and not res["checks"]["enough_reads"]


def test_no_reads_at_all_is_inconclusive():
    res = ex.evaluate(pd.DataFrame(columns=["symbol", "date", "read", "n_bull", "n_bear"]), {"BTCUSDT": _closes()})
    assert res["verdict"].startswith("INCONCLUSIVE")


def test_excess_return_removes_the_symbols_own_drift():
    idx = pd.date_range("2025-01-01", periods=200, freq="D")
    c = pd.Series(100 * np.exp(0.01 * np.arange(200)), index=idx)          # relentless uptrend
    reads = pd.DataFrame({"symbol": ["X"], "date": ["2025-03-01"], "read": [1], "n_bull": [1], "n_bear": [0]})
    v = ex.directional_excess(reads, {"X": c}, 3)
    assert abs(v["value"].iloc[0]) < 1e-9                                 # a bullish read in a steady trend earns nothing extra


def test_entry_is_the_close_of_the_publication_day_not_before_it():
    idx = pd.date_range("2025-01-01", periods=10, freq="D")
    c = pd.Series([100, 100, 200, 200, 200, 200, 200, 200, 200, 200.0], index=idx)   # jump on Jan 3
    reads = pd.DataFrame({"symbol": ["X"], "date": ["2025-01-03"], "read": [1], "n_bull": [1], "n_bear": [0]})
    v = ex.directional_excess(reads, {"X": c}, 1)
    lr_all = np.log(c.shift(-1) / c).dropna()
    assert np.isclose(v["value"].iloc[0], 0.0 - float(lr_all.mean()))       # the Jan 3 jump is not credited to a Jan 3 read


def test_load_reads_applies_context_filter_and_daily_majority(tmp_path):
    db = tmp_path / "n.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE news (datetime_utc TEXT, source TEXT, headline TEXT, url TEXT, summary TEXT, tickers TEXT)")
    rows = [("2026-09-01T10:00:00+00:00", "CoinDesk", "Bitcoin exchange hacked, funds stolen", "u1", "", '["BTCUSDT"]'),
            ("2026-09-01T11:00:00+00:00", "duckduckgo", "Bitcoin exchange hacked, funds stolen", "u2", "", '["BTCUSDT"]'),
            ("2026-09-02T10:00:00+00:00", "Forbes", "What Bitcoin Is And How It Works", "u3", "", '["BTCUSDT"]'),
            ("2026-09-03T10:00:00+00:00", "CoinDesk", "Bitcoin ETFs see inflows", "u4", "", '["BTCUSDT"]')]
    conn.executemany("INSERT INTO news VALUES (?,?,?,?,?,?)", rows)
    conn.commit(); conn.close()
    reads = ex.load_reads(str(db), ["BTCUSDT"])
    assert list(reads["date"]) == ["2026-09-01"] and list(reads["read"]) == [-1]
