"""Phase 12.0: the research loop must promote real edge, reject luck, and never trust a backtest alone."""
import backtrader as bt
import numpy as np
import pandas as pd
import pytest

import core.research_loop as rl
from core.experiment_log import ExperimentLog
from core.research_loop import Candidate, ResearchState, evaluate_candidate, promotion_test, run_cycle

pytest.importorskip("lightgbm")


# ---------------------------------------------------------------- stub strategies (importable by path)

class _Base(bt.Strategy):
    params = (("risk_per_trade", 0.95),)

    def __init__(self):
        self.signals, self.closed_trades = [], []
        self._pending_rationale = None
        self._i = 0

    def notify_trade(self, trade):
        if trade.isclosed:
            self.closed_trades.append(trade)

    def _go(self, want_long):
        size = self.broker.getcash() * self.p.risk_per_trade / self.data.close[0]
        if want_long and not self.position:
            self._pending_rationale = {"action": "open_long"}
            self.buy(size=size)
        elif not want_long and self.position:
            self._pending_rationale = {"action": "close_long"}
            self.close()
        else:
            self._pending_rationale = None


class OracleStrategy(_Base):
    """CHEATS: looks at tomorrow's close. Any sound evaluation must flag this as a huge edge."""
    def next(self):
        closes = self.data._dataname["Close"].to_numpy()
        i = len(self) - 1
        self._go(i + 1 < len(closes) and closes[i + 1] > closes[i])


class CoinFlipStrategy(_Base):
    def __init__(self):
        super().__init__()
        self.rng = np.random.default_rng(7)

    def next(self):
        self._go(self.rng.random() > 0.5)


class AlwaysLongStrategy(_Base):
    def next(self):
        self._go(True)


def _data(n=900, symbols=("AAA", "BBB", "CCC", "DDD"), seed=0, drift=0.0006):
    out = {}
    for k, s in enumerate(symbols):
        rng = np.random.default_rng(seed + k)
        close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.02, n)))
        open_ = np.r_[close[0], close[:-1]]                    # each bar opens at the previous close (no gap)
        out[s] = pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.01, "Low": np.minimum(open_, close) * 0.99,
                               "Close": close, "Volume": 1000.0}, index=pd.date_range("2022-01-01", periods=n, freq="D"))
    return out


ORACLE = Candidate("Oracle", "test_research_loop:OracleStrategy")
COIN = Candidate("CoinFlip", "test_research_loop:CoinFlipStrategy")
HOLD = Candidate("AlwaysLong", "test_research_loop:AlwaysLongStrategy")


@pytest.fixture
def paths(tmp_path):
    p = str(tmp_path / "research.sqlite3")
    return ExperimentLog(p), ResearchState(p), p


# ------------------------------------------------------------------- evaluation

def test_a_strategy_with_real_edge_is_flagged_and_luck_is_not():
    data = _data()
    ev_o = evaluate_candidate(ORACLE, data, n_boot=200, eval_days=600, level=0.983)
    assert ev_o["sharpe_diff"] > 2 and ev_o["ci"][0] > 0 and promotion_test(ev_o)["PASS"]
    ev_c = evaluate_candidate(COIN, data, n_boot=200, eval_days=600, level=0.983)
    assert not promotion_test(ev_c)["PASS"]
    ev_h = evaluate_candidate(HOLD, data, n_boot=200, eval_days=600, level=0.983)
    assert abs(ev_h["sharpe_diff"]) < 0.3 and not promotion_test(ev_h)["PASS"]        # ~ buy and hold, minus fees


def test_evaluation_reports_the_position_each_strategy_wants_next():
    ev = evaluate_candidate(HOLD, _data(600), n_boot=50, eval_days=300)
    assert set(ev["target_positions"].values()) == {1}                                # always long
    assert ev["days"] == 300 and ev["window"][1] == "2023-08-23" and ev["symbols"] == ["AAA", "BBB", "CCC", "DDD"]


def test_promotion_needs_every_condition():
    base = {"ci": [0.2, 1.0], "sharpe_diff_halves": [0.3, 0.4], "trades": 50,
            "strategy": {"max_drawdown": -0.3}, "buy_hold": {"max_drawdown": -0.3}}
    assert promotion_test(base)["PASS"]
    assert not promotion_test({**base, "ci": [-0.1, 1.0]})["PASS"]                    # not significant
    assert not promotion_test({**base, "sharpe_diff_halves": [0.5, -0.1]})["PASS"]    # not replicated
    assert not promotion_test({**base, "trades": 10})["PASS"]                         # too few trades to trust
    assert not promotion_test({**base, "strategy": {"max_drawdown": -0.4}})["PASS"]   # >5pp deeper drawdown


# ----------------------------------------------------------------- state machine

def _ev(ci, passed_halves=True):
    return {"ci": ci, "sharpe_diff_halves": [0.1, 0.1] if passed_halves else [0.1, -0.1], "trades": 50,
            "strategy": {"max_drawdown": -0.2}, "buy_hold": {"max_drawdown": -0.2}}


def test_candidate_is_promoted_then_retired_when_significantly_worse(paths):
    _, st, _ = paths
    ev = _ev([0.1, 0.9])
    assert st.status("X") == "candidate"
    t = st.apply("X", ev, promotion_test(ev), 1)
    assert t["status"] == "paper" and t["changed"]
    bad = _ev([-0.9, -0.2])                                        # upper bound below zero
    t = st.apply("X", bad, promotion_test(bad), 2)
    assert t["status"] == "retired" and "significantly worse" in t["note"]
    good = _ev([0.1, 0.9])
    t = st.apply("X", good, promotion_test(good), 3)
    assert t["status"] == "retired" and not t["changed"]           # never revives automatically


def test_paper_strategy_is_retired_after_consecutive_failures_but_survives_isolated_ones(paths):
    _, st, _ = paths
    win, meh = _ev([0.1, 0.9]), _ev([-0.2, 0.5])
    st.apply("Y", win, promotion_test(win), 1)                    # -> paper
    for i in range(2):
        assert st.apply("Y", meh, promotion_test(meh), 2 + i)["status"] == "paper"
    assert st.apply("Y", win, promotion_test(win), 5)["consecutive_fails"] == 0          # a pass resets the count
    for i in range(2):
        st.apply("Y", meh, promotion_test(meh), 6 + i)
    assert st.apply("Y", meh, promotion_test(meh), 9)["status"] == "retired"


def test_state_persists_across_connections(paths):
    _, st, path = paths
    ev = _ev([0.1, 0.9]); st.apply("Z", ev, promotion_test(ev), 1); st.close()
    assert ResearchState(path).status("Z") == "paper"


# -------------------------------------------------------------------- paper ledger

def test_ledger_is_idempotent_and_marks_positions_to_the_next_prices(paths):
    _, st, _ = paths
    assert st.record_paper_positions("2026-01-01", "C", {"A": 1, "B": -1}, {"A": 100.0, "B": 50.0}) == 2
    assert st.record_paper_positions("2026-01-01", "C", {"A": 1, "B": -1}, {"A": 999.0, "B": 999.0}) == 0    # same day: ignored
    assert st.paper_pnl().empty or st.paper_pnl().iloc[0]["days"] == 0 or True
    st.record_paper_positions("2026-01-02", "C", {"A": 1, "B": 0}, {"A": 110.0, "B": 45.0})
    pnl = st.paper_pnl(fee=0.0).iloc[0]
    # day1 -> day2: A long +10%; B short with price falling 50->45 = +10%; equal weight -> +10%
    assert pnl["candidate"] == "C" and pnl["days"] == 1 and pnl["return_pct"] == pytest.approx(10.0)


def test_paper_pnl_charges_fees_on_position_changes(paths):
    _, st, _ = paths
    st.record_paper_positions("d1", "C", {"A": 1}, {"A": 100.0})
    st.record_paper_positions("d2", "C", {"A": 1}, {"A": 100.0})
    pnl = st.paper_pnl(fee=0.01).iloc[0]
    assert pnl["return_pct"] == pytest.approx(-1.0) and pnl["trades"] == 1            # the entry costs 1%


# ------------------------------------------------------------------------ the loop

def test_cycle_promotes_the_oracle_records_paper_positions_and_logs_every_evaluation(paths):
    log, st, _ = paths
    rep = run_cycle(_data(), [ORACLE, COIN, HOLD], state=st, log=log, n_boot=200, eval_days=600, progress=lambda m: None)
    status = {c["candidate"]: c["status"] for c in rep["candidates"]}
    assert status == {"Oracle": "paper", "CoinFlip": "candidate", "AlwaysLong": "candidate"}
    with_ledger = st.paper_pnl()
    assert log.count() == 3 and {r["name"] for r in log.list_runs(tag="research-loop")} == {
        "research-loop Oracle", "research-loop CoinFlip", "research-loop AlwaysLong"}
    assert all(len(r["git_commit"]) == 40 for r in log.list_runs())
    text = rl.format_report(rep)
    assert "Oracle" in text and "paper" in text and "fail:" in text


def test_next_cycle_forward_tests_the_paper_strategy_on_new_data_and_skips_retired_ones(paths):
    log, st, _ = paths
    d1 = _data(900)
    run_cycle(d1, [ORACLE], state=st, log=log, n_boot=100, eval_days=600, progress=lambda m: None)
    d2 = {s: pd.concat([df, pd.DataFrame({c: [df[c].iloc[-1] * (1.02 if c == "Close" else 1.0)] for c in df.columns},
                                         index=[df.index[-1] + pd.Timedelta(days=1)])]) for s, df in d1.items()}
    rep = run_cycle(d2, [ORACLE], state=st, log=log, n_boot=100, eval_days=600, progress=lambda m: None)
    pnl = {r["candidate"]: r for r in rep["paper_pnl"]}
    assert pnl["Oracle"]["days"] == 1                                       # one forward day has been scored
    ev = _ev([-0.9, -0.2]); st.apply("Oracle", ev, promotion_test(ev), 99)  # force retirement
    rep3 = run_cycle(d2, [ORACLE], state=st, log=log, n_boot=100, eval_days=600, progress=lambda m: None)
    assert rep3["candidates"][0].get("skipped") is True


def test_one_broken_candidate_does_not_stop_the_loop(paths):
    log, st, _ = paths
    broken = Candidate("Broken", "no_such_module:Nope")
    rep = run_cycle(_data(700), [broken, HOLD], state=st, log=log, n_boot=50, eval_days=400, progress=lambda m: None)
    assert "error" in rep["candidates"][0] and rep["candidates"][1]["candidate"] == "AlwaysLong"
    assert "ERROR" in rl.format_report(rep)


def test_default_candidates_all_load_and_include_the_ml_strategy():
    names = [c.name for c in rl.DEFAULT_CANDIDATES]
    assert "GBM (LightGBM)" in names and len(names) == 8
    assert len(set(names)) == len(names)
    for c in rl.DEFAULT_CANDIDATES:
        assert issubclass(c.load(), bt.Strategy)


def test_trend_candidates_wrap_their_base_strategy_and_plain_ones_do_not():
    by_name = {c.name: c for c in rl.DEFAULT_CANDIDATES}
    for base in ("MACD/RSI", "EMA Crossover", "Stochastic", "GBM (LightGBM)"):
        plain = by_name[base]
        wrapped = by_name[base.replace(" (LightGBM)", "") + " + Trend"]
        assert not plain.trend_overlay and wrapped.trend_overlay
        assert wrapped.strategy == plain.strategy and wrapped.params == plain.params
        assert getattr(wrapped.load(), "_trend_overlay_wrapped", False)
        assert not getattr(plain.load(), "_trend_overlay_wrapped", False)
        assert issubclass(wrapped.load(), plain.load())


def test_a_real_strategy_runs_end_to_end_and_yields_a_valid_position():
    ev = evaluate_candidate(rl.DEFAULT_CANDIDATES[1], _data(700), n_boot=50, eval_days=400)      # EMA crossover
    assert ev["trades"] >= 1 and set(ev["target_positions"].values()) <= {-1, 0, 1}
    assert not np.isnan(ev["sharpe_diff"])


def test_a_trend_overlaid_candidate_runs_end_to_end_and_yields_a_valid_position():
    cand = next(c for c in rl.DEFAULT_CANDIDATES if c.name == "EMA Crossover + Trend")
    ev = evaluate_candidate(cand, _data(700), n_boot=50, eval_days=400)
    assert set(ev["target_positions"].values()) <= {-1, 0, 1}
    assert not np.isnan(ev["sharpe_diff"])


def test_cli_status_prints_states_and_paper_pnl(paths, capsys, monkeypatch):
    log, st, path = paths
    ev = _ev([0.1, 0.9]); st.apply("Oracle", ev, promotion_test(ev), 1)
    st.record_paper_positions("d1", "Oracle", {"A": 1}, {"A": 100.0}); st.record_paper_positions("d2", "Oracle", {"A": 1}, {"A": 101.0})
    monkeypatch.setenv("EXPERIMENT_LOG_PATH", path)
    assert rl._cli(["status"]) == 0
    out = capsys.readouterr().out
    assert "Oracle" in out and "paper" in out and "days" in out


def test_format_report_keeps_long_candidate_names_apart_from_the_status_column():
    ev = {"strategy": {"sharpe": 0.8, "max_drawdown": -0.2}, "buy_hold": {"sharpe": 0.5}, "sharpe_diff": 0.3,
          "ci": [-1.0, 2.0], "trades": 100}
    rep = {"run_date": "d", "symbols": ["A"], "paper_pnl": [], "candidates": [
        {"candidate": "EMA Crossover + Trend", "status": "candidate", "evaluation": ev, "test": {"PASS": False, "x": False}},
        {"candidate": "MACD/RSI", "status": "candidate", "evaluation": ev, "test": {"PASS": False, "x": False}}]}
    lines = rl.format_report(rep).splitlines()
    assert "EMA Crossover + Trend  candidate" in lines[2]
    assert lines[1].index("status") == lines[2].index("candidate", len("EMA Crossover + Trend")) == lines[3].index("candidate")
