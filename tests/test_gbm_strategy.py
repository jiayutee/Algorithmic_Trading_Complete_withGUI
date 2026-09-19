"""Phase 6.2: the GBM strategy and its training script."""
import json

import backtrader as bt
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")


@pytest.fixture(autouse=True)
def _isolated_experiment_log(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_LOG_PATH", str(tmp_path / "experiments.sqlite3"))

from core.feature_engineering import build_features, make_target
from core.ml_validation import evaluate_predictions, walk_forward_predict
from strategies.gbm_strategy import GBMStrategy, explain_row, make_lgbm_classifier


def _frame(returns, start="2022-01-01"):
    close = 100 * np.exp(np.cumsum(returns))
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"Open": close, "High": close * 1.005, "Low": close * 0.995, "Close": close,
         "Volume": rng.uniform(500, 1500, len(close))},
        index=pd.date_range(start, periods=len(close), freq="D"),
    )


def _random_walk(n=700, seed=3):
    return _frame(np.random.default_rng(seed).normal(0.0004, 0.02, n))


def _momentum_series(n=900, seed=4, phi=0.45):
    """Returns with strong autocorrelation: yesterday's return really does predict today's sign."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + rng.normal(0, 0.01)
    return _frame(r)


def _oos(df, **kw):
    X, y = build_features(df), make_target(df)
    wf = walk_forward_predict(make_lgbm_classifier, X, y, train_size=250, retrain_every=25, horizon=1, **kw)
    return evaluate_predictions(y, wf.predictions["p_up"]), wf, X


# -------------------------------------------- the model must not invent skill

def test_finds_no_skill_in_pure_noise():
    """If this fails the pipeline is leaking the future: an unpredictable series must score ~chance."""
    m, _, _ = _oos(_random_walk())
    assert 0.40 < m["auc"] < 0.60, m
    assert m["accuracy"] < 0.60, m


def test_finds_real_skill_when_the_data_has_it():
    m, _, _ = _oos(_momentum_series())
    assert m["auc"] > 0.60, m


# ------------------------------------------------------------ inside backtrader

def _run(df, **params):
    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.addstrategy(GBMStrategy, **params)
    cerebro.broker.setcash(100_000)
    return cerebro.run()[0]


def test_trades_and_explains_every_order():
    strat = _run(_momentum_series())
    assert strat.signals, "expected the model to take trades on data it can predict"
    assert strat.oos_metrics["auc"] > 0.55
    for sig in strat.signals:
        r = sig["rationale"]
        assert r["source"] == "strategy" and r["strategy"] == "GBM_LightGBM"
        assert 0.0 <= r["confidence"] <= 1.0
        assert "P(up)=" in r["summary"] and r["features"]["p_up"] is not None
    opened = [s["rationale"] for s in strat.signals if s["rationale"]["action"].startswith("open")]
    assert opened and all(r["feature_importance"] for r in opened)      # per-decision drivers recorded


def test_short_history_means_no_trades_and_no_crash():
    strat = _run(_random_walk(120))
    assert strat.signals == [] and strat._p_up.empty


def test_respects_thresholds_and_can_be_long_only():
    strat = _run(_momentum_series(), allow_short=False)
    assert strat.signals
    assert all(s["type"] in ("buy", "sell") for s in strat.signals)     # never sell_short / buy_cover


def test_nothing_trades_before_the_first_walk_forward_window_closes():
    """Predictions are NaN until train_size (+ purge gap) bars exist, so no order can be placed earlier."""
    df = _momentum_series()
    strat = _run(df, train_size=250)
    assert strat._p_up.iloc[:251].isna().all() and strat._p_up.iloc[251:].notna().any()
    first_signal_date = min(s["date"] for s in strat.signals)
    assert pd.Timestamp(first_signal_date) > df.index[250]


def test_explain_row_returns_the_largest_signed_contributions():
    df = _momentum_series()
    X, y = build_features(df).dropna(), make_target(df)
    Xc = X.join(y.rename("y")).dropna()
    model = make_lgbm_classifier().fit(Xc.drop(columns="y"), Xc["y"])
    drivers = explain_row(model, Xc.drop(columns="y").iloc[[-1]], top=3)
    assert len(drivers) == 3
    mags = [abs(v) for v in drivers.values()]
    assert mags == sorted(mags, reverse=True)


def test_registered_in_the_strategy_manager_and_dash():
    from core.strategy_manager import StrategyManager
    from dash_app.layout import _STRATEGIES
    assert "GBM (LightGBM)" in StrategyManager().get_available_strategies()
    assert "GBM (LightGBM)" in _STRATEGIES


# ------------------------------------------------------------ training script

def test_training_script_end_to_end_on_synthetic_data(tmp_path, monkeypatch, capsys):
    import training_ground.train_gbm as tg
    df = _momentum_series(700)

    class FakeLoader:
        def load_data(self, *a, **k):
            return df
    monkeypatch.setattr("core.data_loader.DataLoader", FakeLoader)
    out = str(tmp_path / "model")
    rc = tg.main(["--symbol", "TEST", "--days", "700", "--train-size", "250", "--retrain-every", "25", "--out", out])
    assert rc == 0
    meta = json.loads((tmp_path / "model.json").read_text())
    assert (tmp_path / "model.txt").exists()
    assert meta["walk_forward"]["metrics"]["auc"] > 0.55
    assert meta["walk_forward"]["auc_ci95"][0] > 0.5           # real skill -> CI clears 0.5
    assert "features" in meta and "rsi_14" in meta["features"]
    assert "VERDICT" in capsys.readouterr().out


def test_training_script_refuses_too_little_data(monkeypatch):
    import training_ground.train_gbm as tg
    class FakeLoader:
        def load_data(self, *a, **k):
            return _random_walk(100)
    monkeypatch.setattr("core.data_loader.DataLoader", FakeLoader)
    assert tg.main(["--train-size", "400"]) == 1


def test_rule_performance_charges_fees_on_every_position_change():
    from training_ground.train_gbm import rule_performance
    idx = pd.date_range("2024-01-01", periods=4)
    p = pd.Series([0.9, 0.9, 0.5, 0.1], index=idx)          # long, long, flat, short
    r = pd.Series([0.01, 0.02, 0.05, -0.03], index=idx)
    perf = rule_performance(p, r, 0.55, 0.45, fee=0.001, periods_per_year=252)
    assert perf["position_changes"] == 3 and perf["bars"] == 4
    gross = (1.01) * (1.02) * 1.0 * (1.03) - 1              # short earns +3% on a -3% bar
    assert perf["strategy_return_pct"] < gross * 100        # fees make it smaller
    assert perf["time_in_market_pct"] == 75.0


def test_bootstrap_ci_brackets_the_point_estimate_and_is_wide_for_noise():
    from sklearn.metrics import roc_auc_score
    from training_ground.train_gbm import bootstrap_auc_ci
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400).astype(float)
    good = y * 0.4 + rng.uniform(0, 0.6, 400)
    lo, hi = bootstrap_auc_ci(y, good)
    assert lo < roc_auc_score(y, good) < hi and lo > 0.6
    lo2, hi2 = bootstrap_auc_ci(y, rng.uniform(0, 1, 400))
    assert lo2 < 0.5 < hi2


# ------------------------------------------------ LSTM deprecation (Phase 6.3 decision)

def test_lstm_is_hidden_from_the_ui_but_still_resolvable_with_a_warning():
    from core.strategy_manager import StrategyManager
    sm = StrategyManager()
    assert "LSTM Predictor" not in sm.get_available_strategies()
    wrapper = sm.get_strategy("LSTM Predictor")
    assert wrapper is not None and wrapper.is_backtrader        # old configs keep working


def test_instantiating_the_lstm_emits_a_deprecation_warning():
    import warnings
    from strategies.ml_strategies import LSTMPredictor
    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=_random_walk(80)))
    cerebro.addstrategy(LSTMPredictor, ticker="NO_MODEL_XYZ")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cerebro.run()
    assert any(issubclass(w.category, DeprecationWarning) and "GBMStrategy" in str(w.message) for w in caught)


def test_training_script_logs_the_run_with_metrics_and_git_commit(tmp_path, monkeypatch):
    import training_ground.train_gbm as tg
    from core.experiment_log import ExperimentLog
    df = _momentum_series(700)

    class FakeLoader:
        def load_data(self, *a, **k):
            return df
    monkeypatch.setattr("core.data_loader.DataLoader", FakeLoader)
    assert tg.main(["--symbol", "TEST", "--train-size", "250", "--retrain-every", "25", "--out", str(tmp_path / "m")]) == 0
    runs = ExperimentLog().list_runs(model_type="lightgbm")
    assert len(runs) == 1
    r = runs[0]
    assert r["metrics"]["auc"] > 0.55 and "auc_ci95_low" in r["metrics"] and r["params"]["n_estimators"] == 150
    assert r["dataset"]["symbol"] == "TEST" and len(r["git_commit"]) == 40
    assert tg.main(["--symbol", "TEST", "--train-size", "250", "--retrain-every", "25", "--out", str(tmp_path / "m2"),
                    "--no-log"]) == 0
    assert len(ExperimentLog().list_runs()) == 1              # --no-log respected
