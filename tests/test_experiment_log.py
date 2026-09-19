"""Phase 11.3: local SQLite experiment log."""
import json
import sqlite3
import subprocess

import numpy as np
import pandas as pd
import pytest

from core.experiment_log import ExperimentLog, _cli, default_path, git_state


@pytest.fixture
def log(tmp_path):
    lg = ExperimentLog(str(tmp_path / "exp.sqlite3"))
    yield lg
    lg.close()


GIT = {"commit": "abc1234def5678", "dirty": False}


def test_a_run_round_trips_with_everything_recorded(log):
    rid = log.log_run(name="gbm btc", model_type="lightgbm", params={"n_estimators": 150, "walk_forward": {"train": 400}},
                      metrics={"auc": 0.491, "auc_ci95": [0.456, 0.523]}, dataset={"symbol": "BTCUSDT", "bars": 1500},
                      tags=["baseline", "btc"], notes="first honest baseline", git=GIT)
    r = log.get_run(rid)
    assert r["name"] == "gbm btc" and r["model_type"] == "lightgbm" and r["status"] == "completed"
    assert r["params"]["walk_forward"]["train"] == 400 and r["metrics"]["auc_ci95"] == [0.456, 0.523]
    assert r["dataset"]["bars"] == 1500 and r["tags"] == ["baseline", "btc"] and r["notes"] == "first honest baseline"
    assert r["git_commit"] == "abc1234def5678" and r["git_dirty"] is False
    assert r["created_at"].endswith("+00:00")


def test_git_commit_is_captured_automatically_in_this_repo(log):
    rid = log.log_run(name="auto", model_type="x")
    r = log.get_run(rid)
    state = git_state()
    assert r["git_commit"] == state["commit"] and len(r["git_commit"]) == 40 and isinstance(r["git_dirty"], bool)


def test_git_state_never_raises_outside_a_repo(tmp_path):
    assert git_state(str(tmp_path)) == {"commit": None, "dirty": None}


def test_numpy_pandas_nan_and_paths_are_stored_safely(log):
    rid = log.log_run(name="np", model_type="x", git=GIT,
                      metrics={"a": np.float64(0.5), "n": np.int64(7), "nan": float("nan"), "inf": float("inf"),
                               "series_mean": pd.Series([1.0, 2.0]).mean()},
                      params={"path": __file__, "when": pd.Timestamp("2026-01-01")})
    r = log.get_run(rid)
    assert r["metrics"]["a"] == 0.5 and r["metrics"]["n"] == 7 and r["metrics"]["series_mean"] == 1.5
    assert r["metrics"]["nan"] is None and r["metrics"]["inf"] is None            # valid JSON, not NaN tokens
    assert isinstance(r["params"]["when"], str)


def test_list_filters_and_orders_newest_first(log):
    a = log.log_run(name="H1a horizon5", model_type="lightgbm", tags=["6.5"], git=GIT)
    b = log.log_run(name="H5 volatility", model_type="lightgbm", tags=["6.5", "vol"], git=GIT)
    c = log.log_run(name="lstm try", model_type="lstm", git=GIT)
    assert [r["id"] for r in log.list_runs()] == [c, b, a]
    assert [r["id"] for r in log.list_runs(model_type="lightgbm")] == [b, a]
    assert [r["id"] for r in log.list_runs(name_contains="volatility")] == [b]
    assert [r["id"] for r in log.list_runs(tag="vol")] == [b]
    assert [r["id"] for r in log.list_runs(limit=1)] == [c]
    assert log.list_runs(since="2999-01-01") == [] and log.count() == 3


def test_best_ranks_by_a_metric_including_nested_ones_and_skips_runs_without_it(log):
    log.log_run(name="low", model_type="m", metrics={"auc": 0.50, "walk_forward": {"auc": 0.48}}, git=GIT)
    top = log.log_run(name="high", model_type="m", metrics={"auc": 0.73, "walk_forward": {"auc": 0.71}}, git=GIT)
    log.log_run(name="none", model_type="m", metrics={"sharpe": 1.0}, git=GIT)
    log.log_run(name="text", model_type="m", metrics={"auc": "n/a"}, git=GIT)
    assert [r["name"] for r in log.best("auc", n=5)] == ["high", "low"]
    assert log.best("walk_forward.auc", n=1)[0]["id"] == top
    assert log.best("auc", higher_is_better=False, n=1)[0]["name"] == "low"
    assert log.best("nonexistent") == []


def test_frame_and_compare_flatten_params_and_metrics(log):
    r1 = log.log_run(name="a", model_type="m", params={"lr": 0.03, "wf": {"train": 400}}, metrics={"auc": 0.5}, git=GIT)
    r2 = log.log_run(name="b", model_type="m", params={"lr": 0.1, "wf": {"train": 400}}, metrics={"auc": 0.6}, git=GIT)
    df = log.to_frame()
    assert {"param.lr", "param.wf.train", "metric.auc"} <= set(df.columns) and len(df) == 2
    cmp_ = log.compare([r1, r2, 999])                                   # unknown ids ignored
    assert list(cmp_.columns) == [r1, r2]
    assert cmp_.loc["param.lr"].tolist() == [0.03, 0.1] and cmp_.loc["metric.auc"].tolist() == [0.5, 0.6]


def test_the_log_persists_across_connections_and_is_plain_sqlite(tmp_path):
    path = str(tmp_path / "persist.sqlite3")
    lg = ExperimentLog(path); rid = lg.log_run(name="keep", model_type="m", git=GIT); lg.close()
    assert ExperimentLog(path).get_run(rid)["name"] == "keep"
    raw = sqlite3.connect(path)                                          # readable without this module
    assert raw.execute("select name from runs").fetchall() == [("keep",)]


def test_env_var_controls_the_default_location(monkeypatch, tmp_path):
    monkeypatch.setenv("EXPERIMENT_LOG_PATH", str(tmp_path / "custom.sqlite3"))
    assert default_path().endswith("custom.sqlite3")
    lg = ExperimentLog(); lg.log_run(name="x", model_type="m", git=GIT); lg.close()
    assert (tmp_path / "custom.sqlite3").exists()


def test_cli_list_show_best_compare(tmp_path, capsys):
    path = str(tmp_path / "cli.sqlite3")
    lg = ExperimentLog(path)
    a = lg.log_run(name="run a", model_type="lightgbm", metrics={"auc": 0.61}, params={"lr": 1}, git=GIT)
    b = lg.log_run(name="run b", model_type="lightgbm", metrics={"auc": 0.55}, params={"lr": 2}, git=GIT)
    lg.close()
    assert _cli(["--path", path, "list"]) == 0
    out = capsys.readouterr().out
    assert "run a" in out and "run b" in out and "(2 shown of 2)" in out and "abc1234" in out
    assert _cli(["--path", path, "show", str(a)]) == 0 and json.loads(capsys.readouterr().out)["name"] == "run a"
    assert _cli(["--path", path, "show", "99"]) == 1
    capsys.readouterr()
    _cli(["--path", path, "best", "auc"])
    assert capsys.readouterr().out.splitlines()[0].startswith(f"#{a}")
    _cli(["--path", path, "compare", str(a), str(b)])
    assert "param.lr" in capsys.readouterr().out
