import numpy as np
import pandas as pd

from training_ground import experiments_6_8 as e


def test_max_drawdown():
    assert abs(e.max_drawdown([0.1, -0.5, 0.2]) - (-0.5)) < 1e-12
    assert e.max_drawdown([0.01, 0.01]) == 0.0


def crash_universe(n=900, seed=0):
    """Assets that trend up, crash hard for ~120 days (a trend filter should dodge most of it), then recover."""
    r = np.random.default_rng(seed)
    base = np.concatenate([np.full(400, 0.004), np.full(120, -0.012), np.full(380, 0.004)])
    cols = {f"S{i}": base + r.normal(0, 0.012, n) for i in range(8)}
    return pd.DataFrame(cols, index=pd.date_range("2022-01-01", periods=n))


def test_confirms_planted_drawdown_reduction():
    res = e.run(crash_universe(), n_boot=300)
    assert res["D1_maxdd_diff"]["value"] > 0.05                  # TSMOM's drawdown clearly shallower
    assert res["stats"]["tsmom"]["max_drawdown"] > res["stats"]["ew"]["max_drawdown"]


def test_pure_noise_is_not_confirmed():
    r = np.random.default_rng(5)
    noise = pd.DataFrame(r.normal(0.0005, 0.03, (800, 8)), index=pd.date_range("2022-01-01", periods=800))
    assert not e.run(noise, n_boot=300)["criteria"]["CONFIRMED"]


def test_void_when_too_few_symbols(monkeypatch):
    monkeypatch.setattr(e, "load_universe", lambda **kw: {"LINKUSDT": {}, "DOTUSDT": {}})
    res = e.run()
    assert res["void"] and "need 5" in res["reason"]
