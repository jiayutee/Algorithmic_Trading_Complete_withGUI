import numpy as np
import pandas as pd

from training_ground import experiments_6_9 as e


def closes(n=1000, k=8, start="2019-03-01", seed=0):
    r = np.random.default_rng(seed)
    return pd.DataFrame(100 * np.cumprod(1 + r.normal(0.0004, 0.03, (n, k)), axis=0),
                        index=pd.date_range(start, periods=n), columns=[f"S{i}" for i in range(k)])


def test_nothing_after_the_cutoff_is_used():
    c = closes(1600)                                              # runs well past 2022-09-09
    assert c.index[-1] > e.CUTOFF
    res = e.run(c, n_boot=100)
    assert pd.Timestamp(res["window"][1]) <= e.CUTOFF


def test_result_is_unchanged_by_post_cutoff_data():
    c = closes(1600)
    a = e.run(c, n_boot=100)
    c2 = c.copy(); c2.loc[c2.index > e.CUTOFF] *= 50              # absurd future prices must not matter
    b = e.run(c2, n_boot=100)
    assert a["stats"]["tsmom"]["sharpe"] == b["stats"]["tsmom"]["sharpe"]


def test_void_with_too_little_data():
    assert e.run(closes(300), n_boot=100)["void"]
    assert e.run(closes(1000, k=4), n_boot=100)["void"]
