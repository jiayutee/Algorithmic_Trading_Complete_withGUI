import numpy as np
import pandas as pd

from training_ground import experiments_9_3 as e


def candle(end, ask, bid):
    return {"end_period_ts": end, "yes_ask": {"close_dollars": str(ask)}, "yes_bid": {"close_dollars": str(bid)}}


def test_snapshot_uses_last_candle_at_or_before_cutoff_never_later():
    cs = [candle(100, 0.30, 0.20), candle(200, 0.50, 0.40), candle(300, 0.99, 0.98)]
    assert e.snapshot_quotes(cs, 250)["yes_ask"] == 0.50
    assert e.snapshot_quotes(cs, 50) is None                    # nothing before cutoff -> dropped, no peeking ahead


def test_snapshot_rejects_bad_quotes():
    assert e.snapshot_quotes([candle(1, 0.0, 0.0)], 5) is None
    assert e.snapshot_quotes([candle(1, 0.4, 0.5)], 5) is None  # crossed
    assert e.snapshot_quotes([{"end_period_ts": 1}], 5) is None


def frame(n, ask, p_win, events, seed=0):
    r = np.random.default_rng(seed)
    return pd.DataFrame({"ticker": [f"T{i}" for i in range(n)], "event": [f"E{i % events}" for i in range(n)],
                         "close_ts": np.arange(n) * 1000, "yes_ask": ask, "yes_bid": ask - 0.02,
                         "outcome": (r.random(n) < p_win).astype(int)})


def test_detects_planted_longshot_overpricing():
    df = frame(600, 0.08, 0.01, 300)                            # priced 8%, wins 1%
    h = e.test_bucket(df, "L", df["yes_ask"] <= 0.10, "negative", n_boot=300)
    assert h["criteria"]["FINDING"] and h["mean_edge"] < 0 and h["TRADABLE"]


def test_calibrated_market_gives_no_finding():
    df = frame(600, 0.08, 0.08, 300)
    h = e.test_bucket(df, "L", df["yes_ask"] <= 0.10, "negative", n_boot=300)
    assert not h["criteria"]["FINDING"]                         # fair pricing: pre-fee edge ~ 0, fees must not create a finding


def test_needs_enough_data():
    df = frame(20, 0.08, 0.0, 5)
    assert not e.test_bucket(df, "L", df["yes_ask"] <= 0.10, "negative", n_boot=100)["criteria"]["FINDING"]


def test_bootstrap_resamples_events_not_rows():
    # one huge-variance event: event-level CI must be wider than the naive row-level one
    df = frame(200, 0.5, 0.5, 2)
    df["outcome"] = (df["event"] == "E0").astype(int)           # outcome perfectly correlated inside an event
    v = e.edge_yes(df)
    lo, hi = e.event_bootstrap_mean(df, v, 500, 0.95)
    assert hi - lo > 4 * v.std() / np.sqrt(len(v))


def test_calibration_table_shape():
    t = e.calibration_table(frame(300, 0.08, 0.08, 100))
    assert t and t[0]["n"] == 300
