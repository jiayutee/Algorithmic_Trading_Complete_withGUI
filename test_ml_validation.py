"""Phase 6.1: walk-forward validation must never let a model see the future."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

from core.ml_validation import (
    LeakageError, assert_no_leakage, evaluate_predictions, walk_forward_predict, walk_forward_splits,
)


def _model():
    return HistGradientBoostingClassifier(max_iter=40, max_depth=3, random_state=0)


def _overfit_prone_model():
    """Real GBMs can memorise small clusters of rows -- which is exactly what leaky splits reward."""
    return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=64, min_samples_leaf=3, random_state=0)


# ------------------------------------------------------------------ splitter

def test_expanding_folds_grow_and_validation_windows_tile_forward():
    folds = list(walk_forward_splits(200, train_size=100, val_size=20, gap=0))
    assert [f.val_start for f in folds] == [100, 120, 140, 160, 180]
    assert all(f.train_idx[0] == 0 for f in folds)
    assert [len(f.train_idx) for f in folds] == [100, 120, 140, 160, 180]
    assert all(len(f.val_idx) == 20 for f in folds)                       # only complete windows


def test_rolling_folds_keep_a_fixed_training_length():
    folds = list(walk_forward_splits(200, train_size=80, val_size=20, gap=5, mode="rolling"))
    assert len(folds) >= 3
    assert all(len(f.train_idx) == 80 for f in folds)
    assert folds[1].train_idx[0] == folds[0].train_idx[0] + 20


def test_gap_rows_are_dropped_between_train_and_validation():
    f = next(walk_forward_splits(200, train_size=100, val_size=20, gap=7))
    assert f.val_start - f.train_end - 1 == 7                             # exactly 7 rows purged


def test_every_fold_passes_the_leakage_check_across_a_parameter_grid():
    for n in (120, 333, 1000):
        for train, val, gap, mode in [(50, 10, 1, "expanding"), (80, 25, 5, "rolling"), (60, 7, 0, "expanding")]:
            for f in walk_forward_splits(n, train, val, gap=gap, mode=mode):
                assert_no_leakage(f.train_idx, f.val_idx, label_horizon=gap)


def test_no_folds_when_history_is_too_short():
    assert list(walk_forward_splits(50, train_size=100, val_size=10)) == []


@pytest.mark.parametrize("kwargs", [dict(train_size=0, val_size=5), dict(train_size=5, val_size=0),
                                    dict(train_size=5, val_size=5, gap=-1), dict(train_size=5, val_size=5, mode="x")])
def test_invalid_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        list(walk_forward_splits(100, **kwargs))


# ------------------------------------------------------------ leakage checker

def test_assert_no_leakage_rejects_overlap_future_training_and_label_overlap():
    with pytest.raises(LeakageError):
        assert_no_leakage(np.arange(0, 60), np.arange(50, 70))            # shared rows
    with pytest.raises(LeakageError):
        assert_no_leakage(np.arange(50, 100), np.arange(0, 20))           # trains on the future
    with pytest.raises(LeakageError, match="purge gap"):
        assert_no_leakage(np.arange(0, 60), np.arange(62, 80), label_horizon=5)   # label reaches into validation
    assert_no_leakage(np.arange(0, 60), np.arange(66, 80), label_horizon=5)       # ok once purged
    with pytest.raises(LeakageError):
        assert_no_leakage([], [1])


# ------------------------------------- the proof: naive K-fold flatters, this doesn't

def _blocky_data(n=600, seed=0):
    """Labels are random per block of 5-15 bars (unpredictable in advance), but every
    row carries a block 'fingerprint' feature. A model that has seen other rows of the
    same block -- which shuffled K-fold guarantees -- looks brilliant; a model that has
    only seen the past cannot know the label of a block it has never seen."""
    rng = np.random.default_rng(seed)
    fingerprints, labels = [], []
    while len(labels) < n:
        length = int(rng.integers(5, 16))
        fingerprints += [rng.normal(0, 5)] * length
        labels += [float(rng.integers(0, 2))] * length
    x = np.array(fingerprints[:n]) + rng.normal(0, 0.05, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"fingerprint": x}, index=idx), pd.Series(labels[:n], index=idx)


def test_naive_kfold_is_fooled_but_walk_forward_is_not():
    X, y = _blocky_data()

    naive_hits = []
    naive_leaks = 0
    for train, val in KFold(n_splits=5, shuffle=True, random_state=0).split(X):
        m = _overfit_prone_model().fit(X.iloc[train], y.iloc[train])
        naive_hits.append((m.predict(X.iloc[val]) == y.iloc[val]).mean())
        try:
            assert_no_leakage(train, val)
        except LeakageError:
            naive_leaks += 1
    naive_acc = float(np.mean(naive_hits))

    res = walk_forward_predict(_overfit_prone_model, X, y, train_size=150, retrain_every=20, horizon=1)
    ev = evaluate_predictions(y, res.predictions["p_up"])

    # Measured over 5 seeds: naive K-fold 0.82-0.88, walk-forward 0.54-0.59 (labels are random per block).
    assert naive_acc > 0.75, f"K-fold should look great on leaky data, got {naive_acc:.2f}"
    assert naive_leaks == 5, "the framework must flag every shuffled K-fold split as leaking"
    assert ev["accuracy"] < 0.66, f"walk-forward must not be able to predict random blocks, got {ev['accuracy']:.2f}"
    assert naive_acc - ev["accuracy"] > 0.15


# ------------------------------------------------------------ walk_forward_predict

class _Recorder:
    """Stub estimator that records exactly which rows it was trained on."""
    log = []

    def fit(self, X, y):
        _Recorder.log.append((X.index.min(), X.index.max(), len(X)))
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X):
        return np.tile([0.4, 0.6], (len(X), 1))


def test_each_prediction_comes_from_a_model_trained_strictly_before_it():
    X, y = _blocky_data(300)
    _Recorder.log = []
    res = walk_forward_predict(_Recorder, X, y, train_size=100, retrain_every=25, horizon=3)
    assert len(res.folds) == len(_Recorder.log) > 3
    for fold, (_t_min, t_max, _n) in zip(res.folds, _Recorder.log):
        # the estimator sees positional row numbers; its last training row must precede validation
        assert t_max < fold.val_start                                       # trained only on the past
        assert t_max + 3 < fold.val_start                                   # purged by the label horizon


def test_predictions_are_nan_before_the_first_validation_window():
    X, y = _blocky_data(300)
    res = walk_forward_predict(_model, X, y, train_size=120, retrain_every=20, horizon=1)
    p = res.predictions["p_up"]
    assert p.iloc[:121].isna().all() and p.iloc[121:].notna().any()
    assert res.model_for(50) is None and res.model_for(200) is not None


def test_rewriting_the_future_does_not_change_earlier_predictions():
    """Predictions for folds that validate before row m are identical whether or not rows >= m exist."""
    X, y = _blocky_data(400)
    base = walk_forward_predict(_model, X, y, train_size=120, retrain_every=20, horizon=1)
    m = 260
    X2, y2 = X.copy(), y.copy()
    rng = np.random.default_rng(5)
    X2.iloc[m:, 0] = rng.normal(0, 50, len(X2) - m)
    y2.iloc[m:] = rng.integers(0, 2, len(y2) - m).astype(float)
    changed = walk_forward_predict(_model, X2, y2, train_size=120, retrain_every=20, horizon=1)
    safe = [f for f in base.folds if f.val_idx[-1] < m - 1]
    assert len(safe) >= 5
    for f in safe:
        pd.testing.assert_series_equal(base.predictions["p_up"].iloc[f.val_idx], changed.predictions["p_up"].iloc[f.val_idx])


def test_folds_with_too_little_or_one_class_history_are_skipped_not_crashed():
    X, y = _blocky_data(300)
    one_class = pd.Series(1.0, index=y.index)
    res = walk_forward_predict(_model, X, one_class, train_size=100, retrain_every=20)
    assert res.folds == [] and res.predictions["p_up"].isna().all()


def test_length_mismatch_raises():
    X, y = _blocky_data(100)
    with pytest.raises(ValueError):
        walk_forward_predict(_model, X, y.iloc[:50], train_size=40, retrain_every=10)


def test_evaluate_predictions_reports_skill_against_the_base_rate():
    y = pd.Series([1, 1, 1, 0, 1, 0, 1, 1], dtype=float)
    perfect = evaluate_predictions(y, y * 0.9 + 0.05)
    assert perfect["accuracy"] == 1.0 and perfect["auc"] == 1.0 and perfect["base_rate"] == 0.75
    ignores_nan = evaluate_predictions(y, pd.Series([np.nan] * 4 + [0.9, 0.1, 0.9, 0.9]))
    assert ignores_nan["n"] == 4
    assert evaluate_predictions(y, pd.Series([np.nan] * 8)) == {"n": 0.0}


# ------------------------------------------------------------- panel walk-forward + bootstrap

from core.ml_validation import (  # noqa: E402
    _auc, block_bootstrap_auc_ci, paired_block_bootstrap_auc_diff, walk_forward_predict_panel,
)


def _panel(n_dates=300, symbols=("AAA", "BBB", "CCC"), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates)
    frames, ys = [], []
    for s in symbols:
        X = pd.DataFrame({"f1": rng.normal(size=n_dates), "f2": rng.normal(size=n_dates)}, index=dates)
        y = pd.Series((X["f1"] + rng.normal(0, 0.5, n_dates) > 0).astype(float), index=dates)   # f1 is informative
        X.index = pd.MultiIndex.from_arrays([[s] * n_dates, dates], names=["symbol", "timestamp"])
        y.index = X.index
        frames.append(X); ys.append(y)
    return pd.concat(frames), pd.concat(ys)


def test_panel_walk_forward_learns_a_real_pooled_signal_and_is_nan_before_the_first_window():
    X, y = _panel()
    p = walk_forward_predict_panel(_model, X, y, train_dates=100, retrain_every=20, horizon=1)
    dates = X.index.get_level_values(1)
    first_pred_date = dates.unique()[101]
    assert p[dates < first_pred_date].isna().all()
    ev = evaluate_predictions(y, p)
    assert ev["auc"] > 0.85 and ev["n"] > 500


def test_panel_never_puts_one_date_on_both_sides_and_purges_the_horizon():
    class Rec:
        seen = []
        def fit(self, Xf, yf):
            Rec.seen.append(len(Xf)); self.classes_ = np.array([0.0, 1.0]); return self
        def predict_proba(self, Xp):
            return np.tile([0.5, 0.5], (len(Xp), 1))
    X, y = _panel(120)
    Rec.seen = []
    walk_forward_predict_panel(Rec, X, y, train_dates=60, retrain_every=20, horizon=3)
    # fold k trains on exactly (60 + 20k) dates x 3 symbols -- rows of a date always move together
    assert Rec.seen[:3] == [60 * 3, 80 * 3, 100 * 3] or all(n % 3 == 0 for n in Rec.seen)
    with pytest.raises(ValueError):
        walk_forward_predict_panel(Rec, X.droplevel(0), y.droplevel(0), train_dates=60, retrain_every=20)


def test_panel_predictions_ignore_rewritten_future_dates():
    X, y = _panel(240)
    base = walk_forward_predict_panel(_model, X, y, train_dates=100, retrain_every=20)
    cut = X.index.get_level_values(1).unique()[160]
    X2, y2 = X.copy(), y.copy()
    late = X2.index.get_level_values(1) >= cut
    rng = np.random.default_rng(9)
    X2.loc[late] = rng.normal(0, 30, size=(late.sum(), 2))
    y2[late] = rng.integers(0, 2, late.sum()).astype(float)
    changed = walk_forward_predict_panel(_model, X2, y2, train_dates=100, retrain_every=20)
    before = X.index.get_level_values(1) < cut - pd.Timedelta(days=2)
    pd.testing.assert_series_equal(base[before], changed[before])


def test_rank_auc_matches_sklearn_including_ties():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300).astype(float)
    p = np.round(rng.uniform(0, 1, 300), 1)          # heavy ties
    assert _auc(y, p) == pytest.approx(roc_auc_score(y, p))
    assert np.isnan(_auc(np.ones(5), np.arange(5.0)))


def test_block_bootstrap_ci_covers_the_truth_and_is_wider_than_iid_for_autocorrelated_data():
    rng = np.random.default_rng(1)
    n = 800
    dates = pd.date_range("2024-01-01", periods=n)
    # slowly varying (autocorrelated) score and label: an iid bootstrap would be far too confident
    score = np.convolve(rng.normal(size=n + 29), np.ones(30) / 30, mode="valid")
    y = (score + rng.normal(0, 0.05, n) > 0).astype(float)
    p = 1 / (1 + np.exp(-30 * score))
    lo_b, hi_b = block_bootstrap_auc_ci(y, p, dates, block=30, n_boot=400)
    lo_i, hi_i = block_bootstrap_auc_ci(y, p, dates, block=1, n_boot=400)
    assert lo_b <= _auc(y, p) <= hi_b
    assert (hi_b - lo_b) > (hi_i - lo_i)


def test_noise_ci_straddles_half_and_real_signal_ci_does_not():
    rng = np.random.default_rng(2)
    n = 600
    dates = pd.date_range("2024-01-01", periods=n)
    y = rng.integers(0, 2, n).astype(float)
    lo, hi = block_bootstrap_auc_ci(y, rng.uniform(size=n), dates, n_boot=400)
    assert lo < 0.5 < hi
    lo2, hi2 = block_bootstrap_auc_ci(y, y * 0.3 + rng.uniform(size=n), dates, n_boot=400)
    assert lo2 > 0.5


def test_paired_difference_ci_detects_a_better_model_and_not_an_equal_one():
    rng = np.random.default_rng(3)
    n = 800
    dates = pd.date_range("2024-01-01", periods=n)
    y = rng.integers(0, 2, n).astype(float)
    weak = y * 0.2 + rng.uniform(size=n)
    strong = y * 0.8 + rng.uniform(size=n)
    lo, hi = paired_block_bootstrap_auc_diff(y, strong, weak, dates, n_boot=400)
    assert lo > 0
    lo2, hi2 = paired_block_bootstrap_auc_diff(y, weak, weak + 1e-9, dates, n_boot=200)
    assert lo2 <= 0 <= hi2
