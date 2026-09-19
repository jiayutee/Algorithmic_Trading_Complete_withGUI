"""Time-series-safe model validation (Phase 6.1).

Ordinary K-fold shuffles rows, so a model is trained on bars that come *after*
the ones it is tested on -- and financial series are autocorrelated, so it
scores far better than it ever could live. Everything here only ever trains on
the past of the bars it predicts.

Two pieces:

* :func:`walk_forward_splits` -- the split generator (expanding or rolling
  window, with a purge gap). Consumed by training (6.2) and backtests.
* :func:`walk_forward_predict` -- runs a model through those splits and returns
  an out-of-sample prediction for every bar that follows the first training
  window, so a strategy can trade them without lookahead.

Purge gap
---------
A label at row ``t`` is computed from prices up to ``t + horizon``. If ``t`` is
just before the validation window those prices lie *inside* it, so training on
that label leaks validation-period information. ``gap`` rows are therefore
dropped between the end of training and the start of validation; ``gap`` must be
at least the label horizon (:func:`assert_no_leakage` enforces this).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd


class LeakageError(ValueError):
    """Raised when a train/validation split would let the model see the future."""


@dataclass(frozen=True)
class Fold:
    index: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    gap: int = 0

    @property
    def train_end(self) -> int:
        return int(self.train_idx[-1])

    @property
    def val_start(self) -> int:
        return int(self.val_idx[0])


def walk_forward_splits(
    n_samples: int,
    train_size: int,
    val_size: int,
    *,
    step: Optional[int] = None,
    gap: int = 0,
    mode: str = "expanding",
) -> Iterator[Fold]:
    """Yield walk-forward folds over ``n_samples`` time-ordered rows.

    Fold *k* validates rows ``[v_k, v_k + val_size)`` with
    ``v_k = train_size + gap + k * step`` and trains on rows ending ``gap`` rows
    before ``v_k`` -- from row 0 (``mode="expanding"``) or only the most recent
    ``train_size`` rows (``mode="rolling"``). Only complete validation windows
    are produced.
    """
    if mode not in ("expanding", "rolling"):
        raise ValueError("mode must be 'expanding' or 'rolling'")
    if train_size < 1 or val_size < 1 or gap < 0:
        raise ValueError("train_size and val_size must be >= 1 and gap >= 0")
    step = step or val_size
    if step < 1:
        raise ValueError("step must be >= 1")

    k = 0
    while True:
        val_start = train_size + gap + k * step
        val_end = val_start + val_size
        if val_end > n_samples:
            return
        train_end = val_start - gap                       # exclusive
        train_start = 0 if mode == "expanding" else max(0, train_end - train_size)
        yield Fold(k, np.arange(train_start, train_end), np.arange(val_start, val_end), gap)
        k += 1


def assert_no_leakage(train_idx, val_idx, label_horizon: int = 0) -> None:
    """Raise :class:`LeakageError` unless every training row -- *and the prices its label
    was computed from* -- precedes every validation row."""
    train_idx, val_idx = np.asarray(train_idx), np.asarray(val_idx)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise LeakageError("empty train or validation set")
    if np.intersect1d(train_idx, val_idx).size:
        raise LeakageError("train and validation share rows")
    if train_idx.max() >= val_idx.min():
        raise LeakageError(
            f"training uses row {train_idx.max()} which is not before validation start {val_idx.min()}: "
            "the model would train on the future"
        )
    if train_idx.max() + label_horizon >= val_idx.min():
        raise LeakageError(
            f"label of training row {train_idx.max()} looks {label_horizon} bar(s) ahead, into the "
            f"validation window (starts {val_idx.min()}); increase the purge gap to >= {label_horizon}"
        )


@dataclass
class WalkForwardResult:
    """Out-of-sample predictions plus the models that made them."""
    predictions: pd.DataFrame                       # index = X.index; columns p_up, fold (NaN before 1st fold)
    models: Dict[int, Any] = field(default_factory=dict)
    folds: List[Fold] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)

    def model_for(self, position: int) -> Optional[Any]:
        """Model that produced the prediction at integer row *position* (None if none)."""
        fold = self.predictions["fold"].iloc[position]
        return None if pd.isna(fold) else self.models.get(int(fold))


def walk_forward_predict(
    model_factory: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    train_size: int,
    retrain_every: int,
    horizon: int = 1,
    mode: str = "expanding",
) -> WalkForwardResult:
    """Train on the past, predict the next ``retrain_every`` bars, slide forward, repeat.

    ``X`` holds features for every bar (NaN rows allowed -- they are excluded from
    training but still get a prediction if complete); ``y`` is the label aligned to
    ``X`` (NaN = unlabelled, e.g. the last ``horizon`` bars). The purge gap equals
    ``horizon``, so a training label never depends on prices from the window it is
    validated against. A row's prediction comes from a model fit only on rows whose
    labels were fully known before it.

    ``model_factory`` returns a fresh estimator with ``fit`` / ``predict_proba``.
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    original_index = X.index
    Xv = X.reset_index(drop=True)
    yv = pd.Series(np.asarray(y, dtype=float))
    complete = Xv.notna().all(axis=1).to_numpy()          # rows with every feature present
    trainable = complete & yv.notna().to_numpy()          # ...and a label

    pred = pd.DataFrame({"p_up": np.nan, "fold": np.nan}, index=original_index)
    result = WalkForwardResult(predictions=pred, feature_names=list(X.columns))
    p_col, f_col = pred.columns.get_loc("p_up"), pred.columns.get_loc("fold")
    min_rows = max(30, train_size // 4)

    for fold in walk_forward_splits(len(Xv), train_size, retrain_every, gap=horizon, mode=mode):
        assert_no_leakage(fold.train_idx, fold.val_idx, label_horizon=horizon)   # runtime guard, not just tests
        train_rows = fold.train_idx[trainable[fold.train_idx]]
        if len(train_rows) < min_rows or yv.iloc[train_rows].nunique() < 2:
            continue                                                             # not enough usable history yet
        model = model_factory()
        model.fit(Xv.iloc[train_rows], yv.iloc[train_rows])
        val_rows = fold.val_idx[complete[fold.val_idx]]
        if len(val_rows):
            up_col = list(model.classes_).index(1.0)
            pred.iloc[val_rows, p_col] = model.predict_proba(Xv.iloc[val_rows])[:, up_col]
            pred.iloc[val_rows, f_col] = fold.index
        result.models[fold.index] = model
        result.folds.append(fold)
    return result


def evaluate_predictions(y_true: pd.Series, p_up: pd.Series) -> Dict[str, float]:
    """Out-of-sample quality of probability forecasts, on rows that have both a label and a prediction.

    ``accuracy`` should be read against ``base_rate`` (the share of up bars): a model that
    always says "up" already scores that much. ``auc`` 0.5 = no skill.
    """
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    both = pd.concat([pd.Series(y_true).rename("y"), pd.Series(p_up).rename("p")], axis=1).dropna()
    out: Dict[str, float] = {"n": float(len(both))}
    if both.empty:
        return out
    y, p = both["y"].to_numpy(), both["p"].to_numpy().clip(1e-6, 1 - 1e-6)
    out["base_rate"] = float(y.mean())
    out["accuracy"] = float(((p > 0.5) == (y == 1)).mean())
    out["brier"] = float(brier_score_loss(y, p))
    if len(np.unique(y)) > 1:
        out["auc"] = float(roc_auc_score(y, p))
        out["logloss"] = float(log_loss(y, p))
    return out


# --------------------------------------------------------------------------- panels

def walk_forward_predict_panel(
    model_factory: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    train_dates: int,
    retrain_every: int,
    horizon: int = 1,
) -> pd.Series:
    """Walk-forward for pooled multi-symbol data: ``X``/``y`` are indexed by (symbol, timestamp).

    Splits are made on the sorted *dates*, not on rows, so every symbol's rows for a given
    date land on the same side of any boundary. Requires all symbols to share one bar
    calendar (true for 24/7 crypto), so "label horizon in bars" = "horizon in dates".
    Returns P(up) per (symbol, timestamp), NaN where no model had been trained yet.
    """
    if not isinstance(X.index, pd.MultiIndex) or X.index.nlevels != 2:
        raise ValueError("X must be indexed by (symbol, timestamp)")
    dates = X.index.get_level_values(1)
    unique_dates = np.sort(dates.unique())
    date_pos = pd.Series(np.arange(len(unique_dates)), index=unique_dates)
    row_date = date_pos.reindex(dates).to_numpy()                 # date position of every row
    yv = np.asarray(y.reindex(X.index), dtype=float)
    complete = X.notna().all(axis=1).to_numpy()
    trainable = complete & ~np.isnan(yv)
    Xv = X.reset_index(drop=True)

    out = pd.Series(np.nan, index=X.index, name="p_up")
    min_rows = max(60, train_dates // 2)
    for fold in walk_forward_splits(len(unique_dates), train_dates, retrain_every, gap=horizon):
        assert_no_leakage(fold.train_idx, fold.val_idx, label_horizon=horizon)
        tr_rows = np.flatnonzero(np.isin(row_date, fold.train_idx) & trainable)
        va_rows = np.flatnonzero(np.isin(row_date, fold.val_idx) & complete)
        if len(tr_rows) < min_rows or len(np.unique(yv[tr_rows])) < 2 or not len(va_rows):
            continue
        model = model_factory().fit(Xv.iloc[tr_rows], yv[tr_rows])
        up = list(model.classes_).index(1.0)
        out.iloc[va_rows] = model.predict_proba(Xv.iloc[va_rows])[:, up]
    return out


# ------------------------------------------------------------------ bootstrap intervals

def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), tie-aware; NaN if only one class."""
    from scipy.stats import rankdata
    n_pos = float(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(p)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _block_resample_dates(n_dates: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap of date positions: concatenated random blocks of ``block`` consecutive dates."""
    n_blocks = int(np.ceil(n_dates / block))
    starts = rng.integers(0, max(1, n_dates - block + 1), n_blocks)
    return np.concatenate([np.arange(s, min(s + block, n_dates)) for s in starts])[:n_dates]


def block_bootstrap_auc_ci(y, p, dates, *, block: int = 10, n_boot: int = 2000,
                           level: float = 0.95, seed: int = 0) -> tuple:
    """Confidence interval for AUC that respects time dependence.

    Whole *dates* are resampled in blocks of ``block`` consecutive dates (all rows -- e.g.
    all symbols -- of a chosen date come along), so autocorrelation and cross-symbol
    correlation are kept, unlike an iid row bootstrap which is too narrow for such data.
    Returns ``(low, high)``; NaNs are dropped first.
    """
    y, p, dates = np.asarray(y, float), np.asarray(p, float), np.asarray(dates)
    keep = ~(np.isnan(y) | np.isnan(p))
    y, p, dates = y[keep], p[keep], dates[keep]
    if len(y) == 0:
        return float("nan"), float("nan")
    uniq, inv = np.unique(dates, return_inverse=True)
    rows_by_date = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        pick = _block_resample_dates(len(uniq), block, rng)
        rows = np.concatenate([rows_by_date[i] for i in pick])
        a = _auc(y[rows], p[rows])
        if not np.isnan(a):
            aucs.append(a)
    a = (1 - level) / 2
    return float(np.quantile(aucs, a)), float(np.quantile(aucs, 1 - a))


def paired_block_bootstrap_auc_diff(y, p_a, p_b, dates, *, block: int = 10, n_boot: int = 2000,
                                    level: float = 0.95, seed: int = 0) -> tuple:
    """CI for AUC(p_a) - AUC(p_b) on the same rows, resampling dates in blocks (paired)."""
    y, p_a, p_b, dates = (np.asarray(v) for v in (y, p_a, p_b, dates))
    keep = ~(np.isnan(y.astype(float)) | np.isnan(p_a.astype(float)) | np.isnan(p_b.astype(float)))
    y, p_a, p_b, dates = y[keep].astype(float), p_a[keep].astype(float), p_b[keep].astype(float), dates[keep]
    uniq, inv = np.unique(dates, return_inverse=True)
    rows_by_date = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        rows = np.concatenate([rows_by_date[i] for i in _block_resample_dates(len(uniq), block, rng)])
        da, db = _auc(y[rows], p_a[rows]), _auc(y[rows], p_b[rows])
        if not (np.isnan(da) or np.isnan(db)):
            diffs.append(da - db)
    a = (1 - level) / 2
    return float(np.quantile(diffs, a)), float(np.quantile(diffs, 1 - a))
