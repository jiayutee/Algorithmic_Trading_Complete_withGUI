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
