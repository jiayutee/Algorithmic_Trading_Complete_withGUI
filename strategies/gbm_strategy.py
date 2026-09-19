"""Gradient-boosted-tree strategy (Phase 6.2).

Predicts whether the next bar closes up, using the Phase 6.0 feature matrix, and
trades the probability. Predictions are produced by the Phase 6.1 walk-forward
framework: the model is retrained every ``retrain_every`` bars on the past only
(purged by the label horizon), so the backtest never trades on a prediction made
by a model that had seen the future.

Decision at the close of bar t -> order fills on bar t+1 (backtrader default).

  flat  + P(up) >= long_threshold             -> open long
  flat  + P(up) <= short_threshold            -> open short  (if allow_short)
  long  + P(up) <= exit_threshold             -> close
  short + P(up) >= exit_threshold             -> close

Every order carries a Phase 11.1 rationale: the probability, and the features that
pushed it there (LightGBM per-row contributions, in log-odds).

To reproduce a trained model from scratch and see honest out-of-sample metrics:

    python training_ground/train_gbm.py --symbol BTCUSDT --days 1500 --interval 1d
"""
from __future__ import annotations

import backtrader as bt
import numpy as np
import pandas as pd

from core.feature_engineering import build_features, make_target
from core.logger import get_logger
from core.ml_validation import evaluate_predictions, walk_forward_predict
from core.trade_rationale import RationaleMixin

logger = get_logger(__name__)

DEFAULT_LGBM_PARAMS = dict(
    n_estimators=150,
    learning_rate=0.03,
    num_leaves=15,
    max_depth=4,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=7,
    n_jobs=1,          # deterministic and safe next to Qt / other OpenMP users
    verbose=-1,
)


def make_lgbm_classifier(**overrides):
    """Fresh, conservatively-regularised LightGBM classifier (small data => shallow trees)."""
    import lightgbm as lgb
    return lgb.LGBMClassifier(**{**DEFAULT_LGBM_PARAMS, **overrides})


def explain_row(model, x_row: pd.DataFrame, top: int = 3) -> dict:
    """Signed per-feature contribution (log-odds) to this row's prediction, largest first."""
    contrib = model.predict(x_row, pred_contrib=True)[0][:-1]      # last column is the bias term
    names = list(x_row.columns)
    order = np.argsort(-np.abs(contrib))[:top]
    return {names[i]: float(contrib[i]) for i in order}


class GBMStrategy(RationaleMixin, bt.Strategy):
    params = (
        ("train_size", 200),          # bars in the first training window
        ("retrain_every", 20),        # bars between retrains (= validation window length)
        ("horizon", 1),               # label looks this many bars ahead (also the purge gap)
        ("long_threshold", 0.55),
        ("short_threshold", 0.45),
        ("exit_threshold", 0.50),
        ("allow_short", True),
        ("include_news", False),      # historical news can't be back-filled, so off by default
        ("risk_per_trade", 0.1),
    )

    def __init__(self):
        self.signals = []
        self.order_count = 0
        self.closed_trades = []
        self._closing_long = False
        self._closing_short = False
        self._p_up = pd.Series(dtype=float)
        self._X = pd.DataFrame()
        self._wf = None
        self.oos_metrics: dict = {}

        frame = self._source_frame()
        min_rows = self.params.train_size + self.params.retrain_every + self.params.horizon + 60
        if frame is None or len(frame) < min_rows:
            logger.warning("GBMStrategy: need >= %d bars of OHLCV (have %s); it will not trade.",
                           min_rows, None if frame is None else len(frame))
            return
        try:
            X = build_features(frame, include_news=self.params.include_news)
            y = make_target(frame, horizon=self.params.horizon)
            self._wf = walk_forward_predict(
                make_lgbm_classifier, X, y,
                train_size=self.params.train_size, retrain_every=self.params.retrain_every,
                horizon=self.params.horizon,
            )
        except Exception as exc:  # noqa: BLE001 -- a modelling failure must not crash the whole backtest
            logger.error("GBMStrategy: model training failed (%s); it will not trade.", exc)
            return
        p = self._wf.predictions["p_up"]
        if getattr(p.index, "tz", None) is not None:
            p.index = p.index.tz_localize(None)
            X.index = X.index.tz_localize(None)
        self._p_up, self._X = p, X
        self.oos_metrics = evaluate_predictions(y.set_axis(p.index), p)
        logger.info("GBMStrategy walk-forward: %d retrains, OOS %s", len(self._wf.folds),
                    {k: round(v, 3) for k, v in self.oos_metrics.items()})

    # ------------------------------------------------------------------ helpers

    def _source_frame(self):
        """The pandas frame behind the data feed (the Backtester always uses PandasData)."""
        frame = getattr(self.data, "_dataname", None)
        return frame if isinstance(frame, pd.DataFrame) else None

    def _explain(self, ts, p: float, direction: str, verb: str, rule: str, action: str, signal: str):
        thresholds = {"long_threshold": self.params.long_threshold, "short_threshold": self.params.short_threshold,
                      "exit_threshold": self.params.exit_threshold}
        drivers, features = {}, {"p_up": p}
        try:
            pos = self._p_up.index.get_loc(ts)
            model = self._wf.model_for(pos)
            x_row = self._X.iloc[[pos]]
            if model is not None:
                drivers = explain_row(model, x_row)
                features.update({name: float(x_row[name].iloc[0]) for name in drivers})
        except Exception as exc:  # noqa: BLE001 -- explanation is best-effort, never blocks a trade
            logger.debug("GBMStrategy explanation failed: %s", exc)
        why = ", ".join(f"{n}={features[n]:.4g} ({c:+.2f})" for n, c in drivers.items()) or "no driver detail"
        self._set_rationale(
            action=action, strategy="GBM_LightGBM", signal=signal,
            summary=f"{verb} {direction}: model P(up)={p:.0%} {rule}; main drivers: {why}",
            features=features, thresholds=thresholds,
            confidence=p if direction == "LONG" else 1.0 - p,       # confidence in this position's direction
            feature_importance=drivers or None,
        )

    # -------------------------------------------------------------- trading loop

    def next(self):
        if self._p_up.empty:
            return
        ts = pd.Timestamp(self.data.datetime.datetime(0))
        p = self._p_up.get(ts)
        if p is None or np.isnan(p):
            return
        prm = self.params
        size = (self.broker.getcash() * prm.risk_per_trade) / self.data.close[0]

        if not self.position:
            if size <= 0.0001:
                return
            if p >= prm.long_threshold:
                self._explain(ts, p, "LONG", "Opened", f">= {prm.long_threshold:.0%}", "open_long", "p_up_above_long_threshold")
                self.buy(size=size)
                self.order_count += 1
            elif prm.allow_short and p <= prm.short_threshold:
                self._explain(ts, p, "SHORT", "Opened", f"<= {prm.short_threshold:.0%}", "open_short", "p_up_below_short_threshold")
                self.sell(size=size)
                self.order_count += 1
        elif self.position.size > 0 and p <= prm.exit_threshold:
            self._explain(ts, p, "LONG", "Closed", f"fell to <= {prm.exit_threshold:.0%}", "close_long", "p_up_below_exit_threshold")
            self._closing_long = True
            self.close()
            self.order_count += 1
        elif self.position.size < 0 and p >= prm.exit_threshold:
            self._explain(ts, p, "SHORT", "Closed", f"rose to >= {prm.exit_threshold:.0%}", "close_short", "p_up_above_exit_threshold")
            self._closing_short = True
            self.close()
            self.order_count += 1

    def notify_order(self, order):
        if order.status != order.Completed:
            return
        if order.isbuy():
            kind = "buy_cover" if self._closing_short else "buy"
            self._closing_short = False
        else:
            kind = "sell" if self._closing_long else "sell_short"
            self._closing_long = False
        self.signals.append({
            "date": self.data.datetime.datetime(0),
            "type": kind,
            "price": order.executed.price,
            "qty": order.executed.size,
        })
        self._attach_rationale_to_last_signal()

    def notify_trade(self, trade):
        if trade.isclosed:
            self.closed_trades.append(trade)

    def stop(self):
        logger.info("GBMStrategy finished. Orders: %d, signals: %d", self.order_count, len(self.signals))
