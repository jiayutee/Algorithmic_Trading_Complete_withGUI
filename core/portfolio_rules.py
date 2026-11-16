"""Long-only crypto portfolio weight rules from Phase 14.0 (docs/PHASE_14_0_PREREGISTRATION.md). Pure functions: no I/O, no orders.

Each rule maps the trailing daily returns of a basket of coins (rows strictly BEFORE the day the weights are applied to) to target
weights that are >= 0 and sum to <= 1; the rest is cash. Nothing here is tuned: the look-backs and caps are the ones fixed in the
pre-registration. Not a profit claim -- see the results section of that document for what was and was not shown.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd

VOTE_LOOKBACKS = (14, 28, 56, 112)
TREND_LOOKBACK = 28
MOMENTUM_LOOKBACK = 56
TOP_K = 4
VOL_WINDOW = 60
VOL_TARGET = 0.40               # annualised, a cap on the portfolio's predicted volatility (never levers up)
HISTORY_NEEDED = max(VOTE_LOOKBACKS)


def _trail(hist: pd.DataFrame, n: int) -> pd.Series:
    return (1 + hist.iloc[-n:]).prod() - 1


def trend_on(hist: pd.DataFrame, n: int = TREND_LOOKBACK) -> pd.Series:
    """1.0 where the last ``n`` days' return is positive, else 0.0."""
    return (_trail(hist, n) > 0).astype(float)


def vote(hist: pd.DataFrame) -> pd.Series:
    """Share of the look-backs (14/28/56/112 days) whose trailing return is positive, per coin (0, .25, .5, .75, 1)."""
    return pd.concat([trend_on(hist, n) for n in VOTE_LOOKBACKS], axis=1).mean(axis=1)


def inverse_vol(hist: pd.DataFrame, window: int = VOL_WINDOW) -> pd.Series:
    """(1/sigma_i) / sum_j(1/sigma_j) over ALL coins, so the weights sum to 1; a coin with no variance gets the average."""
    sigma = hist.iloc[-window:].std(ddof=1).replace(0.0, np.nan)
    inv = 1.0 / sigma
    inv = inv.fillna(inv.mean() if inv.notna().any() else 1.0)
    return inv / inv.sum()


def predicted_vol(hist: pd.DataFrame, w: pd.Series, window: int = VOL_WINDOW) -> float:
    """Annualised volatility of the weights ``w`` under the last ``window`` days' covariance (cash contributes nothing)."""
    cov = hist.iloc[-window:].cov().to_numpy()
    wv = w.reindex(hist.columns).fillna(0.0).to_numpy()
    return float(np.sqrt(max(wv @ cov @ wv, 0.0)) * np.sqrt(365.0))


def w_ew(hist: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0 / hist.shape[1], index=hist.columns)


def w_ts28(hist: pd.DataFrame) -> pd.Series:
    return trend_on(hist) / hist.shape[1]


def w_c1_vote(hist: pd.DataFrame) -> pd.Series:
    return vote(hist) / hist.shape[1]


def w_c2_invvol_trend(hist: pd.DataFrame) -> pd.Series:
    return trend_on(hist) * inverse_vol(hist)


def w_c3_vote_invvol(hist: pd.DataFrame) -> pd.Series:
    return vote(hist) * inverse_vol(hist)


def w_c4_trend_momentum(hist: pd.DataFrame, top_k: int = TOP_K) -> pd.Series:
    w = pd.Series(0.0, index=hist.columns)
    eligible = trend_on(hist) > 0
    if eligible.any():
        ranked = _trail(hist, MOMENTUM_LOOKBACK)[eligible].sort_values(ascending=False, kind="mergesort")
        w[ranked.index[:top_k]] = 1.0 / top_k
    return w


def w_c5_vote_invvol_volcap(hist: pd.DataFrame) -> pd.Series:
    w = w_c3_vote_invvol(hist)
    vol = predicted_vol(hist, w)
    return w * min(1.0, VOL_TARGET / vol) if vol > 0 else w


RULES: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "EW": w_ew, "TS28": w_ts28, "C1": w_c1_vote, "C2": w_c2_invvol_trend, "C3": w_c3_vote_invvol,
    "C4": w_c4_trend_momentum, "C5": w_c5_vote_invvol_volcap,
}
CANDIDATES = ("C1", "C2", "C3", "C4", "C5")          # EW and TS28 are yardsticks, never selected


def target_weights(rule: str, hist: pd.DataFrame) -> pd.Series:
    """Weights for ``rule`` from ``hist`` (needs at least HISTORY_NEEDED rows). Sum <= 1, each >= 0."""
    if len(hist) < HISTORY_NEEDED:
        raise ValueError(f"need {HISTORY_NEEDED} days of history, got {len(hist)}")
    w = RULES[rule](hist).clip(lower=0.0).fillna(0.0)
    total = float(w.sum())
    return w / total if total > 1.0 + 1e-12 else w
