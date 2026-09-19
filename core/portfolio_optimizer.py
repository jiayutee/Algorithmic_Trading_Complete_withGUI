"""Portfolio allocation across assets or strategies (Phase 7.0).

Long-only, fully invested weights from a table of returns (columns = assets/strategies).
Implemented with numpy/scipy directly: skfolio has no distribution for the Python 3.9 this project
runs on, and each method here is small and checkable against closed-form results.

Methods (``weights(returns, method)``)
  equal           1/N
  inverse_vol     weight ~ 1 / volatility
  min_variance    lowest-variance long-only portfolio (covariance shrunk with Ledoit-Wolf)
  risk_parity     equal risk contribution: every asset adds the same share of portfolio variance
  hrp             Hierarchical Risk Parity (Lopez de Prado): cluster by correlation, then split
                  risk down the tree -- needs no covariance inversion, so it is stable with many
                  correlated assets
  max_sharpe      maximum Sharpe with shrunk means (mean-variance is notoriously unstable; this is
                  the least fragile variant, and still the one to trust least)

All weights are non-negative and sum to 1. Black-Litterman is intentionally not included.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

METHODS = ("equal", "inverse_vol", "min_variance", "risk_parity", "hrp", "max_sharpe")


def _cov(returns: pd.DataFrame, shrink: bool = True) -> np.ndarray:
    x = returns.dropna().to_numpy(dtype=float)
    if shrink and x.shape[0] > x.shape[1]:
        from sklearn.covariance import LedoitWolf
        return LedoitWolf().fit(x).covariance_
    return np.cov(x, rowvar=False)


def _normalise(w: np.ndarray) -> np.ndarray:
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    total = w.sum()
    return w / total if total > 0 else np.full(len(w), 1.0 / len(w))


def risk_contributions(w, cov) -> np.ndarray:
    """Each asset's share of portfolio variance (sums to 1)."""
    w, cov = np.asarray(w, float), np.asarray(cov, float)
    marginal = cov @ w
    rc = w * marginal
    return rc / rc.sum()


def equal_weight(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def inverse_volatility(cov) -> np.ndarray:
    return _normalise(1.0 / np.sqrt(np.diag(cov)))


def min_variance(cov) -> np.ndarray:
    n = len(cov)
    res = minimize(lambda w: w @ cov @ w, equal_weight(n), jac=lambda w: 2 * cov @ w,
                   bounds=[(0, 1)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], method="SLSQP")
    return _normalise(res.x if res.success else inverse_volatility(cov))


def risk_parity(cov) -> np.ndarray:
    """Equal risk contribution via the convex formulation: minimise 0.5 w'Cw - (1/n) sum(log w)."""
    n = len(cov)
    x0 = inverse_volatility(cov)
    res = minimize(lambda w: 0.5 * w @ cov @ w - np.log(w).sum() / n, x0, jac=lambda w: cov @ w - 1.0 / (n * w),
                   bounds=[(1e-9, None)] * n, method="L-BFGS-B")
    return _normalise(res.x if res.success else x0)


def _cluster_var(cov: np.ndarray, items: list) -> float:
    sub = cov[np.ix_(items, items)]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp(returns: pd.DataFrame) -> np.ndarray:
    x = returns.dropna()
    cov, corr = np.cov(x.to_numpy(), rowvar=False), np.corrcoef(x.to_numpy(), rowvar=False)
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])
    dist = np.sqrt(np.clip((1 - corr) / 2.0, 0.0, None))
    np.fill_diagonal(dist, 0.0)
    order = [int(i) for i in leaves_list(linkage(squareform(dist, checks=False), method="single"))]
    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:                                             # recursive bisection, breadth first
        clusters = [c[i:j] for c in clusters for i, j in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for k in range(0, len(clusters), 2):
            left, right = clusters[k], clusters[k + 1]
            v_l, v_r = _cluster_var(cov, left), _cluster_var(cov, right)
            alpha = 1 - v_l / (v_l + v_r)
            w[left] *= alpha
            w[right] *= 1 - alpha
    return _normalise(w.sort_index().to_numpy())


def max_sharpe(returns: pd.DataFrame, mean_shrink: float = 0.5) -> np.ndarray:
    """Maximum-Sharpe long-only weights with means shrunk toward their cross-asset average."""
    cov = _cov(returns)
    mu = returns.mean().to_numpy()
    mu = (1 - mean_shrink) * mu + mean_shrink * mu.mean()
    n = len(mu)

    def neg_sharpe(w):
        v = w @ cov @ w
        return -(w @ mu) / np.sqrt(v) if v > 0 else 0.0

    res = minimize(neg_sharpe, equal_weight(n), bounds=[(0, 1)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], method="SLSQP")
    return _normalise(res.x if res.success else equal_weight(n))


def weights(returns: pd.DataFrame, method: str = "risk_parity") -> pd.Series:
    """Weights (index = columns of ``returns``) for one of ``METHODS``."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    r = returns.dropna()
    if r.shape[0] < 3 or r.shape[1] < 1:
        raise ValueError("need at least 3 return observations and 1 asset")
    n = r.shape[1]
    if method == "equal" or n == 1:
        w = equal_weight(n)
    elif method == "hrp":
        w = hrp(r)
    elif method == "max_sharpe":
        w = max_sharpe(r)
    else:
        cov = _cov(r)
        w = {"inverse_vol": inverse_volatility, "min_variance": min_variance, "risk_parity": risk_parity}[method](cov)
    return pd.Series(w, index=returns.columns, name=method)
