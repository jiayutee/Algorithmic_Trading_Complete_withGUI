"""Volatility-targeted position sizing (Phase 6.6).

Scale exposure down when the volatility forecast is high relative to its own history and up
(to a cap) when it is low, so risk per day stays more constant. All functions use only
information up to the current bar.

    log_range   = ln(High / Low)                      daily range-based volatility proxy (Parkinson, up to a constant)
    sigma_hat   = a forecast of tomorrow's log_range  (naive EWMA here; a model in Phase 6.6)
    exposure_t  = min(cap, k * median(sigma_hat[<=t]) / sigma_hat_t)

Comparing a forecast with its own expanding median makes the rule scale-free: a forecast that
is merely biased high or low cannot win or lose because of the bias.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

TRADING_DAYS = 365          # crypto trades every day


def log_range(df: pd.DataFrame) -> pd.Series:
    """ln(High/Low), floored so a zero-range bar does not produce -inf in logs."""
    d = df.copy()
    d.columns = [str(c).lower() for c in d.columns]
    return np.log(d["high"] / d["low"]).clip(lower=1e-5)


def naive_vol_forecast(lr: pd.Series, span: int = 20) -> pd.Series:
    """EWMA of squared log-range, square-rooted. Value at t uses bars <= t (a forecast for t+1)."""
    return np.sqrt((lr ** 2).ewm(span=span, adjust=False).mean())


def vol_target_exposure(sigma_hat: pd.Series, k: float = 0.7, cap: float = 1.0, min_history: int = 30) -> pd.Series:
    """Exposure in [0, cap]: ``min(cap, k * expanding_median(sigma_hat) / sigma_hat)``. NaN until ``min_history`` values exist."""
    med = sigma_hat.expanding(min_periods=min_history).median()
    return (k * med / sigma_hat).clip(lower=0.0, upper=cap)


def apply_no_trade_band(target: pd.Series, band: float = 0.10) -> pd.Series:
    """Only move exposure when the target is more than ``band`` away from where it currently is."""
    out = np.full(len(target), np.nan)
    current = np.nan
    for i, t in enumerate(target.to_numpy(dtype=float)):
        if np.isnan(t):
            continue
        if np.isnan(current) or abs(t - current) > band:
            current = t
        out[i] = current
    return pd.Series(out, index=target.index)


def portfolio_returns(exposure: pd.DataFrame, fwd_returns: pd.DataFrame, fee: float = 0.001) -> pd.Series:
    """Daily net return of an equal-weight portfolio.

    ``exposure`` (dates x symbols) is decided at each close for the *next* bar; ``fwd_returns``
    holds the return from that close to the next. Cost = ``fee`` x change in exposure (the first
    position is charged from zero). Symbols are averaged, i.e. equal capital per symbol.
    """
    e, r = exposure.align(fwd_returns, join="inner")
    gross = (e * r).mean(axis=1)
    turnover = e.diff().abs()
    turnover.iloc[0] = e.iloc[0].abs()
    return (gross - fee * turnover.mean(axis=1)).dropna()


def perf_stats(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> Dict[str, float]:
    r = returns.dropna()
    if r.empty:
        return {}
    equity = (1 + r).cumprod()
    dd = equity / equity.cummax() - 1
    years = len(r) / periods_per_year
    ann_ret = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else float("nan")
    vol = float(r.std() * np.sqrt(periods_per_year))
    max_dd = float(dd.min())
    return {
        "ann_return": ann_ret, "ann_vol": vol,
        "sharpe": float(r.mean() / r.std() * np.sqrt(periods_per_year)) if r.std() > 0 else float("nan"),
        "max_drawdown": max_dd,
        "calmar": float(ann_ret / abs(max_dd)) if max_dd < 0 else float("nan"),
        "total_return": float(equity.iloc[-1] - 1),
    }


def sharpe(r) -> float:
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 2 and r.std(ddof=1) > 0 else float("nan")
