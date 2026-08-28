"""
core/volatility_lab.py — Volatility Clustering & Tail-Risk Research Lab
========================================================================

Provides all analytics backing the "Volatility Clustering" research tab:

  - Rolling annualized volatility (real vs. shuffled comparison)
  - Excess kurtosis (fat-tail magnitude)
  - Autocorrelation of |returns| at multiple lags
  - Ljung-Box test for serial correlation in volatility
  - Same-sign-run rate (momentum proxy)
  - Permutation test: lift over independence
  - Regime tape: calm / normal / turbulent day labels
  - Position-size suggestion from historical VaR / CVaR

All public functions accept returns as list[float], np.ndarray, or
pandas.Series and coerce internally.  Short / degenerate inputs (fewer
points than the requested window or lag) return empty / zero / None
values rather than raising.

Public API summary
------------------
rolling_annualized_volatility(returns, window=21, ann_factor=252) -> list[float]
excess_kurtosis(returns) -> float
acf_abs_returns(returns, lags=(1, 5, 22, 66)) -> dict[int, float]
ljung_box_test(returns, lag=22) -> dict  {statistic, p_value}
same_sign_rate(returns) -> float
permutation_test(returns, n_permutations=500, statistic='acf1_abs', seed=42)
    -> dict  {observed, shuffled_mean, shuffled_std, p_value, lift_pts}
regime_tape(returns, window=21)
    -> dict  {labels: list[str], shuffled_labels: list[str]}
suggest_position_size(returns, capital, risk_budget_pct=0.02, confidence=0.99)
    -> dict  {var_99, cvar_99, suggested_fraction, suggested_notional}
compute_volatility_clustering_report(returns, dates=None,
                                     n_permutations=500, seed=42) -> dict
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.tsa.stattools import acf as sm_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

from core.logger import logger


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _coerce_returns(returns) -> np.ndarray:
    """Convert list / Series / ndarray to a clean float64 numpy array.

    Drops NaN values so downstream computations are not contaminated by
    missing data embedded in the input.
    """
    if isinstance(returns, pd.Series):
        arr = returns.dropna().to_numpy(dtype=float)
    elif isinstance(returns, np.ndarray):
        arr = returns.astype(float)
        arr = arr[~np.isnan(arr)]
    else:
        arr = np.array(returns, dtype=float)
        arr = arr[~np.isnan(arr)]
    return arr


def _acf1_abs(arr: np.ndarray) -> float:
    """Lag-1 autocorrelation of |arr|.  Returns 0.0 for degenerate input."""
    if len(arr) < 3:
        return 0.0
    abs_arr = np.abs(arr)
    try:
        vals = sm_acf(abs_arr, nlags=1, fft=True, missing='drop')
        return float(vals[1])
    except Exception as exc:  # noqa: BLE001
        logger.warning("acf1_abs computation failed: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rolling_annualized_volatility(
    returns,
    window: int = 21,
    ann_factor: int = 252,
) -> list:
    """Compute rolling annualized volatility over a trailing window.

    Parameters
    ----------
    returns    : list[float] | np.ndarray | pd.Series
                 Daily returns (e.g. 0.01 = +1 %).
    window     : int, default 21
                 Look-back period in trading days.
    ann_factor : int, default 252
                 Number of trading days per year for annualisation.

    Returns
    -------
    list[float]
        One value per input bar.  The first ``window - 1`` positions are
        ``None`` (insufficient history).  All subsequent values are the
        annualised standard deviation (e.g. 0.18 = 18 % annual vol).

    Example
    -------
    >>> from core.volatility_lab import rolling_annualized_volatility
    >>> vols = rolling_annualized_volatility([0.01, -0.02, 0.005], window=3)
    """
    arr = _coerce_returns(returns)
    n = len(arr)
    result: list = [None] * n
    if n == 0 or window <= 0:
        return result
    eff_window = min(window, n)
    for i in range(eff_window - 1, n):
        slice_ = arr[max(0, i - eff_window + 1): i + 1]
        if len(slice_) < 2:
            continue
        result[i] = float(np.std(slice_, ddof=1) * np.sqrt(ann_factor))
    return result


def excess_kurtosis(returns) -> float:
    """Return excess (Fisher) kurtosis of the return distribution.

    Uses scipy.stats.kurtosis with fisher=True so that the normal
    distribution scores 0.0.  Positive values indicate heavier tails
    than Gaussian; negative values indicate lighter tails.

    Parameters
    ----------
    returns : list[float] | np.ndarray | pd.Series

    Returns
    -------
    float
        Excess kurtosis, or ``np.nan`` for fewer than 4 observations
        (kurtosis is undefined / meaningless below that threshold).

    Example
    -------
    >>> from core.volatility_lab import excess_kurtosis
    >>> ek = excess_kurtosis([0.01, -0.02, 0.03, -0.01, 0.005])
    """
    arr = _coerce_returns(returns)
    if len(arr) < 4:
        return np.nan
    return float(scipy_stats.kurtosis(arr, fisher=True, bias=False))


def acf_abs_returns(returns, lags: tuple = (1, 5, 22, 66)) -> dict:
    """Autocorrelation of |returns| at the requested lags.

    Volatility clustering implies that large |returns| tend to cluster,
    i.e. ACF(|r|) > 0 at short lags, an effect absent from i.i.d. noise.

    Parameters
    ----------
    returns : list[float] | np.ndarray | pd.Series
    lags    : tuple[int], default (1, 5, 22, 66)
              Lag values (in bars) to report.

    Returns
    -------
    dict[int, float]
        Mapping from lag to autocorrelation coefficient.  Missing lags
        (too few observations) map to ``np.nan``.

    Example
    -------
    >>> from core.volatility_lab import acf_abs_returns
    >>> result = acf_abs_returns(returns, lags=(1, 5, 22))
    """
    arr = _coerce_returns(returns)
    abs_arr = np.abs(arr)
    output: dict = {}
    if len(arr) < 3:
        return {lag: np.nan for lag in lags}
    max_lag = max(lags)
    try:
        acf_vals = sm_acf(abs_arr, nlags=max_lag, fft=True, missing='drop')
    except Exception as exc:  # noqa: BLE001
        logger.warning("acf_abs_returns: statsmodels acf failed: %s", exc)
        return {lag: np.nan for lag in lags}
    for lag in lags:
        if lag < len(acf_vals):
            output[lag] = float(acf_vals[lag])
        else:
            output[lag] = np.nan
    return output


def ljung_box_test(returns, lag: int = 22) -> dict:
    """Ljung-Box portmanteau test for serial correlation in |returns|.

    Tests the null hypothesis that the magnitude series (|returns|) is
    independently distributed up to the given lag.  Rejection (low
    p-value) is consistent with volatility clustering.

    Parameters
    ----------
    returns : list[float] | np.ndarray | pd.Series
    lag     : int, default 22
              Number of lags included in the test statistic.

    Returns
    -------
    dict
        ``{'statistic': float, 'p_value': float}``.
        Both values are ``np.nan`` if the test cannot be computed.

    Example
    -------
    >>> from core.volatility_lab import ljung_box_test
    >>> lb = ljung_box_test(returns, lag=22)
    >>> print(f"LB stat={lb['statistic']:.2f}, p={lb['p_value']:.4f}")
    """
    arr = _coerce_returns(returns)
    nan_result = {'statistic': np.nan, 'p_value': np.nan}
    if len(arr) < lag + 2:
        logger.debug("ljung_box_test: insufficient data (%d obs, lag=%d)", len(arr), lag)
        return nan_result
    abs_arr = np.abs(arr)
    try:
        result_df = acorr_ljungbox(abs_arr, lags=[lag], return_df=True)
        stat = float(result_df['lb_stat'].iloc[-1])
        pval = float(result_df['lb_pvalue'].iloc[-1])
        return {'statistic': stat, 'p_value': pval}
    except Exception as exc:  # noqa: BLE001
        logger.warning("ljung_box_test failed: %s", exc)
        return nan_result


def same_sign_rate(returns) -> float:
    """Fraction of consecutive-day pairs whose returns share the same sign.

    A value above 0.5 signals momentum (runs of gains or losses persist);
    below 0.5 signals mean-reversion.  Exactly 0.5 is consistent with
    i.i.d. returns.

    Parameters
    ----------
    returns : list[float] | np.ndarray | pd.Series

    Returns
    -------
    float
        A value in [0, 1], or ``np.nan`` for fewer than 2 observations.

    Example
    -------
    >>> from core.volatility_lab import same_sign_rate
    >>> ssr = same_sign_rate([0.01, 0.02, -0.01, -0.005, 0.003])
    """
    arr = _coerce_returns(returns)
    if len(arr) < 2:
        return np.nan
    signs = np.sign(arr)
    # Treat zero as positive for parity (rare in real return series)
    signs[signs == 0] = 1
    pairs = len(signs) - 1
    same = int(np.sum(signs[:-1] == signs[1:]))
    return float(same / pairs)


def permutation_test(
    returns,
    n_permutations: int = 500,
    statistic: str = 'acf1_abs',
    seed: int = 42,
) -> dict:
    """Permutation test comparing an observed statistic against shuffled baselines.

    Destroys temporal structure by shuffling the return series
    ``n_permutations`` times and recomputes the chosen statistic on each
    shuffle, yielding an empirical null distribution.

    Parameters
    ----------
    returns       : list[float] | np.ndarray | pd.Series
    n_permutations: int, default 500
    statistic     : str, default 'acf1_abs'
                    Name of the statistic to test.  Currently supported:
                    ``'acf1_abs'`` — lag-1 ACF of |returns|.
    seed          : int, default 42
                    Seed for numpy.random.default_rng (reproducible, never
                    touches global numpy random state).

    Returns
    -------
    dict
        ``{'observed': float, 'shuffled_mean': float, 'shuffled_std': float,
           'p_value': float, 'lift_pts': float}``

        - ``observed``     : statistic value on the original series.
        - ``shuffled_mean``: mean of the null distribution.
        - ``shuffled_std`` : std of the null distribution.
        - ``p_value``      : one-sided fraction of shuffles >= observed.
        - ``lift_pts``     : (observed - shuffled_mean) * 100, in percentage
                             points (e.g. 18.9 → "+18.9 pts lift over
                             independence").

    Example
    -------
    >>> from core.volatility_lab import permutation_test
    >>> pt = permutation_test(returns, n_permutations=1000, seed=0)
    >>> print(f"lift={pt['lift_pts']:+.1f} pts, p={pt['p_value']:.3f}")
    """
    nan_result = {
        'observed': np.nan,
        'shuffled_mean': np.nan,
        'shuffled_std': np.nan,
        'p_value': np.nan,
        'lift_pts': np.nan,
    }
    arr = _coerce_returns(returns)
    if len(arr) < 3:
        logger.debug("permutation_test: insufficient data (%d obs)", len(arr))
        return nan_result

    # Map statistic name to callable
    _stat_fns = {
        'acf1_abs': _acf1_abs,
    }
    if statistic not in _stat_fns:
        logger.error("permutation_test: unsupported statistic '%s'", statistic)
        return nan_result
    stat_fn = _stat_fns[statistic]

    observed = stat_fn(arr)

    rng = np.random.default_rng(seed)
    shuffled_stats = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        shuffled = rng.permutation(arr)
        shuffled_stats[i] = stat_fn(shuffled)

    shuffled_mean = float(np.mean(shuffled_stats))
    shuffled_std = float(np.std(shuffled_stats, ddof=1))
    p_value = float(np.mean(shuffled_stats >= observed))
    lift_pts = (observed - shuffled_mean) * 100.0

    return {
        'observed': float(observed),
        'shuffled_mean': shuffled_mean,
        'shuffled_std': shuffled_std,
        'p_value': p_value,
        'lift_pts': float(lift_pts),
    }


def regime_tape(returns, window: int = 21) -> dict:
    """Label each trading day as 'calm', 'normal', or 'turbulent'.

    Trailing rolling volatility for each day is tercile-bucketed against
    the full-series rolling volatility distribution:

    - Bottom third  → ``'calm'``
    - Middle third  → ``'normal'``
    - Top third     → ``'turbulent'``

    A second labeling is produced on a seed-42 shuffle of the returns to
    act as the shuffled (independence baseline) comparison series in the
    real-vs-shuffled regime tape visualisation.

    Parameters
    ----------
    returns : list[float] | np.ndarray | pd.Series
    window  : int, default 21
              Rolling volatility window in bars.

    Returns
    -------
    dict
        ``{'labels': list[str | None], 'shuffled_labels': list[str | None]}``

        Each list has the same length as ``returns``.  The first
        ``window - 1`` entries are ``None`` (insufficient history for a
        rolling estimate).

    Example
    -------
    >>> from core.volatility_lab import regime_tape
    >>> tape = regime_tape(returns, window=21)
    >>> tape['labels'][:5]
    [None, None, ..., 'calm', 'normal']
    """
    arr = _coerce_returns(returns)
    n = len(arr)
    none_result = {'labels': [None] * n, 'shuffled_labels': [None] * n}
    if n < 2 or window <= 0:
        return none_result

    def _label_from_vols(vols_list: list) -> list:
        """Convert a list of rolling vol values (with None sentinels) to regime labels."""
        valid_vols = np.array([v for v in vols_list if v is not None], dtype=float)
        if len(valid_vols) == 0:
            return [None] * len(vols_list)
        # Tercile boundaries across the non-None subset
        q33 = float(np.percentile(valid_vols, 100 / 3))
        q67 = float(np.percentile(valid_vols, 200 / 3))
        labels: list = []
        for v in vols_list:
            if v is None:
                labels.append(None)
            elif v <= q33:
                labels.append('calm')
            elif v <= q67:
                labels.append('normal')
            else:
                labels.append('turbulent')
        return labels

    real_vols = rolling_annualized_volatility(arr, window=window)
    real_labels = _label_from_vols(real_vols)

    rng = np.random.default_rng(42)
    shuffled_arr = rng.permutation(arr)
    shuffled_vols = rolling_annualized_volatility(shuffled_arr, window=window)
    shuffled_labels = _label_from_vols(shuffled_vols)

    return {'labels': real_labels, 'shuffled_labels': shuffled_labels}


def suggest_position_size(
    returns,
    capital: float,
    risk_budget_pct: float = 0.02,
    confidence: float = 0.99,
) -> dict:
    """Suggest a position size based on historical VaR / CVaR tail risk.

    Historical Value-at-Risk (VaR) is the (1 - confidence) percentile of
    the return distribution — e.g. at 99 % confidence the worst 1 % of
    outcomes define the VaR floor.  Conditional VaR (CVaR, also called
    Expected Shortfall) is the mean of returns at or below that percentile,
    capturing the severity of tail losses beyond the VaR cut-off.

    The suggested fraction of capital is::

        suggested_fraction = min(1.0, risk_budget_pct / |CVaR|)

    Guarded against CVaR ≈ 0 or empty inputs (returns 0.0 fraction).

    Parameters
    ----------
    returns         : list[float] | np.ndarray | pd.Series
    capital         : float
                      Total available capital in account currency.
    risk_budget_pct : float, default 0.02
                      Maximum acceptable single-day loss as a fraction of
                      capital (e.g. 0.02 = 2 %).
    confidence      : float, default 0.99
                      Confidence level for VaR / CVaR (e.g. 0.99 = 99 %).

    Returns
    -------
    dict
        ``{'var_99': float, 'cvar_99': float,
           'suggested_fraction': float, 'suggested_notional': float}``

        - ``var_99``             : VaR at the requested confidence level
                                   (a negative number; larger magnitude =
                                   worse tail).
        - ``cvar_99``            : CVaR / Expected Shortfall (also negative).
        - ``suggested_fraction`` : fraction of capital to deploy, in [0, 1].
        - ``suggested_notional`` : suggested_fraction * capital.

    Example
    -------
    >>> from core.volatility_lab import suggest_position_size
    >>> sizing = suggest_position_size(returns, capital=100_000, risk_budget_pct=0.02)
    >>> print(f"Deploy {sizing['suggested_fraction']*100:.1f}% → "
    ...       f"${sizing['suggested_notional']:,.0f}")
    """
    zero_result = {
        'var_99': np.nan,
        'cvar_99': np.nan,
        'suggested_fraction': 0.0,
        'suggested_notional': 0.0,
    }
    arr = _coerce_returns(returns)
    if len(arr) < 2:
        logger.debug("suggest_position_size: insufficient data (%d obs)", len(arr))
        return zero_result

    percentile_level = (1.0 - confidence) * 100.0  # e.g. 1.0 for 99 % VaR
    var_99 = float(np.percentile(arr, percentile_level))
    tail_returns = arr[arr <= var_99]
    if len(tail_returns) == 0:
        cvar_99 = var_99
    else:
        cvar_99 = float(np.mean(tail_returns))

    abs_cvar = abs(cvar_99)
    if abs_cvar < 1e-12:
        logger.debug(
            "suggest_position_size: CVaR ≈ 0 (abs=%.2e), returning 0 fraction", abs_cvar
        )
        return {
            'var_99': var_99,
            'cvar_99': cvar_99,
            'suggested_fraction': 0.0,
            'suggested_notional': 0.0,
        }

    suggested_fraction = min(1.0, risk_budget_pct / abs_cvar)
    suggested_notional = suggested_fraction * capital

    return {
        'var_99': var_99,
        'cvar_99': cvar_99,
        'suggested_fraction': float(suggested_fraction),
        'suggested_notional': float(suggested_notional),
    }


def compute_volatility_clustering_report(
    returns,
    dates: list = None,
    n_permutations: int = 500,
    seed: int = 42,
) -> dict:
    """Bundle all volatility-clustering analytics into a single report dict.

    This is the canonical entry point for the UI tab — call this once and
    pass the resulting dict to the plotting / display layers.

    Parameters
    ----------
    returns        : list[float] | np.ndarray | pd.Series
                     Daily returns.
    dates          : list[str] | None
                     Optional ISO-format date strings aligned to ``returns``.
                     If ``None``, the 'dates' key in the output is ``None``.
    n_permutations : int, default 500
                     Number of permutations for the permutation test.
    seed           : int, default 42
                     Global seed forwarded to permutation_test and
                     regime_tape for reproducibility.

    Returns
    -------
    dict with keys
        - ``ann_vol_series``  : list[float | None] — rolling annualised vol
          (21-day window, ann_factor=252).
        - ``excess_kurtosis`` : float — Fisher excess kurtosis.
        - ``acf_abs``         : dict[int, float] — ACF of |returns| at lags
          1, 5, 22, 66.
        - ``ljung_box``       : dict {statistic, p_value}.
        - ``same_sign_rate``  : float.
        - ``permutation``     : dict from :func:`permutation_test`.
        - ``regime_tape``     : dict from :func:`regime_tape`.
        - ``dates``           : list[str] | None — echoed back for alignment.

    Example
    -------
    >>> import pandas as pd
    >>> from core.volatility_lab import compute_volatility_clustering_report
    >>> prices = pd.Series([100, 101, 99, 102, 100, 103])
    >>> returns = prices.pct_change().dropna().tolist()
    >>> report = compute_volatility_clustering_report(returns)
    >>> report.keys()
    dict_keys(['ann_vol_series', 'excess_kurtosis', 'acf_abs', 'ljung_box',
               'same_sign_rate', 'permutation', 'regime_tape', 'dates'])
    """
    arr = _coerce_returns(returns)
    logger.info(
        "compute_volatility_clustering_report: %d observations, "
        "%d permutations, seed=%d",
        len(arr),
        n_permutations,
        seed,
    )

    report = {
        'ann_vol_series': rolling_annualized_volatility(arr),
        'excess_kurtosis': excess_kurtosis(arr),
        'acf_abs': acf_abs_returns(arr),
        'ljung_box': ljung_box_test(arr),
        'same_sign_rate': same_sign_rate(arr),
        'permutation': permutation_test(arr, n_permutations=n_permutations, seed=seed),
        'regime_tape': regime_tape(arr),
        'dates': dates,
    }
    return report
