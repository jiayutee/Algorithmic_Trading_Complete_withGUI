"""
core/research_lab.py — Strategy Lab analytics for AlgoTrader.

Provides pure-function analytics for the Strategy Lab backtest-report panel:
drawdown chart, rolling Sharpe, trade P&L distribution, monthly-returns heatmap,
year-by-year performance table, unit economics per trade, strategy comparison
book, and a pass/fail gate verdict.

All eight public functions operate on plain lists / pandas Series and do not
require a live Backtester instance, making them independently testable.  The
only exception is ``build_strategy_book``, which accepts a StrategyManager
instance as its first argument so it can iterate the registered strategies.

Public API
----------
compute_drawdown_series     — % drawdown from running peak, one value per bar.
compute_rolling_sharpe      — rolling annualised Sharpe (63-bar default).
trade_pnl_distribution      — histogram of per-trade net P&L.
monthly_returns_table       — compound monthly returns keyed by year → month.
year_by_year_table          — one row per calendar year with key stats.
unit_economics_per_trade    — avg win/loss, expectancy, profit factor, etc.
build_strategy_book         — run every registered strategy, sort by Sharpe.
evaluate_gate               — pass/fail verdict against configurable thresholds.
"""

import math
import calendar
from typing import Optional

import numpy as np
import pandas as pd

from core.logger import logger


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_series(values: list, dates: list) -> pd.Series:
    """Convert aligned return values and ISO date strings to a dated pd.Series.

    Dates that cannot be parsed are coerced to NaT and can be dropped by the
    caller.  No exception is raised for malformed inputs.
    """
    idx = pd.to_datetime(dates, format='%Y-%m-%d', errors='coerce')
    return pd.Series(values, index=idx, dtype=float)


def _window_sharpe(window_returns: np.ndarray, annualization: int) -> float:
    """Annualized Sharpe for a fixed slice of returns (zero risk-free rate).

    Returns ``math.nan`` when fewer than two observations or when the standard
    deviation of the window is zero (e.g. a run of identical returns).
    """
    if len(window_returns) < 2:
        return math.nan
    std = float(np.std(window_returns, ddof=1))
    if std == 0.0:
        return math.nan
    return float(np.mean(window_returns)) / std * math.sqrt(annualization)


# ---------------------------------------------------------------------------
# 1. Drawdown series
# ---------------------------------------------------------------------------

def compute_drawdown_series(equity_curve: list) -> list:
    """Compute percentage drawdown from the running peak at each bar.

    Values are 0 at a new equity high and negative below the prior peak.
    A return of -12.4 means the portfolio is 12.4% below its running peak.

    Args:
        equity_curve: List of portfolio values, typically
                      ``results['total_asset_value']``.  Must contain
                      positive values to produce meaningful output.

    Returns:
        List of floats, same length as ``equity_curve``, where each element
        is the percentage drawdown from the running peak (0 or negative).
        Returns an empty list when ``equity_curve`` is empty.

    Example::

        dd = compute_drawdown_series([100, 110, 105, 95, 115])
        # → [0.0, 0.0, -4.5455, -13.6364, 0.0]
    """
    if not equity_curve:
        return []

    arr = np.asarray(equity_curve, dtype=float)
    if np.any(arr <= 0):
        logger.warning(
            "compute_drawdown_series: equity curve contains non-positive values; "
            "drawdown percentages may be misleading."
        )

    peak = arr[0]
    result: list = []
    for val in arr:
        if val > peak:
            peak = val
        if peak == 0.0:
            result.append(0.0)
        else:
            result.append(float((val - peak) / peak * 100.0))
    return result


# ---------------------------------------------------------------------------
# 2. Rolling Sharpe
# ---------------------------------------------------------------------------

def compute_rolling_sharpe(
    returns: list,
    window: int = 63,
    annualization: int = 252,
) -> list:
    """Compute the rolling annualized Sharpe ratio (zero risk-free rate).

    Positions before the window accumulates enough bars are returned as
    ``math.nan`` so that charting code can distinguish "no data yet" from a
    genuinely flat Sharpe of zero.

    Args:
        returns:       Per-bar return floats (e.g. ``results['returns']``).
        window:        Rolling look-back period in bars (default 63, roughly
                       one quarter of trading days).
        annualization: Trading bars per year used for annualization
                       (default 252 for daily equity data).

    Returns:
        List of floats, same length as ``returns``.  The first
        ``window - 1`` entries are ``math.nan``; subsequent entries are the
        rolling annualised Sharpe for that bar's trailing window.
        Returns an empty list when ``returns`` is empty.

    Example::

        rs = compute_rolling_sharpe([0.001, -0.002, 0.003] * 30, window=5)
        # rs[:4] are math.nan; rs[4:] are rolling Sharpe values.
    """
    if not returns:
        return []

    arr = np.asarray(returns, dtype=float)
    n = len(arr)
    out: list = [math.nan] * n

    for i in range(window - 1, n):
        out[i] = _window_sharpe(arr[i - window + 1: i + 1], annualization)

    return out


# ---------------------------------------------------------------------------
# 3. Trade P&L distribution
# ---------------------------------------------------------------------------

def trade_pnl_distribution(profit_per_trade: list, bins: int = 30) -> dict:
    """Histogram of per-trade net P&L for the Strategy Lab distribution chart.

    Args:
        profit_per_trade: List of net P&L values per closed trade
                          (``results['profit_per_trade']``).
        bins:             Number of histogram bins (default 30).

    Returns:
        dict with keys:

        ``bin_edges``  — list of ``bins + 1`` floats delimiting each bin.
        ``counts``     — list of ``bins`` ints, trade count per bin.
        ``win_count``  — int, number of profitable trades (P&L > 0).
        ``loss_count`` — int, number of break-even or losing trades (P&L ≤ 0).

        When ``profit_per_trade`` is empty, all numeric fields return 0 and
        ``bin_edges`` / ``counts`` are empty lists.

    Example::

        dist = trade_pnl_distribution([150.0, -80.0, 200.0, -40.0], bins=5)
        dist['win_count']   # → 2
        dist['loss_count']  # → 2
    """
    if not profit_per_trade:
        return {"bin_edges": [], "counts": [], "win_count": 0, "loss_count": 0}

    arr = np.asarray(profit_per_trade, dtype=float)
    win_count = int(np.sum(arr > 0))
    loss_count = int(np.sum(arr <= 0))

    counts, bin_edges = np.histogram(arr, bins=bins)
    return {
        "bin_edges": bin_edges.tolist(),
        "counts": counts.tolist(),
        "win_count": win_count,
        "loss_count": loss_count,
    }


# ---------------------------------------------------------------------------
# 4. Monthly returns table
# ---------------------------------------------------------------------------

def monthly_returns_table(returns: list, dates: list) -> dict:
    """Aggregate per-bar returns into a monthly-returns heatmap structure.

    Each month's return is computed by compounding all daily returns within
    that calendar month: ``(1+r1)*(1+r2)*…-1``.

    Args:
        returns: Per-bar return floats aligned with ``dates``
                 (from ``results['returns']``).
        dates:   ISO ``'YYYY-MM-DD'`` strings, one per bar
                 (from ``results['dates']``).

    Returns:
        dict where integer keys are calendar years (e.g. ``2023``) mapping to
        inner dicts of ``{month_abbrev: pct_return}`` — e.g.
        ``{"Jan": 3.1416, "Feb": -1.22, …}``.  Month abbreviations follow
        Python's ``calendar.month_abbr`` (``"Jan"`` … ``"Dec"``).  Months
        with no data are absent from the inner dict.

        Two additional string keys exist at the top level:

        ``"best_month"``  — e.g. ``"Jan 2023 (+8.45%)"``
        ``"worst_month"`` — e.g. ``"Aug 2022 (-12.30%)"``

        All ``pct_return`` values are expressed as percentages (e.g. 3.14
        means +3.14%), rounded to 4 decimal places.

        Returns ``{"best_month": "N/A", "worst_month": "N/A"}`` when inputs
        are empty or length-mismatched.

    Example::

        tbl = monthly_returns_table(rets, dates)
        tbl[2023]["Jan"]   # → 3.1416
        tbl["best_month"]  # → "Jan 2023 (+3.14%)"
    """
    if not returns or not dates or len(returns) != len(dates):
        return {"best_month": "N/A", "worst_month": "N/A"}

    series = _to_series(returns, dates)
    series = series[series.index.notna()]  # drop parse failures

    if series.empty:
        return {"best_month": "N/A", "worst_month": "N/A"}

    result: dict = {}
    best_val: Optional[float] = None
    best_label: str = "N/A"
    worst_val: Optional[float] = None
    worst_label: str = "N/A"

    grouped = series.groupby([series.index.year, series.index.month])
    for (year, month), grp in grouped:
        compound_ret = round(float((1.0 + grp).prod() - 1.0) * 100.0, 4)

        if year not in result:
            result[year] = {}
        month_abbr = calendar.month_abbr[month]  # "Jan", "Feb", …
        result[year][month_abbr] = compound_ret

        label = f"{month_abbr} {year}"
        sign = "+" if compound_ret >= 0 else ""
        full_label = f"{label} ({sign}{compound_ret:.2f}%)"

        if best_val is None or compound_ret > best_val:
            best_val = compound_ret
            best_label = full_label
        if worst_val is None or compound_ret < worst_val:
            worst_val = compound_ret
            worst_label = full_label

    result["best_month"] = best_label
    result["worst_month"] = worst_label
    return result


# ---------------------------------------------------------------------------
# 5. Year-by-year table
# ---------------------------------------------------------------------------

def year_by_year_table(
    returns: list,
    dates: list,
    benchmark_returns: Optional[list] = None,
) -> list:
    """Build a year-by-year performance breakdown for the Strategy Lab table.

    Args:
        returns:           Per-bar return floats aligned with ``dates``.
        dates:             ISO ``'YYYY-MM-DD'`` strings, one per bar.
        benchmark_returns: Optional list of per-bar benchmark returns, same
                           length as ``returns``.  When ``None`` or
                           length-mismatched, ``benchmark_pct`` is ``None``.

    Returns:
        List of dicts, one per calendar year in ``dates``, sorted ascending
        by year.  Each dict contains:

        ``year``             — int, the calendar year.
        ``return_pct``       — float, compounded strategy return (%), 4 d.p.
        ``benchmark_pct``    — float or ``None``, benchmark compounded return.
        ``sharpe``           — float or ``None``, annualised in-year Sharpe
                               (zero r-f, ``ddof=1``); ``None`` with < 2 bars.
        ``max_drawdown_pct`` — float ≥ 0, largest intra-year drawdown (%),
                               computed on the year's equity sub-curve.
        ``num_trades_note``  — str ``"see signals list"`` — full per-year trade
                               counts require the signals list and are handled
                               by the caller to keep this function pure.

        Returns an empty list when inputs are empty or length-mismatched.

    Example::

        rows = year_by_year_table(rets, dates)
        rows[0]
        # → {'year': 2022, 'return_pct': -18.4, 'benchmark_pct': None,
        #    'sharpe': -1.23, 'max_drawdown_pct': 24.1,
        #    'num_trades_note': 'see signals list'}
    """
    if not returns or not dates or len(returns) != len(dates):
        return []

    series = _to_series(returns, dates)
    series = series[series.index.notna()]
    if series.empty:
        return []

    has_benchmark = (
        benchmark_returns is not None
        and len(benchmark_returns) == len(returns)
    )
    bench_series: Optional[pd.Series] = None
    if has_benchmark:
        bench_series = _to_series(benchmark_returns, dates)
        bench_series = bench_series[bench_series.index.notna()]

    rows: list = []
    for year, grp in series.groupby(series.index.year):
        # Compounded annual return
        ann_ret = round(float((1.0 + grp).prod() - 1.0) * 100.0, 4)

        # Benchmark compounded annual return
        bench_ret: Optional[float] = None
        if bench_series is not None:
            bench_grp = bench_series[bench_series.index.year == year]
            if not bench_grp.empty:
                bench_ret = round(float((1.0 + bench_grp).prod() - 1.0) * 100.0, 4)

        # Annualised in-year Sharpe (zero risk-free)
        sharpe_val: Optional[float] = None
        if len(grp) >= 2:
            std = float(grp.std(ddof=1))
            if std > 0.0:
                sharpe_val = round(float(grp.mean()) / std * math.sqrt(252), 4)

        # Max intra-year drawdown via equity sub-curve starting at 1.0
        equity_year = (1.0 + grp).cumprod().tolist()
        dd_year = compute_drawdown_series(equity_year)
        max_dd = round(abs(min(dd_year)), 4) if dd_year else 0.0

        rows.append({
            "year": int(year),
            "return_pct": ann_ret,
            "benchmark_pct": bench_ret,
            "sharpe": sharpe_val,
            "max_drawdown_pct": max_dd,
            "num_trades_note": "see signals list",
        })

    return sorted(rows, key=lambda r: r["year"])


# ---------------------------------------------------------------------------
# 6. Unit economics per trade
# ---------------------------------------------------------------------------

def unit_economics_per_trade(profit_per_trade: list) -> dict:
    """Compute per-trade unit economics from closed-trade net P&L values.

    Args:
        profit_per_trade: List of net P&L values per closed trade
                          (``results['profit_per_trade']``).

    Returns:
        dict with keys:

        ``avg_pnl``       — float, mean net P&L per trade;
                            ``None`` when ``profit_per_trade`` is empty.
        ``median_pnl``    — float, median net P&L per trade.
        ``win_rate_pct``  — float in [0, 100], percentage of winning trades.
        ``avg_win``       — float, average P&L of profitable trades (P&L > 0);
                            ``None`` when there are no wins.
        ``avg_loss``      — float, average P&L of losing/break-even trades
                            (P&L ≤ 0); ``None`` when there are no losses.
        ``expectancy``    — float, expected P&L per trade
                            ``= p(win)*avg_win + p(loss)*avg_loss``;
                            ``None`` if not computable.
        ``profit_factor`` — float, gross wins / |gross losses|;
                            ``None`` when gross losses are zero
                            (div-by-zero guard).

        All monetary values share the same currency unit as input.

    Example::

        ue = unit_economics_per_trade([200.0, -50.0, 150.0, -30.0])
        ue['win_rate_pct']   # → 50.0
        ue['profit_factor']  # → (200+150) / (50+30) = 4.375
    """
    if not profit_per_trade:
        return {
            "avg_pnl": None,
            "median_pnl": None,
            "win_rate_pct": 0.0,
            "avg_win": None,
            "avg_loss": None,
            "expectancy": None,
            "profit_factor": None,
        }

    arr = np.asarray(profit_per_trade, dtype=float)
    n = len(arr)

    wins = arr[arr > 0]
    losses = arr[arr <= 0]

    win_rate_pct = round(float(len(wins) / n * 100.0), 4)
    loss_rate_pct = 100.0 - win_rate_pct

    avg_win: Optional[float] = round(float(wins.mean()), 4) if len(wins) > 0 else None
    avg_loss: Optional[float] = round(float(losses.mean()), 4) if len(losses) > 0 else None

    # Expectancy: E[P&L per trade]
    if avg_win is not None and avg_loss is not None:
        expectancy: Optional[float] = round(
            (win_rate_pct / 100.0) * avg_win + (loss_rate_pct / 100.0) * avg_loss, 4
        )
    elif avg_win is not None:
        expectancy = avg_win  # 100% win rate
    elif avg_loss is not None:
        expectancy = avg_loss  # 100% loss rate
    else:
        expectancy = None

    # Profit factor
    gross_losses = float(abs(losses.sum())) if len(losses) > 0 else 0.0
    gross_wins = float(wins.sum()) if len(wins) > 0 else 0.0
    profit_factor: Optional[float] = (
        round(gross_wins / gross_losses, 4) if gross_losses != 0.0 else None
    )

    return {
        "avg_pnl": round(float(np.mean(arr)), 4),
        "median_pnl": round(float(np.median(arr)), 4),
        "win_rate_pct": win_rate_pct,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
    }


# ---------------------------------------------------------------------------
# 7. Strategy book
# ---------------------------------------------------------------------------

def build_strategy_book(
    strategy_manager,
    data: "pd.DataFrame",
    benchmark_ticker: str = 'SPY',
    cash: float = 100_000.0,
) -> list:
    """Run every registered strategy against ``data`` and collect key metrics.

    Iterates ``strategy_manager.get_available_strategies()``, runs each
    through ``strategy_manager.run_backtest()``, and records the standard
    backtest metrics.  Any strategy that raises an exception is captured as an
    error entry so one failure cannot prevent the rest from running.

    Successful entries are sorted by ``sharpe`` descending; failed entries
    appear at the end in iteration order.

    Args:
        strategy_manager: A ``StrategyManager`` instance (or compatible object
                          exposing ``get_available_strategies()``,
                          ``get_strategy(name)`` and ``run_backtest()``).
        data:             OHLCV ``pd.DataFrame`` compatible with backtrader's
                          ``PandasData`` feed.
        benchmark_ticker: Ticker for alpha/beta calculation (default ``'SPY'``).
        cash:             Initial cash per strategy run (default 100 000).

    Returns:
        List of dicts.

        Successful entry::

            {"name": str, "sharpe": float, "max_drawdown": float, "win_rate": float}

        Failed entry::

            {"name": str, "error": str}

    Example::

        book = build_strategy_book(sm, df)
        # → [{"name": "EMA Crossover", "sharpe": 1.23, "max_drawdown": 8.4,
        #      "win_rate": 62.5}, ...]
    """
    names = strategy_manager.get_available_strategies()
    successes: list = []
    failures: list = []

    for name in names:
        try:
            wrapper = strategy_manager.get_strategy(name)
            if wrapper is None:
                raise RuntimeError(f"get_strategy returned None for '{name}'")

            report = strategy_manager.run_backtest(
                strategy_wrapper=wrapper,
                data=data.copy(),
                cash=cash,
                broker_mode="simulated",
            )

            if "error" in report:
                raise RuntimeError(report["error"])

            successes.append({
                "name": name,
                "sharpe": float(report.get("sharpe", 0.0)),
                "max_drawdown": float(report.get("max_drawdown", 0.0)),
                "win_rate": float(report.get("win_rate", 0.0)),
            })

        except Exception as exc:
            logger.warning(f"build_strategy_book: strategy '{name}' failed — {exc}")
            failures.append({"name": name, "error": str(exc)})

    successes.sort(key=lambda d: d["sharpe"], reverse=True)
    return successes + failures


# ---------------------------------------------------------------------------
# 8. Gate verdict
# ---------------------------------------------------------------------------

def evaluate_gate(report: dict, thresholds: Optional[dict] = None) -> dict:
    """Evaluate a backtest report against quant-research go/no-go thresholds.

    Runs a fixed suite of named checks and produces a human-readable verdict
    that highlights the most likely kill-switch risk first — written in the
    tone of a real quant-research gate review rather than as a dry list of
    metrics.

    Args:
        report:     The dict returned by ``Backtester.run_backtest()``.  Must
                    include at least ``sharpe``, ``max_drawdown``, ``win_rate``
                    and ``summary["Number of Closed Trades"]``.
        thresholds: Optional override dict.  Recognised keys:

                    ``min_sharpe`` (float, default 0.3)
                      Minimum acceptable annualised Sharpe ratio.

                    ``max_drawdown_pct`` (float, default 40.0)
                      Maximum tolerable max drawdown (%).

                    ``min_trades`` (int, default 30)
                      Minimum closed trades for statistical validity.

                    ``win_rate_bounds`` (tuple[float, float], default (20, 80))
                      Win-rate range outside which a possible overfit signal is
                      flagged.  Values outside this range do NOT automatically
                      fail the gate but contribute to the ``passed`` flag via
                      the check list.

    Returns:
        dict with keys:

        ``passed``       — bool; ``True`` iff **all** checks pass.
        ``checks``       — list of dicts ``{"name": str, "passed": bool,
                           "detail": str}``, one per named check.
        ``verdict_text`` — str; one-paragraph human summary, leading with the
                           most critical failure, or a confidence assessment
                           when all checks pass.

    Example::

        gate = evaluate_gate(report)
        gate['passed']         # → False
        gate['checks'][0]
        # → {'name': 'Sharpe Ratio', 'passed': False,
        #    'detail': 'Sharpe 0.1800 < min 0.30 — risk-adjusted return is insufficient.'}
        print(gate['verdict_text'])
    """
    # --- Resolve thresholds ---
    cfg: dict = {
        "min_sharpe": 0.3,
        "max_drawdown_pct": 40.0,
        "min_trades": 30,
        "win_rate_bounds": (20.0, 80.0),
    }
    if thresholds:
        cfg.update(thresholds)

    min_sharpe = float(cfg["min_sharpe"])
    max_dd_threshold = float(cfg["max_drawdown_pct"])
    min_trades = int(cfg["min_trades"])
    wr_lo = float(cfg["win_rate_bounds"][0])
    wr_hi = float(cfg["win_rate_bounds"][1])

    # --- Extract report values ---
    sharpe = float(report.get("sharpe", 0.0))
    max_dd = float(report.get("max_drawdown", 0.0))
    win_rate = float(report.get("win_rate", 0.0))
    n_trades = int(report.get("summary", {}).get("Number of Closed Trades", 0))

    checks: list = []
    failure_reasons: list = []
    warning_reasons: list = []

    # --- Check 1: Sharpe ratio ---
    sharpe_ok = sharpe >= min_sharpe
    if sharpe_ok:
        sharpe_detail = (
            f"Sharpe {sharpe:.4f} ≥ min {min_sharpe:.2f} — "
            f"acceptable risk-adjusted return."
        )
    else:
        sharpe_detail = (
            f"Sharpe {sharpe:.4f} < min {min_sharpe:.2f} — "
            f"risk-adjusted return is insufficient."
        )
        failure_reasons.append(
            f"Sharpe ratio ({sharpe:.4f}) is below the {min_sharpe:.2f} threshold"
        )
    checks.append({"name": "Sharpe Ratio", "passed": sharpe_ok, "detail": sharpe_detail})

    # --- Check 2: Max drawdown ---
    dd_ok = max_dd <= max_dd_threshold
    if dd_ok:
        dd_detail = (
            f"Max drawdown {max_dd:.2f}% ≤ limit {max_dd_threshold:.1f}% — "
            f"drawdown within tolerance."
        )
    else:
        dd_detail = (
            f"Max drawdown {max_dd:.2f}% exceeds limit {max_dd_threshold:.1f}% — "
            f"position sizing and stop-loss rules need review."
        )
        failure_reasons.append(
            f"max drawdown ({max_dd:.2f}%) exceeds the {max_dd_threshold:.1f}% limit"
        )
    checks.append({"name": "Max Drawdown", "passed": dd_ok, "detail": dd_detail})

    # --- Check 3: Minimum trade count ---
    trades_ok = n_trades >= min_trades
    if trades_ok:
        trades_detail = (
            f"{n_trades} closed trades ≥ min {min_trades} — "
            f"sample is large enough for statistical inference."
        )
    else:
        trades_detail = (
            f"Only {n_trades} closed trades (min {min_trades}) — "
            f"results are statistically unreliable; expand the backtest window "
            f"or loosen entry conditions."
        )
        failure_reasons.append(
            f"trade count ({n_trades}) is too small for statistical confidence "
            f"(min {min_trades})"
        )
    checks.append({"name": "Trade Count", "passed": trades_ok, "detail": trades_detail})

    # --- Check 4: Win-rate overfit signal ---
    wr_ok = wr_lo <= win_rate <= wr_hi
    if wr_ok:
        wr_detail = (
            f"Win rate {win_rate:.2f}% is within [{wr_lo:.0f}%, {wr_hi:.0f}%] — "
            f"no overfit flag."
        )
    else:
        direction = "high" if win_rate > wr_hi else "low"
        wr_detail = (
            f"Win rate {win_rate:.2f}% is outside [{wr_lo:.0f}%, {wr_hi:.0f}%] "
            f"(unusually {direction}). This can indicate data snooping or an "
            f"asymmetric exit rule that may not replicate out-of-sample."
        )
        warning_reasons.append(
            f"win rate ({win_rate:.2f}%) is outside the [{wr_lo:.0f}%–{wr_hi:.0f}%] "
            f"expected range, which is an overfit warning signal"
        )
    checks.append({"name": "Win Rate Overfit Signal", "passed": wr_ok, "detail": wr_detail})

    all_passed = all(c["passed"] for c in checks)

    # --- Compose verdict text ---
    parts: list = []

    if all_passed:
        parts.append(
            f"Strategy clears all four gate checks: Sharpe {sharpe:.2f}, "
            f"max drawdown {max_dd:.1f}%, win rate {win_rate:.1f}%, and "
            f"{n_trades} closed trades."
        )
        parts.append(
            "The most likely erosion risk at this stage is live execution — "
            "slippage, spread, and commission drag can silently eat into a "
            "Sharpe this modest. Confirm that the commission model used in "
            "this backtest matches the broker's worst-case fill cost, "
            "including market-impact on larger sizes."
        )
    else:
        # Lead with the primary failure reason
        if failure_reasons:
            primary = failure_reasons[0]
            parts.append(
                f"This strategy is most likely to fail in live trading because "
                f"{primary}. "
            )

        if not sharpe_ok:
            parts.append(
                f"A Sharpe of {sharpe:.2f} means the return per unit of "
                f"volatility is insufficient to survive realistic transaction "
                f"costs and slippage; the edge will almost certainly disappear "
                f"net of execution friction. "
            )

        if not dd_ok:
            parts.append(
                f"A max drawdown of {max_dd:.1f}% exceeds the "
                f"{max_dd_threshold:.1f}% limit — this level of loss will "
                f"test risk-management rules and may trigger forced liquidation "
                f"or investor redemptions before the strategy can recover. "
                f"Review stop-loss levels and position sizing. "
            )

        if not trades_ok:
            parts.append(
                f"With only {n_trades} closed trades the performance metrics "
                f"lack statistical power; Sharpe and win-rate figures cannot "
                f"be trusted. Extend the backtest window or relax entry "
                f"criteria to generate a credible trade sample. "
            )

        if warning_reasons:
            parts.append(
                f"Additionally, {warning_reasons[0]} — "
                f"out-of-sample validation is required before live deployment. "
            )

    verdict_text = " ".join(parts).strip()
    return {
        "passed": all_passed,
        "checks": checks,
        "verdict_text": verdict_text,
    }
