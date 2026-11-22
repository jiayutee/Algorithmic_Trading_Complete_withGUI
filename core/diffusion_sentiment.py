"""Phase 10.0 slice 1: how a headline's price effect diffuses over time. Pure functions, no I/O, no orders.

Model (docs/PHASE_10_0_PREREGISTRATION.md, approved 2026-11-22):   CAR(h) = beta * s * K(h)
  s     headline sentiment shock (positive - negative), in [-1, 1]
  K(h)  kernel: share of the eventual move that has happened h hours after the first bar that opens after publication
  beta  size of the full move per unit of sentiment
CAR(h) is the sum of the abnormal hourly log returns of bars t0 .. t0+h-1 (so CAR(1) is the return of the bar that opens right after the
headline). Abnormal = coin return minus beta_mkt * BTC return, with beta_mkt estimated on the 720 hourly bars BEFORE t0 only; BTC itself
uses the raw return.

This module reads no news store and no prices: callers pass DataFrames. Slice 1 is tested on synthetic data with a planted signal.
The verdict logic below is the rule fixed in advance in the pre-registration; it is not tuned.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

HORIZONS = np.arange(1, 49)          # hours after t0
PRE_HOURS = 24                       # placebo window: the 24 hours before t0
DECLUSTER_HOURS = 6
BETA_WINDOW_BARS = 24 * 30
MIN_BETA_BARS = BETA_WINDOW_BARS // 2
FIT_FRACTION = 0.6
MIN_HOLDOUT_EVENTS = 200
MARKET = "BTCUSDT"
KINDS = ("null", "instant", "exponential", "bass")


# ------------------------------------------------------------------ kernels
def kernel_null(h):
    return np.zeros_like(np.asarray(h, dtype=float))


def kernel_instant(h):
    return np.ones_like(np.asarray(h, dtype=float))


def kernel_exponential(h, tau):
    return 1.0 - np.exp(-np.asarray(h, dtype=float) / tau)


def kernel_bass(h, p, q):
    e = np.exp(-(p + q) * np.asarray(h, dtype=float))
    return (1.0 - e) / (1.0 + (q / p) * e)


TAU_GRID = np.logspace(np.log10(0.25), np.log10(500), 240)
BASS_P_GRID = np.logspace(-3, 0.5, 36)
BASS_Q_GRID = np.logspace(-3, 1, 36)


def _grid(kind: str):
    """(parameter tuples, kernel matrix [n_params, 48]) for a kernel family."""
    if kind == "instant":
        return [()], kernel_instant(HORIZONS)[None, :]
    if kind == "exponential":
        return [(t,) for t in TAU_GRID], np.array([kernel_exponential(HORIZONS, t) for t in TAU_GRID])
    if kind == "bass":
        params = [(p, q) for p in BASS_P_GRID for q in BASS_Q_GRID]
        return params, np.array([kernel_bass(HORIZONS, p, q) for p, q in params])
    raise ValueError(f"unknown kernel kind {kind!r}")


# ------------------------------------------------------------------ events
@dataclass
class Events:
    """De-clustered events. ``car``: [n, 48] cumulative abnormal returns at h = 1..48; ``pre``: [n] abnormal return over the 24 h before t0."""
    t0: pd.DatetimeIndex
    ticker: np.ndarray
    s: np.ndarray
    car: np.ndarray
    pre: np.ndarray
    dropped: Dict[str, int] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.s)

    @property
    def day(self) -> np.ndarray:
        return np.asarray(self.t0.tz_convert("UTC").normalize().asi8 // 86_400_000_000_000)

    def subset(self, mask) -> "Events":
        mask = np.asarray(mask)
        return Events(self.t0[mask], self.ticker[mask], self.s[mask], self.car[mask], self.pre[mask], dict(self.dropped))


def _tickers_of(value) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            value = parsed if isinstance(parsed, (list, tuple)) else [parsed]
        except ValueError:
            value = [v for v in value.replace(";", ",").split(",")]
    return [str(v).strip().upper() for v in value if str(v).strip()]


def select_headlines(headlines: pd.DataFrame) -> pd.DataFrame:
    """Rule 1: exactly one *USDT ticker, a non-neutral label, and not a date-only (exactly midnight) timestamp.
    Needs columns: published (UTC), tickers, label, shock. Returns rows with a single ``ticker`` column."""
    df = headlines.copy()
    df["published"] = pd.to_datetime(df["published"], utc=True, format="ISO8601")
    usdt = df["tickers"].map(lambda v: [t for t in _tickers_of(v) if t.endswith("USDT")])
    df = df[usdt.map(len) == 1].copy()
    df["ticker"] = usdt[df.index].map(lambda v: v[0])
    df = df[df["label"].astype(str).str.lower() != "neutral"]
    pub = df["published"]
    midnight = (pub.dt.hour == 0) & (pub.dt.minute == 0) & (pub.dt.second == 0) & (pub.dt.microsecond == 0)
    return df[~midnight].sort_values("published").reset_index(drop=True)[["published", "ticker", "shock"]]


def decluster(df: pd.DataFrame, hours: float = DECLUSTER_HOURS) -> pd.DataFrame:
    """Rule 3: per ticker, a headline within ``hours`` of the earlier kept (anchor) headline is merged into it; the event's s is the mean."""
    rows = []
    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("published")
        anchor, group = None, []
        for pub, shock in zip(g["published"], g["shock"]):
            if anchor is not None and pub - anchor <= pd.Timedelta(hours=hours):
                group.append(shock)
            else:
                if anchor is not None:
                    rows.append((anchor, ticker, float(np.mean(group))))
                anchor, group = pub, [shock]
        if anchor is not None:
            rows.append((anchor, ticker, float(np.mean(group))))
    out = pd.DataFrame(rows, columns=["published", "ticker", "shock"])
    return out.sort_values("published").reset_index(drop=True)


def first_bar_after(published: pd.Timestamp) -> pd.Timestamp:
    """Rule 2: the first hourly bar that opens strictly after publication (a headline exactly on the hour waits for the next bar)."""
    return published.floor("h") + pd.Timedelta(hours=1)


def _bar_returns(close: pd.Series) -> pd.Series:
    close = close.sort_index()
    return np.log(close / close.shift(1))      # return of the bar that OPENS at index (previous close ~ this open)


def build_events(headlines: pd.DataFrame, closes: Dict[str, pd.Series]) -> Events:
    """Headlines -> events with abnormal-return paths. ``closes``: ticker -> hourly close, index = bar open time (UTC)."""
    ev = decluster(select_headlines(headlines))
    if MARKET not in closes:
        raise ValueError("BTCUSDT closes are required (market return for the abnormal-return adjustment)")
    rets = {t: _bar_returns(c) for t, c in closes.items()}
    btc = rets[MARKET]
    keep, t0s, tick, ss, cars, pres = 0, [], [], [], [], []
    dropped = {"no_prices": 0, "short_history": 0, "incomplete_window": 0}
    hour = pd.Timedelta(hours=1)
    for pub, ticker, shock in zip(ev["published"], ev["ticker"], ev["shock"]):
        if ticker not in rets:
            dropped["no_prices"] += 1
            continue
        t0 = first_bar_after(pub)
        r = rets[ticker]
        post_idx = pd.date_range(t0, periods=len(HORIZONS), freq="h")
        pre_idx = pd.date_range(t0 - PRE_HOURS * hour, periods=PRE_HOURS, freq="h")
        if ticker == MARKET:
            beta_mkt = 0.0
        else:
            win = pd.date_range(t0 - BETA_WINDOW_BARS * hour, periods=BETA_WINDOW_BARS, freq="h")   # bars strictly before t0
            both = pd.concat([r.reindex(win), btc.reindex(win)], axis=1, keys=["x", "m"]).dropna()
            if len(both) < MIN_BETA_BARS or both["m"].var() == 0:
                dropped["short_history"] += 1
                continue
            beta_mkt = float(both["x"].cov(both["m"]) / both["m"].var())
        x_post = r.reindex(post_idx).to_numpy() - beta_mkt * btc.reindex(post_idx).to_numpy()
        x_pre = r.reindex(pre_idx).to_numpy() - beta_mkt * btc.reindex(pre_idx).to_numpy()
        if np.isnan(x_post).any() or np.isnan(x_pre).any():
            dropped["incomplete_window"] += 1
            continue
        t0s.append(t0)
        tick.append(ticker)
        ss.append(shock)
        cars.append(np.cumsum(x_post))
        pres.append(float(x_pre.sum()))
    idx = pd.DatetimeIndex(t0s, tz="UTC") if t0s else pd.DatetimeIndex([], tz="UTC")
    return Events(idx, np.array(tick, dtype=object), np.array(ss, dtype=float),
                  np.array(cars, dtype=float).reshape(len(ss), len(HORIZONS)), np.array(pres, dtype=float), dropped)


def split_by_time(events: Events, fit_fraction: float = FIT_FRACTION):
    """First ``fit_fraction`` of the event-time RANGE is the fit set, the rest the holdout (never random)."""
    if events.n == 0:
        return events, events
    t = events.t0.asi8
    cut = t.min() + fit_fraction * (t.max() - t.min())
    fit_mask = t <= cut
    return events.subset(fit_mask), events.subset(~fit_mask)


# ------------------------------------------------------------------ fitting (least squares on CAR over all events and horizons)
def _stats(ev: Events):
    """Sufficient statistics: a_h = sum_i s_i CAR_i(h), b = sum_i s_i^2."""
    return (ev.s[:, None] * ev.car).sum(axis=0), float((ev.s ** 2).sum())


def nonparametric_kernel(ev: Events) -> np.ndarray:
    """beta*K(h) with no shape assumed: one least-squares slope of CAR(h) on s per hour."""
    a, b = _stats(ev)
    if b == 0:
        raise ValueError("no events with non-zero sentiment")
    return a / b


def fit_kernel(ev: Events, kind: str) -> dict:
    """Least squares of CAR(h) on beta*s*K(h; theta). For a fixed theta the best beta is closed-form, so theta is found by a fixed grid."""
    if kind == "null":
        return {"kind": "null", "beta": 0.0, "params": ()}
    a, b = _stats(ev)
    if b == 0:
        raise ValueError("no events with non-zero sentiment")
    params, K = _grid(kind)
    num = K @ a
    den = (K ** 2).sum(axis=1)
    gain = num ** 2 / den                       # reduction in squared error for each theta (beta at its best)
    j = int(np.argmax(gain))
    return {"kind": kind, "beta": float(num[j] / (b * den[j])), "params": tuple(float(v) for v in params[j])}


def kernel_of(fit: dict) -> np.ndarray:
    kind, p = fit["kind"], fit["params"]
    if kind == "null":
        return kernel_null(HORIZONS)
    if kind == "instant":
        return kernel_instant(HORIZONS)
    if kind == "exponential":
        return kernel_exponential(HORIZONS, *p)
    return kernel_bass(HORIZONS, *p)


def holdout_mse(fit: dict, ev: Events) -> float:
    pred = fit["beta"] * ev.s[:, None] * kernel_of(fit)[None, :]
    return float(np.mean((ev.car - pred) ** 2))


def _day_groups(ev: Events):
    days = ev.day
    uniq = np.unique(days)
    return uniq, [np.flatnonzero(days == d) for d in uniq]


def _resample_days(ev: Events, rng: np.random.Generator) -> Events:
    """Day-block bootstrap: resample whole UTC days with replacement, keeping every event of a drawn day."""
    _, groups = _day_groups(ev)
    pick = rng.integers(0, len(groups), len(groups))
    idx = np.concatenate([groups[k] for k in pick])
    return Events(ev.t0[idx], ev.ticker[idx], ev.s[idx], ev.car[idx], ev.pre[idx], dict(ev.dropped))


def bootstrap_params(ev: Events, kind: str, n_boot: int = 2000, seed: int = 0) -> Dict[str, tuple]:
    """95% percentile intervals of beta (and the kernel parameters) from day-block resamples of the fit set. Descriptive, no pass/fail threshold."""
    rng = np.random.default_rng(seed)
    draws = [fit_kernel(_resample_days(ev, rng), kind) for _ in range(n_boot)]
    out = {"beta": tuple(np.percentile([d["beta"] for d in draws], [2.5, 97.5]))}
    for j in range(len(draws[0]["params"])):
        out[f"param{j}"] = tuple(np.percentile([d["params"][j] for d in draws], [2.5, 97.5]))
    return out


# ------------------------------------------------------------------ verdict (rule fixed in the pre-registration)
def _interval(x: np.ndarray):
    return tuple(float(v) for v in np.percentile(x, [2.5, 97.5]))


def diffusion_verdict(fit_ev: Events, hold_ev: Events, n_boot: int = 2000, seed: int = 0,
                      min_events: int = MIN_HOLDOUT_EVENTS) -> dict:
    """DIFFUSION SUPPORTED / NOT SUPPORTED / INSUFFICIENT DATA, judged on the holdout only.
    SUPPORTED needs ALL of: (a) the holdout beta interval excludes 0 with the same sign as on the fit set;
    (b) the best non-null kernel beats Null on holdout MSE with an interval of the difference above 0;
    (c) placebo: the 24 h BEFORE the headline shows no drift in the sentiment direction (interval of the pre-event slope includes 0)."""
    if hold_ev.n < min_events:
        return {"verdict": "INSUFFICIENT DATA", "reason": f"{hold_ev.n} holdout events, need {min_events}", "n_holdout": hold_ev.n}
    fits = {k: fit_kernel(fit_ev, k) for k in KINDS}
    mse = {k: holdout_mse(f, hold_ev) for k, f in fits.items()}
    best = min((k for k in KINDS if k != "null"), key=lambda k: mse[k])
    K = kernel_of(fits[best])
    beta_b = fits[best]["beta"]

    uniq, groups = _day_groups(hold_ev)
    A = np.array([(hold_ev.s[g][:, None] * hold_ev.car[g]).sum(axis=0) for g in groups])
    B = np.array([(hold_ev.s[g] ** 2).sum() for g in groups])
    N = np.array([len(g) for g in groups], dtype=float)
    SSE0 = np.array([(hold_ev.car[g] ** 2).sum() for g in groups])
    SSE1 = np.array([((hold_ev.car[g] - beta_b * hold_ev.s[g][:, None] * K[None, :]) ** 2).sum() for g in groups])
    SP = np.array([(hold_ev.s[g] * hold_ev.pre[g]).sum() for g in groups])
    rng = np.random.default_rng(seed)
    betas, diffs, pre = [], [], []
    for _ in range(n_boot):
        k = rng.integers(0, len(groups), len(groups))
        a, b, n = A[k].sum(axis=0), B[k].sum(), N[k].sum()
        betas.append(float(K @ a / (b * (K ** 2).sum())))
        diffs.append(float((SSE0[k].sum() - SSE1[k].sum()) / (n * len(HORIZONS))))
        pre.append(float(SP[k].sum() / b))
    beta_ci, diff_ci, pre_ci = _interval(np.array(betas)), _interval(np.array(diffs)), _interval(np.array(pre))
    a_ok = (beta_ci[0] > 0 or beta_ci[1] < 0) and np.sign(np.mean(betas)) == np.sign(beta_b)
    b_ok = diff_ci[0] > 0
    c_ok = pre_ci[0] <= 0 <= pre_ci[1]
    ok = bool(a_ok and b_ok and c_ok)
    reason = "all three criteria hold" if ok else "failed: " + ", ".join(
        n for n, v in (("(a) holdout beta interval includes 0 or flips sign", a_ok),
                       ("(b) best kernel does not beat Null with an interval above 0", b_ok),
                       ("(c) placebo: the price already moved before the headline", c_ok)) if not v)
    return {"verdict": "DIFFUSION SUPPORTED" if ok else "NOT SUPPORTED", "reason": reason, "best_kernel": best,
            "best_params": fits[best]["params"], "beta_fit": beta_b, "beta_holdout_ci": beta_ci,
            "mse_gain_vs_null_ci": diff_ci, "pre_slope_ci": pre_ci, "holdout_mse": mse, "n_holdout": hold_ev.n}


# ------------------------------------------------------------------ synthetic data (slice 1 tests; no real data anywhere)
def simulate_events(n: int, beta: float, kind: str = "exponential", params: tuple = (6.0,), noise_sd: float = 0.004,
                    days: int = 60, pre_beta: float = 0.0, seed: int = 0) -> Events:
    """Events with a PLANTED effect: CAR(h) = beta*s*K(h) + a random walk of hourly noise. ``pre_beta`` plants drift BEFORE the headline
    (reverse causality) so the placebo check can be tested."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.15, 1.0, n) * rng.choice([-1.0, 1.0], n)
    K = kernel_of({"kind": kind, "params": params})
    car = beta * s[:, None] * K[None, :] + np.cumsum(rng.normal(0, noise_sd, (n, len(HORIZONS))), axis=1)
    pre = pre_beta * s + rng.normal(0, noise_sd * np.sqrt(PRE_HOURS), n)
    t0 = pd.DatetimeIndex(pd.Timestamp("2026-09-01", tz="UTC") + pd.to_timedelta(np.sort(rng.uniform(0, days * 24, n)).astype(int), unit="h"))
    return Events(t0, np.array(["ETHUSDT"] * n, dtype=object), s, car, pre)
