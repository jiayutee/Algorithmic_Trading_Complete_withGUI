# Phase 6.9 pre-registration: does the trend-filter drawdown reduction hold in an EARLIER period?

Written 2026-09-19, **before the pre-2022 data was loaded and before any 6.9 code existed**. Phase 6.8 confirmed the
drawdown reduction on different coins, but in the *same calendar period* as Phase 6.7 (2022-09 -> 2026-09), so it could
not rule out that the effect belongs to that one regime. This tests the identical rule in an earlier, disjoint period
(2019 -> 2022-09-09: the 2019 recovery, the 2020 crash, the 2021 bull run, the 2022 collapse).

## Setup (fixed)
- **Universe (chosen now):** BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, ADAUSDT, LTCUSDT, LINKUSDT, TRXUSDT (all listed on Binance by early 2019).
  A symbol without data covering the window start is dropped and reported; fewer than 6 remaining voids the experiment.
- **Evaluation window:** every common date up to and including **2022-09-09**, i.e. strictly before the 6.7/6.8 windows. Nothing after that date is used.
- **Strategies, tests and criteria: exactly Phase 6.8** -- TSMOM (28-day trend, weekly, 0.1% fees) vs equal weight; D1 maxDD difference
  (30-day-block paired bootstrap, 97.5% interval, both halves) and D2 Sharpe non-inferiority (lower bound > -0.50). K = 2.
- Confirmed only if all three criteria hold; otherwise "not confirmed" and reported as such -- including if the sign flips.

## Limits stated in advance
- Survivorship: all 8 coins survive to today. The 2019-2022 period includes several coins' largest run-ups.
- An interval that includes zero means the earlier period neither supports nor refutes the effect; it does not falsify 6.8.
- A result here applies to crypto only; nothing is implied about equities or other assets.

---

# Results (run 2026-09-19; protocol exactly as above, nothing tuned)

`training_ground/results/phase_6_9.json`. Reproduce: `python training_ground/experiments_6_9.py`.
All 8 symbols had data. Window 2019-02-14 -> 2022-09-09 (1,304 days), nothing after the cutoff used.

| | Sharpe | ann. return | ann. vol | max drawdown |
|---|---|---|---|---|
| Equal weight | 1.25 | 100.8% | 87.3% | -74.3% |
| TSMOM (28-day trend filter) | 1.70 | 125.9% | 58.0% | -47.7% |

| test | value | 97.5% interval | halves |
|---|---|---|---|
| D1 maxDD(TSMOM) - maxDD(EW) | **+26.5 pp shallower** | [+4.3, +46.0] pp | +24.9 / +26.5 pp |
| D2 Sharpe(TSMOM) - Sharpe(EW) | +0.45 | [-0.15, +1.07] (bound -0.50) | - |

**Verdict: drawdown reduction CONFIRMED** on all three pre-registered criteria, in a period disjoint from Phases 6.7/6.8.

## Reading it
- **The drawdown effect now has three looks:** observed on the original 8 (6.7: -67% -> -37.5%), confirmed on 8 other coins in the
  same period (6.8: +14.6 pp), and confirmed in an earlier period (6.9: +26.5 pp, lower bound clearly above zero this time).
  Same sign, same mechanism (sitting in cash through down-trends), varying size.
- **Still a smoother ride, not proven extra return:** the Sharpe difference interval [-0.15, +1.07] includes zero here too.
  The absolute numbers (EW Sharpe 1.25, +100%/yr) show this period was an extraordinary bull run for survivors; the
  filter's Sharpe gain is plausibly regime-dependent and is not claimed.
- **Limits that remain:** crypto only; survivorship (every coin here survived to today, and 2019-2022 flatters them);
  the 28-day lookback was fixed in Phase 6.7's pre-registration before any result existed and was never tuned, but that
  also means it is one arbitrary choice: 14 and 56 days were only reported as untested context in 6.7, so nothing here shows 28 is
  better than its neighbours. A max drawdown of -47.7% is still very large.
- What would be worth doing with this: offer a trend-filter *overlay* (hold a position only while its 28-day return is
  positive) as a risk-reduction option in the app, described as drawdown reduction and not as alpha.
