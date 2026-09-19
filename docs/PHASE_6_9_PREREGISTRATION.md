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
