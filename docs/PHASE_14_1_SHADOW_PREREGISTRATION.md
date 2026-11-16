# Phase 14.1 pre-registration: forward shadow test of the Phase 14.0 portfolio rules

Written 2026-11-16, **before the ledger's first row exists**. Phase 14.0 found the inverse-volatility trend rule (C2) PROMISING but not
statistically supported on historical holdouts. Backtests cannot settle it: every coin in them survived to today, and the rules were chosen after the
fact. The only unseen evidence is the future, so this phase logs what the rules *would have held* each day, with no orders, no broker and no money.

## What is logged (fixed)
- **Universe:** SET1 of Phase 14.0 (BTC, ETH, BNB, XRP, ADA, LTC, LINK, TRX). Fixed; a coin that disappears stops the update loudly (nothing is invented).
- **Rules:** EW and TS28 (yardsticks), **C2 (the pre-registered pick)**, and C3 and C5 labelled **descriptive only** (they looked good in 14.0 but were not the pick).
- **Mechanics = Phase 14.0 exactly:** daily closes of COMPLETED UTC days only; weights for day *t* use only returns before *t*; weekly rebalance (every 7 days
  from the ledger's start day); fee 0.1% of turnover; drifting weights between rebalances; remainder in cash; start NAV 1.0 on the start day.
- Backfill: if updates are missed (Mac asleep), the next update processes every missed completed day in order; the result is identical to having run daily.

## Evaluation (fixed, not before the data exists)
- **No conclusion is drawn before 180 forward days.** Before that, the status prints numbers and says so.
- At **>= 180 days**, apply the 14.0 criteria to C2 against EW and TS28 on the forward window: (a) Sharpe(C2) > Sharpe(EW); (b) max drawdown at least 10
  points shallower than EW; (c) Sharpe(C2) >= Sharpe(TS28). Verdict **FORWARD-CONSISTENT** if all three hold, else **NOT CONFIRMED**. The Sharpe difference
  gets a paired 30-day-block bootstrap interval for information, but 180 days cannot give a statistically supported verdict by itself; the target for that is
  **>= 730 days**, when the 14.0 test (97.5% interval of Sharpe(C2) - Sharpe(EW) above 0) is applied.
- C3 and C5 are reported, never adopted on the basis of this ledger.

## What this is not
No order path, no broker, no change to the paper runner, no profit claim. A FORWARD-CONSISTENT verdict would justify a separate, owner-approved step
(a paper-run of the portfolio rule), not real money.
