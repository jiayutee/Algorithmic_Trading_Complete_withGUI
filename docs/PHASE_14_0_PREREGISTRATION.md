# Phase 14.0 pre-registration: the best long-only crypto portfolio rule I can build, with selection and test kept apart

Written 2026-11-16, **before any 14.0 number exists**. Owner asked for "the best strategy you can come up with". The honest way to
answer that is to generate a small, fixed set of candidates, pick ONE on data that is then never used for the verdict, and judge the pick on
data it has never seen. Everything the earlier phases found stands: no directional edge was shown, the 28-day trend filter cuts drawdown
(three looks) without a significant Sharpe gain, volatility targeting of single assets did not help. This phase asks whether combining those
ingredients at *portfolio* level does better, and says plainly if it does not.

## Data (fixed)
- Binance spot daily closes via `core.order_flow_data.fetch_klines`. Fee 0.1% per unit of turnover (as in 6.7-6.9). Long-only, no leverage, cash earns 0.
- **SET1** (8): BTC, ETH, BNB, XRP, ADA, LTC, LINK, TRX (all listed by early 2019). **SET2** (8, different coins): DOT, AVAX, ATOM, NEAR, UNI, FIL, SOL, DOGE.
- **Window A (selection):** SET1, 2019-02-14 .. 2022-09-09. **Holdout H1:** SET1, 2022-09-10 .. latest. **Holdout H2:** SET2, 2022-09-10 .. latest
  (different coins, same calendar period). A symbol with no data for a window is dropped and reported; fewer than 6 symbols or 800 days voids that window.
- Weights for day *t* use only returns strictly before *t* (up to 120 days of history before a window start may be used for look-back, never after).
- Survivorship: every coin survives to today, so all numbers flatter all strategies; the comparison between strategies is fairer than the levels.

## Candidates (fixed now; no parameter is tuned; weekly rebalance, weights decided from the prior days)
`trail_L` = return over the last L days; `sigma` = std of daily returns over the last 60 days; N = number of coins.
| id | rule |
|---|---|
| EW | benchmark: 1/N each |
| TS28 | baseline (Phase 6.7-6.9): coin i held at 1/N while `trail_28 > 0`, else cash |
| C1 vote | `v_i` = mean of `1[trail_L > 0]` over L in {14, 28, 56, 112}; weight `v_i / N` |
| C2 inv-vol trend | `1[trail_28 > 0] * (1/sigma_i) / sum_j(1/sigma_j)` (cash when trend is off) |
| C3 vote + inv-vol | `v_i * (1/sigma_i) / sum_j(1/sigma_j)` |
| C4 trend-filtered momentum | among coins with `trail_28 > 0`, the top 4 by `trail_56`, weight 1/4 each (empty slots stay cash) |
| C5 C3 + vol cap | C3 weights scaled by `min(1, 0.40 / predicted portfolio vol)` (60-day covariance, annualised); never levered |

## Selection and verdict (fixed)
1. Score C1..C5 on **window A** by Sharpe. The highest is the **selected rule S**. (TS28 and EW are never selected: they are the yardsticks.)
2. Evaluate S **once** on H1 and H2 against EW and against TS28. Nothing is changed after seeing H1/H2. All candidates' holdout numbers are also
   reported, labelled descriptive (a candidate that looks better on a holdout than S is NOT adopted).
3. **PROMISING** only if, on BOTH holdouts: (a) Sharpe(S) > Sharpe(EW); (b) maxDD(S) is shallower than EW by at least 10 points;
   (c) Sharpe(S) >= Sharpe(TS28) (it adds something over the simple trend filter). **SUPPORTED** additionally needs the lower bound of the paired
   block-bootstrap (30-day blocks, 5,000 resamples, seed 0, 97.5% interval for K = 2) of Sharpe(S) - Sharpe(EW) to be above 0 on both holdouts.
   Otherwise **NOT SUPPORTED**. A holdout window that is void makes the verdict INCONCLUSIVE.

## What a pass would and would not mean
- Not a profit forecast: survivorship-biased coins, one regime (2022-2026 was a bear/range market for most coins), 4-8 years of daily data.
- Selecting among five candidates on one window and testing once is a fair single test, but five candidates were tried: a pass is weaker than a
  pre-registered single hypothesis, and a SUPPORTED verdict would still need a forward paper test before anything real.
- No execution path is added: the rule would be a pure function (`core/portfolio_rules.py`) and wiring it into the paper runner is a separate, owner-approved step.

# Results
(appended after the run)
