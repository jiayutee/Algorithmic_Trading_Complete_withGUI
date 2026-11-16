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

# Results (run 2026-11-16; protocol exactly as above, nothing changed after seeing numbers; `training_ground/results/phase_14_0.json`)
**Selected on window A: C2 (inverse-volatility trend).** Verdict: **PROMISING (not statistically supported)**.

| Sharpe / max drawdown | EW | TS28 | **C2** | C1 | C3 | C4 | C5 |
|---|---|---|---|---|---|---|---|
| A, 2019-05..2022-09 (selection) | 1.16 / -74% | 1.61 / -48% | **1.67 / -45%** | 1.53 / -47% | 1.60 / -44% | 1.45 / -72% | 1.56 / -35% |
| H1, SET1, 2022-09..2026-11 | 0.59 / -61% | 0.67 / -39% | **0.80 / -35%** | 0.69 / -31% | 0.83 / -29% | 0.45 / -55% | 0.87 / -24% |
| H2, SET2, 2022-09..2026-11 | 0.21 / -84% | 0.50 / -57% | **0.51 / -55%** | 0.47 / -57% | 0.46 / -56% | 0.29 / -72% | 0.50 / -41% |

Criteria on BOTH holdouts: (a) Sharpe above EW: yes (0.80 vs 0.59; 0.51 vs 0.21). (b) max drawdown at least 10 points shallower than EW: yes (26 and 29
points). (c) Sharpe at least TS28's: yes, but on H2 only by 0.01 (0.51 vs 0.50). (d) the 97.5% interval of Sharpe(C2) - Sharpe(EW) above zero: **no**
(H1 [-0.48, +0.69], H2 [-0.43, +0.83]). Against TS28 the Sharpe difference is [+0.01, +0.28] on H1 and [-0.09, +0.10] on H2.

## Reading it
- **What is solid:** the drawdown reduction again (26-29 points shallower than equal weight on coins and a period it was not selected on), and
  the Sharpe point estimates are higher in every comparison that counts. This is the fourth look at the same drawdown effect.
- **What is not:** the Sharpe advantage over equal weight is not distinguishable from zero in either holdout, and over the plain 28-day trend
  filter it is zero on H2 and at most small on H1. The incremental value of inverse-volatility weighting over TS28 is therefore **not demonstrated**.
- **Not adopted, but worth knowing (descriptive, selected after the fact would be cherry-picking):** C5 (vote + inverse-vol + 40% volatility cap) and
  C3 had the best holdout risk numbers (H1 max drawdown -24% / -29%, Sharpe 0.87 / 0.83) but gave up the most return in window A. They were not the
  pre-registered pick, so they carry no claim; a fresh pre-registration on new data would be the way to test them.
- **Momentum tilt (C4) is the worst everywhere** (concentration into recent winners hurt in both bear-ish holdouts), consistent with 6.7's XSMOM.
- **Limits:** survivorship (all coins survive to today), 2022-2026 was a hard period for most coins so absolute levels are low and
  everything beats equal weight on drawdown there, five candidates were tried on one selection window, costs fixed at 0.1% (no sensitivity run).

## Verdict for the app
The best-supported rule I can offer is "**hold only the coins in a 28-day uptrend, weighted by inverse volatility, rest in cash**": same family as
the running Trend Filter, with a smoother ride than equal weight, and no demonstrated return edge. It is **not** wired into the paper runner (the runner
trades one single-asset rule per symbol; a portfolio rule needs a runner change that needs owner approval). A forward, order-free shadow ledger is the
low-risk next step: log the rule's weights each day and mark them to prices, so there is genuinely unseen evidence.
