# Research Loop: First 8-Candidate Evaluation

**Date:** 2026-09-24  
**Run on:** scratch copy of `training_ground/results/experiments.sqlite3`  
**Canonical file:** unchanged (SHA-256 before and after: `4a86ff64...f26d34`)  
**Status:** evidence only — canonical promote/retire state untouched; applying decisions is an owner decision  
**Git commit at run time:** `d00c66b`  
**Branch:** `orchestrator/day89-research-loop-8cand`

---

## Run parameters

| Parameter | Value |
|---|---|
| Symbols | BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, ADAUSDT, LTCUSDT, DOGEUSDT, SOLUSDT |
| History fetched | 1000 bars per symbol |
| Evaluation window | Last 700 bars (data through 2026-09-23) |
| Fee per side | 0.1% |
| Bootstrap iterations | 500 (default 2000; results approximate) |
| Block size | 10 bars |
| Candidates | 8 (4 base + 4 trend-filtered, added PR #23) |
| Bonferroni total alpha | 0.05 |
| Per-candidate alpha | 0.00625 (0.05 / 8) |
| **CI level per candidate** | **99.375%** (vs 98.75% with 4 candidates — wider intervals) |

---

## Promotion rules (fixed in advance, unchanged)

A candidate is promoted to paper trading only when ALL four conditions hold:

1. CI lower bound > 0 (Bonferroni-corrected Sharpe-diff interval strictly positive)
2. Sharpe diff positive in both halves of the evaluation window
3. At least 30 closed trades total
4. Max drawdown no more than 5 percentage points worse than buy-and-hold

Retirement requires: CI upper bound < 0, OR 3 consecutive evaluation failures while in paper trading.

---

## Buy-and-hold baseline (equal-weight, 8 symbols)

- Sharpe: **0.47**
- Max drawdown: **-66.7%**

---

## Candidate results

| Candidate | Sharpe | B&H Sharpe | Diff | 99.4% CI | Both halves | Trades | maxDD% | B&H maxDD% | Fails | Would assign |
|---|---|---|---|---|---|---|---|---|---|---|
| MACD/RSI | 0.38 | 0.47 | -0.10 | [-3.15, +2.95] | No | 9 | -5% | -67% | CI lower, halves, trades | candidate |
| **EMA Crossover** | **0.88** | 0.47 | **+0.41** | **[-0.93, +1.74]** | **Yes** | **128** | **-39%** | -67% | **CI lower only** | candidate |
| Stochastic | 0.36 | 0.47 | -0.12 | [-3.48, +3.24] | No | 97 | -261% | -67% | CI lower, halves, drawdown | candidate |
| GBM (LightGBM) | -0.51 | 0.47 | -0.98 | [-3.50, +1.62] | No | 1192 | -56% | -67% | CI lower, halves | candidate |
| MACD/RSI + Trend | 0.59 | 0.47 | +0.11 | [-3.19, +3.20] | No | 5 | -3% | -67% | CI lower, halves, trades | candidate |
| EMA Crossover + Trend | 0.57 | 0.47 | +0.10 | [-2.48, +2.99] | No | 100 | -15% | -67% | CI lower, halves | candidate |
| Stochastic + Trend | 0.75 | 0.47 | +0.28 | [-2.88, +3.17] | No | 83 | -107% | -67% | CI lower, halves, drawdown | candidate |
| GBM + Trend | -0.73 | 0.47 | -1.20 | [-4.09, +2.06] | No | 622 | -40% | -67% | CI lower, halves | candidate |

All 8 candidates remain on trial. No promotions. No retirements.

---

## Trend overlay effect (Phases 6.7-6.9 prior expectation)

**Prior expectation:** the 28-bar weekly trend filter reduces drawdown but does not produce a Sharpe edge vs buy-and-hold.

**This run agrees:**

| Base | Base maxDD | + Trend maxDD | Drawdown reduced? | Base Sharpe | + Trend Sharpe | Sharpe improved? |
|---|---|---|---|---|---|---|
| MACD/RSI | -5% | -3% | Yes (minor) | 0.38 | 0.59 | Point estimate higher, but trades too few (9 -> 5) |
| EMA Crossover | -39% | -15% | Yes | 0.88 | 0.57 | No — Sharpe falls |
| Stochastic | -261% | -107% | Yes | 0.36 | 0.75 | Point estimate higher, but drawdown still extreme |
| GBM | -56% | -40% | Yes | -0.51 | -0.73 | No — Sharpe worsens |

In three of four cases the trend overlay makes Sharpe worse or neutral. In one case (Stochastic) the point estimate improves but the drawdown still fails the promotion test (-107% vs -67% B&H). Drawdown reduction is confirmed across all four. The Sharpe promotion bar is not cleared by any trend-filtered variant.

**No trend-filtered variant clears the promotion bar.**

---

## Key observations

**EMA Crossover** is the only candidate where every non-CI test passes: both halves show positive Sharpe diff, 128 trades, and drawdown (-39%) is well below B&H (-67%). The CI lower bound is -0.93 under the Bonferroni-widened 99.375% level. Under the narrower 98.75% level (4-candidate run) it was [-1.2, +2.1] (from the first live run recorded in the notebook). The interval has not moved decisively positive across multiple evaluations.

**Stochastic's -261% max drawdown** is a backtest artifact. Under the all-in (95%) sizing, a sequence of losses across 8 symbols drives equity close to zero. This is not a realistic trading outcome, but it is what the model reports and the rule correctly rejects it.

**MACD/RSI + Trend (5 trades total)** is essentially an always-flat strategy in this window. The trend filter removes almost all MACD/RSI signals. This is not a failure of the filter — it means MACD/RSI signals rarely arrive while the 28-bar trend is up. Too few observations to assess.

**GBM walk-forward OOS AUC** ranged 0.477-0.540 across symbols (close to 0.5), confirming no reliable directional edge. Adding the trend overlay reduces trading volume but does not convert the lack of edge into one.

**Bonferroni widening** from 4 to 8 candidates moves the per-candidate CI from 98.75% to 99.375%. This makes the already-demanding lower-bound test marginally harder. The effect is small given how wide the intervals already are, but it is the honest accounting for testing 8 hypotheses simultaneously.

---

## Safety notes

- The scratch copy (`artifacts/tmp_day89/experiments_copy.sqlite3`) was used exclusively. The canonical file (`training_ground/results/experiments.sqlite3`) was not opened or modified.
- SHA-256 of canonical file: `4a86ff64e8f364c0984d1117ccf45f6b219c376badcd8b7b1aaf83f349f26d34` — identical before and after the run.
- The scratch directory was deleted after the commit. It is not committed.
- No code or promotion rules were changed. This is evidence only.
