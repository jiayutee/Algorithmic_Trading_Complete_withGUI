---
name: Backtest Metrics Prompt
description: Improve backtest correctness, analyzers, commission models, alpha or beta calculations, or report stability.
agent: "Backtest and Metrics Agent"
argument-hint: Describe the metric issue, benchmark context, and expected reporting behavior.
tools: [read, search, edit, execute]
---
You are working on backtest or performance reporting logic.

Task:
<describe the metric or backtest issue>

Context:
- Benchmark: <SPY, BTC-USD, etc.>
- Dataset: <historical range, symbol, frequency>
- Problem: <alpha, beta, Sharpe, commission, empty output, etc.>

What I want:
1. Validate analyzer assumptions and return handling.
2. Fix the smallest metric or report bug.
3. Preserve comparability across runs.
4. Add tests for summary keys and edge conditions.
5. Confirm reproducibility with a verification step.

Output format:
- What changed
- Why
- Validation run
- Residual risks
- Next actions
