---
name: Strategy Change Prompt
description: Implement or adjust strategy logic, signal generation, or strategy registration.
agent: "Strategy Agent"
argument-hint: Describe target strategy, expected signals, and contract constraints.
tools: [read, search, edit, execute]
---
You are working on a trading strategy change.

Task:
<describe the strategy behavior or registration change>

Context:
- Strategy name: <name>
- Expected signals: <buy, sell, hold, thresholds, etc.>
- Interface constraints: <backtrader, custom instance, wrapper behavior>

What I want:
1. Verify the current contract and call sites.
2. Implement the smallest correct strategy change.
3. Preserve compatibility with manager and backtester paths.
4. Add tests for buy/sell/hold edge cases.
5. Report any assumptions you had to make.

Output format:
- What changed
- Why
- Validation run
- Residual risks
- Next actions
