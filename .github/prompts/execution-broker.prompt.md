---
name: Broker Execution Prompt
description: Implement broker routing, execution safeguards, connector setup, or fee alignment changes.
agent: "Execution Broker Agent"
argument-hint: Describe broker path, environment assumptions, and observed execution issue.
tools: [read, search, edit, execute]
---
You are working on broker execution.

Task:
<describe the broker or order execution change>

Context:
- Broker path: <Simulator, Alpaca, Binance, IB, etc.>
- Environment: <paper, live, testnet, local>
- Issue: <routing, credentials, order type, commission, failover>

What I want:
1. Inspect initialization and routing.
2. Add safe guards for missing credentials or invalid states.
3. Keep simulator and live behavior consistent where possible.
4. Add tests or mocks for failure cases.
5. Summarize execution safety implications.

Output format:
- What changed
- Why
- Validation run
- Residual risks
- Next actions
