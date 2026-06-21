---
name: Data Pipeline Fix Prompt
description: Diagnose or implement market data, news data, streaming, or normalization changes.
agent: "Data Pipeline Agent"
argument-hint: Describe the data source, symbol types, interval, and reliability issue.
tools: [read, search, edit, execute]
---
You are working on data ingestion or normalization.

Task:
<describe the data problem or feature>

Context:
- Source: <Yahoo, CCXT, WebSocket, news, etc.>
- Symbols: <stocks, crypto, indexes, etc.>
- Interval: <1m, 5m, 1h, 1d, etc.>
- Failure mode: <empty data, timezone mismatch, bad schema, etc.>

What I want:
1. Trace the affected data path.
2. Fix fallback or normalization behavior.
3. Keep downstream dataframe shape stable.
4. Add or update tests for edge cases.
5. Show the verification you ran.

Output format:
- What changed
- Why
- Validation run
- Residual risks
- Next actions
