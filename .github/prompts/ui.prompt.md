---
name: UI Change Prompt
description: Implement or fix UI behavior, state transitions, charts, timers, or user-facing messaging.
agent: "UI Agent"
argument-hint: Describe the user flow, expected UI state transitions, and backend linkage.
tools: [read, search, edit, execute]
---
You are working on the desktop UI.

Task:
<describe the UI behavior or bug>

Context:
- Screen or control: <chart, buttons, status bar, simulation, trading panel>
- State flow: <load, play, pause, backtest, trade>
- Problem: <wrong state, freeze, chart issue, bad messaging>

What I want:
1. Trace the event and signal flow.
2. Make the smallest UI-safe change.
3. Avoid blocking the UI thread.
4. Preserve existing semantics unless a change is requested.
5. Add smoke coverage if needed.

Output format:
- What changed
- Why
- Validation run
- Residual risks
- Next actions
