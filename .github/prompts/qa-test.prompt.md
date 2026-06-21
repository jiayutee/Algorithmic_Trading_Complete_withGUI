---
name: QA Test Prompt
description: Create or update deterministic tests for a changed behavior.
agent: "QA Test Agent"
argument-hint: Describe changed behavior and required regression coverage.
tools: [read, search, edit, execute]
---
You are working on tests for a repository change.

Task:
<describe the behavior that needs coverage>

Context:
- Target files: <paths>
- Risk areas: <flaky network, edge case, regression, integration>
- Expected behavior: <what should pass or fail>

What I want:
1. Build a focused test matrix.
2. Add deterministic happy-path and failure-path tests.
3. Avoid live external calls unless explicitly required.
4. Mark network-dependent tests clearly.
5. Report what test command you ran and what failed or passed.

Output format:
- What changed
- Why
- Validation run
- Residual risks
- Next actions
