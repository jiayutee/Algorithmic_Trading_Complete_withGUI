---
name: Reliability Release Prompt
description: Run a release-readiness pass, dependency check, or startup health check.
agent: "Reliability Release Agent"
argument-hint: Describe release scope, environment target, and required verification depth.
tools: [read, search, execute, todo]
---
You are working on release readiness.

Task:
<describe the release or health-check focus>

Context:
- Environment: <local, CI, dev container, production-like>
- Scope: <dependencies, startup, smoke checks, docs, setup>
- Known concerns: <version mismatch, missing config, manual steps>

What I want:
1. Check dependency and environment assumptions.
2. Run the relevant verification steps.
3. Summarize operational risks and mitigations.
4. Call out any manual setup needed for API keys or brokers.
5. End with a release checklist.

Output format:
- What changed
- Why
- Validation run
- Residual risks
- Next actions
