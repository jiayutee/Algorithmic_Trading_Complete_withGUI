---
name: Orchestrator Plan Prompt
description: Triage a repository change, split it into specialist tasks, and produce an execution plan.
agent: "Orchestrator Agent"
argument-hint: Describe the change request, constraints, and desired outcomes.
tools: [read, search, todo]
---
You are planning a change for this repository.

Task:
<describe the feature, bug, or refactor>

Context:
- Relevant modules: <list the files or folders involved>
- Constraints: <list safety, timing, or compatibility constraints>
- Risks: <list anything that could break>

What I want:
1. Break the work into specialist subtasks.
2. Assign the right agent to each subtask.
3. State merge order and dependencies.
4. Call out blockers and acceptance criteria.
5. End with a concise execution plan.

Output format:
- Plan
- Routing decisions
- Risks and mitigations
- Validation approach
- Final signoff checklist
