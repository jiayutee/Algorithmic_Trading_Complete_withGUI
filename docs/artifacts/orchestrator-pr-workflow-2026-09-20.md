# Orchestrator workflow evidence — 2026-09-20

Base: `0fb5fba`. Branch: `codex/orchestrator-pr-workflow`.
Notion: https://app.notion.com/p/3e0d2ab050d981218af5e78bd6a3fb90

## Change
Removed the scheduled direct-main exception and the release specialist's main-push command. The runbook requires owned task worktrees, explicit task-branch pushes, one PR per task, full local tests and latest-head CI status. It prohibits automatic merging/deployment, preserves incomplete work, distinguishes pending review from Done, and includes branch PRs in evening reporting. The Telegram wrapper follows the same workflow.

## Validation
- Reviewed scheduled work-loop, ad-hoc task, retry, failed CI, pending review and evening reporting paths.
- Checked launcher: scripts/orchestrator-local.sh loads the canonical runbook with --append-system-prompt-file on each invocation. Launcher work-loop prompt now explicitly requests the helper/PR workflow; scheduling is unchanged.
- Checked canonical/mirrored agent definitions for remaining direct-main push instructions.
- Whitespace and notebook structure checks passed.
- Focused helper suite: 15 passed, including preservation of unpushed commits when recreating a removed worktree.
- Shell syntax: bash -n passed for helper and launcher.
- Full local suite: 1,284 passed, 1 skipped, 14 warnings in 236.41 seconds (Miniconda Python 3.9). Latest-head GitHub CI is recorded on the PR and Notion before merge.

## Activation and limits
The new instructions apply to future sessions once this PR is merged and the main runtime checkout is fast-forwarded. An already-running agent keeps its previously loaded instructions. No scheduled run is triggered as a test; no services are restarted. Claude PR #13 supplies the tested helper; this change wires it into the instructions and preserves unpushed local commits when a removed worktree is recreated. The helper constrains its own pushes; GitHub branch protection is not configured by this change. The published claude.ai notebook artifact requires separate republishing; its repository HTML source is updated here.
