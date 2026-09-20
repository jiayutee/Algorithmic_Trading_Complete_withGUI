#!/bin/bash
# Git plumbing for the UNATTENDED orchestrator's code-writing runs (work-loop and task).
# Rule: the orchestrator never edits the shared project folder and never pushes to main. Each day it works in its own
# worktree on its own branch, and hands the result over as a pull request that the owner (or an interactive session) merges.
#
#   scripts/orchestrator_git.sh setup   <name>           -> prints the worktree path (creates or reuses it)
#   scripts/orchestrator_git.sh push    <name>           -> pushes orchestrator/<name> (never force, never main)
#   scripts/orchestrator_git.sh pr      <name> <title> [body]  -> opens the PR for that branch, or comments on it if one is open; prints the URL
#   scripts/orchestrator_git.sh cleanup <name>           -> removes the worktree if it is clean and fully pushed (else leaves it)
#   scripts/orchestrator_git.sh guard                    -> prints the shared folder's `git status --porcelain` (should be empty)
# <name> identifies the unit of work: a task id plus timestamp, or a day number (letters, digits, . _ - only).
# Branch: orchestrator/<name>. Worktree: <project>/.claude/worktrees/orchestrator-<name> (INSIDE the project on purpose: an
# unattended run is only allowed to edit files under the project folder, so a sibling folder outside it can fail).
# Calling setup again for the same name reuses the worktree (retry / second firing) and merges in the newest main when that is clean.
# Env overrides (used by the tests): ORCH_REPO, ORCH_REMOTE, ORCH_SLUG.
set -euo pipefail
REPO="${ORCH_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
REMOTE="${ORCH_REMOTE:-Algorithmic-Trading-Complete-with-GUI}"
SLUG="${ORCH_SLUG:-jiayutee/Algorithmic_Trading_Complete_withGUI}"
PY="${ORCH_PYTHON:-$HOME/miniconda3/bin/python3}"
cmd="${1:-}"; name="${2:-}"

die() { echo "orchestrator_git: $*" >&2; exit "${2:-1}"; }
[ "$cmd" = "guard" ] && { git -C "$REPO" status --porcelain; exit 0; }
[ -n "$cmd" ] && [ -n "$name" ] || die "usage: $0 setup|push|pr|cleanup <name> ... | guard" 2
[[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] || die "bad name '$name'" 2
BRANCH="orchestrator/$name"
WT="$REPO/.claude/worktrees/orchestrator-$name"

case "$cmd" in
  setup)
    git -C "$REPO" fetch -q "$REMOTE"
    if [ -d "$WT" ]; then
      [ "$(git -C "$WT" rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || die "$WT exists but is not on $BRANCH" 3
    elif git -C "$REPO" rev-parse --verify -q "refs/remotes/$REMOTE/$BRANCH" >/dev/null; then
      git -C "$REPO" worktree add -q -B "$BRANCH" "$WT" "$REMOTE/$BRANCH"     # the earlier firing tonight already pushed it
    else
      git -C "$REPO" worktree add -q -b "$BRANCH" "$WT" "$REMOTE/main"
    fi
    # bring the branch up to date with main when that is safe (clean tree, no conflict); otherwise leave it and say so
    if [ -z "$(git -C "$WT" status --porcelain)" ]; then
      if ! git -C "$WT" merge -q --no-edit "$REMOTE/main" >/dev/null 2>&1; then
        git -C "$WT" merge --abort >/dev/null 2>&1 || true
        echo "orchestrator_git: could not merge $REMOTE/main into $BRANCH cleanly; continuing on the older base" >&2
      fi
    fi
    if [ -f "$REPO/config/settings.py" ] && [ ! -f "$WT/config/settings.py" ]; then     # git-ignored, but some tests import it
      mkdir -p "$WT/config" && cp "$REPO/config/settings.py" "$WT/config/settings.py"
    fi
    echo "$WT"
    ;;
  push)
    [ -d "$WT" ] || die "no worktree at $WT (run setup first)" 3
    [ "$(git -C "$WT" rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || die "worktree is not on $BRANCH" 3
    case "$BRANCH" in orchestrator/*) ;; *) die "refusing to push $BRANCH" 3 ;; esac
    git -C "$WT" push -q -u "$REMOTE" "$BRANCH"
    echo "pushed $BRANCH"
    ;;
  pr)
    title="${3:-}"; body="${4:-Opened by the unattended orchestrator work-loop. Review and merge, or close.}"
    [ -n "$title" ] || die "pr needs a title" 2
    [ -f "$REPO/.env" ] || die "no .env for GITHUB_PAT" 4
    set -a; . "$REPO/.env"; set +a
    [ -n "${GITHUB_PAT:-}" ] || die "GITHUB_PAT missing in .env" 4
    ORCH_TITLE="$title" ORCH_BODY="$body" ORCH_BRANCH="$BRANCH" ORCH_SLUG="$SLUG" GITHUB_PAT="$GITHUB_PAT" "$PY" - <<'PYEOF'
import json, os, urllib.request
slug, branch, tok = os.environ["ORCH_SLUG"], os.environ["ORCH_BRANCH"], os.environ["GITHUB_PAT"]
hdr = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json", "User-Agent": "algotrader-orchestrator"}
def call(url, data=None):
    req = urllib.request.Request(url, headers=hdr, data=None if data is None else json.dumps(data).encode(),
                                 method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)
owner = slug.split("/")[0]
open_prs = call(f"https://api.github.com/repos/{slug}/pulls?head={owner}:{branch}&state=open")
footer = "\n\n(Never merged automatically: the orchestrator only opens PRs.)"
if open_prs:
    n = open_prs[0]["number"]
    call(f"https://api.github.com/repos/{slug}/issues/{n}/comments", {"body": os.environ["ORCH_BODY"]})
    print(open_prs[0]["html_url"])
else:
    pr = call(f"https://api.github.com/repos/{slug}/pulls",
              {"title": os.environ["ORCH_TITLE"], "head": branch, "base": "main", "body": os.environ["ORCH_BODY"] + footer})
    print(pr["html_url"])
PYEOF
    ;;
  cleanup)
    [ -d "$WT" ] || { echo "nothing to clean"; exit 0; }
    if [ -n "$(git -C "$WT" status --porcelain)" ]; then echo "left in place: uncommitted changes in $WT"; exit 0; fi
    if git -C "$WT" rev-parse -q --verify "refs/remotes/$REMOTE/$BRANCH" >/dev/null \
       && [ "$(git -C "$WT" rev-list --count "$REMOTE/$BRANCH..HEAD")" = "0" ]; then
      git -C "$REPO" worktree remove "$WT" && echo "removed $WT"
    else
      echo "left in place: unpushed commits on $BRANCH"
    fi
    ;;
  *) die "unknown command '$cmd'" 2 ;;
esac
