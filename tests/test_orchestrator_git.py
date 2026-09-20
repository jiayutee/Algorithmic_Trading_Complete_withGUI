"""scripts/orchestrator_git.sh: the unattended orchestrator works in its own worktree on orchestrator/day-N and never touches main."""
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "orchestrator_git.sh"


def git(cwd, *args, check=True):
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=check, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})


def run(repo, *args, env_extra=None):
    env = {**os.environ, "ORCH_REPO": str(repo), "ORCH_REMOTE": "origin", "ORCH_SLUG": "owner/repo", **(env_extra or {})}
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=env)


@pytest.fixture
def world(tmp_path):
    """A bare 'GitHub' remote plus the shared project folder, with one commit on main."""
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
    repo = tmp_path / "project"
    subprocess.run(["git", "clone", "-q", str(remote), str(repo)], check=True, capture_output=True)
    git(repo, "config", "user.email", "t@t"); git(repo, "config", "user.name", "t")
    git(repo, "checkout", "-q", "-b", "main")
    (repo / "README.md").write_text("hello\n")
    (repo / ".gitignore").write_text(".claude/\nconfig/settings.py\n.env\n")
    git(repo, "add", "-A"); git(repo, "commit", "-q", "-m", "init"); git(repo, "push", "-q", "-u", "origin", "main")
    (repo / "config").mkdir(); (repo / "config" / "settings.py").write_text("SECRET = 1\n")     # git-ignored, like the real one
    return repo, remote


def shared_state(repo):
    return git(repo, "rev-parse", "HEAD").stdout, git(repo, "branch", "--show-current").stdout, git(repo, "status", "--porcelain").stdout


def test_setup_makes_a_worktree_on_its_own_branch_and_leaves_the_shared_folder_alone(world):
    repo, _ = world
    before = shared_state(repo)
    r = run(repo, "setup", "day-7")
    assert r.returncode == 0, r.stderr
    wt = Path(r.stdout.strip().splitlines()[-1])
    assert wt == repo / ".claude" / "worktrees" / "orchestrator-day-7" and wt.is_dir()
    assert git(wt, "branch", "--show-current").stdout.strip() == "orchestrator/day-7"
    assert (wt / "config" / "settings.py").read_text() == "SECRET = 1\n"          # ignored config copied so tests can import app
    assert not (wt / ".env").exists()
    assert shared_state(repo) == before                                            # branch, HEAD and status of the shared folder unchanged
    assert git(repo, "branch", "--show-current").stdout.strip() == "main"


def test_setup_is_idempotent(world):
    repo, _ = world
    a = run(repo, "setup", "day-7").stdout.strip().splitlines()[-1]
    b = run(repo, "setup", "day-7")
    assert b.returncode == 0 and b.stdout.strip().splitlines()[-1] == a


def test_push_publishes_the_branch_and_never_main(world):
    repo, remote = world
    wt = Path(run(repo, "setup", "day-7").stdout.strip().splitlines()[-1])
    (wt / "new.py").write_text("x = 1\n")
    git(wt, "add", "new.py"); git(wt, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "work")
    main_before = git(remote, "rev-parse", "main").stdout
    r = run(repo, "push", "day-7")
    assert r.returncode == 0 and "pushed orchestrator/day-7" in r.stdout
    assert git(remote, "rev-parse", "--verify", "orchestrator/day-7").returncode == 0
    assert git(remote, "rev-parse", "main").stdout == main_before                  # main on the remote did not move


def test_push_refuses_a_worktree_that_is_not_on_the_expected_branch(world):
    repo, remote = world
    wt = Path(run(repo, "setup", "day-7").stdout.strip().splitlines()[-1])
    git(wt, "checkout", "-q", "-b", "something-else")
    r = run(repo, "push", "day-7")
    assert r.returncode == 3 and "not on orchestrator/day-7" in r.stderr
    assert git(remote, "rev-parse", "--verify", "something-else", check=False).returncode != 0


def test_second_firing_continues_the_same_branch_and_picks_up_new_main_commits(world, tmp_path):
    repo, remote = world
    wt = Path(run(repo, "setup", "day-7").stdout.strip().splitlines()[-1])
    (wt / "first.py").write_text("1\n"); git(wt, "add", "-A")
    git(wt, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "first firing"); run(repo, "push", "day-7")
    assert "removed" in run(repo, "cleanup", "day-7").stdout                            # first firing tidies up after itself
    other = tmp_path / "other"                                                     # meanwhile the owner merges something into main
    subprocess.run(["git", "clone", "-q", str(remote), str(other)], check=True, capture_output=True)
    (other / "owner.txt").write_text("merged\n"); git(other, "add", "-A")
    git(other, "-c", "user.email=o@o", "-c", "user.name=o", "commit", "-q", "-m", "owner change"); git(other, "push", "-q", "origin", "main")
    wt2 = Path(run(repo, "setup", "day-7").stdout.strip().splitlines()[-1])            # the 00:20 firing
    assert (wt2 / "first.py").exists() and (wt2 / "owner.txt").exists()            # its own earlier work AND the new main


def test_cleanup_removes_only_a_clean_fully_pushed_worktree(world):
    repo, _ = world
    wt = Path(run(repo, "setup", "day-7").stdout.strip().splitlines()[-1])
    (wt / "wip.py").write_text("wip\n")
    assert "left in place: uncommitted" in run(repo, "cleanup", "day-7").stdout and wt.exists()
    git(wt, "add", "-A"); git(wt, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "wip")
    assert "left in place: unpushed" in run(repo, "cleanup", "day-7").stdout and wt.exists()
    run(repo, "push", "day-7")
    assert "removed" in run(repo, "cleanup", "day-7").stdout and not wt.exists()
    assert run(repo, "cleanup", "day-7").stdout.strip() == "nothing to clean"


@pytest.mark.parametrize("bad", ["../x", "a b", "1;rm", "", "x/y"])
def test_bad_day_values_are_rejected(world, bad):
    repo, _ = world
    r = run(repo, "setup", bad)
    assert r.returncode == 2


def test_pr_needs_credentials_and_a_title_and_does_nothing_without_them(world):
    repo, _ = world
    assert run(repo, "pr", "day-7", "title").returncode == 4                            # no .env at all
    (repo / ".env").write_text("SOMETHING=1\n")
    r = run(repo, "pr", "day-7", "title")
    assert r.returncode == 4 and "GITHUB_PAT" in r.stderr
    assert run(repo, "pr", "day-7").returncode == 2                                    # no title


def test_guard_reports_a_dirty_shared_folder(world):
    repo, _ = world
    assert run(repo, "guard", "x").stdout.strip() == ""
    (repo / "stray.txt").write_text("someone edited the shared folder\n")
    assert "stray.txt" in run(repo, "guard", "x").stdout


def test_the_script_contains_no_way_to_push_main_or_force():
    src = SCRIPT.read_text()
    assert "--force" not in src and "push -f" not in src and " main\"" not in src.split("push)")[1].split(";;")[0]
