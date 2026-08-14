"""Provisioning the repository worktree isolation needs.

The contract is narrow and easy to break: fleet may create a repository, but the
workspace's own files must never be tracked, staged or committed by it, and agents'
worktrees must stay fully usable so stage-to-stage chaining can still commit.
"""

from __future__ import annotations

import subprocess

import pytest

from research_fleet import isolation


def _git(*args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


@pytest.fixture()
def workspace(tmp_path):
    ws = tmp_path / "project"
    (ws / "prompts").mkdir(parents=True)
    (ws / "prompts" / "judge.md").write_text("grade this")
    (ws / "myc-discovery.yaml").write_text("name: myc\n")
    return ws


def test_a_plain_directory_becomes_usable(workspace, tmp_path):
    usable, detail = isolation.ensure_repo(workspace, tmp_path / "root")
    assert usable
    assert "created" in detail
    assert isolation.is_repo(workspace)


def test_the_git_directory_lives_under_fleets_root_not_the_project(workspace, tmp_path):
    root = tmp_path / "root"
    isolation.ensure_repo(workspace, root)
    assert isolation.gitdir_for(root, workspace).is_dir()
    # Only a pointer file is left in the project.
    assert (workspace / ".git").is_file()


def test_the_users_files_are_never_tracked(workspace, tmp_path):
    isolation.ensure_repo(workspace, tmp_path / "root")
    tracked = _git("ls-files", cwd=workspace).stdout.strip()
    assert tracked == "", f"fleet tracked the user's files: {tracked}"


def test_the_workspace_stays_quiet(workspace, tmp_path):
    """`git status` in the project must not turn into a wall of untracked prompts."""
    isolation.ensure_repo(workspace, tmp_path / "root")
    assert _git("status", "--short", cwd=workspace).stdout.strip() == ""


def test_an_agents_worktree_can_still_stage_its_own_work(workspace, tmp_path):
    """The exclusion is scoped to the workspace. If it leaked into linked worktrees,
    chaining one stage onto the next would commit nothing and silently lose files."""
    isolation.ensure_repo(workspace, tmp_path / "root")
    empty = _git("hash-object", "-t", "tree", "/dev/null", cwd=workspace).stdout.strip()
    root_commit = _git("commit-tree", empty, "-m", "root", cwd=workspace).stdout.strip()
    wt = tmp_path / "wt"
    assert _git("worktree", "add", "-b", "fleet/x", str(wt), root_commit, cwd=workspace).returncode == 0

    (wt / "findings.md").write_text("F1: ...")
    assert "findings.md" in _git("status", "--short", cwd=wt).stdout
    _git("add", "-A", cwd=wt)
    assert "findings.md" in _git("diff", "--cached", "--name-only", cwd=wt).stdout


def test_an_agents_worktree_starts_empty(workspace, tmp_path):
    """Nothing is committed, so a worktree is scratch space -- the user's prompts and
    yamls are not carried into it."""
    isolation.ensure_repo(workspace, tmp_path / "root")
    empty = _git("hash-object", "-t", "tree", "/dev/null", cwd=workspace).stdout.strip()
    root_commit = _git("commit-tree", empty, "-m", "root", cwd=workspace).stdout.strip()
    wt = tmp_path / "wt"
    _git("worktree", "add", "-b", "fleet/x", str(wt), root_commit, cwd=workspace)
    assert [p.name for p in wt.iterdir()] == [".git"]


def test_an_existing_repository_is_left_alone(tmp_path):
    ws = tmp_path / "repo"
    ws.mkdir()
    _git("init", "-q", cwd=ws)
    (ws / "code.py").write_text("x = 1")
    _git("add", "-A", cwd=ws)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init", cwd=ws)

    usable, detail = isolation.ensure_repo(ws, tmp_path / "root")
    assert usable and "already" in detail
    assert (ws / ".git").is_dir(), "fleet replaced the user's own git directory"
    assert not isolation.gitdir_for(tmp_path / "root", ws).exists()
    assert _git("ls-files", cwd=ws).stdout.strip() == "code.py"


def test_provisioning_twice_changes_nothing(workspace, tmp_path):
    first = isolation.ensure_repo(workspace, tmp_path / "root")
    second = isolation.ensure_repo(workspace, tmp_path / "root")
    assert first[0] and second[0]
    assert "already" in second[1]


def test_a_missing_workspace_is_reported_not_created(tmp_path):
    usable, detail = isolation.ensure_repo(tmp_path / "nope", tmp_path / "root")
    assert not usable
    assert "does not exist" in detail
    assert not (tmp_path / "nope").exists()


def test_a_dangling_pointer_left_by_a_moved_root_is_healed(workspace, tmp_path):
    """Fleet's git directory lives under its root. Move or clear that root and the
    workspace holds a pointer to nothing -- git then refuses everything, `git init`
    included, and isolation would quietly switch off on the next run."""
    root = tmp_path / "root"
    assert isolation.ensure_repo(workspace, root)[0]
    import shutil
    shutil.rmtree(root)                       # as if fleet-logs/ were moved away

    assert not isolation.is_repo(workspace)
    usable, detail = isolation.ensure_repo(workspace, root)
    assert usable, detail
    assert "stale pointer" in detail
    assert isolation.is_repo(workspace)
    assert _git("ls-files", cwd=workspace).stdout.strip() == ""


def test_a_real_git_directory_is_never_removed(tmp_path):
    ws = tmp_path / "repo"
    ws.mkdir()
    _git("init", "-q", cwd=ws)
    assert isolation.clear_dangling_pointer(ws) is None
    assert (ws / ".git").is_dir()


def test_a_pointer_whose_target_exists_is_never_removed(workspace, tmp_path):
    isolation.ensure_repo(workspace, tmp_path / "root")
    assert isolation.clear_dangling_pointer(workspace) is None
    assert (workspace / ".git").is_file()


def test_an_unrelated_dot_git_file_is_left_alone(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / ".git").write_text("not a gitdir pointer at all")
    assert isolation.clear_dangling_pointer(ws) is None
    assert (ws / ".git").exists()
