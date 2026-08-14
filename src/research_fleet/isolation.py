"""Provisioning the git repository that worktree isolation runs on.

Isolation gives every agent job its own git worktree, so a run's mistakes cost a branch
rather than the live tree. That needs a repository at the workspace root, which a
research directory -- data, prompts, YAML -- has no other reason to be.

So fleet creates one, and takes care that it is fleet's repository and not a claim on
the user's files:

  * the git directory lives under fleet's root, leaving a single pointer file behind;
  * nothing is tracked, staged or committed, so no prompt or config enters a history;
  * a per-worktree `core.excludesFile` keeps `git status` silent in the workspace while
    leaving agents' own worktrees fully usable -- a blanket `info/exclude` would be
    shared with the linked worktrees and would silently stop stage-to-stage chaining
    from committing anything.

A workspace that is already inside a repository is left completely alone: that is the
user's version control, and fleet only ever adds `fleet/*` branches to it.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

EXCLUDE_ALL = "# Written by research-fleet. The workspace is not fleet's to version.\n/*\n"


def _git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(cwd) if cwd else None,
        capture_output=True, text=True, timeout=30,
    )


def is_repo(workspace: Path) -> bool:
    """Whether the workspace already sits inside a git repository."""
    try:
        return _git("rev-parse", "--git-dir", cwd=workspace).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def clear_dangling_pointer(workspace: Path) -> Path | None:
    """Remove a `.git` file pointing at a git directory that no longer exists.

    Fleet keeps its git directory under its own root, so moving or clearing that root
    leaves the workspace holding a pointer to nothing. Git then refuses every command
    there, including `git init`, and isolation would silently switch itself off for the
    next run. Deleting the dangling pointer is safe precisely because it references
    nothing: a real `.git` directory, or a pointer whose target exists, is never touched.

    Returns the path removed, or None.
    """
    pointer = Path(workspace) / ".git"
    if not pointer.is_file():           # a real repository is a directory
        return None
    try:
        content = pointer.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    if not content.startswith("gitdir:"):
        return None
    target = Path(content[len("gitdir:"):].strip()).expanduser()
    if not target.is_absolute():
        target = (Path(workspace) / target).resolve()
    if target.exists():
        return None
    try:
        pointer.unlink()
    except OSError:
        return None
    return pointer


def gitdir_for(root: Path, workspace: Path) -> Path:
    """Where fleet keeps the isolation repository for one workspace."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", workspace.name).strip("-.") or "workspace"
    return root / "isolation" / f"{name}.git"


def ensure_repo(workspace: Path, root: Path) -> tuple[bool, str]:
    """Make sure `workspace` can host worktrees. Returns (usable, explanation).

    Idempotent: an already-provisioned workspace takes the same path as a user's own
    repository and is left untouched.
    """
    workspace = Path(workspace).expanduser().resolve()
    if not workspace.is_dir():
        return False, f"workspace {workspace} does not exist"
    if is_repo(workspace):
        return True, "workspace is already a git repository"

    # Left behind if fleet's root was moved or cleared. Until it goes, every git
    # command here fails, `git init` included.
    stale = clear_dangling_pointer(workspace)

    gitdir = gitdir_for(Path(root).expanduser().resolve(), workspace)
    gitdir.parent.mkdir(parents=True, exist_ok=True)
    try:
        init = _git("init", f"--separate-git-dir={gitdir}", str(workspace))
        if init.returncode != 0:
            return False, f"git init failed: {init.stderr.strip() or init.stdout.strip()}"

        # Nothing is ever committed from the workspace itself, so an identity is only
        # needed for the empty root commit and the chaining commits made in worktrees.
        _git("config", "user.name", "research-fleet", cwd=workspace)
        _git("config", "user.email", "fleet@localhost", cwd=workspace)

        # Scoped to this worktree, so the exclusion covers the user's directory and not
        # the agents' worktrees, which must stay able to stage their own work.
        exclude = gitdir / "fleet-exclude"
        exclude.write_text(EXCLUDE_ALL)
        _git("config", "extensions.worktreeConfig", "true", cwd=workspace)
        scoped = _git("config", "--worktree", "core.excludesFile", str(exclude), cwd=workspace)
        if scoped.returncode != 0:
            # Older git without worktree config: leave the files merely untracked
            # rather than risk excluding the agents' work too.
            exclude.unlink(missing_ok=True)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"could not provision isolation repository: {exc}"

    made = f"created an empty isolation repository ({gitdir})"
    return True, f"{made}; replaced a stale pointer to a removed one" if stale else made
