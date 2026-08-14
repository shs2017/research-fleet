"""Where a run's artifacts land on disk.

Run and job ids are uuids, which is right for a ledger key and wrong for a directory
you have to find six weeks later. This module maps a run onto a readable path instead:

    <results_dir>/<workflow>/<NNN>/<stage>/

`<NNN>` counts attempts of that workflow, so running the same pipeline again nests a
new numbered directory beside the last rather than scattering siblings. A continuation
(`--resume`) writes into the directory of the run it continues, because it is the same
attempt carrying on; `--from-run` re-executes everything and so opens a new one.

Nothing here reaches across attempts. A fresh run allocates an empty directory and is
told nothing about its predecessors -- inheritance only happens where a caller asks for
it by run id.

Directories written before this layout existed are flat (`<results_dir>/<run_id>/`), so
`run_dir_for` falls back to that shape and old runs stay readable.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

ATTEMPT = re.compile(r"^\d+$")
MANIFEST = "run.json"


def slug(name: str) -> str:
    """A name safe to use as one path segment, never empty."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", (name or "").strip()).strip("-.")
    return cleaned[:64] or "run"


def allocate_run_dir(results_dir: Path, name: str, *, run_id: str, based_on: str | None = None) -> Path:
    """Claim the next free attempt directory for `name` and describe it in `run.json`.

    Claiming is `mkdir` without `exist_ok`, retried on collision, so two fleets racing
    for the same workflow get different attempts rather than one silently adopting the
    other's directory.
    """
    base = results_dir / slug(name)
    base.mkdir(parents=True, exist_ok=True)
    used = [int(p.name) for p in base.iterdir() if p.is_dir() and ATTEMPT.match(p.name)]
    attempt = max(used, default=0) + 1
    while True:
        candidate = base / f"{attempt:03d}"
        try:
            candidate.mkdir()
        except FileExistsError:
            attempt += 1
            continue
        write_manifest(candidate, run_id=run_id, workflow=name, attempt=attempt, based_on=based_on)
        return candidate


def write_manifest(
    run_dir: Path,
    *,
    run_id: str,
    workflow: str,
    attempt: int,
    based_on: str | None = None,
) -> None:
    (run_dir / MANIFEST).write_text(json.dumps({
        "run_id": run_id,
        "workflow": workflow,
        "attempt": attempt,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "based_on": based_on,
        "continued_by": [],
    }, indent=2) + "\n")


def note_continuation(run_dir: Path, run_id: str) -> None:
    """Record that `run_id` resumed into an existing attempt.

    The attempt keeps the run id that opened it, so without this the ledger and the
    directory disagree about who wrote the later half of it.
    """
    path = run_dir / MANIFEST
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    continued = manifest.setdefault("continued_by", [])
    if run_id not in continued:
        continued.append(run_id)
        try:
            path.write_text(json.dumps(manifest, indent=2) + "\n")
        except OSError:
            pass


def run_dir_for(results_dir: Path, run_id: str) -> Path | None:
    """Find a finished run's directory by its run id, or None if it left none.

    Checks the flat legacy path first because that is one `is_dir()`, then scans the
    manifests. A run that both opened an attempt and later resumed another is found
    under either id.
    """
    legacy = results_dir / run_id
    if legacy.is_dir():
        return legacy
    if not results_dir.is_dir():
        return None
    for workflow_dir in sorted(results_dir.iterdir()):
        if not workflow_dir.is_dir():
            continue
        for attempt_dir in sorted(workflow_dir.iterdir(), reverse=True):
            if not (ATTEMPT.match(attempt_dir.name) and attempt_dir.is_dir()):
                continue
            try:
                manifest = json.loads((attempt_dir / MANIFEST).read_text())
            except (OSError, ValueError):
                continue
            if manifest.get("run_id") == run_id or run_id in (manifest.get("continued_by") or []):
                return attempt_dir
    return None


def stage_dirs(run_dir: Path, names: dict[str, str] | None = None) -> dict[str, Path]:
    """A finished run's stage directories, keyed by stage name.

    New-layout runs are already named for their stages. Runs written under the old flat
    layout have `job_<id>` directories, which tell a later agent nothing about which one
    holds the findings and which the review; pass `names` (job id -> stage name, from
    the ledger) to recover that. Unmapped directories keep their id rather than being
    dropped -- an oddly named input beats a missing one.
    """
    stages: dict[str, Path] = {}
    if not run_dir.is_dir():
        return stages
    for path in sorted(run_dir.iterdir()):
        # Dot-directories are fleet's own bookkeeping (see SUPERSEDED), never stages.
        if not path.is_dir() or path.name.startswith("."):
            continue
        stages[slug((names or {}).get(path.name, path.name))] = path
    return stages


def input_stages(mounts) -> list[str]:
    """Stage names under `/inputs`, one per distinct directory.

    A single-copy stage is mounted under both `research` and its `research-0` alias.
    Listing both in a brief only invites the agent to wonder what the difference is,
    so the shortest name for each directory wins.
    """
    by_source: dict[str, str] = {}
    for mount in mounts:
        if not mount.target.startswith("/inputs/"):
            continue
        name = mount.target[len("/inputs/"):]
        current = by_source.get(mount.source)
        if current is None or (len(name), name) < (len(current), current):
            by_source[mount.source] = name
    return sorted(by_source.values())


def render_environment_brief(mounts, *, isolated: bool = False, chained_from: str | None = None) -> str:
    """Tell the agent which directories it has and what each one means.

    Without this an agent has to infer the filesystem from its task prompt. That fails
    in a specific, silent way: a stage told to "write the write-up to /results" reports
    that path in its final message, the next stage reads the same string, and finds its
    own empty `/results` -- the two are different directories. Naming the stages that
    are actually mounted, and saying outright that a quoted `/results` path belongs to
    the job that wrote it, is what stops the next stage from guessing.
    """
    stages = input_stages(mounts)
    targets = {m.target for m in mounts}
    lines = ["", "## Files and directories", ""]

    if isolated:
        # Sequential isolated stages are chained: each branches from the previous one's
        # committed worktree, so file edits do carry forward. An agent told only "you
        # are isolated" would wrongly assume the tree is pristine.
        inherited = (
            f" It was branched from the previous stage's tree, so files an earlier "
            f"stage wrote are already here."
            if chained_from else
            " It starts from the project's committed state."
        )
        lines.append(
            "- `/workspace` -- the project, and your working directory. It is your own "
            "git worktree on its own branch, so nothing you write reaches the live "
            f"checkout or a parallel job.{inherited} Only tracked files are present; "
            "large inputs may be bind-mounted in separately."
        )
    else:
        lines.append(
            "- `/workspace` -- the project, and your working directory. It is shared "
            "with every other job, so keep scratch files out of the way and expect "
            "files you did not create."
        )
    lines.append(
        "- `/results` -- write your deliverables here. It is private to this job, and "
        "anything outside `/workspace` and `/results` is discarded when the container "
        "exits."
    )

    if stages:
        lines.append(
            "- `/inputs/` -- read-only results of earlier stages of this workflow, one "
            f"directory each: {', '.join('`/inputs/' + s + '`' for s in stages)}."
        )
    previous = sorted(
        t[len("/previous/"):] for t in targets if t.startswith("/previous/")
    )
    if previous:
        lines.append(
            "- `/previous/` -- read-only results of the *earlier run* this one builds "
            f"on, one directory per stage: {', '.join('`/previous/' + s + '`' for s in previous)}. "
            "That run answered the same question before you; read it and go further "
            "rather than repeating it. `/previous-results` holds the same run whole."
        )
    elif "/previous-results" in targets:
        lines.append(
            "- `/previous-results` -- read-only artifacts of the earlier run this one "
            "builds on, laid out the same way (one directory per stage)."
        )

    if stages:
        lines += [
            "",
            "Because `/results` is per-job, a path an earlier stage quotes in its message "
            "is *its* `/results`, not yours: the same files are under "
            "`/inputs/<that stage>/`. Check there before reporting a promised file "
            "missing.",
        ]
    return "\n".join(lines)


SUPERSEDED = ".superseded"
"""Where a re-run stage's previous attempt is moved. Dotted so it is not a stage."""


def reclaim(results_dir: Path) -> Path | None:
    """Free `results_dir` for a stage that is running again, keeping what was there.

    A resume re-runs the stage that failed, and that stage must get its own name back:
    `/inputs/<stage>` and `/previous/<stage>` are resolved by directory name, so leaving
    the failed attempt in place would hand it to every later reader as if it were the
    result. The old contents move to `.superseded/` rather than being deleted, since the
    trace of the failure is usually the reason you resumed.

    Returns where the previous attempt went, or None if there was nothing to move.
    """
    if not results_dir.is_dir() or not any(results_dir.iterdir()):
        return None
    archive = results_dir.parent / SUPERSEDED
    archive.mkdir(parents=True, exist_ok=True)
    for n in range(1, 1000):
        target = archive / f"{results_dir.name}-{n}"
        if not target.exists():
            results_dir.rename(target)
            return target
    return None


def job_dir_name(name: str, taken: set[str]) -> str:
    """A directory name for one job: its stage name, kept unique within the run.

    Workflow job names already carry the distinguishing suffix -- `review-2` for a
    cycle's second pass, `probe-0`/`probe-1` for a fan-out -- so the common single-pass
    stage stays plain `research`. The counter is only reached by a caller that submits
    two jobs under one name.
    """
    base = slug(name)
    if base not in taken:
        return base
    for n in range(2, 1000):
        candidate = f"{base}~{n}"
        if candidate not in taken:
            return candidate
    return f"{base}~{len(taken)}"
