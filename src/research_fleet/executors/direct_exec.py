"""Run Fleet jobs directly on the host, with the same paths and worktree semantics."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

from ..policy import Policy
from ..spec import JobResult, JobSpec, JobState
from .base import LineHandler, Placement, run_process


class DirectExecutor:
    kind = "direct"

    def __init__(self, project_dir: str):
        self.project_dir = str(Path(project_dir).expanduser().resolve())
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def available_gpus(self) -> list[str]:
        try:
            output = subprocess.run(
                ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=15, check=True,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return []
        return [line.strip() for line in output.splitlines() if line.strip()]

    @staticmethod
    def _git(path: str, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", path, *args], capture_output=True, text=True, check=False,
        )

    def _worktree(self, spec: JobSpec) -> tuple[str, str, str] | None:
        if not spec.isolate:
            return None
        branch = f"fleet/{spec.run_id}-{spec.name}"[:80]
        root = Path(self.project_dir).parent / ".fleet-worktrees" / Path(self.project_dir).name
        path = root / branch.replace("/", "-")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            if spec.worktree_base:
                base_run = spec.worktree_base_run_id or spec.run_id
                base = f"fleet/{base_run}-{spec.worktree_base}"[:80]
                start = base
            else:
                head = self._git(self.project_dir, "rev-parse", "--verify", "HEAD")
                if head.returncode:
                    tree = self._git(self.project_dir, "hash-object", "-t", "tree", "/dev/null").stdout.strip()
                    start = self._git(
                        self.project_dir, "-c", "user.name=fleet", "-c",
                        "user.email=fleet@localhost", "commit-tree", tree, "-m", "fleet: empty root",
                    ).stdout.strip()
                else:
                    start = "HEAD"
            made = self._git(self.project_dir, "worktree", "add", "-b", branch, str(path), start)
            if made.returncode:
                raise RuntimeError(made.stderr.strip() or f"could not create worktree {branch}")
        base_commit = self._git(str(path), "rev-parse", "HEAD").stdout.strip()
        return str(path), branch, base_commit

    @staticmethod
    def _mapping(spec: JobSpec, workspace: str) -> list[tuple[str, str]]:
        mounts = sorted(
            ((mount.target, str(Path(mount.source).resolve())) for mount in spec.mounts),
            key=lambda pair: len(pair[0]), reverse=True,
        )
        return mounts + [("/workspace", workspace)]

    @staticmethod
    def _translate(value: str, mapping: list[tuple[str, str]]) -> str:
        if not mapping:
            return value
        sources = dict(mapping)
        pattern = re.compile("|".join(re.escape(target) for target in sources))
        return pattern.sub(lambda match: sources[match.group(0)], value)

    @staticmethod
    def _sandbox_codex(command: list[str], spec: JobSpec) -> list[str]:
        """Replace Codex's container-only bypass with a host filesystem sandbox.

        The working directory is writable under ``workspace-write``. Fleet mounts
        marked rw (normally just /results) must also be declared explicitly. Read-only
        inputs need no exception and therefore remain protected from agent writes.
        """
        bypass = "--dangerously-bypass-approvals-and-sandbox"
        if bypass not in command:
            return command
        writable = [
            str(Path(mount.source).expanduser().resolve())
            for mount in spec.mounts
            if mount.mode == "rw"
        ]
        # Bubblewrap needs mount namespaces that are commonly disabled on managed
        # research hosts. Landlock enforces this simple workspace-write policy in the
        # kernel without that requirement; forcing it avoids a run that starts normally
        # but fails on every agent shell command with `bwrap: ... Permission denied`.
        replacement = [
            "--sandbox", "workspace-write", "--enable", "use_legacy_landlock",
        ]
        for path in dict.fromkeys(writable):
            replacement += ["--add-dir", path]
        index = command.index(bypass)
        return command[:index] + replacement + command[index + 1:]

    def _snapshot(self, spec: JobSpec, result: JobResult) -> None:
        path = result.worktree_path
        results = next((Path(m.source) for m in spec.mounts if m.target == "/results"), None)
        if not path or results is None:
            return
        status = self._git(path, "status", "--porcelain").stdout.strip()
        if status:
            self._git(path, "add", "-A")
            committed = self._git(
                path, "-c", "user.name=fleet", "-c", "user.email=fleet@localhost",
                "commit", "-m", f"fleet: snapshot {spec.run_id}/{spec.name}",
            )
            if committed.returncode:
                raise RuntimeError(committed.stderr.strip() or "could not snapshot direct worktree")
        commit = self._git(path, "rev-parse", "HEAD").stdout.strip()
        ref = f"refs/fleet-snapshots/{spec.run_id}/{re.sub(r'[^A-Za-z0-9._-]+', '-', spec.id)}"
        self._git(path, "update-ref", ref, commit)
        result.worktree_commit = commit
        result.worktree_snapshot_ref = ref
        base = result.worktree_base_commit
        patch = self._git(path, "diff", "--binary", f"{base}..{commit}").stdout if base else ""
        (results / "snapshot.patch").write_text(patch)
        (results / "snapshot.json").write_text(json.dumps({
            "run_id": spec.run_id, "job_id": spec.id, "stage": spec.name,
            "branch": result.worktree_branch, "base_commit": base, "commit": commit,
            "ref": ref, "changed": base != commit,
            "status_before_snapshot": status.splitlines(),
        }, indent=2) + "\n")

    def run(
        self, spec: JobSpec, *, argv: list[str], env: dict[str, str],
        placement: Placement, policy: Policy, on_line: LineHandler,
    ) -> JobResult:
        started = time.time()
        result = JobResult(job_id=spec.id, state=JobState.RUNNING, started_at=started,
                           node="local", gpu_ids=list(placement.gpu_ids))
        try:
            worktree = self._worktree(spec)
            workspace = worktree[0] if worktree else self.project_dir
            if worktree:
                result.worktree_path, result.worktree_branch, result.worktree_base_commit = worktree
            mapping = self._mapping(spec, workspace)
            command = [self._translate(part, mapping) for part in argv]
            command = self._sandbox_codex(command, spec)
            direct_env = {key: self._translate(value, mapping) for key, value in env.items()}
            direct_env = {**os.environ, **direct_env}
            if placement.gpu_ids:
                direct_env["CUDA_VISIBLE_DEVICES"] = placement.cuda_visible_devices

            def register(proc: subprocess.Popen) -> None:
                with self._lock:
                    self._procs[spec.id] = proc

            outcome = run_process(
                command, env=direct_env, cwd=workspace, timeout_s=spec.timeout_s,
                on_line=on_line, on_start=register,
                on_timeout=lambda: self.cancel(spec.id), stderr_tail_size=12,
            )
            with self._lock:
                self._procs.pop(spec.id, None)
            result.exit_code = outcome.exit_code
            result.state = JobState.SUCCEEDED if outcome.exit_code == 0 else JobState.FAILED
            if outcome.timed_out:
                result.state = JobState.FAILED
                result.error = f"timed out after {spec.timeout_s}s"
            elif outcome.exit_code:
                result.error = f"exit code {outcome.exit_code}" + (
                    f": {outcome.stderr_tail[-1]}" if outcome.stderr_tail else ""
                )
            if worktree:
                self._snapshot(spec, result)
        except Exception as exc:
            result.state = JobState.FAILED
            result.error = f"{type(exc).__name__}: {exc}"
        result.ended_at = time.time()
        return result

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            process = self._procs.get(job_id)
        if process is None:
            return False
        try:
            process.send_signal(signal.SIGTERM)
        except OSError:
            return False
        return True

    def drop_worktree(self, branch: str) -> None:
        listing = self._git(self.project_dir, "worktree", "list", "--porcelain").stdout.splitlines()
        path = None
        current = None
        for line in listing:
            if line.startswith("worktree "):
                current = line[9:]
            elif line == f"branch refs/heads/{branch}":
                path = current
        if path:
            self._git(self.project_dir, "worktree", "remove", "--force", path)
        self._git(self.project_dir, "branch", "-D", branch)

    def close(self) -> None:
        with self._lock:
            jobs = list(self._procs)
        for job in jobs:
            self.cancel(job)
