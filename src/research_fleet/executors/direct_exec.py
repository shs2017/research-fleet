"""Run Fleet jobs directly on the host, with the same paths and worktree semantics."""

from __future__ import annotations

import json
import os
import re
import shutil
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

    def __init__(self, project_dir: str, *, nono_binary: str | None = None,
                 state_dir: str | None = None):
        self.project_dir = str(Path(project_dir).expanduser().resolve())
        self.state_dir = str(Path(state_dir or self.project_dir).expanduser().resolve())
        self.nono = shutil.which(nono_binary) if nono_binary else None
        if nono_binary and self.nono is None:
            raise RuntimeError(
                f"{nono_binary!r} not found on PATH; install nono or use executor kind 'ship'"
            )
        if self.nono:
            self.kind = "nono"
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
        # Translate container paths at the start of a value or after punctuation,
        # but never a matching segment inside an already-absolute host path.
        pattern = re.compile(
            r"(?<![A-Za-z0-9._/\\-])(?:"
            + "|".join(re.escape(target) for target in sources)
            + ")"
        )
        return pattern.sub(lambda match: sources[match.group(0)], value)

    @staticmethod
    def _sandbox_codex(
        command: list[str], spec: JobSpec, workspace: str | None = None
    ) -> list[str]:
        """Replace Codex's container-only bypass with a host filesystem sandbox.

        The working directory is writable under ``workspace-write``. Fleet mounts
        marked rw (normally just /results) must also be declared explicitly. Read-only
        inputs need no exception and therefore remain protected from agent writes.
        """
        bypass = "--dangerously-bypass-approvals-and-sandbox"
        if bypass not in command:
            return command
        writable = ([str(Path(workspace).resolve())] if workspace else []) + [
            str(Path(mount.source).expanduser().resolve())
            for mount in spec.mounts
            if mount.mode == "rw"
        ]
        roots = list(dict.fromkeys(writable))
        # Use config overrides instead of `--sandbox`/`--add-dir`: unlike those
        # top-level exec flags, `-c` is also accepted by `codex exec resume`, which is
        # how persistent Fleet actors run every stage after their first one.
        replacement = [
            "-c", 'sandbox_mode="workspace-write"',
            "-c", f"sandbox_workspace_write.writable_roots={json.dumps(roots)}",
            "-c", "sandbox_workspace_write.exclude_slash_tmp=true",
            "-c", "sandbox_workspace_write.exclude_tmpdir_env_var=true",
        ]
        index = command.index(bypass)
        return command[:index] + replacement + command[index + 1:]

    def _nono_command(self, command: list[str], spec: JobSpec, workspace: str) -> list[str]:
        """Wrap a host command in nono's kernel-enforced Landlock sandbox."""
        if not self.nono:
            return command
        profile = Path(__file__).resolve().parents[1] / "data" / "nono-fleet.json"
        wrapped = [
            self.nono, "run", "--silent", "--profile", str(profile),
            "--sandbox-policy", "landlock",
            "--no-audit", "--no-rollback", "--workdir", workspace,
            "--allow", workspace,
        ]
        if "--dangerously-bypass-approvals-and-sandbox" in command:
            state = str(Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser().resolve())
            wrapped += ["--allow", state, "--bypass-protection", state]
        elif "--dangerously-skip-permissions" in command:
            state = str(Path("~/.claude").expanduser().resolve())
            wrapped += ["--allow", state, "--bypass-protection", state]
        for mount in spec.mounts:
            source = str(Path(mount.source).expanduser().resolve())
            wrapped += ["--allow" if mount.mode == "rw" else "--read", source]
        # Native package builds need the system compiler headers and libraries.
        # This is read-only and does not expose writable host state.
        wrapped += ["--read", "/usr"]
        # Matplotlib/fontconfig need only the system font configuration; keep it
        # read-only rather than exposing the rest of /etc.
        wrapped += ["--read", "/etc/fonts"]
        # The conservative profile intentionally removes broad system writes.
        # These device nodes are still needed by ordinary shells and Python
        # (for example, `2>/dev/null` and `os.urandom`) and do not expose a
        # directory or a host data tree.
        for device in ("/dev/null", "/dev/zero", "/dev/random", "/dev/urandom", "/dev/tty"):
            wrapped += ["--allow-file", device]
        return wrapped + ["--", *command]

    def _nono_state_home(self) -> str:
        """Keep nono's own state out of world-writable scratch trees.

        nono's default policy grants `/tmp`; placing its protected state below
        `/tmp` therefore conflicts with its safety checks. Fleet results may
        legitimately live there in tests or on a scratch node, so use the
        user's normal state directory for nono metadata in that case.
        """
        state = Path(self.state_dir).resolve()
        scratch_roots = [Path(os.environ.get("TMPDIR", "/tmp")).resolve(), Path("/tmp")]
        if any(state == root or root in state.parents for root in scratch_roots):
            return str(Path("~/.local/state/research-fleet").expanduser().resolve())
        return self.state_dir

    def preflight(self, policy: Policy, image: str = "") -> None:
        if not self.nono:
            return
        checked = subprocess.run(
            [self.nono, "run", "--silent", "--profile",
             str(Path(__file__).resolve().parents[1] / "data" / "nono-fleet.json"),
             "--sandbox-policy", "landlock",
             "--no-audit", "--no-rollback", "--allow", self.project_dir,
             "--allow-file", "/dev/null", "--allow-file", "/dev/zero",
             "--allow-file", "/dev/random", "--allow-file", "/dev/urandom",
             "--read", "/usr",
             "--", "/bin/true"],
            cwd=self.project_dir, capture_output=True, text=True, check=False,
            env={**os.environ, "XDG_STATE_HOME": self._nono_state_home()},
        )
        if checked.returncode:
            detail = checked.stderr.strip().splitlines()
            raise RuntimeError(
                "nono could not start its Landlock sandbox"
                + (f": {detail[-1]}" if detail else "")
            )

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
        pid_file: Path | None = None
        try:
            worktree = self._worktree(spec)
            workspace = worktree[0] if worktree else self.project_dir
            if worktree:
                result.worktree_path, result.worktree_branch, result.worktree_base_commit = worktree
            mapping = self._mapping(spec, workspace)
            command = [self._translate(part, mapping) for part in argv]
            if self.nono:
                command = self._nono_command(command, spec, workspace)
            else:
                command = self._sandbox_codex(command, spec, workspace)
            direct_env = {key: self._translate(value, mapping) for key, value in env.items()}
            direct_env = {**os.environ, **direct_env}
            if self.nono:
                direct_env["XDG_STATE_HOME"] = self._nono_state_home()
            if placement.gpu_ids:
                direct_env["CUDA_VISIBLE_DEVICES"] = placement.cuda_visible_devices

            def register(proc: subprocess.Popen) -> None:
                with self._lock:
                    self._procs[spec.id] = proc
                if self.nono:
                    pid_file_dir = Path(self.state_dir) / "pids" / spec.run_id
                    pid_file_dir.mkdir(parents=True, exist_ok=True)
                    nonlocal pid_file
                    pid_file = pid_file_dir / f"{spec.id}.pid"
                    pid_file.write_text(str(proc.pid) + "\n", encoding="ascii")

            outcome = run_process(
                command, env=direct_env, cwd=workspace, timeout_s=spec.timeout_s,
                on_line=on_line, on_start=register,
                on_timeout=lambda: self.cancel(spec.id), stderr_tail_size=12,
            )
            with self._lock:
                self._procs.pop(spec.id, None)
            if pid_file:
                pid_file.unlink(missing_ok=True)
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
            if pid_file:
                pid_file.unlink(missing_ok=True)
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
            os.killpg(process.pid, signal.SIGTERM)
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
