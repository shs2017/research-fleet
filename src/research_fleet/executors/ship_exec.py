"""Run jobs in research-ship containers with fleet resource limits."""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

from ..policy import Policy
from ..spec import JobKind, JobResult, JobSpec, JobState
from .base import LineHandler, Placement, run_process


class ShipUnavailable(RuntimeError):
    pass


_WORKTREE_RE = re.compile(r"^\[ship\] isolated in worktree (\S+) on branch (\S+)$", re.MULTILINE)


def _explain(code: int | None, stderr_tail) -> str:
    """Turn an exit code into something a person can act on."""
    detail = next(
        (line for line in reversed(stderr_tail) if "Error" in line or "error" in line),
        stderr_tail[-1] if stderr_tail else "",
    )
    hint = ""
    if code == 125:
        hint = " (docker could not start the container; is the image built? `ship build`)"
    elif code == 127:
        hint = " (command not found inside the container)"
    return f"exit code {code}{hint}" + (f": {detail}" if detail else "")


class ShipExecutor:
    kind = "ship"

    def __init__(
        self,
        *,
        ship_binary: str = "ship",
        project_dir: str | None = None,
        docker_binary: str = "docker",
    ):
        self.ship = shutil.which(ship_binary) or ship_binary
        self.docker = docker_binary
        self.project_dir = str(Path(project_dir).expanduser().resolve()) if project_dir else None
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

        if shutil.which(ship_binary) is None:
            raise ShipUnavailable(
                f"{ship_binary!r} not found on PATH. Install research-ship and symlink its "
                "`ship` script somewhere on your PATH."
            )

    def available_gpus(self) -> list[str]:
        """GPU UUIDs on this host, in index order."""
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=15, check=True,
            ).stdout
        except (FileNotFoundError, subprocess.SubprocessError):
            return []
        return [line.strip() for line in out.splitlines() if line.strip()]

    def preflight(self, policy: Policy, image: str = "") -> None:
        """Ensure ship's project environment exists before submitting anything."""
        spec = JobSpec(name="preflight", command=["true"], image=image)
        argv = self.build_argv(
            spec, argv=["true"], env={}, placement=Placement(gpu_ids=()),
            policy=policy, name="fleet-preflight",
        )
        resolved_image = self._image_from(argv)
        if resolved_image is None or self._image_exists(resolved_image):
            return

        if image:
            raise ShipUnavailable(
                f"configured image {resolved_image!r} does not exist. Build or pull it, or remove "
                "`image:` from fleet.yaml to let fleet prepare the project automatically."
            )

        project = Path(self.project_dir or os.getcwd())
        if not (project / ".ship.conf").exists():
            self._run_ship_setup("init", project)
        self._run_ship_setup("build", project)
        if not self._image_exists(resolved_image):
            raise ShipUnavailable(
                f"automatic `ship build` completed, but image {resolved_image!r} still does not exist. "
                f"Run `ship build` in {project} to inspect the problem."
            )

    def _image_exists(self, image: str) -> bool:
        return subprocess.run(
            [self.docker, "image", "inspect", image],
            capture_output=True, text=True, check=False,
        ).returncode == 0

    def _ship_env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.project_dir:
            env["SHIP_PROJECT_DIR"] = self.project_dir
        return env

    def _run_ship_setup(self, command: str, project: Path) -> None:
        """Run one automatic setup step and report its useful output, not Python internals."""
        try:
            done = subprocess.run(
                [self.ship, command], cwd=project, env=self._ship_env(),
                capture_output=True, text=True,
            )
        except OSError as exc:
            raise ShipUnavailable(f"could not run `ship {command}`: {exc}") from None
        if done.returncode:
            detail = (done.stderr or done.stdout).strip()
            suffix = f"\n{detail}" if detail else ""
            raise ShipUnavailable(
                f"automatic `ship {command}` failed in {project} (exit {done.returncode}).{suffix}"
            )

    def credential_volume(self, policy: Policy, image: str = "") -> str | None:
        """The Docker volume research-ship mounts for agent credentials, if any."""
        spec = JobSpec(kind=JobKind.AGENT, name="probe", image=image,
                       agent={"task": "probe", "backend": "claude-cli"})
        argv = self.build_argv(
            spec, argv=["true"], env={}, placement=Placement(gpu_ids=()),
            policy=policy, name="fleet-probe",
        )
        for token in argv:
            if token.endswith(":/home/dev/.claude"):
                return token.split(":", 1)[0]
        return None

    def credentials_present(self, policy: Policy, image: str = "") -> bool | None:
        """Return whether credentials exist, or None when this cannot be determined."""
        volume = self.credential_volume(policy, image)
        if volume is None:
            return None
        argv = self.build_argv(
            JobSpec(name="probe", command=["true"], image=image), argv=["true"], env={},
            placement=Placement(gpu_ids=()), policy=policy, name="fleet-probe",
        )
        resolved = self._image_from(argv)
        if resolved is None:
            return None
        image = resolved
        done = subprocess.run(
            [self.docker, "run", "--rm", "-v", f"{volume}:/probe", image,
             "bash", "-lc", "test -s /probe/.credentials.json || test -s /probe/.claude.json"],
            capture_output=True, text=True, check=False,
        )
        return done.returncode == 0

    @staticmethod
    def _image_from(argv: list[str]) -> str | None:
        """The image is the token just before `bash -lc` in what ship emits."""
        for i, token in enumerate(argv):
            if token == "bash" and i + 1 < len(argv) and argv[i + 1] == "-lc":
                return argv[i - 1] if i else None
        return None

    def _ship_flags(
        self, spec: JobSpec, env: dict[str, str], placement: Placement, policy: Policy, name: str
    ) -> list[str]:
        flags = ["--name", name, "--shm-size", f"{int(spec.resources.shm_size_gb)}g"]
        if spec.image:
            flags += ["--image", spec.image]

        flags += ["--creds"] if spec.kind is JobKind.AGENT else ["--no-creds"]

        if spec.isolate:
            flags += ["--worktree", f"fleet/{spec.run_id}-{spec.name}"[:80]]
            if spec.worktree_base:
                flags += ["--worktree-base", f"fleet/{spec.run_id}-{spec.worktree_base}"[:80]]

        if placement.gpu_ids:
            flags += ["--gpus", f'"device={",".join(placement.gpu_ids)}"']
        else:
            flags += ["--gpus", "0"]

        mode = policy.network.mode
        if mode == "none":
            flags += ["--network", "none", "--no-firewall"]
        elif mode == "limited":
            flags += ["--firewall"]
        else:
            flags += ["--no-firewall"]

        for m in spec.mounts:
            flags += ["--mount", m.to_docker_arg()]
        for key, value in env.items():
            flags += ["--env", key if value == "" else f"{key}={value}"]

        flags += ["--label", f"fleet.job={spec.id}", "--label", f"fleet.run={spec.run_id}"]
        return flags

    def _resolve_docker_args(
        self,
        spec: JobSpec,
        *,
        argv: list[str],
        env: dict[str, str],
        placement: Placement,
        policy: Policy,
        name: str,
    ) -> subprocess.CompletedProcess:
        """Run `ship docker-args`, unchecked, so callers can inspect stderr too --
        that's where an isolated job's worktree location is reported."""
        cmd = [self.ship, "docker-args", *self._ship_flags(spec, env, placement, policy, name), "--", *argv]
        proc_env = self._ship_env()
        if policy.network.allowed_hosts:
            proc_env["FIREWALL_EXTRA_DOMAINS"] = " ".join(policy.network.allowed_hosts)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=proc_env)

    def _finish_argv(self, spec: JobSpec, policy: Policy, out: str) -> list[str]:
        tokens = out.splitlines()
        if len(tokens) < 2 or tokens[0] != "docker":
            raise ShipUnavailable(f"unexpected output from ship docker-args: {out[:200]!r}")

        limits = [
            "--cpus", str(spec.resources.cpus),
            "--memory", f"{int(spec.resources.memory_gb * 1024)}m",
            *policy.container.docker_args(),
        ]
        return [self.docker, "run", *limits, *tokens[2:]]

    def _docker_argv(
        self,
        spec: JobSpec,
        *,
        argv: list[str],
        env: dict[str, str],
        placement: Placement,
        policy: Policy,
        name: str,
    ) -> tuple[list[str], str]:
        proc = self._resolve_docker_args(
            spec, argv=argv, env=env, placement=placement, policy=policy, name=name
        )
        if proc.returncode:
            raise ShipUnavailable(
                f"`ship docker-args` failed ({proc.returncode}): {proc.stderr.strip()}"
            )
        return self._finish_argv(spec, policy, proc.stdout), proc.stderr

    def build_argv(
        self,
        spec: JobSpec,
        *,
        argv: list[str],
        env: dict[str, str],
        placement: Placement,
        policy: Policy,
        name: str,
    ) -> list[str]:
        """Ask research-ship for the container argv, then layer the fleet's limits on."""
        docker_argv, _ = self._docker_argv(
            spec, argv=argv, env=env, placement=placement, policy=policy, name=name
        )
        return docker_argv

    def run(
        self,
        spec: JobSpec,
        *,
        argv: list[str],
        env: dict[str, str],
        placement: Placement,
        policy: Policy,
        on_line: LineHandler,
    ) -> JobResult:
        name = f"fleet-{spec.id}"
        started = time.time()
        result = JobResult(
            job_id=spec.id, state=JobState.RUNNING, started_at=started,
            node=placement.node, gpu_ids=list(placement.gpu_ids), container_id=name,
        )

        try:
            docker_argv, ship_stderr = self._docker_argv(
                spec, argv=argv, env=env, placement=placement, policy=policy, name=name
            )
        except ShipUnavailable as exc:
            result.state = JobState.FAILED
            result.error = str(exc)
            result.ended_at = time.time()
            return result

        if spec.isolate:
            match = _WORKTREE_RE.search(ship_stderr)
            if match:
                result.worktree_path, result.worktree_branch = match.group(1), match.group(2)

        child_env = dict(os.environ)
        for key, value in env.items():
            if value == "":
                child_env.setdefault(key, os.environ.get(key, ""))

        def register(proc: subprocess.Popen) -> None:
            with self._lock:
                self._procs[spec.id] = proc

        try:
            outcome = run_process(
                docker_argv, env=child_env, timeout_s=spec.timeout_s, on_line=on_line,
                on_start=register, on_timeout=lambda: self._stop_container(name),
                stderr_tail_size=12,
            )
        except OSError as exc:
            result.state = JobState.FAILED
            result.error = f"failed to start container: {exc}"
            result.ended_at = time.time()
            return result
        with self._lock:
            self._procs.pop(spec.id, None)

        result.exit_code = outcome.exit_code
        result.ended_at = time.time()
        if outcome.timed_out:
            result.state = JobState.FAILED
            result.error = f"timed out after {spec.timeout_s}s"
        elif outcome.exit_code == 0:
            result.state = JobState.SUCCEEDED
        else:
            result.state = JobState.FAILED
            result.error = _explain(outcome.exit_code, outcome.stderr_tail)
        return result

    CANCEL_GRACE_S = 5

    def _stop_container(self, name: str, timeout: int = 20) -> None:
        subprocess.run(
            [self.docker, "stop", "-t", str(timeout), name],
            capture_output=True, text=True, check=False,
        )

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            proc = self._procs.get(job_id)
        self._stop_container(f"fleet-{job_id}", timeout=self.CANCEL_GRACE_S)
        if proc is not None:
            try:
                proc.send_signal(signal.SIGTERM)
            except OSError:
                pass
            return True
        return False

    def drop_worktree(self, branch: str) -> None:
        """Ask ship to discard an isolated worktree and branch."""
        subprocess.run(
            [self.ship, "worktree-drop", branch],
            capture_output=True, text=True, timeout=30, env=self._ship_env(), check=False,
        )

    def close(self) -> None:
        with self._lock:
            job_ids = list(self._procs)
        for jid in job_ids:
            self.cancel(jid)


class DryRunExecutor:
    """Validates and records what *would* run. Used by tests and `--executor dry-run`."""

    kind = "dry-run"

    def __init__(self, fake_gpus: int = 2):
        self._gpus = [f"GPU-dryrun-{i}" for i in range(fake_gpus)]

    def available_gpus(self) -> list[str]:
        return list(self._gpus)

    def run(self, spec: JobSpec, *, argv, env, placement, policy, on_line: LineHandler) -> JobResult:
        on_line("stdout", f"[dry-run] would execute: {' '.join(argv)}")
        on_line("stdout", f"[dry-run] gpus={placement.gpu_ids} node={placement.node}")
        now = time.time()
        return JobResult(
            job_id=spec.id, state=JobState.SUCCEEDED, exit_code=0,
            started_at=now, ended_at=now, node=placement.node,
            gpu_ids=list(placement.gpu_ids),
        )

    def cancel(self, job_id: str) -> bool:
        return True

    def close(self) -> None:
        return None
