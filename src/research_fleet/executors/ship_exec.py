"""Ship executor: runs jobs in research-ship containers.

research-fleet does not define what a GPU container looks like. `research-ship`
does: base image, CUDA, the uv-managed venv, the model cache volumes, the
non-root user, the egress firewall. This executor asks it for the resolved
`docker run` argv (`ship docker-args`), appends the fleet's own per-job
hardening, and owns the process from there: streaming output, enforcing the
timeout, and stopping the container on cancel.

Keeping container configuration in one place matters more than it sounds: the
alternative is two tools that each think they know the right mount layout, and
they drift the first time either changes.

Requires `ship` on PATH: see https://github.com/<you>/research-ship.
"""

from __future__ import annotations

import collections
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
from .base import LineHandler, Placement


class ShipUnavailable(RuntimeError):
    pass


# What `resolve_worktree()` in `ship` prints on stderr once it isolates a job, so a
# result can say where the work actually landed instead of that line vanishing with
# the rest of `ship docker-args`' stderr.
_WORKTREE_RE = re.compile(r"^\[ship\] isolated in worktree (\S+) on branch (\S+)$", re.MULTILINE)


def _explain(code: int | None, stderr_tail) -> str:
    """Turn an exit code into something a person can act on."""
    detail = next(
        (line for line in reversed(stderr_tail) if "Error" in line or "error" in line),
        stderr_tail[-1] if stderr_tail else "",
    )
    hint = ""
    if code == 125:
        # The daemon refused to start the container at all, which is nearly always a
        # missing image.
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

    # ------------------------------------------------------------- discovery

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
        """Fail before submitting anything if the project's image is missing.

        Without this, every job dies with a bare `exit code 125` from the Docker daemon
        and the operator has to go digging for the reason.
        """
        # Honour a configured image override, or ship's default for the project.
        spec = JobSpec(name="preflight", command=["true"], image=image)
        try:
            argv = self.build_argv(
                spec, argv=["true"], env={}, placement=Placement(gpu_ids=()),
                policy=policy, name="fleet-preflight",
            )
        except ShipUnavailable:
            raise
        image = self._image_from(argv)
        if image is None:
            return
        found = subprocess.run(
            [self.docker, "image", "inspect", image],
            capture_output=True, text=True, check=False,
        )
        if found.returncode != 0:
            where = self.project_dir or os.getcwd()
            raise ShipUnavailable(
                f"the image {image!r} does not exist, so no job could run.\n"
                f"research-ship derives it from the project at {where}.\n"
                f"Build it with:  cd {where} && ship init && ship build\n"
                f"Or point `image:` in fleet.yaml at an image you already have."
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
        """True, False, or None when it cannot be determined.

        Agent jobs fail one by one with "Not logged in" otherwise, which costs a run and
        tells the operator nothing about how to fix it.
        """
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

    # -------------------------------------------------------------- argv build

    def _ship_flags(
        self, spec: JobSpec, env: dict[str, str], placement: Placement, policy: Policy, name: str
    ) -> list[str]:
        flags = ["--name", name, "--shm-size", f"{int(spec.resources.shm_size_gb)}g"]
        if spec.image:
            flags += ["--image", spec.image]

        # research-ship withholds credentials unless asked. Only agent jobs need them;
        # a training run or a sweep point has no use for an API token, and every
        # container that holds one is another place it can leak from.
        flags += ["--creds"] if spec.kind is JobKind.AGENT else ["--no-creds"]

        # research-ship checks out a git worktree on this branch, so the job cannot
        # touch the live working tree. The branch is derived from the job, so parallel
        # agents on one question each land on their own reviewable branch.
        if spec.isolate:
            flags += ["--worktree", f"fleet/{spec.run_id}-{spec.name}"[:80]]
            if spec.worktree_base:
                # Same derivation as above, applied to the job this one continues from,
                # so a chain of steps (plan -> implement -> review) keeps each other's
                # file edits without sharing a single branch.
                flags += ["--worktree-base", f"fleet/{spec.run_id}-{spec.worktree_base}"[:80]]

        if placement.gpu_ids:
            # Quoted form is what Docker's --gpus parser expects for a device list.
            flags += ["--gpus", f'"device={",".join(placement.gpu_ids)}"']
        else:
            flags += ["--gpus", "0"]

        # The fleet's network policy maps onto research-ship's egress allowlist.
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
            # A bare KEY inherits from the daemon environment, so a secret never
            # lands in the argv we are about to write to the audit ledger.
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
        proc_env = dict(os.environ)
        if self.project_dir:
            proc_env["SHIP_PROJECT_DIR"] = self.project_dir
        if policy.network.allowed_hosts:
            proc_env["FIREWALL_EXTRA_DOMAINS"] = " ".join(policy.network.allowed_hosts)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=proc_env)

    def _finish_argv(self, spec: JobSpec, policy: Policy, out: str) -> list[str]:
        tokens = out.splitlines()
        if len(tokens) < 2 or tokens[0] != "docker":
            raise ShipUnavailable(f"unexpected output from ship docker-args: {out[:200]!r}")

        # Insert the fleet's resource ceilings after `docker run`. These are the
        # limits research-ship has no opinion about, because they are a scheduling
        # concern rather than an environment one.
        limits = [
            "--cpus", str(spec.resources.cpus),
            "--memory", f"{int(spec.resources.memory_gb * 1024)}m",
            *policy.container.docker_args(),
        ]
        return [self.docker, "run", *limits, *tokens[2:]]

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
        proc = self._resolve_docker_args(
            spec, argv=argv, env=env, placement=placement, policy=policy, name=name
        )
        if proc.returncode != 0:
            raise ShipUnavailable(
                f"`ship docker-args` failed ({proc.returncode}): {proc.stderr.strip()}"
            )
        return self._finish_argv(spec, policy, proc.stdout)

    # ------------------------------------------------------------------- run

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
            resolve_proc = self._resolve_docker_args(
                spec, argv=argv, env=env, placement=placement, policy=policy, name=name
            )
            if resolve_proc.returncode != 0:
                raise ShipUnavailable(
                    f"`ship docker-args` failed ({resolve_proc.returncode}): {resolve_proc.stderr.strip()}"
                )
            docker_argv = self._finish_argv(spec, policy, resolve_proc.stdout)
        except ShipUnavailable as exc:
            result.state = JobState.FAILED
            result.error = str(exc)
            result.ended_at = time.time()
            return result

        if spec.isolate:
            match = _WORKTREE_RE.search(resolve_proc.stderr)
            if match:
                result.worktree_path, result.worktree_branch = match.group(1), match.group(2)

        # Secret-bearing vars reach the daemon through the subprocess environment,
        # never through the command line.
        child_env = dict(os.environ)
        for key, value in env.items():
            if value == "":
                child_env.setdefault(key, os.environ.get(key, ""))

        try:
            proc = subprocess.Popen(
                docker_argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1, env=child_env,
            )
        except OSError as exc:
            result.state = JobState.FAILED
            result.error = f"failed to start container: {exc}"
            result.ended_at = time.time()
            return result

        with self._lock:
            self._procs[spec.id] = proc

        # Kept so a failure can say why, instead of only reporting an exit code.
        stderr_tail: collections.deque[str] = collections.deque(maxlen=12)

        def pump(stream, label: str) -> None:
            try:
                for line in iter(stream.readline, ""):
                    text = line.rstrip("\n")
                    if label == "stderr" and text.strip():
                        stderr_tail.append(text.strip())
                    on_line(label, text)
            finally:
                stream.close()

        threads = [
            threading.Thread(target=pump, args=(proc.stdout, "stdout"), daemon=True),
            threading.Thread(target=pump, args=(proc.stderr, "stderr"), daemon=True),
        ]
        for t in threads:
            t.start()

        timed_out = False
        try:
            proc.wait(timeout=spec.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._stop_container(name)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()

        for t in threads:
            t.join(timeout=10)
        with self._lock:
            self._procs.pop(spec.id, None)

        result.exit_code = proc.returncode
        result.ended_at = time.time()
        if timed_out:
            result.state = JobState.FAILED
            result.error = f"timed out after {spec.timeout_s}s"
        elif proc.returncode == 0:
            result.state = JobState.SUCCEEDED
        else:
            result.state = JobState.FAILED
            result.error = _explain(proc.returncode, stderr_tail)
        return result

    # ------------------------------------------------------------- lifecycle

    # A cancelled job does not need a long grace period; the operator asked for it to
    # stop now. A timeout is different: there the container may be mid-write.
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
        """Ask ship to discard an isolated job's worktree/branch. Fire-and-forget:
        the branch's content is either already folded into something else's history
        or was never needed again, so a failure here just leaves disk unreclaimed
        rather than losing anything."""
        proc_env = dict(os.environ)
        if self.project_dir:
            proc_env["SHIP_PROJECT_DIR"] = self.project_dir
        subprocess.run(
            [self.ship, "worktree-drop", branch],
            capture_output=True, text=True, timeout=30, env=proc_env, check=False,
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
