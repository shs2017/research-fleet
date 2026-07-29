"""Ship executor — runs jobs in research-ship containers.

research-fleet does not define what a GPU container looks like. `research-ship`
does: base image, CUDA, the uv-managed venv, the model cache volumes, the
non-root user, the egress firewall. This executor asks it for the resolved
`docker run` argv (`ship docker-args`), appends the fleet's own per-job
hardening, and owns the process from there — streaming output, enforcing the
timeout, and stopping the container on cancel.

Keeping container configuration in one place matters more than it sounds: the
alternative is two tools that each think they know the right mount layout, and
they drift the first time either changes.

Requires `ship` on PATH — see https://github.com/<you>/research-ship.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

from ..policy import Policy
from ..spec import JobResult, JobSpec, JobState
from .base import LineHandler, Placement


class ShipUnavailable(RuntimeError):
    pass


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

    # -------------------------------------------------------------- argv build

    def _ship_flags(
        self, spec: JobSpec, env: dict[str, str], placement: Placement, policy: Policy, name: str
    ) -> list[str]:
        flags = ["--name", name, "--shm-size", f"{int(spec.resources.shm_size_gb)}g"]
        if spec.image:
            flags += ["--image", spec.image]

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
        cmd = [self.ship, "docker-args", *self._ship_flags(spec, env, placement, policy, name), "--", *argv]
        proc_env = dict(os.environ)
        if self.project_dir:
            proc_env["SHIP_PROJECT_DIR"] = self.project_dir
        if policy.network.allowed_hosts:
            proc_env["FIREWALL_EXTRA_DOMAINS"] = " ".join(policy.network.allowed_hosts)

        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=60, env=proc_env
            ).stdout
        except subprocess.CalledProcessError as exc:
            raise ShipUnavailable(
                f"`ship docker-args` failed ({exc.returncode}): {exc.stderr.strip()}"
            ) from exc

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
            docker_argv = self.build_argv(
                spec, argv=argv, env=env, placement=placement, policy=policy, name=name
            )
        except ShipUnavailable as exc:
            result.state = JobState.FAILED
            result.error = str(exc)
            result.ended_at = time.time()
            return result

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

        def pump(stream, label: str) -> None:
            try:
                for line in iter(stream.readline, ""):
                    on_line(label, line.rstrip("\n"))
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
            result.error = f"exit code {proc.returncode}"
        return result

    # ------------------------------------------------------------- lifecycle

    def _stop_container(self, name: str, timeout: int = 20) -> None:
        subprocess.run(
            [self.docker, "stop", "-t", str(timeout), name],
            capture_output=True, text=True, check=False,
        )

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            proc = self._procs.get(job_id)
        self._stop_container(f"fleet-{job_id}")
        if proc is not None:
            try:
                proc.send_signal(signal.SIGTERM)
            except OSError:
                pass
            return True
        return False

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
