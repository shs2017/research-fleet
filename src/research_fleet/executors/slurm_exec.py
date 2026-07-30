"""Slurm executor: hand placement to a cluster scheduler.

Jobs run through `srun`, which blocks, streams output, and returns an exit code, so it
fits the executor protocol without a polling loop or scratch files. The fleet already
runs each job on its own thread, so blocking is free.

Two things differ from the local path:

  * **Slurm owns placement.** It picks the node and binds the GPUs, so the fleet does
    not hand out device UUIDs here. `slots` caps how many `srun` processes the fleet
    holds open at once; anything beyond that queues in Slurm, which is where queueing
    belongs.
  * **Containers are Apptainer, not Docker.** Shared clusters rarely expose the Docker
    daemon. Set `container_image` to a `.sif` and jobs run inside it; leave it empty and
    they run directly in the allocation, using whatever environment the cluster
    provides.

Fractional GPUs are not expressible in Slurm's `--gres`, so a request below one whole
GPU is rounded up rather than silently packed.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import threading
import time

from ..policy import Policy
from ..spec import JobResult, JobSpec, JobState
from .base import LineHandler, Placement


class SlurmUnavailable(RuntimeError):
    pass


class SlurmExecutor:
    kind = "slurm"

    def __init__(
        self,
        *,
        srun_binary: str = "srun",
        scancel_binary: str = "scancel",
        partition: str | None = None,
        account: str | None = None,
        qos: str | None = None,
        slots: int = 8,
        container_image: str | None = None,
        container_binary: str = "apptainer",
        extra_args: list[str] | None = None,
        workspace: str | None = None,
    ):
        if shutil.which(srun_binary) is None:
            raise SlurmUnavailable(
                f"{srun_binary!r} not found on PATH. The slurm executor needs Slurm's "
                "client tools on the machine submitting jobs."
            )
        self.srun = srun_binary
        self.scancel = scancel_binary
        self.partition = partition
        self.account = account
        self.qos = qos
        self.slots = max(1, slots)
        self.container_image = container_image
        self.container_binary = container_binary
        self.extra_args = list(extra_args or [])
        self.workspace = workspace
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------- discovery

    def available_gpus(self) -> list[str]:
        """Placeholder slots. Slurm does the real allocation; this only bounds how
        many submissions the fleet keeps in flight."""
        return [f"slurm-slot-{i}" for i in range(self.slots)]

    # -------------------------------------------------------------- argv build

    def job_name(self, spec: JobSpec) -> str:
        return f"fleet-{spec.id}"

    def build_argv(self, spec: JobSpec, argv: list[str], policy: Policy) -> list[str]:
        gpus = math.ceil(spec.resources.gpus)
        minutes = max(1, math.ceil(spec.timeout_s / 60))

        cmd = [
            self.srun,
            f"--job-name={self.job_name(spec)}",
            f"--cpus-per-task={max(1, int(spec.resources.cpus))}",
            f"--mem={int(spec.resources.memory_gb * 1024)}M",
            # Slurm enforces the wall clock too, so a hung job is reaped even if the
            # submitting process dies.
            f"--time={minutes}",
        ]
        if gpus:
            cmd.append(f"--gres=gpu:{gpus}")
        if self.partition:
            cmd.append(f"--partition={self.partition}")
        if self.account:
            cmd.append(f"--account={self.account}")
        if self.qos:
            cmd.append(f"--qos={self.qos}")
        cmd += self.extra_args

        if not self.container_image:
            return cmd + argv

        inner = [self.container_binary, "exec"]
        if gpus:
            inner.append("--nv")
        workdir = self.workspace or os.getcwd()
        inner += ["--bind", f"{workdir}:/workspace", "--pwd", "/workspace"]
        for mount in spec.mounts:
            suffix = ":ro" if mount.mode == "ro" else ""
            inner += ["--bind", f"{mount.source}:{mount.target}{suffix}"]
        inner.append(self.container_image)
        return cmd + inner + argv

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
        started = time.time()
        result = JobResult(
            job_id=spec.id, state=JobState.RUNNING, started_at=started,
            node=placement.node, container_id=self.job_name(spec),
        )

        # srun inherits the environment, so secrets travel through it rather than the
        # command line, exactly as on the local path.
        child_env = dict(os.environ)
        for key, value in env.items():
            child_env[key] = value or os.environ.get(key, "")

        try:
            proc = subprocess.Popen(
                self.build_argv(spec, argv, policy),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1, env=child_env,
            )
        except OSError as exc:
            result.state = JobState.FAILED
            result.error = f"failed to submit to slurm: {exc}"
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
            # A little longer than Slurm's own limit, so its message arrives first.
            proc.wait(timeout=spec.timeout_s + 60)
        except subprocess.TimeoutExpired:
            timed_out = True
            self.cancel(spec.id)
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

    def cancel(self, job_id: str) -> bool:
        subprocess.run(
            [self.scancel, "--name", f"fleet-{job_id}"],
            capture_output=True, text=True, check=False,
        )
        with self._lock:
            proc = self._procs.get(job_id)
        if proc is not None:
            proc.terminate()
            return True
        return False

    def close(self) -> None:
        with self._lock:
            job_ids = list(self._procs)
        for job_id in job_ids:
            self.cancel(job_id)
