"""Executor protocol: where a job physically runs.

An executor receives a fully-validated `JobSpec` (policy has already run) plus
the GPU slot it was scheduled onto, and is responsible for actually starting the
container and streaming its output back line by line.

`on_line` is called from the executor's own thread for every line of output.
The scheduler uses it to feed the ledger, so the trace is durable before the
job finishes. Executors must not buffer output to the end.
"""

from __future__ import annotations

import collections
import subprocess
import threading
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from ..policy import Policy
from ..spec import JobResult, JobSpec

LineHandler = Callable[[str, str], None]  # (stream, line) where stream is "stdout"|"stderr"


@dataclass
class ProcessOutcome:
    exit_code: int | None
    timed_out: bool
    stderr_tail: list[str]


def run_process(
    argv: list[str],
    *,
    env: dict[str, str],
    timeout_s: int,
    on_line: LineHandler,
    on_start: Callable[[subprocess.Popen], None],
    on_timeout: Callable[[], None],
    stderr_tail_size: int = 0,
    cwd: str | None = None,
) -> ProcessOutcome:
    """Run and stream a subprocess with the lifecycle shared by local executors."""
    proc = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, env=env, cwd=cwd,
    )
    on_start(proc)
    stderr_tail: collections.deque[str] = collections.deque(maxlen=stderr_tail_size)

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
    for thread in threads:
        thread.start()

    timed_out = False
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        on_timeout()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()

    for thread in threads:
        thread.join(timeout=10)
    return ProcessOutcome(proc.returncode, timed_out, list(stderr_tail))


@dataclass
class Placement:
    """Where the scheduler decided this job goes."""

    node: str = "local"
    gpu_ids: tuple[str, ...] = ()

    @property
    def cuda_visible_devices(self) -> str:
        return ",".join(self.gpu_ids)


@runtime_checkable
class Executor(Protocol):
    kind: str

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
        ...

    def cancel(self, job_id: str) -> bool:
        ...

    def available_gpus(self) -> list[str]:
        ...

    def preflight(self, policy: Policy) -> None:
        """Optional. Raise if nothing could possibly run, before anything is submitted."""
        ...

    def drop_worktree(self, branch: str) -> None:
        """Optional. Discard an isolated job's worktree/branch once nothing will use
        it as a --worktree-base anymore. Only the caller knows that -- a branch can
        have more than one consumer (parallel siblings sharing a base), so this is
        never called automatically by the executor itself. A no-op if unsupported or
        the branch is already gone."""
        ...

    def close(self) -> None:
        ...
