"""Executor protocol: where a job physically runs.

An executor receives a fully-validated `JobSpec` (policy has already run) plus
the GPU slot it was scheduled onto, and is responsible for actually starting the
container and streaming its output back line by line.

`on_line` is called from the executor's own thread for every line of output.
The scheduler uses it to feed the ledger, so the trace is durable before the
job finishes. Executors must not buffer output to the end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from ..policy import Policy
from ..spec import JobResult, JobSpec

LineHandler = Callable[[str, str], None]  # (stream, line) where stream is "stdout"|"stderr"


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

    def close(self) -> None:
        ...
