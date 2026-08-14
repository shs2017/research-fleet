"""Execution backends.

`ship` runs jobs in research-ship containers. `dry-run` starts nothing, which
keeps plans and tests independent of Docker and GPUs.
"""

from .base import Executor, LineHandler, Placement  # noqa: F401
from .ship_exec import DryRunExecutor, ShipExecutor, ShipUnavailable  # noqa: F401


def build_executor(cfg) -> Executor:
    """Construct the executor named by `cfg.executor.kind`."""
    kind = cfg.executor.kind
    if kind == "dry-run":
        return DryRunExecutor()
    if kind == "ship":
        return ShipExecutor(
            ship_binary=cfg.executor.ship_binary,
            project_dir=cfg.executor.project_dir or cfg.workspace,
            docker_binary=cfg.executor.docker_binary,
        )
    raise ValueError(f"unknown executor kind {kind!r}")


__all__ = [
    "Executor", "LineHandler", "Placement",
    "ShipExecutor", "ShipUnavailable", "DryRunExecutor",
    "build_executor",
]
