"""Execution backends.

`ship` runs jobs in research-ship containers. `dry-run` starts nothing, which
keeps plans and tests independent of Docker and GPUs.
"""

from .base import Executor, LineHandler, Placement  # noqa: F401
from .direct_exec import DirectExecutor  # noqa: F401
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
    if kind == "direct":
        return DirectExecutor(project_dir=cfg.executor.project_dir or cfg.workspace)
    if kind == "nono":
        return DirectExecutor(
            project_dir=cfg.executor.project_dir or cfg.workspace,
            nono_binary=cfg.executor.nono_binary,
            state_dir=str(cfg.root_path / "nono"),
        )
    raise ValueError(f"unknown executor kind {kind!r}")


__all__ = [
    "Executor", "LineHandler", "Placement",
    "ShipExecutor", "ShipUnavailable", "DirectExecutor", "DryRunExecutor",
    "build_executor",
]
