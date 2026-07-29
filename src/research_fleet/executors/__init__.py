"""Execution backends.

`ship` is the real one — it runs jobs in research-ship containers. `ray` wraps
it for multi-node placement. `dry-run` starts nothing, which is what lets the
test suite run on a machine with no GPU and no Docker.
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
    if kind == "ray":
        from .ray_exec import RayExecutor

        return RayExecutor(
            address=cfg.executor.ray_address or "auto",
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
