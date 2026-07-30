"""Execution backends.

`ship` runs jobs in research-ship containers on this host. `ray` and `slurm` hand
placement to a cluster scheduler instead. `dry-run` starts nothing, which is what lets
the test suite run on a machine with no GPU and no Docker.
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
    if kind == "slurm":
        from .slurm_exec import SlurmExecutor

        sl = cfg.executor.slurm
        return SlurmExecutor(
            partition=sl.partition, account=sl.account, qos=sl.qos, slots=sl.slots,
            container_image=sl.container_image, container_binary=sl.container_binary,
            extra_args=sl.extra_args, workspace=cfg.workspace,
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
