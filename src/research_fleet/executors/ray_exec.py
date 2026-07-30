"""Ray executor: the same research-ship container, placed across a multi-node cluster.

Ray owns placement and GPU accounting; research-ship still owns the container. The
remote task asks Ray which GPUs it was given, resolves those indices to stable
device UUIDs on that node, and hands them to the identical `ShipExecutor`
used in single-host mode. One container definition, two placement strategies.

Every node must have `ship` on PATH and the project's image built locally
(`ship build`), since each worker launches its own container.

Output streaming crosses the process boundary through a `ray.util.queue.Queue`,
so the driver's ledger receives agent reasoning live from remote nodes rather
than only at job exit.
"""

from __future__ import annotations

import subprocess
import threading
import time
from typing import Any

from ..policy import Policy
from ..spec import JobResult, JobSpec, JobState
from .base import LineHandler, Placement
from .ship_exec import ShipExecutor

_SENTINEL = "__fleet_done__"


def _resolve_gpu_uuids(indices: list[Any]) -> tuple[str, ...]:
    """Map Ray's GPU indices to device UUIDs on the local node."""
    if not indices:
        return ()
    if all(isinstance(i, str) and i.startswith("GPU-") for i in indices):
        return tuple(indices)  # already UUIDs
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return tuple(str(i) for i in indices)
    table = {}
    for line in out.splitlines():
        if "," in line:
            idx, uuid = (p.strip() for p in line.split(",", 1))
            table[idx] = uuid
    return tuple(table.get(str(i), str(i)) for i in indices)


def _remote_body(spec_dict, argv, env, policy_dict, exec_kwargs, out_queue, node_id) -> dict:
    """Runs inside the Ray worker. Kept module-level so it pickles cleanly."""
    import ray

    from ..spec import JobSpec as _JobSpec
    from ..policy import Policy as _Policy

    spec = _JobSpec(**spec_dict)
    policy = _Policy(**policy_dict)
    gpu_ids = _resolve_gpu_uuids(list(ray.get_gpu_ids()))
    placement = Placement(node=node_id, gpu_ids=gpu_ids)

    def on_line(stream: str, line: str) -> None:
        try:
            out_queue.put((stream, line), timeout=5)
        except Exception:
            pass  # never let a full queue kill the job

    executor = ShipExecutor(**exec_kwargs)
    try:
        result = executor.run(
            spec, argv=argv, env=env, placement=placement, policy=policy, on_line=on_line
        )
    finally:
        try:
            out_queue.put(("control", _SENTINEL), timeout=5)
        except Exception:
            pass
        executor.close()
    return result.model_dump(mode="json")


class RayExecutor:
    """Multi-node placement. Requires `pip install 'research-fleet[ray]'`."""

    kind = "ray"

    def __init__(
        self,
        *,
        address: str | None = "auto",
        ship_binary: str = "ship",
        project_dir: str | None = None,
        docker_binary: str = "docker",
        queue_maxsize: int = 10_000,
    ):
        try:
            import ray
            from ray.util.queue import Queue
        except ImportError as exc:  # pragma: no cover - depends on extra
            raise RuntimeError(
                "RayExecutor needs Ray: pip install 'research-fleet[ray]'"
            ) from exc

        self._ray = ray
        if not ray.is_initialized():
            # address=None starts a local cluster; "auto" attaches to an existing one.
            ray.init(address=address, ignore_reinit_error=True, log_to_driver=False)

        self._Queue = Queue
        self._queue_maxsize = queue_maxsize
        self._exec_kwargs = {
            "ship_binary": ship_binary,
            "project_dir": project_dir,
            "docker_binary": docker_binary,
        }
        self._refs: dict[str, Any] = {}
        self._lock = threading.Lock()

    def available_gpus(self) -> list[str]:
        total = int(self._ray.cluster_resources().get("GPU", 0))
        return [f"ray-gpu-{i}" for i in range(total)]

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
        ray = self._ray
        remote_fn = ray.remote(num_gpus=spec.resources.gpus, num_cpus=spec.resources.cpus)(_remote_body)

        # One queue per job. A shared queue would force every drainer to inspect
        # and re-enqueue lines belonging to other jobs, which burns CPU and can
        # reorder output; a private queue makes the drain loop a plain read.
        queue = self._Queue(maxsize=self._queue_maxsize)
        ref = remote_fn.remote(
            spec.model_dump(mode="json"),
            argv,
            env,
            policy.model_dump(mode="json"),
            self._exec_kwargs,
            queue,
            placement.node,
        )
        with self._lock:
            self._refs[spec.id] = ref

        stop = threading.Event()

        def drain() -> None:
            while not stop.is_set():
                try:
                    stream, line = queue.get(timeout=0.5)
                except Exception:
                    continue
                if stream == "control" and line == _SENTINEL:
                    return
                on_line(stream, line)

        pump = threading.Thread(target=drain, daemon=True)
        pump.start()

        try:
            payload = ray.get(ref, timeout=spec.timeout_s + 120)
            result = JobResult(**payload)
        except Exception as exc:  # includes GetTimeoutError and worker crashes
            result = JobResult(
                job_id=spec.id,
                state=JobState.FAILED,
                error=f"ray task failed: {type(exc).__name__}: {exc}",
                started_at=time.time(),
                ended_at=time.time(),
                node=placement.node,
            )
        finally:
            stop.set()
            pump.join(timeout=5)
            with self._lock:
                self._refs.pop(spec.id, None)
            try:
                queue.shutdown()
            except Exception:
                pass
        return result

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            ref = self._refs.get(job_id)
        if ref is None:
            return False
        self._ray.cancel(ref, force=True)
        return True

    def close(self) -> None:
        with self._lock:
            refs = list(self._refs.values())
        for ref in refs:
            try:
                self._ray.cancel(ref, force=True)
            except Exception:
                pass
