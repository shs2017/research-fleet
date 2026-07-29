"""The public Python API.

    from research_fleet import Fleet

    with Fleet(workspace=".", max_usd=20) as fleet:
        fleet.run_agents("Try 3 attention variants on TinyStories", n=4)
        report = fleet.wait()
        print(report.summary())

Everything the CLI and the MCP server do goes through this class, so the three
entry points can never drift apart in behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .budget import Quote, cost_menu, quote
from .config import FleetConfig, load_config
from .executors import build_executor
from .ledger import Ledger, Redactor
from .scheduler import JobRecord, Scheduler
from .spec import AgentConfig, JobKind, JobResult, JobSpec, Mount, Resources
from .sweep import build_sweep


@dataclass
class RunReport:
    run_id: str
    results: dict[str, JobResult]
    status: dict[str, Any]

    @property
    def succeeded(self) -> list[JobResult]:
        return [r for r in self.results.values() if r.state.value == "succeeded"]

    @property
    def failed(self) -> list[JobResult]:
        return [r for r in self.results.values() if r.state.value in {"failed", "denied", "cancelled"}]

    @property
    def total_cost_usd(self) -> float:
        root = self.status.get("budget", {}).get(self.run_id, {})
        return float(root.get("spent_usd", 0.0))

    @property
    def total_tokens(self) -> int:
        root = self.status.get("budget", {}).get(self.run_id, {})
        return int(root.get("spent_tokens", 0))

    @property
    def awaiting_approval(self) -> list[str]:
        return [
            j["id"] for j in self.status.get("jobs", []) if j["state"] == "awaiting_approval"
        ]

    def summary(self) -> str:
        lines = [
            f"run {self.run_id}: {len(self.succeeded)} succeeded, {len(self.failed)} failed",
            f"spend: ${self.total_cost_usd:,.2f} / {self.total_tokens:,} tokens",
        ]
        if self.awaiting_approval:
            lines.append(
                f"blocked: {len(self.awaiting_approval)} job(s) need approval — "
                f"call fleet.approve({self.awaiting_approval[0]!r}) then wait() again, "
                "or run `fleet pending` to see why"
            )
        for jid, res in self.results.items():
            dur = f"{res.duration_s:.1f}s" if res.duration_s else "-"
            lines.append(f"  {jid}  {res.state.value:<10} {dur:>8}  {res.error or ''}".rstrip())
        return "\n".join(lines)


class Fleet:
    """Orchestrates containerized research jobs across GPUs, with audit + budget.

    Jobs carry no workspace mount: research-ship already bind-mounts the project
    directory at /workspace, so adding one here would duplicate the mount target.
    """

    def __init__(
        self,
        config: FleetConfig | str | Path | None = None,
        *,
        run_id: str | None = None,
        on_event: Callable[[str, dict], None] | None = None,
        **overrides: Any,
    ):
        if isinstance(config, FleetConfig):
            self.config = config
        else:
            self.config = load_config(config, **overrides)

        self.config.root_path.mkdir(parents=True, exist_ok=True)
        self.config.results_path.mkdir(parents=True, exist_ok=True)

        self.ledger = Ledger(
            self.config.root_path,
            redactor=Redactor() if self.config.policy.redact_secrets else None,
        )
        self.executor = build_executor(self.config)
        self.scheduler = Scheduler(
            self.config, self.executor, self.ledger, run_id=run_id, on_event=on_event
        )

    # ------------------------------------------------------------------ props

    @property
    def run_id(self) -> str:
        return self.scheduler.run_id

    # --------------------------------------------------------------- submit

    def submit(self, spec: JobSpec) -> JobRecord:
        return self.scheduler.submit(spec)

    def run_agents(
        self,
        task: str,
        *,
        n: int = 1,
        model: str | None = None,
        backend: str | None = None,
        effort: str | None = None,
        gpus: float = 1.0,
        image: str | None = None,
        timeout_s: int = 3600,
        max_turns: int | None = None,
        allowed_tools: Sequence[str] | None = None,
        system_prompt: str | None = None,
        mounts: Sequence[Mount] | None = None,
        env: dict[str, str] | None = None,
        name_prefix: str = "agent",
    ) -> list[JobRecord]:
        """Launch `n` agents on the same task, each in its own container and GPU slot.

        Running several on one task is the point: they explore different
        approaches, and you compare their results and their traces afterwards.
        """
        specs = []
        for i in range(n):
            specs.append(
                JobSpec(
                    kind=JobKind.AGENT,
                    name=f"{name_prefix}-{i}" if n > 1 else name_prefix,
                    image=image or self.config.image,
                    agent=AgentConfig(
                        backend=backend or self.config.agent.name,
                        model=model or self.config.budget.default_model,
                        task=task,
                        system_prompt=system_prompt,
                        max_turns=max_turns,
                        allowed_tools=list(allowed_tools) if allowed_tools else None,
                    ),
                    resources=Resources(gpus=gpus),
                    timeout_s=timeout_s,
                    mounts=list(mounts or []),
                    env=dict(env or {}),
                    labels={"effort": effort or self.config.budget.default_effort},
                )
            )
        return [self.submit(s) for s in specs]

    def run_sweep(
        self,
        command: Sequence[str],
        grid: dict[str, Sequence[Any]] | None = None,
        *,
        points: Sequence[dict[str, Any]] | None = None,
        gpus: float = 1.0,
        image: str | None = None,
        timeout_s: int = 3600,
        env: dict[str, str] | None = None,
        name_prefix: str = "sweep",
    ) -> list[JobRecord]:
        specs = build_sweep(
            command, grid, points=points,
            image=image or self.config.image,
            resources=Resources(gpus=gpus),
            name_prefix=name_prefix,
            timeout_s=timeout_s,
            env=env,
        )
        return [self.submit(s) for s in specs]

    def run_command(
        self,
        command: Sequence[str],
        *,
        name: str = "cmd",
        gpus: float = 1.0,
        image: str | None = None,
        timeout_s: int = 3600,
        env: dict[str, str] | None = None,
    ) -> JobRecord:
        return self.submit(
            JobSpec(
                name=name,
                command=list(command),
                image=image or self.config.image,
                resources=Resources(gpus=gpus),
                timeout_s=timeout_s,
                env=dict(env or {}),
            )
        )

    # ---------------------------------------------------------------- control

    def wait(self, timeout: float | None = None) -> RunReport:
        results = self.scheduler.wait(timeout=timeout)
        return RunReport(self.run_id, results, self.scheduler.status())

    def status(self) -> dict[str, Any]:
        return self.scheduler.status()

    def approve(self, job_id: str) -> bool:
        return self.scheduler.approve(job_id)

    def deny(self, job_id: str, reason: str = "denied by operator") -> bool:
        return self.scheduler.deny(job_id, reason=reason)

    def cancel(self, reason: str = "operator cancelled") -> None:
        self.scheduler.cancel_all(reason)

    # ----------------------------------------------------------- introspection

    def quote(self, model: str | None = None, *, effort: str = "high", process: str = "agent_standard") -> Quote:
        """What would this cost before I run it?"""
        return quote(model or self.config.budget.default_model, effort=effort, process=process)

    def cost_menu(self) -> list[dict[str, Any]]:
        return cost_menu(self.config.budget.delegation_models)

    def verify_audit(self) -> tuple[bool, str | None]:
        return self.ledger.verify()

    def trace(self, job_id: str, limit: int = 5000) -> list[dict[str, Any]]:
        """The reasoning trace for one job, in order."""
        return [
            {"seq": e.seq, "ts": e.ts, "type": e.type, **e.payload}
            for e in self.ledger.events(job_id=job_id, limit=limit)
        ]

    def close(self) -> None:
        self.scheduler.close()
        self.ledger.close()

    def __enter__(self) -> "Fleet":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
