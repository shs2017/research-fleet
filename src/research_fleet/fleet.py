"""The public Python API.

    from research_fleet import Fleet

    with Fleet(workspace=".", max_usd=20) as fleet:
        fleet.run_agents("Try 3 attention variants on TinyStories", n=4)
        report = fleet.wait()
        print(report.summary())

The CLI is a thin wrapper around this class.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

from .budget import Quote, cost_menu, quote
from .backends import get_backend
from .config import FleetConfig, load_config
from .executors import build_executor
from .ledger import Ledger, Redactor
from .scheduler import JobRecord, Scheduler
from .spec import AgentConfig, JobKind, JobResult, JobSpec, Mount, Resources

if TYPE_CHECKING:
    from .workflow import Workflow


class CredentialsUnavailable(RuntimeError):
    """An agent job cannot start until the project has credentials."""


def _default_cpus(n: int = 1) -> float:
    """A per-job CPU request that fits the host, split across `n` jobs meant to run
    at once.

    Docker rejects `--cpus` outright when it exceeds the host's core count (unlike
    GPUs, there is no queueing fallback), so callers that do not pass `cpus`
    explicitly must still end up with a value the host can satisfy rather than
    `Resources`' fixed default.
    """
    total = os.cpu_count() or 1
    return max(0.01, total / max(1, n))


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

    @property
    def worktrees(self) -> list[tuple[str, str]]:
        """(path, branch) for isolated jobs whose worktree still exists on disk.

        A chained job's predecessor is deliberately pruned once its changes are
        folded forward (see ship's --worktree-base), so filtering on existence -
        rather than listing every isolated job - naturally surfaces just the
        current tips instead of paths that no longer exist.
        """
        seen: set[str] = set()
        out: list[tuple[str, str]] = []
        for r in self.succeeded:
            if r.worktree_path and r.worktree_path not in seen and Path(r.worktree_path).exists():
                seen.add(r.worktree_path)
                out.append((r.worktree_path, r.worktree_branch or "?"))
        return out

    def summary(self) -> str:
        lines = [
            f"run {self.run_id}: {len(self.succeeded)} succeeded, {len(self.failed)} failed",
            f"spend: ${self.total_cost_usd:,.2f} / {self.total_tokens:,} tokens",
        ]
        if self.awaiting_approval:
            lines.append(
                f"blocked: {len(self.awaiting_approval)} job(s) need approval: "
                f"call fleet.approve({self.awaiting_approval[0]!r}) then wait() again, "
                "or run `fleet jobs --state pending` to see why"
            )
        for jid, res in self.results.items():
            dur = f"{res.duration_s:.1f}s" if res.duration_s else "-"
            lines.append(f"  {jid}  {res.state.value:<10} {dur:>8}  {res.error or ''}".rstrip())
        wts = self.worktrees
        if wts:
            lines.append("")
            lines.append("isolated work landed in (not your working tree):")
            for path, branch in wts:
                lines.append(f"  {path}  (branch {branch})")
            lines.append(f"bring it in with, e.g.: git merge {wts[-1][1]}")
        return "\n".join(lines)


@dataclass
class WorkflowReport:
    workflow: str
    outcomes: list[Any]
    steps: dict[str, JobResult]
    run: RunReport

    def summary(self) -> str:
        lines = [f"workflow {self.workflow}"]
        for outcome in self.outcomes:
            detail = f"{outcome.iterations} iteration(s)" if outcome.iterations > 1 else ""
            if outcome.stopped_early:
                detail += ", stopped early"
            lines.append(f"  {outcome.stage:<24} {len(outcome.job_ids)} job(s) {detail}".rstrip())
        lines.append(self.run.summary())
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
        # Fail now, with a reason, rather than letting every job die on the daemon.
        preflight = getattr(self.executor, "preflight", None)
        if callable(preflight):
            preflight(self.config.policy, self.config.image)
        self.scheduler = Scheduler(
            self.config, self.executor, self.ledger, run_id=run_id, on_event=on_event
        )
        self._checked_credentials = False

    @property
    def run_id(self) -> str:
        return self.scheduler.run_id

    def submit(self, spec: JobSpec) -> JobRecord:
        # Executor mounts are the stable project inputs shared by every stage. Stage
        # mounts follow them so a more specific dependency path can override one.
        spec.mounts = [*self.config.executor.mounts, *spec.mounts]
        return self.scheduler.submit(spec)

    def drop_worktree(self, branch: str) -> None:
        """Discard an isolated job's worktree/branch. Only call this once nothing
        will use it as a --worktree-base anymore -- a branch can have more than one
        consumer, so it is never dropped automatically. No-op on executors that
        don't support isolation (for example, dry-run)."""
        drop = getattr(self.executor, "drop_worktree", None)
        if callable(drop):
            drop(branch)

    def run_agents(
        self,
        task: str,
        *,
        n: int = 1,
        model: str | None = None,
        backend: str | None = None,
        effort: str | None = None,
        gpus: float = 1.0,
        cpus: float | None = None,
        image: str | None = None,
        timeout_s: int = 3600,
        max_turns: int | None = None,
        allowed_tools: Sequence[str] | None = None,
        disallowed_tools: Sequence[str] | None = None,
        system_prompt: str | None = None,
        mounts: Sequence[Mount] | None = None,
        env: dict[str, str] | None = None,
        isolate: bool | None = None,
        worktree_base: str | None = None,
        worktree_base_run_id: str | None = None,
        resume_from: str | None = None,
        session_id: str | None = None,
        labels: dict[str, str] | None = None,
        name_prefix: str = "agent",
    ) -> list[JobRecord]:
        """Launch `n` agents on the same task, each in its own container and GPU slot.

        Running several on one task is the point: they explore different
        approaches, and you compare their results and their traces afterwards.

        `worktree_base` only matters when isolated: it names another isolated job in
        this run whose branch this one should continue from (see `JobSpec.worktree_base`),
        rather than starting fresh from the live tree's HEAD.
        """
        self._require_credentials()
        if resume_from and not self.scheduler.continue_run(resume_from):
            raise ValueError(f"run {resume_from!r} has no results directory to continue")
        selected_backend = backend or self.config.agent.name
        selected_model = self.agent_model(model, selected_backend)
        specs = []
        for i in range(n):
            specs.append(
                JobSpec(
                    kind=JobKind.AGENT,
                    name=f"{name_prefix}-{i}" if n > 1 else name_prefix,
                    image=image or self.config.image,
                    agent=AgentConfig(
                        backend=selected_backend,
                        model=selected_model,
                        task=task,
                        system_prompt=system_prompt,
                        session_id=session_id,
                        # Passed to the harness as well as recorded as a label: the
                        # label drives cost estimation, this drives the actual run.
                        effort=effort or self.config.budget.default_effort,
                        max_turns=max_turns,
                        allowed_tools=list(allowed_tools) if allowed_tools else None,
                        disallowed_tools=list(disallowed_tools or []),
                    ),
                    resources=Resources(gpus=gpus, cpus=cpus if cpus is not None else _default_cpus(n)),
                    timeout_s=timeout_s,
                    mounts=list(mounts or []),
                    env=dict(env or {}),
                    isolate=self.config.isolate_agents if isolate is None else isolate,
                    worktree_base=worktree_base,
                    worktree_base_run_id=worktree_base_run_id,
                    labels={"effort": effort or self.config.budget.default_effort,
                            **(labels or {})},
                )
            )
        return [self.submit(s) for s in specs]

    def agent_model(self, model: str | None = None, backend: str | None = None) -> str:
        """Resolve a model without ever sending one provider's default to another."""
        if model:
            return model
        selected = backend or self.config.agent.name
        configured = self.config.budget.default_model
        mismatched = (selected == "codex-cli" and configured.startswith("claude-")) or (
            selected == "claude-cli" and configured.startswith(("gpt-", "codex-"))
        )
        return get_backend(selected).default_model() if mismatched else configured

    def run_command(
        self,
        command: Sequence[str],
        *,
        name: str = "cmd",
        gpus: float = 1.0,
        cpus: float | None = None,
        image: str | None = None,
        timeout_s: int = 3600,
        env: dict[str, str] | None = None,
        mounts: Sequence[Mount] | None = None,
        isolate: bool | None = None,
        worktree_base: str | None = None,
        worktree_base_run_id: str | None = None,
        labels: dict[str, str] | None = None,
    ) -> JobRecord:
        return self.submit(
            JobSpec(
                name=name,
                command=list(command),
                image=image or self.config.image,
                resources=Resources(gpus=gpus, cpus=cpus if cpus is not None else _default_cpus()),
                timeout_s=timeout_s,
                env=dict(env or {}),
                mounts=list(mounts or []),
                isolate=bool(isolate),
                worktree_base=worktree_base,
                worktree_base_run_id=worktree_base_run_id,
                labels=dict(labels or {}),
            )
        )

    def run_workflow(
        self,
        workflow: "Workflow | dict | str | Path",
        *,
        predicates: dict[str, Any] | None = None,
        resume_from: str | None = None,
        base_run: str | None = None,
    ) -> "WorkflowReport":
        """Run a multi-step pipeline defined in YAML, a dict, or Python.

        `predicates` maps a loop name to a callable taking the results so far, for
        stopping conditions that declarative YAML cannot express.
        """
        from .workflow import Workflow, WorkflowRunner

        if isinstance(workflow, (str, Path)):
            workflow = Workflow.from_yaml(workflow)
        elif isinstance(workflow, dict):
            workflow = Workflow.from_dict(workflow)

        self.ledger.append(
            "workflow.started",
            {"name": workflow.name, "stages": [st.name for st in workflow.stages],
             "parameters": workflow.parameters},
            run_id=self.run_id,
        )
        if resume_from and base_run:
            raise ValueError("use either resume_from or base_run, not both")

        # Name the attempt directory after the workflow, and put a continuation back
        # into the attempt it continues. `base_run` re-executes every stage, so it is a
        # new attempt that merely knows what it was built on.
        self.scheduler.run_name = workflow.name
        self.scheduler.based_on = base_run
        if resume_from and not self.scheduler.continue_run(resume_from):
            self.scheduler.based_on = resume_from

        runner = WorkflowRunner(
            self, workflow, predicates=predicates,
            prior_run=resume_from or base_run,
            resume=resume_from is not None,
        )
        deadline_timer = None
        if workflow.max_duration_s:
            deadline_timer = threading.Timer(
                workflow.max_duration_s,
                self.cancel,
                kwargs={"reason": f"workflow exceeded {workflow.max_duration_s}s deadline"},
            )
            deadline_timer.daemon = True
            deadline_timer.start()
        try:
            outcomes = runner.run()
        finally:
            if deadline_timer is not None:
                deadline_timer.cancel()
        self.ledger.append(
            "workflow.finished",
            {"name": workflow.name, "outcomes": [o.model_dump() for o in outcomes]},
            run_id=self.run_id,
        )
        return WorkflowReport(
            workflow=workflow.name,
            outcomes=outcomes,
            steps=dict(runner.results),
            run=self.wait(),
        )

    def _require_credentials(self) -> None:
        """Refuse to launch agents with no credentials, once per run.

        Without this every agent burns a container and reports "Not logged in", which
        looks like a model failure rather than a setup step.
        """
        if self._checked_credentials:
            return
        check = getattr(self.executor, "credentials_present", None)
        if not callable(check):
            self._checked_credentials = True
            return
        if check(self.config.policy, self.config.image) is False:
            raise CredentialsUnavailable(
                "This project is not logged in yet (agents would report \"Not logged in\").\n"
                "Run `fleet login --import` to use your existing host login, or "
                "`fleet login` to sign in interactively."
            )
        self._checked_credentials = True

    def wait(self, timeout: float | None = None) -> RunReport:
        results = self.scheduler.wait(timeout=timeout)
        return RunReport(self.run_id, results, self.scheduler.status())

    def status(self) -> dict[str, Any]:
        return self.scheduler.status()

    def approve(self, job_id: str) -> bool:
        return self.scheduler.approve(job_id)

    def deny(self, job_id: str, reason: str = "denied by operator") -> bool:
        return self.scheduler.deny(job_id, reason=reason)

    def cancel(self, reason: str = "operator cancelled") -> int:
        """Stop every unfinished job in this run. Returns how many were stopped."""
        return self.scheduler.cancel_all(reason)

    def quote(self, model: str | None = None, *, effort: str = "high", process: str = "agent_standard") -> Quote:
        """What would this cost before I run it?

        Prefers what past jobs on the same model actually cost, falling back to the
        token profile for `process` when there is no history yet.
        """
        chosen = model or self.config.budget.default_model
        return quote(
            chosen, effort=effort, process=process,
            observed=self.ledger.observed_cost(chosen),
        )

    def cost_menu(self) -> list[dict[str, Any]]:
        models = list(dict.fromkeys([
            self.config.budget.default_model, *self.config.budget.delegation_models
        ]))
        return cost_menu(models)

    def usage(
        self,
        group_by: str | None = None,
        *,
        run_id: str | None = None,
        since: float | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Compute and spend across every run this root has recorded.

        With `group_by`, totals per bucket (comma separate keys for a finer grain, for
        example `run,stage,attempt` so a repeated node is counted per attempt). Without
        it, one row of overall totals.
        """
        if group_by:
            return self.ledger.usage_by(group_by, run_id=run_id, since=since, kind=kind)
        return self.ledger.usage_totals(run_id=run_id, since=since, kind=kind)

    def usage_jobs(self, *, run_id: str | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        """Per-job usage rows, newest first."""
        return self.ledger.usage_rows(run_id=run_id, limit=limit)

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
