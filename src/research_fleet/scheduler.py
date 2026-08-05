"""Schedule jobs across GPU slots with policy, budgets, and an audit trail.

Agents submit child jobs through per-job spool directories. Every child returns through
the normal validation path and receives a budget scope bounded by its parent.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TextIO

from .backends import AgentEvent, get_backend
from .budget import BudgetExceeded, BudgetTracker, Usage, quote, render_cost_brief
from .config import FleetConfig
from .executors import Executor, Placement
from .ledger import Ledger
from .policy import Decision, Policy
from .spec import JobKind, JobResult, JobSpec, JobState, Mount, new_id


class SlotPool:
    """Fractional GPU allocator keyed by device UUID."""

    def __init__(self, gpu_ids: list[str]):
        self._capacity: dict[str, float] = {g: 1.0 for g in gpu_ids}
        self._cv = threading.Condition()

    @property
    def device_count(self) -> int:
        return len(self._capacity)

    def acquire(self, amount: float, timeout: float | None = None) -> tuple[str, ...] | None:
        """Block until `amount` GPUs are free. Returns the UUIDs granted."""
        if amount <= 0:
            return ()
        deadline = None if timeout is None else time.time() + timeout
        with self._cv:
            while True:
                granted = self._try_locked(amount)
                if granted is not None:
                    return granted
                remaining = None if deadline is None else deadline - time.time()
                if remaining is not None and remaining <= 0:
                    return None
                self._cv.wait(timeout=remaining if remaining is not None else 1.0)

    def _try_locked(self, amount: float) -> tuple[str, ...] | None:
        if amount < 1.0:
            # Pack onto the fullest device that still fits, to keep whole cards free.
            candidates = [(cap, g) for g, cap in self._capacity.items() if cap >= amount]
            if not candidates:
                return None
            candidates.sort()
            _, gid = candidates[0]
            self._capacity[gid] -= amount
            return (gid,)

        whole = [g for g, cap in self._capacity.items() if cap >= 1.0]
        need = int(amount)
        if len(whole) < need:
            return None
        chosen = tuple(whole[:need])
        for g in chosen:
            self._capacity[g] -= 1.0
        return chosen

    def release(self, gpu_ids: tuple[str, ...], amount: float) -> None:
        if not gpu_ids:
            return
        per = amount / len(gpu_ids) if amount < 1.0 else 1.0
        with self._cv:
            for g in gpu_ids:
                self._capacity[g] = min(1.0, self._capacity[g] + per)
            self._cv.notify_all()


@dataclass
class JobRecord:
    spec: JobSpec
    state: JobState = JobState.PENDING
    result: JobResult | None = None
    decision: Decision | None = None
    usage: Usage = field(default_factory=Usage)
    output: str = ""            # the agent's final message, or the command's last lines
    agent_seconds: float | None = None   # what the harness said it spent
    tail: list[str] = field(default_factory=list)
    reserved_usd: float = 0.0
    reserved_tokens: int = 0
    budget_scope: str = ""
    owns_scope: bool = False
    depth: int = 0
    future: Future | None = None
    children: list[str] = field(default_factory=list)
    results_dir: Path | None = None
    stream_log: TextIO | None = None    # open handle on <results>/stream.log


class Scheduler:
    """Runs a DAG of jobs against an executor, under policy and budget."""

    def __init__(
        self,
        config: FleetConfig,
        executor: Executor,
        ledger: Ledger,
        *,
        run_id: str | None = None,
        on_event: Callable[[str, dict], None] | None = None,
    ):
        self.config = config
        self.executor = executor
        self.ledger = ledger
        self.policy: Policy = config.policy
        self.run_id = run_id or new_id("run")
        self._on_event = on_event

        gpus = executor.available_gpus()
        self.slots = SlotPool(gpus)

        self.budget = BudgetTracker()
        self.budget.open(
            self.run_id,
            max_usd=config.budget.max_usd,
            max_tokens=config.budget.max_tokens,
        )

        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(
            max_workers=self.policy.max_concurrent_jobs, thread_name_prefix="fleet-job"
        )
        self._spool_root = config.root_path / "spool"
        self._spool_root.mkdir(parents=True, exist_ok=True)
        self._stop = threading.Event()
        self._watcher: threading.Thread | None = None

        self.ledger.append(
            "run.started",
            {
                "run_id": self.run_id,
                "executor": executor.kind,
                "gpus": gpus,
                "image": config.image,
                "policy": self.policy.model_dump(mode="json"),
                "budget": {"max_usd": config.budget.max_usd, "max_tokens": config.budget.max_tokens},
            },
            run_id=self.run_id,
        )

    def submit(self, spec: JobSpec, *, parent: JobRecord | None = None) -> JobRecord:
        spec.run_id = self.run_id
        depth = (parent.depth + 1) if parent else 0
        if parent:
            spec.parent_job_id = parent.spec.id

        est = self._estimate(spec)
        scope = parent.budget_scope if parent else self.run_id

        with self._lock:
            siblings = len(parent.children) if parent else 0

        decision = self.policy.check(
            spec,
            depth=depth,
            sibling_count=siblings,
            budget=self.budget,
            budget_scope=scope,
            estimate=est,
            workspace_roots=self.policy.allowed_mount_roots,
        )

        rec = JobRecord(spec=spec, decision=decision, depth=depth, budget_scope=scope)
        if est is not None:
            rec.reserved_usd = est.est_cost_usd
            rec.reserved_tokens = est.est_input_tokens + est.est_output_tokens

        self.ledger.append(
            "job.submitted",
            {
                "spec": spec.model_dump(mode="json"),
                "fingerprint": spec.fingerprint(),
                "depth": depth,
                "parent": spec.parent_job_id,
                "estimate": est.to_dict() if est else None,
                "decision": decision.to_dict(),
            },
            run_id=self.run_id,
            job_id=spec.id,
        )

        for key, value in decision.mutations.items():
            if key == "timeout_s":
                spec.timeout_s = value
            elif key == "agent_disallowed_tools" and spec.agent:
                spec.agent.disallowed_tools = value

        with self._lock:
            self._jobs[spec.id] = rec
            if parent:
                parent.children.append(spec.id)

        if decision.verdict == "deny":
            self._finish_denied(rec, decision)
            return rec
        if decision.verdict == "require_approval":
            rec.state = JobState.AWAITING_APPROVAL
            self._set_state(rec, JobState.AWAITING_APPROVAL, {"reasons": decision.reasons})
            return rec

        self._enqueue(rec)
        return rec

    def approve(self, job_id: str, *, approver: str = "cli") -> bool:
        with self._lock:
            rec = self._jobs.get(job_id)
        if rec is None or rec.state is not JobState.AWAITING_APPROVAL:
            return False
        self.ledger.append("job.approved", {"approver": approver}, run_id=self.run_id, job_id=job_id)
        self._enqueue(rec)
        return True

    def deny(self, job_id: str, *, reason: str = "denied by operator") -> bool:
        with self._lock:
            rec = self._jobs.get(job_id)
        if rec is None or rec.state is not JobState.AWAITING_APPROVAL:
            return False
        self._finish_denied(rec, Decision("deny", [reason]))
        return True

    def _enqueue(self, rec: JobRecord) -> None:
        self._set_state(rec, JobState.QUEUED)
        rec.future = self._pool.submit(self._run_job, rec)

    def _finish_denied(self, rec: JobRecord, decision: Decision) -> None:
        if rec.reserved_usd or rec.reserved_tokens:
            self.budget.release(rec.budget_scope, usd=rec.reserved_usd, tokens=rec.reserved_tokens)
            rec.reserved_usd = rec.reserved_tokens = 0
        rec.result = JobResult(job_id=rec.spec.id, state=JobState.DENIED, error="; ".join(decision.reasons))
        self._set_state(rec, JobState.DENIED, {"reasons": decision.reasons})

    def _estimate(self, spec: JobSpec):
        if spec.kind is not JobKind.AGENT or spec.agent is None:
            return None
        process = spec.labels.get("process") or (
            "agent_long" if spec.timeout_s > 4 * 3600 else "agent_standard"
        )
        model = spec.agent.model or self.config.budget.default_model
        return quote(
            model,
            effort=spec.labels.get("effort", self.config.budget.default_effort),
            process=process,
            observed=self.ledger.observed_cost(model),
        )

    def _wait_for_deps(self, rec: JobRecord) -> bool:
        """Returns False if a dependency failed, which cancels this job."""
        deps = list(rec.spec.depends_on)
        while deps and not self._stop.is_set():
            pending = []
            with self._lock:
                records = {dep: self._jobs.get(dep) for dep in deps}
            for dep in deps:
                d = records[dep]
                if d is None:
                    pending.append(dep)
                elif d.state is JobState.SUCCEEDED:
                    continue
                elif d.state.terminal:
                    self._set_state(rec, JobState.CANCELLED, {"reason": f"dependency {dep} ended {d.state.value}"})
                    return False
                else:
                    pending.append(dep)
            deps = pending
            if deps:
                time.sleep(0.25)
        return not self._stop.is_set()

    def _run_job(self, rec: JobRecord) -> JobResult:
        spec = rec.spec
        if not self._wait_for_deps(rec):
            return rec.result or JobResult(job_id=spec.id, state=JobState.CANCELLED)

        gpu_ids = self.slots.acquire(spec.resources.gpus, timeout=spec.timeout_s)
        if gpu_ids is None:
            self._set_state(rec, JobState.FAILED, {"reason": "timed out waiting for a GPU slot"})
            rec.result = JobResult(job_id=spec.id, state=JobState.FAILED, error="no GPU slot available")
            return rec.result

        try:
            argv, env = self._prepare(rec)
            placement = Placement(node="local", gpu_ids=gpu_ids)
            self._set_state(rec, JobState.RUNNING, {"gpu_ids": list(gpu_ids), "argv_preview": argv[:2]})

            result = self.executor.run(
                spec, argv=argv, env=env, placement=placement,
                policy=self.policy,
                on_line=self._make_line_handler(
                    rec, get_backend(spec.agent.backend) if spec.agent else None
                ),
            )

            if spec.kind is JobKind.AGENT:
                self._drain_spool(rec, self._spool_root / spec.id)

            result.usage = rec.usage.to_dict()
            result.output = rec.output or "\n".join(rec.tail)
            result.agent_seconds = rec.agent_seconds

            if rec.state is JobState.CANCELLED:
                result.state = JobState.CANCELLED
                result.error = rec.result.error if rec.result else "cancelled"
                rec.result = result
                self._settle_budget(rec)
                return result

            rec.result = result
            self._settle_budget(rec)
            self._set_state(rec, result.state, {"result": result.model_dump(mode="json")})
            return result
        except Exception as exc:  # a scheduler bug must not lose the job record
            rec.result = JobResult(
                job_id=spec.id, state=JobState.FAILED,
                error=f"{type(exc).__name__}: {exc}", ended_at=time.time(),
            )
            self._settle_budget(rec)
            self._set_state(rec, JobState.FAILED, {"error": str(exc)})
            return rec.result
        finally:
            self.slots.release(gpu_ids, spec.resources.gpus)
            self._maybe_close_scope(rec)
            self._write_result_files(rec)

    def _write_result_files(self, rec: JobRecord) -> None:
        """Write a best-effort result summary beside the job's artifacts."""
        if rec.stream_log is not None:
            try:
                rec.stream_log.close()
            except OSError:
                pass
            rec.stream_log = None

        if rec.results_dir is None:
            return
        try:
            result = rec.result
            if result is not None:
                (rec.results_dir / "result.json").write_text(
                    json.dumps(result.model_dump(mode="json"), indent=2, default=str)
                )
            text = (result.output if result else "") or rec.output
            if text:
                (rec.results_dir / "output.md").write_text(text)
        except OSError:
            pass

    def _prepare(self, rec: JobRecord) -> tuple[list[str], dict[str, str]]:
        """Build the argv and env for a job, mounting its results and (for agent
        jobs) its spool directory, and opening its child budget scope."""
        spec = rec.spec
        env = dict(self.config.env)
        env.update(spec.env)
        env["FLEET_RUN_ID"] = self.run_id
        env["FLEET_JOB_ID"] = spec.id

        # Fleet mounts are trusted because they are added after policy validation.
        results_dir = self.config.results_path / self.run_id / spec.id
        results_dir.mkdir(parents=True, exist_ok=True)
        spec.mounts.append(Mount(source=str(results_dir), target="/results", mode="rw"))
        env["FLEET_RESULTS_DIR"] = "/results"
        rec.results_dir = results_dir
        rec.stream_log = (results_dir / "stream.log").open("w", buffering=1)

        if spec.kind is JobKind.COMMAND:
            return list(spec.command), env

        assert spec.agent is not None
        backend = get_backend(spec.agent.backend)

        for key in set(backend.required_env()) | set(self.config.agent.passthrough_env):
            env.setdefault(key, "")

        parent_scope = rec.budget_scope
        parent_node = self.budget.get(parent_scope)
        frac = self.config.budget.max_child_grant_fraction
        grant_usd = min(self.policy.max_usd_per_job, parent_node.remaining_usd * frac + rec.reserved_usd)
        grant_tokens = int(
            min(self.policy.max_tokens_per_job, parent_node.remaining_tokens * frac + rec.reserved_tokens)
        )
        child_scope = f"{spec.id}"
        # Move the reservation into the child scope instead of charging it twice.
        self.budget.release(parent_scope, usd=rec.reserved_usd, tokens=rec.reserved_tokens)
        rec.reserved_usd = rec.reserved_tokens = 0
        try:
            self.budget.open(child_scope, max_usd=grant_usd, max_tokens=grant_tokens, parent=parent_scope)
        except BudgetExceeded:
            self.budget.open(
                child_scope,
                max_usd=min(grant_usd, parent_node.remaining_usd),
                max_tokens=min(grant_tokens, parent_node.remaining_tokens),
                parent=parent_scope,
            )
        rec.budget_scope = child_scope
        rec.owns_scope = True
        node = self.budget.get(child_scope)

        # Keep the spool outside /workspace so it cannot shadow project files.
        spool_dir = self._spool_root / spec.id
        spool_dir.mkdir(parents=True, exist_ok=True)
        spec.mounts.append(Mount(source=str(spool_dir), target="/spool", mode="rw"))
        env["FLEET_SUBMIT_DIR"] = "/spool"
        env["FLEET_BUDGET_USD"] = f"{node.remaining_usd:.4f}"
        env["FLEET_BUDGET_TOKENS"] = str(node.remaining_tokens)
        env["FLEET_DEPTH"] = str(rec.depth)
        env["FLEET_MAX_DEPTH"] = str(self.policy.max_agent_depth)

        brief = self._agent_brief(rec, node.remaining_usd, node.remaining_tokens)
        return backend.build_command(spec.agent, brief=brief), env

    def _agent_brief(self, rec: JobRecord, remaining_usd: float, remaining_tokens: int) -> str:
        """Everything the agent needs to spend and delegate responsibly."""
        can_delegate = rec.depth < self.policy.max_agent_depth
        parts = [
            render_cost_brief(
                remaining_usd,
                remaining_tokens,
                models=self.config.budget.delegation_models,
            )
        ]
        if can_delegate:
            parts.append(
                "\n".join(
                    [
                        "",
                        "## Delegating work",
                        f"You may launch up to {self.policy.max_children_per_agent} sub-jobs "
                        f"(you are at depth {rec.depth} of {self.policy.max_agent_depth}).",
                        "",
                        "To launch one, write a JSON file into `$FLEET_SUBMIT_DIR`:",
                        "",
                        "```bash",
                        'cat > "$FLEET_SUBMIT_DIR/try-baseline.json" <<\'EOF\'',
                        json.dumps(
                            {
                                "kind": "agent",
                                "name": "try-baseline",
                                "agent": {
                                    "backend": "claude-cli",
                                    "model": "claude-haiku-4-5",
                                    "task": "Run the baseline config and report val loss.",
                                },
                                "resources": {"gpus": 1},
                                "labels": {"effort": "low", "process": "agent_short"},
                            },
                            indent=2,
                        ),
                        "EOF",
                        "```",
                        "",
                        "Or, for a plain training run, use `\"kind\": \"command\"` with a "
                        "`\"command\": [...]` array instead of an `agent` block.",
                        "",
                        "Each sub-job is checked against the same policy and debits the budget "
                        "above before it starts. A launch that would overspend is rejected with a "
                        "reason written back to `$FLEET_SUBMIT_DIR/<name>.rejected.json`: read it "
                        "and pick a cheaper model or lower effort rather than retrying blindly.",
                        "Write results to `$FLEET_RESULTS_DIR`; anything else is discarded when "
                        "the container exits.",
                    ]
                )
            )
        else:
            parts.append(
                f"\n## Delegating work\nYou are at the maximum depth "
                f"({self.policy.max_agent_depth}) and cannot launch sub-agents. "
                "Complete this task directly."
            )
        return "\n".join(parts)

    def _make_line_handler(self, rec: JobRecord, backend) -> Callable[[str, str], None]:
        spec_id = rec.spec.id

        def handle(stream: str, line: str) -> None:
            if rec.stream_log is not None:
                try:
                    rec.stream_log.write(f"{stream}\t{line}\n")
                except (ValueError, OSError):
                    pass

            if backend is None:
                if stream == "stdout" and line.strip():
                    rec.tail = (rec.tail + [line])[-20:]
                self.ledger.append(
                    "job.output", {"stream": stream, "line": line},
                    run_id=self.run_id, job_id=spec_id,
                )
                self._emit("job.output", {"job_id": spec_id, "stream": stream, "line": line})
                return

            if stream == "stderr":
                self.ledger.append(
                    "job.output", {"stream": "stderr", "line": line},
                    run_id=self.run_id, job_id=spec_id,
                )
                return

            event: AgentEvent | None = backend.parse_line(line)
            if event is None:
                return
            if event.usage is not None:
                with self._lock:
                    rec.usage = rec.usage.merge(event.usage)
            if event.type == "result":
                if event.text:
                    rec.output = event.text
                reported = event.payload.get("duration_ms")
                if reported:
                    rec.agent_seconds = float(reported) / 1000.0
            self.ledger.append(
                f"agent.{event.type}", event.to_ledger(),
                run_id=self.run_id, job_id=spec_id,
            )
            self._emit(f"agent.{event.type}", {"job_id": spec_id, **event.to_ledger()})

        return handle

    def _emit(self, type_: str, payload: dict) -> None:
        if self._on_event:
            try:
                self._on_event(type_, payload)
            except Exception:
                pass

    def _drain_spool(self, parent: JobRecord, spool_dir: Path) -> list[JobRecord]:
        """Ingest child job requests an agent wrote to its spool directory."""
        submitted: list[JobRecord] = []
        for path in sorted(spool_dir.glob("*.json")):
            if path.name.endswith(".rejected.json"):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                self._reject(path, f"unreadable submission: {exc}")
                continue
            try:
                payload.pop("id", None)
                payload.pop("run_id", None)
                payload.pop("parent_job_id", None)
                # Child jobs cannot replace the operator-selected image.
                payload["image"] = self.config.image
                child = JobSpec(**payload)
            except Exception as exc:
                self._reject(path, f"invalid job spec: {exc}")
                continue

            self.ledger.append(
                "agent.job_requested",
                {"parent": parent.spec.id, "source_file": path.name,
                 "spec": child.model_dump(mode="json")},
                run_id=self.run_id, job_id=parent.spec.id,
            )
            rec = self.submit(child, parent=parent)
            if rec.state is JobState.DENIED:
                self._reject(path, rec.result.error if rec.result else "denied by policy")
            else:
                path.unlink(missing_ok=True)
            submitted.append(rec)
        return submitted

    def _reject(self, path: Path, reason: str) -> None:
        out = path.with_suffix(".rejected.json")
        try:
            out.write_text(json.dumps({"rejected": True, "reason": reason}, indent=2), encoding="utf-8")
            path.unlink(missing_ok=True)
        except OSError:
            pass
        self.ledger.append("agent.job_rejected", {"file": path.name, "reason": reason}, run_id=self.run_id)

    def _maybe_close_scope(self, rec: JobRecord) -> None:
        """Return an agent's unused grant to its parent: but only once every
        descendant is terminal, so a still-running child stays bounded by the
        ceiling its parent was granted."""
        if not rec.state.terminal:
            return
        if not rec.owns_scope:
            self._close_parent_scope(rec)
            return
        with self._lock:
            pending = [c for c in rec.children if not self._jobs[c].state.terminal]
        if pending:
            return
        try:
            self.budget.close(rec.budget_scope)
        except KeyError:
            pass
        rec.owns_scope = False
        self.ledger.append(
            "budget.scope_closed",
            {"scope": rec.budget_scope, "final": self.budget.snapshot().get(rec.budget_scope)},
            run_id=self.run_id, job_id=rec.spec.id,
        )
        self._close_parent_scope(rec)

    def _close_parent_scope(self, rec: JobRecord) -> None:
        parent_id = rec.spec.parent_job_id
        if not parent_id:
            return
        with self._lock:
            parent = self._jobs.get(parent_id)
        if parent is not None:
            self._maybe_close_scope(parent)

    def _settle_budget(self, rec: JobRecord) -> None:
        usage = rec.usage
        if usage.total_tokens == 0 and not rec.reserved_usd:
            return
        if not usage.model:
            usage.model = (
                (rec.spec.agent.model if rec.spec.agent else None) or self.config.budget.default_model
            )
        cost = self.budget.commit(
            rec.budget_scope, usage,
            release_usd=rec.reserved_usd, release_tokens=rec.reserved_tokens,
        )
        rec.reserved_usd = rec.reserved_tokens = 0
        self.ledger.append(
            "budget.committed",
            {"scope": rec.budget_scope, "usage": usage.to_dict(), "cost_usd": round(cost, 6),
             "run_remaining_usd": round(self.budget.get(self.run_id).remaining_usd, 4)},
            run_id=self.run_id, job_id=rec.spec.id,
        )

    def _set_state(self, rec: JobRecord, state: JobState, extra: dict | None = None) -> None:
        rec.state = state
        self.ledger.append(
            f"job.{state.value}", {"name": rec.spec.name, **(extra or {})},
            run_id=self.run_id, job_id=rec.spec.id,
        )
        self.ledger.upsert_job(rec.spec, state.value, rec.result)
        self._emit(f"job.{state.value}", {"job_id": rec.spec.id, "name": rec.spec.name})

    def wait(self, timeout: float | None = None) -> dict[str, JobResult]:
        """Block until every job (including ones agents spawned) is terminal.

        Returns early if the only work left is waiting on a human: a job parked
        in AWAITING_APPROVAL will never progress on its own, so blocking on it
        would hang forever rather than telling the operator there is a decision
        to make.
        """
        deadline = None if timeout is None else time.time() + timeout
        while True:
            with self._lock:
                pending = [r for r in self._jobs.values() if not r.state.terminal]
                if not pending:
                    break
                if all(r.state is JobState.AWAITING_APPROVAL for r in pending):
                    self.ledger.append(
                        "run.blocked_on_approval",
                        {"job_ids": [r.spec.id for r in pending]},
                        run_id=self.run_id,
                    )
                    break
                futures = [r.future for r in pending if r.future is not None]
            if deadline is not None and time.time() > deadline:
                break
            for fut in futures:
                try:
                    fut.result(timeout=1.0)
                except Exception:
                    pass
            time.sleep(0.05)

        with self._lock:
            results = {jid: r.result for jid, r in self._jobs.items() if r.result}
        self.ledger.append(
            "run.finished",
            {
                "jobs": len(self._jobs),
                "succeeded": sum(1 for r in self._jobs.values() if r.state is JobState.SUCCEEDED),
                "failed": sum(1 for r in self._jobs.values() if r.state is JobState.FAILED),
                "denied": sum(1 for r in self._jobs.values() if r.state is JobState.DENIED),
                "budget": self.budget.snapshot().get(self.run_id),
            },
            run_id=self.run_id,
        )
        return results

    def cancel_all(self, reason: str = "operator cancelled") -> int:
        """Stop every job that has not finished. Returns how many were stopped."""
        self._stop.set()
        with self._lock:
            active = [r for r in self._jobs.values() if not r.state.terminal]
        for rec in active:
            self.executor.cancel(rec.spec.id)
            rec.result = rec.result or JobResult(
                job_id=rec.spec.id, state=JobState.CANCELLED, error=reason
            )
            self._set_state(rec, JobState.CANCELLED, {"reason": reason})
        self.ledger.append(
            "run.cancelled", {"reason": reason, "jobs": len(active)}, run_id=self.run_id
        )
        return len(active)

    def status(self) -> dict[str, Any]:
        with self._lock:
            jobs = [
                {
                    "id": r.spec.id, "name": r.spec.name, "kind": r.spec.kind.value,
                    "state": r.state.value, "depth": r.depth,
                    "parent": r.spec.parent_job_id,
                    "usage": r.usage.to_dict(),
                }
                for r in self._jobs.values()
            ]
        return {
            "run_id": self.run_id,
            "gpus": self.slots.device_count,
            "jobs": jobs,
            "budget": self.budget.snapshot(),
        }

    def close(self) -> None:
        """Stop executors before waiting for their worker threads."""
        self._stop.set()
        self.executor.close()
        self._pool.shutdown(wait=True, cancel_futures=True)
        shutil.rmtree(self._spool_root, ignore_errors=True)
