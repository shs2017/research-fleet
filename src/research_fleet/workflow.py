"""Multi-step agent and command workflows.

Stages run sequentially unless `needs` makes dependencies explicit. Steps can fan out
with `for_each` or `copies`, and cycles repeat up to their configured limit. Templates
use simple `{{ path.to.value }}` substitution against earlier results.
"""

from __future__ import annotations

import re
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, Callable

import yaml
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from .spec import JobKind, JobResult, JobState, Mount
from . import runlayout

_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_.]+)\s*\}\}")


def render(text: str, context: dict[str, Any]) -> str:
    """Substitute `{{ a.b }}` from a nested context. Unknown names are left alone,
    so a stray brace in a prompt survives rather than raising mid-run."""

    def lookup(match: re.Match) -> str:
        cursor: Any = context
        for part in match.group(1).split("."):
            if isinstance(cursor, dict) and part in cursor:
                cursor = cursor[part]
            else:
                return match.group(0)
        return str(cursor)

    return _PLACEHOLDER.sub(lookup, text)


class Condition(BaseModel):
    """When repetition should stop. Every field given must hold."""

    step: str | None = Field(
        None, description="Whose result to inspect. Defaults to the node holding it."
    )
    output_contains: str | None = None
    output_not_contains: str | None = None
    succeeded: bool | None = None

    def met(self, results: dict[str, JobResult], default_step: str | None = None) -> bool:
        result = results.get(self.step or default_step or "")
        if result is None:
            return False
        text = result.output or ""
        if self.output_contains is not None and self.output_contains.lower() not in text.lower():
            return False
        if self.output_not_contains is not None and self.output_not_contains.lower() in text.lower():
            return False
        if self.succeeded is not None and (result.state.value == "succeeded") != self.succeeded:
            return False
        return True


class Step(BaseModel):
    name: str
    kind: JobKind = JobKind.AGENT
    task: str = ""                          # agent steps
    task_file: str = ""
    context_files: list[str] = Field(
        default_factory=list,
        description="Private reference files appended to this stage's task prompt.",
    )
    command: list[str] = Field(default_factory=list)   # command steps

    model: str | None = None
    effort: str | None = None
    gpus: float = 1.0
    timeout_s: int = 3600
    isolate: bool | None = None
    needs: list[str] = Field(default_factory=list)
    actor: str | None = Field(
        None,
        description="Named workflow actor whose provider conversation runs this agent step.",
    )

    until: Condition | None = Field(
        None, description="Stop repeating once this holds of the node's own output."
    )
    max_iterations: int | None = Field(None, ge=1, description="Cap on repeats of this node's cycle.")

    for_each: list[Any] = Field(default_factory=list, description="One run per item; use {{ item }}.")
    copies: int = Field(1, ge=1, description="Run this many identical copies in parallel.")

    @model_validator(mode="after")
    def _check(self) -> Step:
        if self.kind is JobKind.AGENT and not self.task:
            raise ValueError(f"step {self.name!r}: agent steps need a `task` or `task_file`")
        if self.kind is JobKind.COMMAND and not self.command:
            raise ValueError(f"step {self.name!r}: command steps need a `command`")
        if self.kind is JobKind.COMMAND and self.actor is not None:
            raise ValueError(f"step {self.name!r}: command steps cannot name an actor")
        if self.for_each and self.copies > 1:
            raise ValueError(f"step {self.name!r}: use either `for_each` or `copies`, not both")
        return self


class Loop(BaseModel):
    """Repeat `steps` until `until` holds, or `max_iterations` is reached."""

    name: str
    steps: list[Step]
    max_iterations: int = Field(3, ge=1)
    until: Condition | None = None
    needs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> Loop:
        if not self.steps:
            raise ValueError(f"loop {self.name!r}: needs at least one step")
        if self.until and self.until.step not in {s.name for s in self.steps}:
            raise ValueError(
                f"loop {self.name!r}: until.step {self.until.step!r} is not one of its steps"
            )
        return self


class Actor(BaseModel):
    """A logical agent identity reused by any number of workflow stages."""

    persistent: bool = True
    backend: str | None = None
    model: str | None = None
    effort: str | None = None
    system_prompt: str | None = None


class Workflow(BaseModel):
    model_config = {"extra": "forbid"}

    name: str = "workflow"
    description: str = ""
    stages: list[Step | Loop] = Field(default_factory=list)
    actors: dict[str, Actor] = Field(default_factory=dict)

    model: str | None = None
    effort: str | None = None
    gpus: float | None = None
    isolate: bool | None = None
    max_iterations: int = Field(
        3, ge=1, description="Default cap on how many times a cycle repeats."
    )

    _graph_nodes: set[str] = PrivateAttr(default_factory=set)

    @model_validator(mode="after")
    def _check(self) -> Workflow:
        if not self.stages:
            raise ValueError("a workflow needs at least one stage")
        names = [st.name for st in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("stage names must be unique")
        known = set(names)
        for stage in self.stages:
            for need in stage.needs:
                if need not in known:
                    raise ValueError(f"stage {stage.name!r} needs {need!r}, which is not a stage")
            steps = stage.steps if isinstance(stage, Loop) else [stage]
            for step in steps:
                if step.actor is not None and step.actor not in self.actors:
                    raise ValueError(
                        f"step {step.name!r} uses unknown actor {step.actor!r}"
                    )
                if step.actor is not None and (step.for_each or step.copies > 1):
                    raise ValueError(
                        f"step {step.name!r}: actor steps cannot fan out; give each copy its own actor"
                    )
        return self

    def graph(self) -> dict[str, set[str]]:
        """Node name to the nodes it waits for.

        Nodes declared under `graph:` use exactly the edges they state. Nodes declared
        under `stages:` without `needs` get an implicit edge to everything above them,
        and that implicit edge is skipped where it would invent a cycle the author did
        not write. So a reported cycle is always a real, intended one.
        """
        explicit = {st.name: set(st.needs) for st in self.stages}

        def depends_on(start: str, target: str, seen: set[str] | None = None) -> bool:
            seen = seen if seen is not None else set()
            for dep in explicit.get(start, ()):
                if dep == target or (dep not in seen and depends_on(dep, target, seen | {dep})):
                    return True
            return False

        graph_nodes = getattr(self, "_graph_nodes", set())
        edges: dict[str, set[str]] = {}
        earlier: list[str] = []
        for stage in self.stages:
            if stage.needs or stage.name in graph_nodes:
                edges[stage.name] = set(stage.needs)
            else:
                edges[stage.name] = {
                    prior for prior in earlier if not depends_on(prior, stage.name)
                }
            earlier.append(stage.name)
        return edges

    def components(self) -> list[list[str]]:
        """Nodes grouped into strongly connected components, in dependency order.

        A component of one node with no self-edge runs once. Any larger component, or a
        self-edge, is a cycle: its nodes repeat together. Condensing to components is
        what lets a cyclic graph be scheduled at all, since the cycle becomes a single
        unit with a well-defined place in the order.
        """
        edges = self.graph()
        order = [st.name for st in self.stages]
        index: dict[str, int] = {}
        low: dict[str, int] = {}
        on_stack: dict[str, bool] = {}
        stack: list[str] = []
        found: list[list[str]] = []
        counter = [0]

        def strong_connect(node: str) -> None:
            index[node] = low[node] = counter[0]
            counter[0] += 1
            stack.append(node)
            on_stack[node] = True
            for dep in sorted(edges.get(node, ())):
                if dep not in index:
                    strong_connect(dep)
                    low[node] = min(low[node], low[dep])
                elif on_stack.get(dep):
                    low[node] = min(low[node], index[dep])
            if low[node] == index[node]:
                group = []
                while True:
                    top = stack.pop()
                    on_stack[top] = False
                    group.append(top)
                    if top == node:
                        break
                found.append(group)

        for name in order:
            if name not in index:
                strong_connect(name)

        # Tarjan returns dependencies first. Preserve declaration order inside cycles.
        rank = {name: i for i, name in enumerate(order)}
        return [sorted(group, key=lambda n: rank[n]) for group in found]

    def is_cyclic_component(self, group: list[str]) -> bool:
        if len(group) > 1:
            return True
        only = group[0]
        return only in self.graph().get(only, set())

    def cycles(self) -> list[list[str]]:
        """Every cyclic component. Empty means the graph is acyclic."""
        return [g for g in self.components() if self.is_cyclic_component(g)]

    def repeat_limit(self, group: list[str]) -> int:
        """How many times a cyclic component may repeat."""
        by_name = {st.name: st for st in self.stages}
        caps = [
            by_name[n].max_iterations
            for n in group
            if isinstance(by_name.get(n), Step) and by_name[n].max_iterations
        ]
        return min(caps) if caps else self.max_iterations

    def stop_conditions(self, group: list[str]) -> list[tuple[str, Condition]]:
        by_name = {st.name: st for st in self.stages}
        return [
            (n, by_name[n].until)
            for n in group
            if isinstance(by_name.get(n), Step) and by_name[n].until is not None
        ]

    def levels(self) -> list[list[str]]:
        """Waves of components that can run together.

        Components with no outstanding dependency on each other share a wave. A cyclic
        component takes a wave of its own, because its nodes repeat as a unit.
        """
        comps = self.components()
        edges = self.graph()
        place = {n: i for i, g in enumerate(comps) for n in g}

        comp_deps: list[set[int]] = []
        for i, group in enumerate(comps):
            comp_deps.append(
                {place[d] for n in group for d in edges.get(n, ()) if place[d] != i}
            )

        waves: list[list[str]] = []
        done: set[int] = set()
        while len(done) < len(comps):
            ready = [i for i in range(len(comps)) if i not in done and comp_deps[i] <= done]
            if not ready:                                    # pragma: no cover
                raise ValueError("could not order components; this is a bug")
            cyclic = [i for i in ready if self.is_cyclic_component(comps[i])]
            if cyclic:
                waves.append(comps[cyclic[0]])
                done.add(cyclic[0])
            else:
                waves.append(sorted(n for i in ready for n in comps[i]))
                done.update(ready)
        return waves

    def warnings(self) -> list[str]:
        """Things worth saying before anything runs."""
        notes: list[str] = []
        for group in self.cycles():
            limit = self.repeat_limit(group)
            stops = self.stop_conditions(group)
            note = (
                f"cycle {' -> '.join(group)} -> {group[0]} repeats, at most "
                f"{limit} time(s)."
            )
            if stops:
                note += " Stops early when " + ", ".join(n for n, _ in stops) + " is satisfied."
            else:
                note += (
                    " No stop condition, so it will always run all "
                    f"{limit} rounds. Add `until` to a node in the cycle to end it sooner."
                )
            notes.append(note)
        return notes

    @classmethod
    def from_yaml(cls, path: str | Path) -> Workflow:
        p = Path(path).expanduser()
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw, base_dir=p.parent)

    @staticmethod
    def _resolve_task_file(entry: dict[str, Any], base_dir: Path | None) -> dict[str, Any]:
        """Fold a step's `task_file` into its `task`.

        Paths are relative to the workflow file, not the working directory, so a
        workflow keeps working when it is run from somewhere else.
        """
        name = entry.get("name", "?")
        ref = entry.get("task_file")
        if ref and entry.get("task"):
            raise ValueError(f"step {name!r}: set either `task` or `task_file`, not both")
        refs = ([ref] if ref else []) + list(entry.get("context_files", []))
        parts: list[str] = []
        for item in refs:
            path = Path(item).expanduser()
            if not path.is_absolute() and base_dir is not None:
                path = base_dir / path
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(
                    f"step {name!r}: cannot read prompt file {str(path)!r}: {exc}"
                ) from exc
            if not text.strip():
                raise ValueError(f"step {name!r}: prompt file {str(path)!r} is empty")
            parts.append(text)
        if parts:
            entry["task"] = "\n\n---\n\n".join(parts)
        return entry

    @classmethod
    def _load_step(cls, data: dict[str, Any], base: Path | None) -> Step:
        return Step(**cls._resolve_task_file(dict(data), base))

    @classmethod
    def _load_loop(cls, data: dict[str, Any], base: Path | None) -> Loop:
        body = dict(data)
        body["steps"] = [cls._load_step(step, base) for step in body.get("steps", []) or []]
        return Loop(**body)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], base_dir: str | Path | None = None) -> Workflow:
        """Accepts two equivalent spellings, and a mix of them.

        `stages:` is an ordered list, where a node without `needs` waits for everything
        above it. `graph:` is a mapping of name to node, with no implicit ordering at
        all, for when the dependencies are the point. Nodes from `graph:` are appended
        after any `stages:`, and `needs` may point in either direction.
        """
        data = dict(raw)
        base = Path(base_dir) if base_dir is not None else None
        stages: list[Step | Loop] = []

        for entry in data.pop("stages", []) or []:
            if "loop" in entry:
                body = dict(entry["loop"])
                body.setdefault("name", entry.get("name", f"loop{len(stages)}"))
                stages.append(cls._load_loop(body, base))
            else:
                stages.append(cls._load_step(entry, base))

        for name, body in (data.pop("graph", {}) or {}).items():
            body = dict(body or {})
            body["name"] = name
            body.setdefault("needs", [])
            if "loop" in body:
                inner = dict(body.pop("loop"))
                inner["name"] = name
                inner["needs"] = body["needs"]
                stages.append(cls._load_loop(inner, base))
            else:
                stages.append(cls._load_step(body, base))
            data.setdefault("_explicit", set())
            data["_explicit"].add(name)

        explicit = data.pop("_explicit", set())
        wf = cls(stages=stages, **data)
        wf._graph_nodes = explicit
        return wf


Predicate = Callable[[dict[str, JobResult]], bool]


class StageOutcome(BaseModel):
    stage: str
    iterations: int = 1
    job_ids: list[str] = Field(default_factory=list)
    stopped_early: bool = False


class WorkflowRunner:
    """Executes a Workflow against a Fleet, one stage at a time.

    Stages run in declaration order and each waits for the ones it needs, which keeps
    the execution order the same as the reading order. Parallelism is explicit, inside
    a fan-out step, rather than implied by the graph.
    """

    def __init__(self, fleet, workflow: Workflow, *, predicates: dict[str, Predicate] | None = None,
                 prior_run: str | None = None, resume: bool = False):
        self.fleet = fleet
        self.workflow = workflow
        self.predicates = predicates or {}
        self.results: dict[str, JobResult] = {}
        self.outcomes: list[StageOutcome] = []
        self._worktree_tip: str | None = None
        self._worktree_tip_run: str | None = None
        self._completed: set[str] = set()
        self._prior_run: str | None = prior_run
        self._restored_outcomes: dict[str, StageOutcome] = {}
        self._actor_sessions: dict[str, str] = {}
        self._actors_started: set[str] = set()
        self._fingerprint = hashlib.sha256(json.dumps(
            workflow.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        if prior_run:
            self._restore(prior_run, resume=resume)

    def _restore(self, prior_run: str, *, resume: bool) -> None:
        events = self.fleet.ledger.events(
            run_id=prior_run, types=["workflow.checkpoint"], limit=100000
        )
        matches = [e for e in events if e.payload.get("fingerprint") == self._fingerprint]
        if not matches:
            if resume:
                # Resuming skips completed stages, so a changed definition would leave
                # this run half-built from a DAG that no longer exists.
                raise ValueError(
                    f"run {prior_run!r} has no compatible checkpoint for workflow "
                    f"{self.workflow.name!r}"
                )
            # `--from-run` re-executes everything and only inherits files, so a revised
            # workflow is the normal case: building on a previous attempt usually means
            # having changed the prompts in light of it. Take the prior run's artifacts
            # and none of its step state.
            self.fleet.ledger.append(
                "workflow.restored",
                {"from_run": prior_run, "resume": False, "completed": [],
                 "definition_changed": True},
                run_id=self.fleet.run_id,
            )
            return
        state = matches[-1].payload
        self.results = {
            name: JobResult.model_validate(value)
            for name, value in state.get("results", {}).items()
        }
        self._completed = set(state.get("completed", [])) if resume else set()
        self._restored_outcomes = {
            name: StageOutcome.model_validate(value)
            for name, value in state.get("outcomes", {}).items()
        } if resume else {}
        self._worktree_tip = state.get("worktree_tip")
        self._worktree_tip_run = state.get("worktree_tip_run") or prior_run
        self._actor_sessions = dict(state.get("actor_sessions", {})) if resume else {}
        self._actors_started = set(state.get("actors_started", [])) if resume else set()
        self.fleet.ledger.append(
            "workflow.restored",
            {"from_run": prior_run, "resume": resume,
             "completed": sorted(self._completed)},
            run_id=self.fleet.run_id,
        )

    def _checkpoint(self, completed: set[str], outcomes: dict[str, StageOutcome]) -> None:
        self.fleet.ledger.append(
            "workflow.checkpoint",
            {
                "workflow": self.workflow.name,
                "fingerprint": self._fingerprint,
                "completed": sorted(completed),
                "results": {k: v.model_dump(mode="json") for k, v in self.results.items()},
                "outcomes": {k: v.model_dump(mode="json") for k, v in outcomes.items()},
                "worktree_tip": self._worktree_tip,
                "worktree_tip_run": self._worktree_tip_run,
                "actor_sessions": dict(self._actor_sessions),
                "actors_started": sorted(self._actors_started),
            },
            run_id=self.fleet.run_id,
        )

    def _context(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        steps = {
            name: {"output": r.output, "state": r.state.value, "job_id": r.job_id}
            for name, r in self.results.items()
        }
        return {"steps": steps, "workflow": self.workflow.name, **(extra or {})}

    def _stage_mounts(self) -> list[Mount]:
        """Every finished stage's `/results`, mounted read-only at `/inputs/<stage>`.

        A stage's `/results` is private to its own container, so without this the only
        thing that crosses a stage boundary is the final message -- and a stage that
        answers "the write-up is in /results/findings.md" hands the next one a path that
        resolves to an empty directory there.

        The naming mirrors templating: whatever is addressable as
        `{{ steps.X.output }}` is readable as `/inputs/X/`. Read-only, so a later stage
        cannot rewrite the evidence it is reviewing.
        """
        mounts = []
        for name, result in self.results.items():
            # Each result carries the directory it actually wrote to, which is what
            # makes a restored result from an earlier run resolve correctly here.
            if result.results_dir and Path(result.results_dir).is_dir():
                mounts.append(
                    Mount(source=result.results_dir, target=f"/inputs/{name}", mode="ro")
                )
        return mounts

    def _previous_run_mounts(self) -> list[Mount]:
        """The run this one builds on, read-only: whole at `/previous-results`, and
        stage by stage at `/previous/<stage>`.

        The per-stage view is what makes a prior run usable rather than merely present.
        A researcher starting a second attempt wants last time's `findings.md` *and* the
        judge's `review.md`, and it should not have to open `job_4319620b5f63/` to find
        out which one that is.
        """
        if not self._prior_run:
            return []
        prior_dir = runlayout.run_dir_for(self.fleet.config.results_path, self._prior_run)
        if prior_dir is None:
            return []

        mounts = [Mount(source=str(prior_dir), target="/previous-results", mode="ro")]
        for stage, path in runlayout.stage_dirs(prior_dir, self._prior_job_names()).items():
            mounts.append(Mount(source=str(path), target=f"/previous/{stage}", mode="ro"))
        return mounts

    def _prior_job_names(self) -> dict[str, str]:
        """job id -> stage name for the prior run, for runs whose directories are ids."""
        try:
            jobs = self.fleet.ledger.jobs(run_id=self._prior_run)
        except Exception:
            return {}
        return {
            job_id: name
            for job in jobs
            if (job_id := job.get("job_id")) and (name := job.get("name"))
        }

    def _submit(self, step: Step, label: str, extra_ctx: dict[str, Any]) -> tuple[list[str], str | None]:
        """Submit a step and return its job ids and optional worktree chain label."""
        ctx = self._context(extra_ctx)
        model = step.model or self.workflow.model
        effort = step.effort or self.workflow.effort
        actor = self.workflow.actors.get(step.actor) if step.actor else None
        if actor is not None:
            model = actor.model or model
            effort = actor.effort or effort
            if actor.persistent and step.actor in self._actors_started \
                    and step.actor not in self._actor_sessions:
                raise RuntimeError(
                    f"persistent actor {step.actor!r} produced no resumable session id"
                )
        gpus = self.workflow.gpus if step.gpus == 1.0 and self.workflow.gpus is not None else step.gpus
        isolate = step.isolate if step.isolate is not None else self.workflow.isolate
        if isolate is None:
            isolate = self.fleet.config.isolate_agents
        items = list(step.for_each) if step.for_each else list(range(step.copies))

        chainable = len(items) == 1 and isolate
        base = self._worktree_tip if chainable else None
        base_run = self._worktree_tip_run if chainable else None

        records = []
        inherited_mounts = self._stage_mounts() + self._previous_run_mounts()
        for index, item in enumerate(items):
            item_ctx = {**ctx, "item": item, "index": index}
            name = label if len(items) == 1 else f"{label}-{index}"
            labels = {
                "workflow": self.workflow.name,
                "stage": step.name,
                "attempt": str(extra_ctx.get("iteration", 1)),
            }
            if len(items) > 1:
                labels["copy"] = str(index)
            if step.kind is JobKind.AGENT:
                records += self.fleet.run_agents(
                    render(step.task, item_ctx),
                    n=1, name_prefix=name, labels=labels,
                    model=model, backend=actor.backend if actor else None,
                    effort=effort, system_prompt=actor.system_prompt if actor else None,
                    session_id=(self._actor_sessions.get(step.actor)
                                if actor is not None and actor.persistent else None),
                    gpus=gpus,
                    timeout_s=step.timeout_s, isolate=isolate, worktree_base=base,
                    worktree_base_run_id=base_run,
                    mounts=inherited_mounts,
                )
            else:
                records.append(
                    self.fleet.run_command(
                        [render(part, item_ctx) for part in step.command],
                        name=name, gpus=gpus, timeout_s=step.timeout_s,
                        isolate=isolate, worktree_base=base,
                        worktree_base_run_id=base_run, mounts=inherited_mounts, labels=labels,
                    )
                )
        return [r.spec.id for r in records], (label if chainable else None)

    def _advance_tip(self, chain_label: str | None, result_key: str) -> None:
        """Advance the isolated worktree chain after a successful single-job step."""
        if chain_label is None:
            return
        result = self.results.get(result_key)
        if result is not None and result.state == JobState.SUCCEEDED:
            old_tip = self._worktree_tip
            old_tip_run = self._worktree_tip_run
            self._worktree_tip = chain_label
            self._worktree_tip_run = self.fleet.run_id
            if old_tip is not None and old_tip_run in (None, self.fleet.run_id):
                self.fleet.drop_worktree(old_tip)

    def _outcomes_succeeded(self, names: list[str], outcomes: dict[str, StageOutcome]) -> bool:
        ids = [job_id for name in names for job_id in outcomes[name].job_ids]
        return bool(ids) and all(
            (record := self.fleet.scheduler._jobs.get(job_id)) is not None
            and record.result is not None
            and record.result.state is JobState.SUCCEEDED
            for job_id in ids
        )

    def _record(self, step_name: str, job_ids: list[str], report) -> None:
        """Record a step's results from a finished report.

        A fan-out records the first job under the bare step name so later templates can
        refer to it simply, plus each copy under `name-N`.
        """
        for index, job_id in enumerate(job_ids):
            result = report.results.get(job_id)
            if result is None:
                continue
            if index == 0:
                self.results[step_name] = result
            self.results[f"{step_name}-{index}"] = result
        step = self._step_named(step_name)
        if step is not None and step.actor is not None:
            actor = self.workflow.actors[step.actor]
            first = self.results.get(step_name)
            self._actors_started.add(step.actor)
            if actor.persistent and first is not None and first.session_id:
                self._actor_sessions[step.actor] = first.session_id

    def _step_named(self, name: str) -> Step | None:
        for stage in self.workflow.stages:
            if isinstance(stage, Step) and stage.name == name:
                return stage
            if isinstance(stage, Loop):
                for step in stage.steps:
                    if step.name == name:
                        return step
        return None

    def _collect(self, step_name: str, job_ids: list[str]) -> None:
        self._record(step_name, job_ids, self.fleet.wait())

    def run(self) -> list[StageOutcome]:
        """Execute the dependency graph, component by component.

        Independent nodes share a wave and run together. A cyclic component repeats its
        nodes in declaration order until one of their `until` conditions holds or the
        repeat limit is reached, so a cycle is how you express "keep going until".
        """
        for note in self.workflow.warnings():
            warnings.warn(f"{self.workflow.name}: {note}", stacklevel=2)
            self.fleet.ledger.append(
                "workflow.warning", {"workflow": self.workflow.name, "note": note},
                run_id=self.fleet.run_id,
            )

        by_name: dict[str, Step | Loop] = {st.name: st for st in self.workflow.stages}
        outcomes: dict[str, StageOutcome] = dict(self._restored_outcomes)
        completed = set(self._completed)

        cyclic = {tuple(group) for group in self.workflow.cycles()}

        for wave in self.workflow.levels():
            if set(wave) <= completed:
                continue
            if tuple(wave) in cyclic:
                outcomes.update(self._run_cycle(wave, by_name))
                if not self._outcomes_succeeded(wave, outcomes):
                    self._checkpoint(completed, outcomes)
                    break
                completed.update(wave)
                self._checkpoint(completed, outcomes)
                continue

            stages = [by_name[name] for name in wave]
            steps = [st for st in stages if isinstance(st, Step)]
            loops = [st for st in stages if isinstance(st, Loop)]

            persistent_actors = [
                step.actor for step in steps
                if step.actor is not None and self.workflow.actors[step.actor].persistent
            ]
            duplicates = sorted({name for name in persistent_actors
                                 if persistent_actors.count(name) > 1})
            if duplicates:
                raise ValueError(
                    "persistent actors cannot run concurrent stages in one wave: "
                    + ", ".join(duplicates)
                )

            submitted = {st.name: self._submit(st, st.name, {"iteration": 1}) for st in steps}
            if submitted:
                report = self.fleet.wait()
                for name, (job_ids, _chain_label) in submitted.items():
                    self._record(name, job_ids, report)
                    outcomes[name] = StageOutcome(stage=name, job_ids=job_ids)
                if len(steps) == 1:
                    only = steps[0].name
                    self._advance_tip(submitted[only][1], only)

            for loop in loops:
                outcomes[loop.name] = self._run_loop(loop)

            if not self._outcomes_succeeded(wave, outcomes):
                self._checkpoint(completed, outcomes)
                break
            completed.update(wave)
            self._checkpoint(completed, outcomes)

        self.outcomes = [outcomes[st.name] for st in self.workflow.stages if st.name in outcomes]
        return self.outcomes

    def _run_cycle(
        self, group: list[str], by_name: dict[str, Step | Loop]
    ) -> dict[str, StageOutcome]:
        """Repeat a cyclic component until a stop condition holds or the limit is hit."""
        limit = self.workflow.repeat_limit(group)
        stops = self.workflow.stop_conditions(group)
        outcomes = {name: StageOutcome(stage=name, iterations=0) for name in group}

        for iteration in range(1, limit + 1):
            for name in group:
                stage = by_name[name]
                outcomes[name].iterations = iteration
                if isinstance(stage, Loop):
                    inner = self._run_loop(stage)
                    outcomes[name].job_ids += inner.job_ids
                    continue
                label = f"{name}-{iteration}" if iteration > 1 else name
                job_ids, chain_label = self._submit(stage, label, {"iteration": iteration})
                outcomes[name].job_ids += job_ids
                self._collect(name, job_ids)
                self._advance_tip(chain_label, name)

            met = [n for n, cond in stops if cond.met(self.results, default_step=n)]
            predicate = self.predicates.get(group[0])
            if met or (predicate is not None and predicate(self.results)):
                for outcome in outcomes.values():
                    outcome.stopped_early = True
                break

        return outcomes

    def _run_loop(self, loop: Loop) -> StageOutcome:
        outcome = StageOutcome(stage=loop.name, iterations=0)
        predicate = self.predicates.get(loop.name)

        for iteration in range(1, loop.max_iterations + 1):
            outcome.iterations = iteration
            for step in loop.steps:
                label = f"{loop.name}-{step.name}-{iteration}"
                job_ids, chain_label = self._submit(step, label, {"iteration": iteration})
                outcome.job_ids += job_ids
                self._collect(step.name, job_ids)
                self._advance_tip(chain_label, step.name)

            if predicate is not None and predicate(self.results):
                outcome.stopped_early = True
                break
            if loop.until is not None and loop.until.met(self.results):
                outcome.stopped_early = True
                break

        return outcome
