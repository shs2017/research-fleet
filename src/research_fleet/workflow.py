"""Workflows: multi-step research pipelines, in YAML or in Python.

A workflow is an ordered list of stages. Each stage is one of:

  * a **step**, which runs an agent or a command,
  * a **loop**, which repeats its steps until a condition holds or it runs out of
    iterations (a coder/reviewer cycle is the canonical case),
  * a **fan-out**, which is a step with `for_each` or `copies`, run in parallel.

Stages form a graph, and the rule for reading one is deliberately simple:

  * a stage with `needs` runs once those stages are done, so siblings sharing a
    dependency run in parallel;
  * a stage without `needs` runs after everything declared before it, which keeps a
    plain list sequential and makes a bare stage a natural barrier.

That means a flat list behaves like a sequence, adding `needs` opts into parallelism,
and the two mix without a mode switch. Stages read earlier results through `{{ }}`
templating, so a reviewer can be handed exactly what the coder said.

Two deliberate limits keep this predictable. Templating is plain substitution, not an
expression language, so a prompt cannot compute. Loop conditions are declarative
(`output_contains`, `succeeded`) rather than arbitrary code, so reading the YAML tells
you when the loop stops. Python callers who want real logic pass a callable instead.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Callable

import yaml
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from .spec import JobKind, JobResult

# ----------------------------------------------------------------- templating

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


# ------------------------------------------------------------------ conditions


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


# ---------------------------------------------------------------------- stages


class Step(BaseModel):
    name: str
    kind: JobKind = JobKind.AGENT
    task: str = ""                          # agent steps
    command: list[str] = Field(default_factory=list)   # command steps

    model: str | None = None
    effort: str | None = None
    gpus: float = 1.0
    timeout_s: int = 3600
    isolate: bool | None = None
    needs: list[str] = Field(default_factory=list)

    # When this node is part of a cycle, these govern how long the cycle repeats.
    until: Condition | None = Field(
        None, description="Stop repeating once this holds of the node's own output."
    )
    max_iterations: int | None = Field(None, ge=1, description="Cap on repeats of this node's cycle.")

    # Fan-out: run the same step many times, in parallel.
    for_each: list[Any] = Field(default_factory=list, description="One run per item; use {{ item }}.")
    copies: int = Field(1, ge=1, description="Run this many identical copies in parallel.")

    @model_validator(mode="after")
    def _check(self) -> Step:
        if self.kind is JobKind.AGENT and not self.task:
            raise ValueError(f"step {self.name!r}: agent steps need a `task`")
        if self.kind is JobKind.COMMAND and not self.command:
            raise ValueError(f"step {self.name!r}: command steps need a `command`")
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


class Workflow(BaseModel):
    model_config = {"extra": "forbid"}

    name: str = "workflow"
    description: str = ""
    stages: list[Step | Loop] = Field(default_factory=list)

    # Defaults applied to any step that does not set them itself.
    model: str | None = None
    effort: str | None = None
    gpus: float | None = None
    isolate: bool | None = None
    max_iterations: int = Field(
        3, ge=1, description="Default cap on how many times a cycle repeats."
    )

    # Names that came from a `graph:` block, so they keep exactly the edges they state.
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
        return self

    # ------------------------------------------------------------------ graph

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

        # Tarjan emits components in reverse topological order of the condensation, so
        # dependencies come first already. Sort within a component by declaration order
        # so a cycle repeats in the order it reads.
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
        raw = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Workflow:
        """Accepts two equivalent spellings, and a mix of them.

        `stages:` is an ordered list, where a node without `needs` waits for everything
        above it. `graph:` is a mapping of name to node, with no implicit ordering at
        all, for when the dependencies are the point. Nodes from `graph:` are appended
        after any `stages:`, and `needs` may point in either direction.
        """
        data = dict(raw)
        stages: list[Any] = []

        for entry in data.pop("stages", []) or []:
            if "loop" in entry:
                body = dict(entry["loop"])
                body.setdefault("name", entry.get("name", f"loop{len(stages)}"))
                stages.append(Loop(**body))
            else:
                stages.append(Step(**entry))

        for name, body in (data.pop("graph", {}) or {}).items():
            body = dict(body or {})
            body["name"] = name
            # A graph node is explicit about its edges, so it never gets an implicit one.
            body.setdefault("needs", [])
            if "loop" in body:
                inner = dict(body.pop("loop"))
                inner["name"] = name
                inner["needs"] = body["needs"]
                stages.append(Loop(**inner))
            else:
                stages.append(Step(**body))
            data.setdefault("_explicit", set())
            data["_explicit"].add(name)

        explicit = data.pop("_explicit", set())
        wf = cls(stages=stages, **data)
        wf._graph_nodes = explicit
        return wf


# ---------------------------------------------------------------------- runner

# A Python caller can pass a predicate instead of a declarative Condition.
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

    def __init__(self, fleet, workflow: Workflow, *, predicates: dict[str, Predicate] | None = None):
        self.fleet = fleet
        self.workflow = workflow
        self.predicates = predicates or {}
        self.results: dict[str, JobResult] = {}
        self.outcomes: list[StageOutcome] = []

    # --------------------------------------------------------------- helpers

    def _context(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        steps = {
            name: {"output": r.output, "state": r.state.value, "job_id": r.job_id}
            for name, r in self.results.items()
        }
        return {"steps": steps, "workflow": self.workflow.name, **(extra or {})}

    def _defaults(self, step: Step) -> dict[str, Any]:
        wf = self.workflow
        return {
            "model": step.model or wf.model,
            "effort": step.effort or wf.effort,
            "gpus": step.gpus if step.gpus != 1.0 else (wf.gpus if wf.gpus is not None else step.gpus),
            "isolate": step.isolate if step.isolate is not None else wf.isolate,
        }

    def _submit(self, step: Step, label: str, extra_ctx: dict[str, Any]) -> list[str]:
        """Submit one step, expanding a fan-out into several jobs."""
        ctx = self._context(extra_ctx)
        opts = self._defaults(step)

        items: list[Any]
        if step.for_each:
            items = list(step.for_each)
        else:
            items = list(range(step.copies))

        records = []
        for index, item in enumerate(items):
            item_ctx = {**ctx, "item": item, "index": index}
            name = label if len(items) == 1 else f"{label}-{index}"
            # The logical stage and the attempt travel as labels, so a repeated node
            # stays one row per attempt in the usage table rather than being merged.
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
                    model=opts["model"], effort=opts["effort"], gpus=opts["gpus"],
                    timeout_s=step.timeout_s, isolate=opts["isolate"],
                )
            else:
                records.append(
                    self.fleet.run_command(
                        [render(part, item_ctx) for part in step.command],
                        name=name, gpus=opts["gpus"], timeout_s=step.timeout_s,
                        isolate=opts["isolate"], labels=labels,
                    )
                )
        return [r.spec.id for r in records]

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

    def _collect(self, step_name: str, job_ids: list[str]) -> None:
        self._record(step_name, job_ids, self.fleet.wait())

    # ------------------------------------------------------------------- run

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
        outcomes: dict[str, StageOutcome] = {}

        # `levels` gives a cyclic component a wave to itself, so a wave repeats only
        # when it *is* that component. A wave of several independent nodes is not a
        # cycle, however many nodes it holds.
        cyclic = {tuple(group) for group in self.workflow.cycles()}

        for wave in self.workflow.levels():
            if tuple(wave) in cyclic:
                outcomes.update(self._run_cycle(wave, by_name))
                continue

            stages = [by_name[name] for name in wave]
            steps = [st for st in stages if isinstance(st, Step)]
            loops = [st for st in stages if isinstance(st, Loop)]

            # Submit every plain node in the wave before waiting, so independent work
            # genuinely overlaps.
            submitted = {st.name: self._submit(st, st.name, {"iteration": 1}) for st in steps}
            if submitted:
                report = self.fleet.wait()
                for name, job_ids in submitted.items():
                    self._record(name, job_ids, report)
                    outcomes[name] = StageOutcome(stage=name, job_ids=job_ids)

            for loop in loops:
                outcomes[loop.name] = self._run_loop(loop)

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
                job_ids = self._submit(stage, label, {"iteration": iteration})
                outcomes[name].job_ids += job_ids
                # Recorded under the bare name, so `{{ steps.review.output }}` always
                # means the current round.
                self._collect(name, job_ids)

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
                job_ids = self._submit(step, label, {"iteration": iteration})
                outcome.job_ids += job_ids
                # Collected under the plain step name so `{{ steps.review.output }}`
                # always refers to the current iteration.
                self._collect(step.name, job_ids)

            if predicate is not None and predicate(self.results):
                outcome.stopped_early = True
                break
            if loop.until is not None and loop.until.met(self.results):
                outcome.stopped_early = True
                break

        return outcome
