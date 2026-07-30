"""Workflows: multi-step research pipelines, in YAML or in Python.

A workflow is an ordered list of stages. Each stage is one of:

  * a **step**, which runs an agent or a command,
  * a **loop**, which repeats its steps until a condition holds or it runs out of
    iterations (a coder/reviewer cycle is the canonical case),
  * a **fan-out**, which is a step with `for_each` or `copies`, run in parallel.

Steps declare `needs` to run after earlier stages, and read earlier results through
`{{ }}` templating, so a reviewer can be handed exactly what the coder said.

Two deliberate limits keep this predictable. Templating is plain substitution, not an
expression language, so a prompt cannot compute. Loop conditions are declarative
(`output_contains`, `succeeded`) rather than arbitrary code, so reading the YAML tells
you when the loop stops. Python callers who want real logic pass a callable instead.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

import yaml
from pydantic import BaseModel, Field, model_validator

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
    """When a loop should stop. Every field given must hold."""

    step: str = Field(..., description="Which step's result to inspect.")
    output_contains: str | None = None
    output_not_contains: str | None = None
    succeeded: bool | None = None

    def met(self, results: dict[str, JobResult]) -> bool:
        result = results.get(self.step)
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
    name: str = "workflow"
    description: str = ""
    stages: list[Step | Loop] = Field(default_factory=list)

    # Defaults applied to any step that does not set them itself.
    model: str | None = None
    effort: str | None = None
    gpus: float | None = None
    isolate: bool | None = None

    @model_validator(mode="after")
    def _check(self) -> Workflow:
        if not self.stages:
            raise ValueError("a workflow needs at least one stage")
        names = [st.name for st in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("stage names must be unique")
        known: set[str] = set()
        for stage in self.stages:
            for need in stage.needs:
                if need not in known:
                    raise ValueError(f"stage {stage.name!r} needs {need!r}, which is not defined above it")
            known.add(stage.name)
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> Workflow:
        raw = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Workflow:
        """Accepts `stages:` entries that are either steps or `loop:` blocks."""
        data = dict(raw)
        stages: list[Any] = []
        for entry in data.pop("stages", []) or []:
            if "loop" in entry:
                body = dict(entry["loop"])
                body.setdefault("name", entry.get("name", f"loop{len(stages)}"))
                stages.append(Loop(**body))
            else:
                stages.append(Step(**entry))
        return cls(stages=stages, **data)


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
            if step.kind is JobKind.AGENT:
                records += self.fleet.run_agents(
                    render(step.task, item_ctx),
                    n=1, name_prefix=name,
                    model=opts["model"], effort=opts["effort"], gpus=opts["gpus"],
                    timeout_s=step.timeout_s, isolate=opts["isolate"],
                )
            else:
                records.append(
                    self.fleet.run_command(
                        [render(part, item_ctx) for part in step.command],
                        name=name, gpus=opts["gpus"], timeout_s=step.timeout_s,
                        isolate=opts["isolate"],
                    )
                )
        return [r.spec.id for r in records]

    def _collect(self, step_name: str, job_ids: list[str]) -> None:
        """Wait for the submitted jobs and record the step's result.

        A fan-out records the first job under the bare step name so later templates can
        refer to it simply, plus each copy under `name-N`.
        """
        report = self.fleet.wait()
        for index, job_id in enumerate(job_ids):
            result = report.results.get(job_id)
            if result is None:
                continue
            if index == 0:
                self.results[step_name] = result
            self.results[f"{step_name}-{index}"] = result

    # ------------------------------------------------------------------- run

    def run(self) -> list[StageOutcome]:
        for stage in self.workflow.stages:
            if isinstance(stage, Loop):
                self.outcomes.append(self._run_loop(stage))
            else:
                job_ids = self._submit(stage, stage.name, {"iteration": 1})
                self._collect(stage.name, job_ids)
                self.outcomes.append(StageOutcome(stage=stage.name, job_ids=job_ids))
        return self.outcomes

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
