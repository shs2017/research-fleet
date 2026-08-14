"""Workflows: parsing, templating, loop control, and fan-out.

The stub executor returns scripted output per step name, which is what lets a
coder/reviewer loop be tested deterministically: the reviewer objects once, then
approves, and the loop must stop on its own.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from research_fleet import Condition, Fleet, Loop, Step, Workflow
from research_fleet.spec import JobKind, JobResult, JobState
from research_fleet.workflow import render


class ScriptedExecutor:
    """Returns queued output for each step, keyed by a substring of the job name."""

    kind = "scripted"

    def __init__(self, script: dict[str, list[str]] | None = None, delay: float = 0.0):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.delay = delay
        self.calls: list[str] = []          # job names, in execution order
        self.tasks: list[str] = []          # the prompt each agent received
        self.systems: list[str] = []        # the system prompt each agent received
        self.isolated: list[bool] = []
        self.spans: dict[str, tuple[float, float]] = {}   # name -> (start, end)
        self.sessions: list[str | None] = []
        self._lock = threading.Lock()

    def _output_for(self, name: str) -> str:
        for key, queue in self.script.items():
            if key in name:
                if len(queue) > 1:
                    return queue.pop(0)
                return queue[0] if queue else ""
        return f"output of {name}"

    def available_gpus(self) -> list[str]:
        return ["GPU-0", "GPU-1", "GPU-2", "GPU-3"]

    def run(self, spec, *, argv, env, placement, policy, on_line) -> JobResult:
        began = time.time()
        with self._lock:
            self.calls.append(spec.name)
            self.isolated.append(spec.isolate)
            text = self._output_for(spec.name)
        if self.delay:
            time.sleep(self.delay)

        if spec.kind is JobKind.AGENT:
            prompt = argv[argv.index("-p") + 1] if "-p" in argv else ""
            self.tasks.append(prompt)
            flag = "--append-system-prompt"
            self.systems.append(argv[argv.index(flag) + 1] if flag in argv else "")
            resume = argv[argv.index("--resume") + 1] if "--resume" in argv else None
            self.sessions.append(resume)
            session_id = resume or f"session-{spec.name}"
            on_line("stdout", json.dumps({"type": "system", "session_id": session_id}))
            on_line("stdout", json.dumps({
                "type": "assistant",
                "message": {"model": "claude-sonnet-5", "content": [{"type": "text", "text": "working"}],
                            "usage": {"input_tokens": 100, "output_tokens": 10}},
            }))
            on_line("stdout", json.dumps({"type": "result", "subtype": "success", "result": text}))
        else:
            self.tasks.append(" ".join(argv))
            on_line("stdout", text)

        now = time.time()
        with self._lock:
            self.spans[spec.name] = (began, now)
        return JobResult(job_id=spec.id, state=JobState.SUCCEEDED, exit_code=0,
                         started_at=began, ended_at=now)

    def cancel(self, job_id: str) -> bool:
        return True

    def close(self) -> None:
        return None


def _fleet(tmp_path, executor, **overrides):
    fleet = Fleet(root=str(tmp_path / "state"), workspace=str(tmp_path),
                  executor={"kind": "dry-run"}, **overrides)
    fleet.executor = executor
    fleet.scheduler.executor = executor
    return fleet


# ---------------------------------------------------------------- templating


def test_render_substitutes_nested_values():
    ctx = {"steps": {"code": {"output": "did the thing"}}, "item": 7}
    assert render("saw: {{ steps.code.output }} / {{ item }}", ctx) == "saw: did the thing / 7"


def test_named_actors_resume_only_their_own_conversations(tmp_path):
    executor = ScriptedExecutor()
    fleet = _fleet(tmp_path, executor)
    workflow = Workflow.from_dict({
        "name": "alternating",
        "gpus": 0,
        "actors": {
            "researcher": {"persistent": True, "model": "claude-sonnet-5"},
            "judge": {"persistent": True, "model": "claude-sonnet-5"},
        },
        "stages": [
            {"name": "research_1", "actor": "researcher", "task": "first"},
            {"name": "judge_1", "actor": "judge", "task": "judge first"},
            {"name": "research_2", "actor": "researcher", "task": "adversarial"},
            {"name": "judge_2", "actor": "judge", "task": "judge second"},
        ],
    })
    try:
        fleet.run_workflow(workflow)
    finally:
        fleet.close()

    assert executor.tasks == ["first", "judge first", "adversarial", "judge second"]
    assert executor.sessions == [
        None,
        None,
        "session-research_1",
        "session-judge_1",
    ]


def test_nonpersistent_actor_always_starts_fresh(tmp_path):
    executor = ScriptedExecutor()
    fleet = _fleet(tmp_path, executor)
    workflow = Workflow.from_dict({
        "name": "fresh",
        "gpus": 0,
        "actors": {"researcher": {"persistent": False}},
        "stages": [
            {"name": "one", "actor": "researcher", "task": "first"},
            {"name": "two", "actor": "researcher", "task": "second"},
        ],
    })
    try:
        fleet.run_workflow(workflow)
    finally:
        fleet.close()
    assert executor.sessions == [None, None]


def test_unknown_actor_is_rejected():
    with pytest.raises(ValueError, match="unknown actor"):
        Workflow.from_dict({
            "actors": {"known": {}},
            "stages": [{"name": "step", "actor": "missing", "task": "work"}],
        })


def test_persistent_actor_cannot_run_two_parallel_stages(tmp_path):
    executor = ScriptedExecutor()
    fleet = _fleet(tmp_path, executor)
    workflow = Workflow.from_dict({
        "actors": {"researcher": {"persistent": True}},
        "graph": {
            "left": {"actor": "researcher", "task": "left"},
            "right": {"actor": "researcher", "task": "right"},
        },
    })
    try:
        with pytest.raises(ValueError, match="cannot run concurrent"):
            fleet.run_workflow(workflow)
    finally:
        fleet.close()


def test_render_leaves_unknown_placeholders_untouched():
    assert render("{{ steps.missing.output }} and {{ nope }}", {"steps": {}}) == \
        "{{ steps.missing.output }} and {{ nope }}"


def test_render_tolerates_whitespace_variants():
    assert render("{{item}} {{  item  }}", {"item": "x"}) == "x x"


# ---------------------------------------------------------------- conditions


def _result(output="", state=JobState.SUCCEEDED):
    return JobResult(job_id="j", state=state, output=output)


def test_condition_matches_on_output_case_insensitively():
    cond = Condition(step="review", output_contains="approved")
    assert cond.met({"review": _result("Looks good. APPROVED")})
    assert not cond.met({"review": _result("needs work")})


def test_condition_can_require_absence():
    cond = Condition(step="review", output_not_contains="BLOCKER")
    assert cond.met({"review": _result("all fine")})
    assert not cond.met({"review": _result("found a BLOCKER")})


def test_condition_can_require_success():
    cond = Condition(step="build", succeeded=True)
    assert cond.met({"build": _result(state=JobState.SUCCEEDED)})
    assert not cond.met({"build": _result(state=JobState.FAILED)})


def test_condition_is_unmet_when_the_step_has_not_run():
    assert not Condition(step="review", output_contains="x").met({})


# ------------------------------------------------------------------- parsing


def test_workflow_parses_steps_and_loops_from_yaml(tmp_path):
    path = tmp_path / "wf.yaml"
    path.write_text("""
name: demo
description: two stages
model: claude-sonnet-5
stages:
  - name: first
    task: do something
  - name: cycle
    loop:
      max_iterations: 2
      until:
        step: check
        output_contains: OK
      steps:
        - name: check
          task: check it
""")
    wf = Workflow.from_yaml(path)
    assert wf.name == "demo" and wf.model == "claude-sonnet-5"
    assert [s.name for s in wf.stages] == ["first", "cycle"]
    loop = wf.stages[1]
    assert isinstance(loop, Loop) and loop.max_iterations == 2
    assert loop.until.output_contains == "OK"


def test_command_steps_are_parsed(tmp_path):
    wf = Workflow.from_dict({
        "stages": [{"name": "test", "kind": "command", "command": ["pytest", "-q"]}]
    })
    assert wf.stages[0].kind is JobKind.COMMAND
    assert wf.stages[0].command == ["pytest", "-q"]


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"stages": []}, "at least one stage"),
        ({"stages": [{"name": "a", "kind": "agent"}]}, "need a `task`"),
        ({"stages": [{"name": "a", "kind": "command"}]}, "need a `command`"),
        ({"stages": [{"name": "a", "task": "t"}, {"name": "a", "task": "t"}]}, "unique"),
        ({"stages": [{"name": "a", "task": "t", "needs": ["nope"]}]}, "not a stage"),
        ({"stages": [{"name": "a", "task": "t", "for_each": [1], "copies": 2}]}, "not both"),
    ],
)
def test_invalid_workflows_are_rejected_with_a_reason(raw, message):
    with pytest.raises(ValueError, match=message):
        Workflow.from_dict(raw)


def test_a_loop_condition_must_name_one_of_its_own_steps():
    with pytest.raises(ValueError, match="is not one of its steps"):
        Workflow.from_dict({
            "stages": [{
                "name": "l",
                "loop": {"steps": [{"name": "a", "task": "t"}],
                         "until": {"step": "elsewhere", "output_contains": "x"}},
            }]
        })


def test_an_empty_loop_is_rejected():
    with pytest.raises(ValueError, match="at least one step"):
        Loop(name="l", steps=[])


# ------------------------------------------------------------------ execution


def test_stages_run_in_declaration_order(tmp_path):
    stub = ScriptedExecutor()
    fleet = _fleet(tmp_path, stub)
    try:
        report = fleet.run_workflow({
            "name": "ordered",
            "stages": [
                {"name": "one", "task": "a", "gpus": 0},
                {"name": "two", "task": "b", "gpus": 0},
                {"name": "three", "task": "c", "gpus": 0},
            ],
        })
        assert stub.calls == ["one", "two", "three"]
        assert [o.stage for o in report.outcomes] == ["one", "two", "three"]
        assert "workflow ordered" in report.summary()
    finally:
        fleet.close()


def test_a_step_can_read_an_earlier_steps_output(tmp_path):
    stub = ScriptedExecutor({"first": ["the answer is 42"]})
    fleet = _fleet(tmp_path, stub)
    try:
        fleet.run_workflow({
            "stages": [
                {"name": "first", "task": "compute", "gpus": 0},
                {"name": "second", "task": "given {{ steps.first.output }}, continue", "gpus": 0},
            ],
        })
        assert "the answer is 42" in stub.tasks[1]
    finally:
        fleet.close()


def test_a_loop_stops_as_soon_as_the_reviewer_approves(tmp_path):
    # Objects once, then approves: the loop must run exactly twice, not three times.
    stub = ScriptedExecutor({"review": ["needs work: add tests", "APPROVED"]})
    fleet = _fleet(tmp_path, stub)
    try:
        report = fleet.run_workflow({
            "stages": [{
                "name": "cycle",
                "loop": {
                    "max_iterations": 5,
                    "until": {"step": "review", "output_contains": "APPROVED"},
                    "steps": [
                        {"name": "implement", "task": "implement. {{ steps.review.output }}", "gpus": 0},
                        {"name": "review", "task": "review it", "gpus": 0},
                    ],
                },
            }],
        })
        outcome = report.outcomes[0]
        assert outcome.iterations == 2 and outcome.stopped_early
        assert len(stub.calls) == 4          # implement, review, implement, review

        # The second implement round must have been handed the reviewer's objection.
        second_implement = stub.tasks[2]
        assert "add tests" in second_implement
    finally:
        fleet.close()


def test_a_loop_gives_up_at_max_iterations(tmp_path):
    stub = ScriptedExecutor({"review": ["still not right"]})
    fleet = _fleet(tmp_path, stub)
    try:
        report = fleet.run_workflow({
            "stages": [{
                "name": "cycle",
                "loop": {
                    "max_iterations": 3,
                    "until": {"step": "review", "output_contains": "APPROVED"},
                    "steps": [{"name": "review", "task": "review", "gpus": 0}],
                },
            }],
        })
        outcome = report.outcomes[0]
        assert outcome.iterations == 3 and not outcome.stopped_early
        assert len(stub.calls) == 3
    finally:
        fleet.close()


def test_iteration_number_is_available_to_prompts(tmp_path):
    stub = ScriptedExecutor()
    fleet = _fleet(tmp_path, stub)
    try:
        fleet.run_workflow({
            "stages": [{
                "name": "cycle",
                "loop": {"max_iterations": 2,
                         "steps": [{"name": "s", "task": "attempt {{ iteration }}", "gpus": 0}]},
            }],
        })
        assert "attempt 1" in stub.tasks[0]
        assert "attempt 2" in stub.tasks[1]
    finally:
        fleet.close()


def test_a_python_predicate_can_stop_a_loop(tmp_path):
    stub = ScriptedExecutor({"s": ["0.42"]})
    fleet = _fleet(tmp_path, stub)
    try:
        # Stop once the reported metric parses below a threshold.
        def good_enough(results):
            try:
                return float(results["s"].output) < 0.5
            except (KeyError, ValueError):
                return False

        report = fleet.run_workflow(
            {"stages": [{"name": "cycle", "loop": {
                "max_iterations": 4, "steps": [{"name": "s", "task": "measure", "gpus": 0}]}}]},
            predicates={"cycle": good_enough},
        )
        assert report.outcomes[0].iterations == 1 and report.outcomes[0].stopped_early
    finally:
        fleet.close()


# -------------------------------------------------------------------- fan-out


def test_copies_run_the_same_step_in_parallel(tmp_path):
    stub = ScriptedExecutor()
    fleet = _fleet(tmp_path, stub)
    try:
        report = fleet.run_workflow({
            "stages": [{"name": "explore", "task": "try something", "copies": 3, "gpus": 0}],
        })
        assert sorted(stub.calls) == ["explore-0", "explore-1", "explore-2"]
        assert len(report.outcomes[0].job_ids) == 3
        # Both the bare name and the indexed names are addressable afterwards.
        assert "explore" in report.steps and "explore-2" in report.steps
    finally:
        fleet.close()


def test_for_each_substitutes_the_item(tmp_path):
    stub = ScriptedExecutor()
    fleet = _fleet(tmp_path, stub)
    try:
        fleet.run_workflow({
            "stages": [{"name": "ablate", "task": "run variant {{ item }}",
                        "for_each": ["rmsnorm", "layernorm"], "gpus": 0}],
        })
        assert any("variant rmsnorm" in t for t in stub.tasks)
        assert any("variant layernorm" in t for t in stub.tasks)
    finally:
        fleet.close()


# ------------------------------------------------------------------- defaults


def test_workflow_defaults_apply_unless_a_step_overrides_them(tmp_path):
    stub = ScriptedExecutor()
    fleet = _fleet(tmp_path, stub)
    try:
        fleet.run_workflow({
            "model": "claude-haiku-4-5",
            "isolate": True,
            "stages": [
                {"name": "cheap", "task": "t", "gpus": 0},
                {"name": "dear", "task": "t", "model": "claude-opus-5", "gpus": 0},
            ],
        })
        jobs = {r.spec.name: r.spec for r in fleet.scheduler._jobs.values()}
        assert jobs["cheap"].agent.model == "claude-haiku-4-5"
        assert jobs["dear"].agent.model == "claude-opus-5"
        assert all(stub.isolated), "isolate: true should reach every job"
    finally:
        fleet.close()


def test_command_steps_render_their_arguments(tmp_path):
    stub = ScriptedExecutor({"measure": ["loss=0.3"]})
    fleet = _fleet(tmp_path, stub)
    try:
        fleet.run_workflow({
            "stages": [
                {"name": "measure", "kind": "command", "command": ["echo", "loss=0.3"], "gpus": 0},
                {"name": "report", "kind": "command",
                 "command": ["echo", "saw {{ steps.measure.output }}"], "gpus": 0},
            ],
        })
        assert any("saw loss=0.3" in t for t in stub.tasks)
    finally:
        fleet.close()


# --------------------------------------------------------------------- audit


def test_a_workflow_run_is_recorded_in_the_ledger(tmp_path):
    stub = ScriptedExecutor()
    fleet = _fleet(tmp_path, stub)
    try:
        fleet.run_workflow({"name": "audited", "stages": [{"name": "s", "task": "t", "gpus": 0}]})
        types = {e.type for e in fleet.ledger.events(limit=500)}
        assert "workflow.started" in types and "workflow.finished" in types

        started = next(e for e in fleet.ledger.events(types=["workflow.started"]))
        assert started.payload["name"] == "audited"
        assert started.payload["stages"] == ["s"]

        ok, msg = fleet.verify_audit()
        assert ok, msg
    finally:
        fleet.close()


def test_a_workflow_can_be_loaded_from_a_path(tmp_path):
    path = tmp_path / "wf.yaml"
    path.write_text("name: fromfile\nstages:\n  - name: s\n    task: t\n    gpus: 0\n")
    stub = ScriptedExecutor()
    fleet = _fleet(tmp_path, stub)
    try:
        report = fleet.run_workflow(path)
        assert report.workflow == "fromfile"
        assert stub.calls == ["s"]
    finally:
        fleet.close()


def test_workflow_can_resume_completed_waves_in_a_new_run(tmp_path):
    workflow = {"name": "resumable", "stages": [
        {"name": "first", "task": "one", "gpus": 0},
        {"name": "second", "task": "use {{ steps.first.output }}", "gpus": 0},
    ]}
    first_exec = ScriptedExecutor()
    first = _fleet(tmp_path, first_exec)
    try:
        first_report = first.run_workflow(workflow)
        prior_run = first.run_id
        assert len(first_report.outcomes) == 2
    finally:
        first.close()

    resumed_exec = ScriptedExecutor()
    resumed = _fleet(tmp_path, resumed_exec)
    try:
        report = resumed.run_workflow(workflow, resume_from=prior_run)
        assert resumed.run_id != prior_run
        assert resumed_exec.calls == []
        assert [o.stage for o in report.outcomes] == ["first", "second"]
        restored = resumed.ledger.events(
            run_id=resumed.run_id, types=["workflow.restored"]
        )[-1]
        assert restored.payload["from_run"] == prior_run
    finally:
        resumed.close()


def test_from_run_restores_context_but_reruns_all_stages(tmp_path):
    workflow = {"name": "based", "stages": [
        {"name": "first", "task": "one", "gpus": 0},
    ]}
    first = _fleet(tmp_path, ScriptedExecutor())
    try:
        first.run_workflow(workflow)
        prior_run = first.run_id
    finally:
        first.close()

    based_exec = ScriptedExecutor()
    based = _fleet(tmp_path, based_exec)
    try:
        based.run_workflow(workflow, base_run=prior_run)
        assert based_exec.calls == ["first"]
    finally:
        based.close()


def test_resume_rejects_a_changed_workflow(tmp_path):
    first = _fleet(tmp_path, ScriptedExecutor())
    try:
        first.run_workflow({"stages": [{"name": "a", "task": "old", "gpus": 0}]})
        prior_run = first.run_id
    finally:
        first.close()

    resumed = _fleet(tmp_path, ScriptedExecutor())
    try:
        with pytest.raises(ValueError, match="no compatible checkpoint"):
            resumed.run_workflow(
                {"stages": [{"name": "a", "task": "changed", "gpus": 0}]},
                resume_from=prior_run,
            )
    finally:
        resumed.close()


# ------------------------------------------------------------------- the graph


def _wf(*stages, **kw):
    return Workflow.from_dict({"stages": list(stages), **kw})


def test_a_plain_list_stays_sequential():
    """No `needs` anywhere means each stage waits for everything before it."""
    wf = _wf({"name": "a", "task": "t"}, {"name": "b", "task": "t"}, {"name": "c", "task": "t"})
    assert wf.levels() == [["a"], ["b"], ["c"]]


def test_shared_needs_put_siblings_in_one_wave():
    wf = _wf(
        {"name": "plan", "task": "t"},
        {"name": "left", "task": "t", "needs": ["plan"]},
        {"name": "right", "task": "t", "needs": ["plan"]},
    )
    assert wf.levels() == [["plan"], ["left", "right"]]


def test_a_stage_without_needs_acts_as_a_barrier():
    """This is what makes a trailing `verify` wait for every branch."""
    wf = _wf(
        {"name": "plan", "task": "t"},
        {"name": "left", "task": "t", "needs": ["plan"]},
        {"name": "right", "task": "t", "needs": ["plan"]},
        {"name": "verify", "task": "t"},
    )
    assert wf.levels() == [["plan"], ["left", "right"], ["verify"]]
    assert wf.graph()["verify"] == {"plan", "left", "right"}


def test_sequence_and_graph_styles_mix():
    wf = _wf(
        {"name": "setup", "task": "t"},                              # sequence
        {"name": "x", "task": "t", "needs": ["setup"]},              # graph
        {"name": "y", "task": "t", "needs": ["setup"]},              # graph
        {"name": "merge", "task": "t", "needs": ["x", "y"]},          # graph
        {"name": "report", "task": "t"},                             # sequence barrier
    )
    assert wf.levels() == [["setup"], ["x", "y"], ["merge"], ["report"]]


def test_needs_may_point_at_a_stage_declared_later():
    """Order in the file is presentation; the graph decides execution."""
    wf = _wf({"name": "second", "task": "t", "needs": ["first"]}, {"name": "first", "task": "t"})
    assert wf.levels() == [["first"], ["second"]]
    assert wf.cycles() == []


def test_a_dag_reports_no_warnings():
    wf = _wf({"name": "a", "task": "t"}, {"name": "b", "task": "t", "needs": ["a"]})
    assert wf.cycles() == []
    assert wf.warnings() == []


# ------------------------------------------------------------------- cycles


def test_a_two_stage_cycle_is_detected():
    wf = _wf({"name": "x", "task": "t", "needs": ["y"]},
             {"name": "y", "task": "t", "needs": ["x"]})
    assert wf.cycles() == [["x", "y"]]


def test_a_longer_cycle_is_detected_and_named():
    wf = _wf({"name": "a", "task": "t", "needs": ["c"]},
             {"name": "b", "task": "t", "needs": ["a"]},
             {"name": "c", "task": "t", "needs": ["b"]})
    assert set(wf.cycles()[0]) == {"a", "b", "c"}


def test_a_self_dependency_is_a_cycle_of_one():
    """A node that needs itself is the shortest way to say "repeat this"."""
    wf = Workflow(name="w", stages=[Step(name="tune", task="t", needs=["tune"])])
    assert wf.cycles() == [["tune"]]
    assert wf.is_cyclic_component(["tune"])


def test_a_cycle_is_schedulable_and_sits_in_dependency_order():
    wf = _wf(
        {"name": "plan", "task": "t"},
        {"name": "implement", "task": "t", "needs": ["plan", "review"]},
        {"name": "review", "task": "t", "needs": ["implement"]},
        {"name": "ship", "task": "t", "needs": ["review"]},
    )
    assert wf.components() == [["plan"], ["implement", "review"], ["ship"]]
    assert wf.levels() == [["plan"], ["implement", "review"], ["ship"]]


def test_a_cycle_warns_about_repeating_and_names_its_limit():
    wf = _wf({"name": "x", "task": "t", "needs": ["y"]},
             {"name": "y", "task": "t", "needs": ["x"]}, max_iterations=5)
    note = wf.warnings()[0]
    assert "repeats" in note and "5 time(s)" in note
    assert "No stop condition" in note, "must say it will use every round"


def test_a_cycle_with_a_stop_condition_says_so():
    wf = _wf(
        {"name": "implement", "task": "t", "needs": ["review"]},
        {"name": "review", "task": "t", "needs": ["implement"],
         "until": {"output_contains": "APPROVED"}},
    )
    note = wf.warnings()[0]
    assert "Stops early when review" in note


def test_the_tightest_max_iterations_in_a_cycle_wins():
    wf = _wf(
        {"name": "a", "task": "t", "needs": ["b"], "max_iterations": 7},
        {"name": "b", "task": "t", "needs": ["a"], "max_iterations": 2},
        max_iterations=9,
    )
    assert wf.repeat_limit(["a", "b"]) == 2


# ------------------------------------------------------------ cycles that run


def test_a_cycle_repeats_until_a_node_is_satisfied(tmp_path):
    """The reviewer objects once, then approves, so the cycle runs twice."""
    stub = ScriptedExecutor({"review": ["needs work: add tests", "APPROVED"]})
    fleet = _fleet(tmp_path, stub)
    try:
        with pytest.warns(UserWarning, match="repeats"):
            report = fleet.run_workflow({
                "max_iterations": 5,
                "graph": {
                    "implement": {"task": "implement. {{ steps.review.output }}",
                                  "needs": ["review"], "gpus": 0},
                    "review": {"task": "review it", "needs": ["implement"], "gpus": 0,
                               "until": {"output_contains": "APPROVED"}},
                },
            })
        assert stub.calls == ["implement", "review", "implement-2", "review-2"]
        outcome = {o.stage: o for o in report.outcomes}["review"]
        assert outcome.iterations == 2 and outcome.stopped_early
        # The second round saw the objection from the first.
        assert "add tests" in stub.tasks[2]
    finally:
        fleet.close()


def test_a_cycle_without_a_stop_condition_uses_every_round(tmp_path):
    stub = ScriptedExecutor()
    fleet = _fleet(tmp_path, stub)
    try:
        with pytest.warns(UserWarning, match="No stop condition"):
            report = fleet.run_workflow({
                "max_iterations": 3,
                "graph": {"a": {"task": "t", "needs": ["b"], "gpus": 0},
                          "b": {"task": "t", "needs": ["a"], "gpus": 0}},
            })
        assert len(stub.calls) == 6              # two nodes, three rounds
        assert all(not o.stopped_early for o in report.outcomes)
    finally:
        fleet.close()


def test_a_self_dependency_repeats_one_node(tmp_path):
    stub = ScriptedExecutor({"tune": ["0.9", "0.4"]})
    fleet = _fleet(tmp_path, stub)
    try:
        with pytest.warns(UserWarning):
            fleet.run_workflow({
                "max_iterations": 4,
                "graph": {"tune": {"task": "tune it", "needs": ["tune"], "gpus": 0,
                                   "until": {"output_contains": "0.4"}}},
            })
        assert stub.calls == ["tune", "tune-2"]
    finally:
        fleet.close()


def test_a_cycle_runs_after_what_it_depends_on_and_before_what_follows(tmp_path):
    stub = ScriptedExecutor({"review": ["APPROVED"]})
    fleet = _fleet(tmp_path, stub)
    try:
        with pytest.warns(UserWarning):
            fleet.run_workflow({
                "graph": {
                    "plan": {"task": "t", "gpus": 0},
                    "implement": {"task": "t", "needs": ["plan", "review"], "gpus": 0},
                    "review": {"task": "t", "needs": ["implement"], "gpus": 0,
                               "until": {"output_contains": "APPROVED"}},
                    "ship": {"task": "t", "needs": ["review"], "gpus": 0},
                },
            })
        assert stub.calls == ["plan", "implement", "review", "ship"]
    finally:
        fleet.close()


def test_a_predicate_can_stop_a_graph_cycle(tmp_path):
    stub = ScriptedExecutor({"measure": ["0.42"]})
    fleet = _fleet(tmp_path, stub)
    try:
        with pytest.warns(UserWarning):
            report = fleet.run_workflow(
                {"max_iterations": 4, "graph": {
                    "measure": {"task": "t", "needs": ["adjust"], "gpus": 0},
                    "adjust": {"task": "t", "needs": ["measure"], "gpus": 0}}},
                # Keyed on the first node of the cycle, in declaration order.
                predicates={"measure": lambda r: float(r["measure"].output or 1) < 0.5},
            )
        assert report.outcomes[0].stopped_early
        assert len(stub.calls) == 2
    finally:
        fleet.close()


# ------------------------------------------------------- the graph spelling


def test_graph_nodes_use_only_the_edges_they_declare():
    """Unlike a `stages` list, a graph node gets no implicit barrier edge."""
    wf = Workflow.from_dict({"graph": {
        "a": {"task": "t"},
        "b": {"task": "t"},
    }})
    assert wf.graph() == {"a": set(), "b": set()}
    assert wf.levels() == [["a", "b"]], "independent graph nodes run together"


def test_stages_and_graph_can_be_combined():
    wf = Workflow.from_dict({
        "stages": [{"name": "setup", "task": "t"}],
        "graph": {
            "left": {"task": "t", "needs": ["setup"]},
            "right": {"task": "t", "needs": ["setup"]},
        },
    })
    assert wf.levels() == [["setup"], ["left", "right"]]


def test_a_graph_node_can_hold_a_loop():
    wf = Workflow.from_dict({"graph": {
        "prep": {"task": "t"},
        "cycle": {"needs": ["prep"], "loop": {"steps": [{"name": "s", "task": "t"}]}},
    }})
    assert wf.levels() == [["prep"], ["cycle"]]
    assert isinstance({st.name: st for st in wf.stages}["cycle"], Loop)


# --------------------------------------------------------- parallel execution


def test_independent_stages_run_at_the_same_time(tmp_path):
    stub = ScriptedExecutor(delay=0.4)
    fleet = _fleet(tmp_path, stub)
    try:
        fleet.run_workflow({"stages": [
            {"name": "plan", "task": "t", "gpus": 0},
            {"name": "left", "task": "t", "needs": ["plan"], "gpus": 0},
            {"name": "right", "task": "t", "needs": ["plan"], "gpus": 0},
        ]})
        left, right = stub.spans["left"], stub.spans["right"]
        assert left[0] < right[1] and right[0] < left[1], "the siblings should overlap"
    finally:
        fleet.close()


def test_a_dependent_stage_waits_for_its_dependency(tmp_path):
    stub = ScriptedExecutor(delay=0.2)
    fleet = _fleet(tmp_path, stub)
    try:
        fleet.run_workflow({"stages": [
            {"name": "first", "task": "t", "gpus": 0},
            {"name": "second", "task": "t", "needs": ["first"], "gpus": 0},
        ]})
        assert stub.spans["first"][1] <= stub.spans["second"][0]
    finally:
        fleet.close()


def test_outcomes_are_reported_in_declaration_order(tmp_path):
    stub = ScriptedExecutor()
    fleet = _fleet(tmp_path, stub)
    try:
        report = fleet.run_workflow({"stages": [
            {"name": "plan", "task": "t", "gpus": 0},
            {"name": "left", "task": "t", "needs": ["plan"], "gpus": 0},
            {"name": "right", "task": "t", "needs": ["plan"], "gpus": 0},
        ]})
        assert [o.stage for o in report.outcomes] == ["plan", "left", "right"]
    finally:
        fleet.close()


def test_a_loop_can_depend_on_an_earlier_stage(tmp_path):
    stub = ScriptedExecutor({"review": ["APPROVED"]})
    fleet = _fleet(tmp_path, stub)
    try:
        report = fleet.run_workflow({"stages": [
            {"name": "plan", "task": "t", "gpus": 0},
            {"name": "cycle", "needs": ["plan"], "loop": {
                "max_iterations": 2,
                "until": {"step": "review", "output_contains": "APPROVED"},
                "steps": [{"name": "review", "task": "t", "gpus": 0}]}},
        ]})
        assert stub.calls[0] == "plan"
        assert report.outcomes[1].stopped_early
    finally:
        fleet.close()


# ------------------------------------------------- artifacts across stages


class ArtifactExecutor(ScriptedExecutor):
    """Writes a file into its own `/results` and records the mounts it was given.

    Each job's `/results` is a private directory, so this is what proves a later
    stage can actually open what an earlier one wrote.
    """

    def __init__(self, filename: str = "findings.md", **kwargs):
        super().__init__(**kwargs)
        self.filename = filename
        self.mounts: dict[str, list] = {}

    def run(self, spec, *, argv, env, placement, policy, on_line):
        self.mounts[spec.name] = list(spec.mounts)
        results = next((m.source for m in spec.mounts if m.target == "/results"), None)
        if results is not None:
            (Path(results) / self.filename).write_text(f"written by {spec.name}")
        return super().run(spec, argv=argv, env=env, placement=placement,
                           policy=policy, on_line=on_line)


def test_a_later_stage_can_read_an_earlier_stages_results(tmp_path):
    stub = ArtifactExecutor()
    fleet = _fleet(tmp_path, stub)
    try:
        fleet.run_workflow({"stages": [
            {"name": "research", "task": "analyse", "gpus": 0},
            {"name": "judge", "task": "review", "needs": ["research"], "gpus": 0},
        ]})
        inputs = {m.target: m for m in stub.mounts["judge"]}
        assert "/inputs/research" in inputs, "judge got no view of the research stage"
        assert inputs["/inputs/research"].mode == "ro"
        assert (Path(inputs["/inputs/research"].source) / "findings.md").read_text() == \
            "written by research"
    finally:
        fleet.close()


def test_the_first_stage_gets_no_input_mounts(tmp_path):
    stub = ArtifactExecutor()
    fleet = _fleet(tmp_path, stub)
    try:
        fleet.run_workflow({"stages": [
            {"name": "research", "task": "analyse", "gpus": 0},
            {"name": "judge", "task": "review", "needs": ["research"], "gpus": 0},
        ]})
        assert [m.target for m in stub.mounts["research"] if m.target.startswith("/inputs")] == []
    finally:
        fleet.close()


def test_each_fanned_out_copy_keeps_its_own_results_directory(tmp_path):
    stub = ArtifactExecutor()
    fleet = _fleet(tmp_path, stub)
    try:
        fleet.run_workflow({"stages": [
            {"name": "probe", "task": "t", "copies": 2, "gpus": 0},
        ]})
        sources = {
            name: next(m.source for m in mounts if m.target == "/results")
            for name, mounts in stub.mounts.items()
        }
        assert len(set(sources.values())) == 2, "copies shared a results directory"
    finally:
        fleet.close()


def test_a_stage_sees_every_completed_stage_by_name(tmp_path):
    stub = ArtifactExecutor()
    fleet = _fleet(tmp_path, stub)
    try:
        fleet.run_workflow({"stages": [
            {"name": "one", "task": "t", "gpus": 0},
            {"name": "two", "task": "t", "needs": ["one"], "gpus": 0},
            {"name": "three", "task": "t", "needs": ["two"], "gpus": 0},
        ]})
        targets = {m.target for m in stub.mounts["three"]}
        assert {"/inputs/one", "/inputs/two"} <= targets
    finally:
        fleet.close()


def test_a_long_final_answer_reaches_the_next_stage_whole(tmp_path):
    """The judge's prompt interpolates the researcher's answer. If that answer is
    clipped on the way, findings vanish from the review without anyone noticing."""
    answer = "## F1: first\n" + ("filler " * 3000) + "\n## F7: last"
    stub = ScriptedExecutor({"research": [answer]})
    fleet = _fleet(tmp_path, stub)
    try:
        report = fleet.run_workflow({"stages": [
            {"name": "research", "task": "analyse", "gpus": 0},
            {"name": "judge", "task": "grade this:\n{{ steps.research.output }}", "gpus": 0},
        ]})
        assert report.steps["research"].output == answer
        judge_prompt = stub.tasks[1]
        assert "## F7: last" in judge_prompt, "the tail of the answer never reached the judge"
    finally:
        fleet.close()


# ------------------------------------------------------------ run directories


def _results_root(fleet):
    return fleet.config.results_path


def test_a_run_lands_under_its_workflow_name_and_attempt(tmp_path):
    stub = ArtifactExecutor()
    fleet = _fleet(tmp_path, stub)
    try:
        fleet.run_workflow({"name": "myc-discovery", "stages": [
            {"name": "research", "task": "t", "gpus": 0},
            {"name": "judge", "task": "t", "needs": ["research"], "gpus": 0},
        ]})
        run_dir = _results_root(fleet) / "myc-discovery" / "001"
        assert (run_dir / "research" / "findings.md").exists()
        assert (run_dir / "judge" / "findings.md").exists()
    finally:
        fleet.close()


def test_running_the_same_workflow_again_opens_the_next_attempt(tmp_path):
    workflow = {"name": "myc-discovery", "stages": [{"name": "research", "task": "t", "gpus": 0}]}
    first = _fleet(tmp_path, ArtifactExecutor())
    try:
        first.run_workflow(workflow)
    finally:
        first.close()
    second = _fleet(tmp_path, ArtifactExecutor())
    try:
        second.run_workflow(workflow)
        attempts = sorted(p.name for p in (_results_root(second) / "myc-discovery").iterdir())
        assert attempts == ["001", "002"]
    finally:
        second.close()


def test_a_repeat_run_is_told_nothing_about_the_last_one(tmp_path):
    """Isolation is the default: only an explicit resume or base run inherits."""
    workflow = {"name": "wf", "stages": [
        {"name": "research", "task": "t", "gpus": 0},
        {"name": "judge", "task": "t", "needs": ["research"], "gpus": 0},
    ]}
    first = _fleet(tmp_path, ArtifactExecutor())
    try:
        first.run_workflow(workflow)
    finally:
        first.close()

    stub = ArtifactExecutor()
    second = _fleet(tmp_path, stub)
    try:
        second.run_workflow(workflow)
        for name, mounts in stub.mounts.items():
            sources = [m.source for m in mounts]
            assert not any("/001/" in s for s in sources), f"{name} reached into attempt 001"
            assert "/previous-results" not in [m.target for m in mounts]
        # ...and the judge still sees this run's own research stage.
        assert "/inputs/research" in {m.target for m in stub.mounts["judge"]}
    finally:
        second.close()


def test_a_resume_continues_in_the_same_directory(tmp_path):
    workflow = {"name": "wf", "stages": [
        {"name": "first", "task": "t", "gpus": 0},
        {"name": "second", "task": "t", "needs": ["first"], "gpus": 0},
    ]}
    first = _fleet(tmp_path, ArtifactExecutor())
    try:
        first.run_workflow(workflow)
        prior_run = first.run_id
    finally:
        first.close()

    resumed = _fleet(tmp_path, ArtifactExecutor())
    try:
        resumed.run_workflow(workflow, resume_from=prior_run)
        root = _results_root(resumed) / "wf"
        assert sorted(p.name for p in root.iterdir()) == ["001"]
        manifest = json.loads((root / "001" / "run.json").read_text())
        assert manifest["run_id"] == prior_run
        assert resumed.run_id in manifest["continued_by"]
    finally:
        resumed.close()


def test_a_base_run_opens_a_new_attempt_and_keeps_the_old_one(tmp_path):
    workflow = {"name": "wf", "stages": [{"name": "first", "task": "t", "gpus": 0}]}
    first = _fleet(tmp_path, ArtifactExecutor())
    try:
        first.run_workflow(workflow)
        prior_run = first.run_id
    finally:
        first.close()

    stub = ArtifactExecutor()
    based = _fleet(tmp_path, stub)
    try:
        based.run_workflow(workflow, base_run=prior_run)
        root = _results_root(based) / "wf"
        assert sorted(p.name for p in root.iterdir()) == ["001", "002"]
        assert json.loads((root / "002" / "run.json").read_text())["based_on"] == prior_run
        assert (root / "001" / "first" / "findings.md").exists(), "the earlier attempt was clobbered"
        previous = [m for m in stub.mounts["first"] if m.target == "/previous-results"]
        assert previous and previous[0].source.endswith("/001")
    finally:
        based.close()


def test_a_run_that_submits_nothing_leaves_no_directory(tmp_path):
    fleet = _fleet(tmp_path, ArtifactExecutor())
    try:
        assert not _results_root(fleet).exists() or list(_results_root(fleet).iterdir()) == []
    finally:
        fleet.close()


def test_cycle_iterations_and_copies_get_their_own_directories(tmp_path):
    stub = ArtifactExecutor(script={"review": ["needs work", "APPROVED"]})
    fleet = _fleet(tmp_path, stub)
    try:
        fleet.run_workflow({"name": "wf", "stages": [
            {"name": "probe", "task": "t", "copies": 2, "gpus": 0},
            {"name": "cycle", "needs": ["probe"], "loop": {
                "max_iterations": 2,
                "until": {"step": "review", "output_contains": "APPROVED"},
                "steps": [{"name": "review", "task": "t", "gpus": 0}]}},
        ]})
        names = sorted(p.name for p in (_results_root(fleet) / "wf" / "001").iterdir() if p.is_dir())
        assert names == ["cycle-review-1", "cycle-review-2", "probe-0", "probe-1"]
    finally:
        fleet.close()


def test_isolation_is_provisioned_without_the_user_asking(tmp_path):
    """A research directory is not a git repo and should not have to become one by
    hand for isolation to work."""
    from research_fleet import isolation
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "prompts.md").write_text("a prompt")

    fleet = Fleet(root=str(tmp_path / "state"), workspace=str(workspace),
                  executor={"kind": "dry-run"}, isolate_agents=True)
    fleet.executor = fleet.scheduler.executor = ScriptedExecutor()
    try:
        fleet.run_workflow({"name": "wf", "stages": [{"name": "one", "task": "t", "gpus": 0}]})
        assert isolation.is_repo(workspace)
        assert (workspace / ".git").is_file(), "the git directory should not live in the project"
        import subprocess
        tracked = subprocess.run(["git", "ls-files"], cwd=workspace,
                                 capture_output=True, text=True).stdout
        assert tracked.strip() == "", "fleet tracked the user's files"
    finally:
        fleet.close()


def test_a_run_with_no_isolated_jobs_never_touches_the_workspace(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    fleet = Fleet(root=str(tmp_path / "state"), workspace=str(workspace),
                  executor={"kind": "dry-run"}, isolate_agents=False)
    fleet.executor = fleet.scheduler.executor = ScriptedExecutor()
    try:
        fleet.run_workflow({"name": "wf", "stages": [{"name": "one", "task": "t", "gpus": 0}]})
        assert not (workspace / ".git").exists()
    finally:
        fleet.close()


def test_an_unusable_workspace_degrades_instead_of_failing_the_run(tmp_path):
    """Losing isolation is better than losing the run."""
    fleet = Fleet(root=str(tmp_path / "state"), workspace=str(tmp_path / "gone"),
                  executor={"kind": "dry-run"}, isolate_agents=True)
    stub = ScriptedExecutor()
    fleet.executor = fleet.scheduler.executor = stub
    try:
        with pytest.warns(UserWarning, match="isolation"):
            report = fleet.run_workflow({"name": "wf", "stages": [
                {"name": "one", "task": "t", "gpus": 0}]})
        assert report.steps["one"].state is JobState.SUCCEEDED
        assert stub.isolated == [False]
    finally:
        fleet.close()


def test_a_based_run_sees_the_earlier_runs_stages_by_name(tmp_path):
    workflow = {"name": "wf", "stages": [
        {"name": "research", "task": "t", "gpus": 0},
        {"name": "judge", "task": "t", "needs": ["research"], "gpus": 0},
    ]}
    first = _fleet(tmp_path, ArtifactExecutor())
    try:
        first.run_workflow(workflow)
        prior_run = first.run_id
    finally:
        first.close()

    stub = ArtifactExecutor()
    based = _fleet(tmp_path, stub)
    try:
        based.run_workflow(workflow, base_run=prior_run)
        targets = {m.target: m for m in stub.mounts["research"]}
        assert "/previous/research" in targets and "/previous/judge" in targets
        assert targets["/previous/research"].mode == "ro"
        assert (Path(targets["/previous/judge"].source) / "findings.md").exists()
        # The whole prior run stays available under its documented name.
        assert "/previous-results" in targets
    finally:
        based.close()


def test_a_fresh_run_is_given_no_previous_stages(tmp_path):
    workflow = {"name": "wf", "stages": [{"name": "research", "task": "t", "gpus": 0}]}
    first = _fleet(tmp_path, ArtifactExecutor())
    try:
        first.run_workflow(workflow)
    finally:
        first.close()

    stub = ArtifactExecutor()
    second = _fleet(tmp_path, stub)
    try:
        second.run_workflow(workflow)
        assert not [m for m in stub.mounts["research"] if m.target.startswith("/previous")]
    finally:
        second.close()


def test_a_based_run_tolerates_a_revised_workflow(tmp_path):
    """Building on an earlier attempt usually means having changed the prompts in
    light of it, so `--from-run` must not require an unchanged definition."""
    first = _fleet(tmp_path, ArtifactExecutor())
    try:
        first.run_workflow({"name": "wf", "stages": [{"name": "research", "task": "v1", "gpus": 0}]})
        prior_run = first.run_id
    finally:
        first.close()

    stub = ArtifactExecutor()
    based = _fleet(tmp_path, stub)
    try:
        based.run_workflow(
            {"name": "wf", "stages": [
                {"name": "research", "task": "v2, now referencing /previous", "gpus": 0}]},
            base_run=prior_run,
        )
        assert stub.calls == ["research"]
        assert "/previous/research" in {m.target for m in stub.mounts["research"]}
    finally:
        based.close()


def test_a_resume_still_refuses_a_revised_workflow(tmp_path):
    """Resuming skips completed stages, so a changed DAG would half-build the run."""
    first = _fleet(tmp_path, ArtifactExecutor())
    try:
        first.run_workflow({"name": "wf", "stages": [{"name": "a", "task": "v1", "gpus": 0}]})
        prior_run = first.run_id
    finally:
        first.close()

    resumed = _fleet(tmp_path, ArtifactExecutor())
    try:
        with pytest.raises(ValueError, match="no compatible checkpoint"):
            resumed.run_workflow(
                {"name": "wf", "stages": [{"name": "a", "task": "v2", "gpus": 0}]},
                resume_from=prior_run,
            )
    finally:
        resumed.close()


def test_every_stage_receives_the_projects_shared_prompt(tmp_path):
    """The point of the file: standing rules stop being copied into each task prompt."""
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "FLEET.md").write_text("Never quote a number you did not compute.")

    stub = ScriptedExecutor()
    fleet = Fleet(root=str(tmp_path / "state"), workspace=str(workspace),
                  executor={"kind": "dry-run"})
    fleet.executor = fleet.scheduler.executor = stub
    try:
        fleet.run_workflow({"name": "wf", "stages": [
            {"name": "research", "task": "analyse", "gpus": 0},
            {"name": "judge", "task": "review", "needs": ["research"], "gpus": 0},
        ]})
        assert len(stub.systems) == 2
        for system in stub.systems:
            assert "Never quote a number you did not compute." in system
        # ...and it did not displace the task itself.
        assert "analyse" in stub.tasks[0] and "review" in stub.tasks[1]
    finally:
        fleet.close()


def test_a_project_without_a_file_still_gets_the_default_instructions(tmp_path):
    """The generic guidance ships with fleet, so a new project is not left with none."""
    workspace = tmp_path / "project"
    workspace.mkdir()
    stub = ScriptedExecutor()
    fleet = Fleet(root=str(tmp_path / "state"), workspace=str(workspace),
                  executor={"kind": "dry-run"})
    fleet.executor = fleet.scheduler.executor = stub
    try:
        fleet.run_workflow({"name": "wf", "stages": [{"name": "a", "task": "t", "gpus": 0}]})
        assert "Project instructions" in stub.systems[0]
        assert "How a run is organised" in stub.systems[0]
    finally:
        fleet.close()


def test_the_shared_prompt_is_read_once_per_run(tmp_path):
    """A file edited mid-run would otherwise split a workflow's stages across two sets
    of rules -- the drift the shared prompt exists to prevent."""
    workspace = tmp_path / "project"
    workspace.mkdir()
    shared = workspace / "FLEET.md"
    shared.write_text("version one")

    class Editing(ScriptedExecutor):
        def run(self, spec, **kw):
            shared.write_text("version two")
            return super().run(spec, **kw)

    stub = Editing()
    fleet = Fleet(root=str(tmp_path / "state"), workspace=str(workspace),
                  executor={"kind": "dry-run"})
    fleet.executor = fleet.scheduler.executor = stub
    try:
        fleet.run_workflow({"name": "wf", "stages": [
            {"name": "a", "task": "t", "gpus": 0},
            {"name": "b", "task": "t", "needs": ["a"], "gpus": 0},
        ]})
        assert all("version one" in s for s in stub.systems)
    finally:
        fleet.close()


def test_a_resumed_stage_overwrites_its_failed_attempt_not_its_neighbour(tmp_path):
    """The whole point of stage-named directories is that `/inputs/second` is the
    second stage's result. A resume must not leave the failure sitting there."""
    from research_fleet import runlayout

    class Gated(ArtifactExecutor):
        """`second` fails until a gate file appears in `first`'s results."""

        def run(self, spec, **kw):
            result = super().run(spec, **kw)
            if spec.name == "second":
                results = next(m.source for m in spec.mounts if m.target == "/results")
                gate = [m.source for m in spec.mounts if m.target == "/inputs/first"]
                if not (gate and (Path(gate[0]) / "gate").exists()):
                    result.state = JobState.FAILED
                    result.exit_code = 1
                else:
                    (Path(results) / "second.txt").write_text("resumed ok")
            return result

    workflow = {"name": "wf", "stages": [
        {"name": "first", "task": "t", "gpus": 0},
        {"name": "second", "task": "t", "needs": ["first"], "gpus": 0},
    ]}

    first = _fleet(tmp_path, Gated())
    try:
        first.run_workflow(workflow)
        prior_run = first.run_id
    finally:
        first.close()

    run_dir = _results_root(first) / "wf" / "001"
    assert not (run_dir / "second" / "second.txt").exists(), "the stage should have failed"
    (run_dir / "first" / "gate").write_text("go")

    resumed = _fleet(tmp_path, Gated())
    try:
        resumed.run_workflow(workflow, resume_from=prior_run)
    finally:
        resumed.close()

    assert (run_dir / "second" / "second.txt").exists(), \
        "the successful re-run did not land in the stage's own directory"
    assert not (run_dir / "second~2").exists(), "the re-run was pushed aside by a suffix"
    superseded = run_dir / runlayout.SUPERSEDED
    assert superseded.is_dir() and any(superseded.iterdir()), "the failed attempt was lost"
    # ...and a later run building on this one gets the success, not the failure.
    assert "second.txt" in {p.name for p in runlayout.stage_dirs(run_dir)["second"].iterdir()}


def test_a_harness_failure_stops_the_workflow_instead_of_feeding_the_next_stage(tmp_path):
    """A refused prompt exits 0 with the refusal as its 'answer'. Marked succeeded, the
    run carries on and spends the next stage judging an error message."""
    class Refusing(ScriptedExecutor):
        def run(self, spec, *, argv, env, placement, policy, on_line):
            if spec.name == "research":
                self.calls.append(spec.name)
                self.tasks.append("")
                self.systems.append("")
                on_line("stdout", json.dumps({
                    "type": "result", "subtype": "success", "is_error": False,
                    "num_turns": 0,
                    "result": "Goal condition is limited to 4000 characters (got 4184)",
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                }))
                return JobResult(job_id=spec.id, state=JobState.SUCCEEDED, exit_code=0)
            return super().run(spec, argv=argv, env=env, placement=placement,
                               policy=policy, on_line=on_line)

    stub = Refusing()
    fleet = _fleet(tmp_path, stub)
    try:
        report = fleet.run_workflow({"name": "wf", "stages": [
            {"name": "research", "task": "t", "gpus": 0},
            {"name": "judge", "task": "grade {{ steps.research.output }}", "gpus": 0},
        ]})
        assert report.steps["research"].state is JobState.FAILED
        assert "4000 characters" in (report.steps["research"].error or "")
        assert "judge" not in stub.calls, "the judge ran against an error message"
    finally:
        fleet.close()
