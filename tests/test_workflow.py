"""Workflows: parsing, templating, loop control, and fan-out.

The stub executor returns scripted output per step name, which is what lets a
coder/reviewer loop be tested deterministically: the reviewer objects once, then
approves, and the loop must stop on its own.
"""

from __future__ import annotations

import json
import threading
import time

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
        self.isolated: list[bool] = []
        self.spans: dict[str, tuple[float, float]] = {}   # name -> (start, end)
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
