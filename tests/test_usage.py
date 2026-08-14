"""Cross-run usage accounting.

The point of this table is answering "what has this cost me" over time, so the tests
care about two things: that a repeated node is counted once per attempt rather than
merged, and that the whole thing can be rebuilt from the append-only log.
"""

from __future__ import annotations

import json
import time

import pytest

from research_fleet import Fleet, Ledger
from research_fleet.spec import JobKind, JobResult, JobSpec, JobState, Resources

from test_workflow import ScriptedExecutor


def _fleet(tmp_path, executor=None, **overrides):
    fleet = Fleet(root=str(tmp_path / "state"), workspace=str(tmp_path),
                  executor={"kind": "dry-run"}, **overrides)
    if executor is not None:
        fleet.executor = executor
        fleet.scheduler.executor = executor
    return fleet


def _spec(**kw):
    base = dict(name="j", command=["true"], run_id="run_1", resources=Resources(gpus=2))
    base.update(kw)
    return JobSpec(**base)


def _result(spec, *, cost=0.5, tokens=1000, duration=10.0, model="claude-opus-5"):
    now = time.time()
    return JobResult(
        job_id=spec.id, state=JobState.SUCCEEDED, started_at=now, ended_at=now + duration,
        usage={"model": model, "requests": 2, "input_tokens": tokens, "output_tokens": 100,
               "cache_read_tokens": 50, "cache_write_tokens": 0,
               "total_tokens": tokens + 150, "cost_usd": cost, "unpriced_model": False},
    )


# ------------------------------------------------------------------ recording


def test_a_finished_job_lands_in_the_usage_table(tmp_path):
    ledger = Ledger(tmp_path)
    spec = _spec(name="train")
    ledger.upsert_job(spec, "succeeded", _result(spec, cost=0.25, duration=30.0))

    rows = ledger.usage_rows()
    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == "train" and row["state"] == "succeeded"
    assert row["cost_usd"] == 0.25
    assert row["duration_s"] == pytest.approx(30.0)
    assert row["gpu_seconds"] == pytest.approx(60.0), "2 GPUs for 30s"
    assert row["requests"] == 2
    ledger.close()


def test_a_job_without_a_result_records_no_usage(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.upsert_job(_spec(), "queued")
    assert ledger.usage_rows() == []
    ledger.close()


def test_updating_a_job_replaces_rather_than_duplicates(tmp_path):
    ledger = Ledger(tmp_path)
    spec = _spec()
    ledger.upsert_job(spec, "running", _result(spec, cost=0.1))
    ledger.upsert_job(spec, "succeeded", _result(spec, cost=0.4))
    rows = ledger.usage_rows()
    assert len(rows) == 1 and rows[0]["cost_usd"] == 0.4
    ledger.close()


def test_an_unpriced_model_is_flagged_not_hidden(tmp_path):
    ledger = Ledger(tmp_path)
    spec = _spec()
    result = _result(spec, cost=0.0)
    result.usage["unpriced_model"] = True
    ledger.upsert_job(spec, "succeeded", result)

    assert ledger.usage_rows()[0]["unpriced"] == 1
    assert ledger.usage_totals()["unpriced_jobs"] == 1
    ledger.close()


# ------------------------------------------------------------------- grouping


def _populate(ledger):
    for run, name, model, cost in [
        ("run_a", "plan", "claude-opus-5", 1.0),
        ("run_a", "code", "claude-sonnet-5", 0.5),
        ("run_b", "plan", "claude-opus-5", 2.0),
        ("run_b", "code", "claude-haiku-4-5", 0.1),
    ]:
        spec = _spec(name=name, run_id=run, kind=JobKind.COMMAND)
        ledger.upsert_job(spec, "succeeded", _result(spec, cost=cost, model=model))


def test_totals_sum_everything(tmp_path):
    ledger = Ledger(tmp_path)
    _populate(ledger)
    total = ledger.usage_totals()
    assert total["jobs"] == 4
    assert total["cost_usd"] == pytest.approx(3.6)
    ledger.close()


def test_totals_can_be_scoped_to_one_run(tmp_path):
    ledger = Ledger(tmp_path)
    _populate(ledger)
    assert ledger.usage_totals(run_id="run_a")["cost_usd"] == pytest.approx(1.5)
    ledger.close()


def test_grouping_by_model_ranks_by_cost(tmp_path):
    ledger = Ledger(tmp_path)
    _populate(ledger)
    rows = ledger.usage_by("model")
    assert rows[0]["model"] == "claude-opus-5"
    assert rows[0]["cost_usd"] == pytest.approx(3.0) and rows[0]["jobs"] == 2
    ledger.close()


def test_grouping_by_run_and_name_together(tmp_path):
    ledger = Ledger(tmp_path)
    _populate(ledger)
    rows = ledger.usage_by("run,name")
    keyed = {(r["run"], r["name"]): r["cost_usd"] for r in rows}
    assert keyed[("run_b", "plan")] == pytest.approx(2.0)
    assert keyed[("run_a", "code")] == pytest.approx(0.5)
    ledger.close()


def test_grouping_by_day_buckets_by_date(tmp_path):
    ledger = Ledger(tmp_path)
    _populate(ledger)
    rows = ledger.usage_by("day")
    assert len(rows) == 1 and rows[0]["jobs"] == 4
    ledger.close()


def test_an_unknown_grouping_key_lists_the_valid_ones(tmp_path):
    ledger = Ledger(tmp_path)
    with pytest.raises(ValueError, match="try one or more of"):
        ledger.usage_by("colour")
    with pytest.raises(ValueError, match="try one or more of"):
        ledger.usage_by("")
    ledger.close()


def test_usage_can_be_filtered_by_kind(tmp_path):
    ledger = Ledger(tmp_path)
    cmd = _spec(name="train", kind=JobKind.COMMAND)
    ledger.upsert_job(cmd, "succeeded", _result(cmd, cost=0.0))
    agent = _spec(name="think", kind=JobKind.AGENT, command=[],
                  agent={"task": "t", "backend": "claude-cli"})
    ledger.upsert_job(agent, "succeeded", _result(agent, cost=1.0))

    assert ledger.usage_totals(kind="agent")["cost_usd"] == pytest.approx(1.0)
    assert ledger.usage_totals(kind="command")["cost_usd"] == pytest.approx(0.0)
    ledger.close()


# ------------------------------------------------- repeats counted separately


def test_a_repeated_node_is_recorded_once_per_attempt(tmp_path):
    """A cycle that runs three times must not collapse into one row."""
    stub = ScriptedExecutor({"review": ["again", "again", "APPROVED"]})
    fleet = _fleet(tmp_path, stub, budget={"max_usd": 50.0})
    try:
        with pytest.warns(UserWarning):
            fleet.run_workflow({
                "max_iterations": 3,
                "graph": {
                    "implement": {"task": "do it", "needs": ["review"], "gpus": 0},
                    "review": {"task": "check it", "needs": ["implement"], "gpus": 0,
                               "until": {"output_contains": "APPROVED"}},
                },
            })
        run_id = fleet.run_id
    finally:
        fleet.close()

    ledger = Ledger(tmp_path / "state")
    try:
        rows = ledger.usage_by("stage,attempt", run_id=run_id)
        by_key = {(r["stage"], r["attempt"]): r for r in rows}
        # Three attempts of each node, each its own row.
        for attempt in (1, 2, 3):
            assert ("implement", attempt) in by_key
            assert ("review", attempt) in by_key
        assert all(r["jobs"] == 1 for r in rows), "one job per attempt, not merged"

        # Rolled up by stage, the attempts sum.
        per_stage = {r["stage"]: r for r in ledger.usage_by("stage", run_id=run_id)}
        assert per_stage["review"]["jobs"] == 3
    finally:
        ledger.close()


def test_fan_out_copies_are_recorded_separately(tmp_path):
    stub = ScriptedExecutor()
    fleet = _fleet(tmp_path, stub)
    try:
        fleet.run_workflow({"stages": [
            {"name": "explore", "task": "try it", "copies": 3, "gpus": 0}]})
        run_id = fleet.run_id
    finally:
        fleet.close()

    ledger = Ledger(tmp_path / "state")
    try:
        rows = ledger.usage_by("stage", run_id=run_id)
        assert len(rows) == 1 and rows[0]["jobs"] == 3
        assert len(ledger.usage_rows(run_id=run_id)) == 3
    finally:
        ledger.close()


def test_the_same_stage_in_two_runs_stays_separate(tmp_path):
    stub = ScriptedExecutor()
    runs = []
    for _ in range(2):
        fleet = _fleet(tmp_path, stub)
        try:
            fleet.run_workflow({"stages": [{"name": "step", "task": "t", "gpus": 0}]})
            runs.append(fleet.run_id)
        finally:
            fleet.close()

    ledger = Ledger(tmp_path / "state")
    try:
        rows = ledger.usage_by("run,stage")
        assert {r["run"] for r in rows} == set(runs)
        assert all(r["jobs"] == 1 for r in rows)
        # And across both runs, aggregated.
        assert ledger.usage_by("stage")[0]["jobs"] == 2
    finally:
        ledger.close()


def test_workflow_name_is_recorded(tmp_path):
    stub = ScriptedExecutor()
    fleet = _fleet(tmp_path, stub)
    try:
        fleet.run_workflow({"name": "tagged", "stages": [
            {"name": "s", "task": "t", "gpus": 0}]})
    finally:
        fleet.close()

    ledger = Ledger(tmp_path / "state")
    try:
        assert ledger.usage_by("workflow")[0]["workflow"] == "tagged"
    finally:
        ledger.close()


# ------------------------------------------------------------------- rebuild


def test_usage_survives_losing_the_index(tmp_path):
    """The JSONL is the source of truth, so a deleted index costs nothing."""
    fleet = _fleet(tmp_path)
    try:
        for i in range(3):
            fleet.run_command(["true"], name=f"job-{i}", gpus=0)
        fleet.wait(timeout=60)
        run_id = fleet.run_id
    finally:
        fleet.close()

    ledger = Ledger(tmp_path / "state")
    before = ledger.usage_totals(run_id=run_id)
    assert before["jobs"] == 3

    ledger._db.execute("DELETE FROM usage")
    ledger._db.commit()
    assert ledger.usage_totals(run_id=run_id)["jobs"] == 0

    ledger.reindex()
    after = ledger.usage_totals(run_id=run_id)
    assert after["jobs"] == 3
    assert after["duration_s"] == pytest.approx(before["duration_s"], rel=0.01)
    ledger.close()


def test_reindex_also_restores_the_jobs_table(tmp_path):
    fleet = _fleet(tmp_path)
    try:
        fleet.run_command(["true"], name="one", gpus=0)
        fleet.run_command(["true"], name="two", gpus=0)
        fleet.wait(timeout=60)
        run_id = fleet.run_id
    finally:
        fleet.close()

    ledger = Ledger(tmp_path / "state")
    ledger._db.execute("DELETE FROM jobs")
    ledger._db.commit()
    assert ledger.jobs(run_id) == []

    ledger.reindex()
    assert len(ledger.jobs(run_id)) == 2
    ledger.close()


# ------------------------------------------------------------ the public API


def test_fleet_exposes_totals_and_groupings(tmp_path):
    stub = ScriptedExecutor()
    fleet = _fleet(tmp_path, stub)
    try:
        fleet.run_agents("do something", n=2, gpus=0)
        fleet.wait(timeout=60)

        totals = fleet.usage()
        assert totals["jobs"] == 2
        assert totals["total_tokens"] > 0, "the stub reports token usage"

        by_model = fleet.usage("model")
        assert by_model[0]["model"] == "claude-sonnet-5"
        assert len(fleet.usage_jobs()) == 2
    finally:
        fleet.close()


# ----------------------------------------------- agent time vs execution time


def test_agent_time_and_wall_time_are_stored_separately(tmp_path):
    """Wall clock includes starting the container; agent time is what the harness
    reported working. Both matter, for different questions."""
    ledger = Ledger(tmp_path)
    spec = _spec(name="think", kind=JobKind.AGENT, command=[],
                 agent={"task": "t", "backend": "claude-cli"})
    result = _result(spec, duration=42.0)
    result.agent_seconds = 30.0
    ledger.upsert_job(spec, "succeeded", result)

    row = ledger.usage_rows()[0]
    assert row["duration_s"] == pytest.approx(42.0)
    assert row["agent_seconds"] == pytest.approx(30.0)
    totals = ledger.usage_totals()
    assert totals["duration_s"] == pytest.approx(42.0)
    assert totals["agent_seconds"] == pytest.approx(30.0)
    ledger.close()


def test_command_jobs_have_wall_time_but_no_agent_time(tmp_path):
    ledger = Ledger(tmp_path)
    spec = _spec(name="train", kind=JobKind.COMMAND)
    ledger.upsert_job(spec, "succeeded", _result(spec, duration=12.0))
    row = ledger.usage_rows()[0]
    assert row["duration_s"] == pytest.approx(12.0)
    assert row["agent_seconds"] is None
    ledger.close()


def test_agent_time_comes_from_the_harness_and_is_aggregated_per_run(tmp_path):
    """End to end: the harness reports duration_ms, and it reaches the per-run totals."""
    class Timed(ScriptedExecutor):
        def run(self, spec, *, argv, env, placement, policy, on_line):
            on_line("stdout", json.dumps({
                "type": "result", "subtype": "success", "result": "done",
                "duration_ms": 2500,
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }))
            now = time.time()
            return JobResult(job_id=spec.id, state=JobState.SUCCEEDED, exit_code=0,
                             started_at=now, ended_at=now + 4.0)

    stub = Timed()
    fleet = _fleet(tmp_path, stub)
    try:
        fleet.run_agents("do it", n=2, gpus=0)
        fleet.wait(timeout=60)
        per_run = fleet.usage("run")
        assert len(per_run) == 1
        assert per_run[0]["agent_seconds"] == pytest.approx(5.0), "2 jobs x 2.5s reported"
        assert per_run[0]["duration_s"] == pytest.approx(8.0, rel=0.1)
    finally:
        fleet.close()


def test_token_counts_are_not_mistaken_for_secrets(tmp_path):
    """`input_tokens` contains the word "token"; masking it destroyed the accounting."""
    ledger = Ledger(tmp_path)
    ledger.append("budget.committed", {
        "usage": {"input_tokens": 1234, "total_tokens": 1300, "cost_usd": 0.5},
        "auth_token": "sk-ant-should-be-masked",
    })
    payload = ledger.events(types=["budget.committed"])[0].payload
    assert payload["usage"]["input_tokens"] == 1234
    assert payload["usage"]["total_tokens"] == 1300
    assert payload["auth_token"] == "[REDACTED]"
    ledger.close()


# --------------------------------------------------- estimates from history


def test_observed_cost_needs_data_before_it_says_anything(tmp_path):
    ledger = Ledger(tmp_path)
    assert ledger.observed_cost("claude-opus-5") == {"samples": 0}
    ledger.close()


def test_observed_cost_summarises_past_jobs(tmp_path):
    ledger = Ledger(tmp_path)
    for cost in [0.10, 0.20, 0.30, 0.40]:
        spec = _spec(name="a", kind=JobKind.AGENT, command=[],
                     agent={"task": "t", "backend": "claude-cli"})
        ledger.upsert_job(spec, "succeeded", _result(spec, cost=cost, model="claude-opus-5"))

    seen = ledger.observed_cost("claude-opus-5")
    assert seen["samples"] == 4
    # p75 is conservative without inheriting the tail of one runaway job.
    assert seen["cost_usd"] == pytest.approx(0.40)
    assert seen["median_cost_usd"] == pytest.approx(0.30)
    ledger.close()


def test_observed_cost_ignores_other_models_and_failures(tmp_path):
    ledger = Ledger(tmp_path)
    a = _spec(name="a", kind=JobKind.AGENT, command=[], agent={"task": "t"})
    ledger.upsert_job(a, "succeeded", _result(a, cost=1.0, model="claude-opus-5"))
    b = _spec(name="b", kind=JobKind.AGENT, command=[], agent={"task": "t"})
    ledger.upsert_job(b, "succeeded", _result(b, cost=9.0, model="claude-haiku-4-5"))
    c = _spec(name="c", kind=JobKind.AGENT, command=[], agent={"task": "t"})
    ledger.upsert_job(c, "failed", _result(c, cost=9.0, model="claude-opus-5"))

    assert ledger.observed_cost("claude-opus-5")["samples"] == 1
    ledger.close()


def test_a_quote_prefers_measurement_once_there_is_enough_of_it():
    from research_fleet.budget import quote

    guessed = quote("claude-opus-5", process="agent_standard")
    assert guessed.source == "estimated"

    measured = quote("claude-opus-5", process="agent_standard",
                     observed={"samples": 5, "cost_usd": 0.11,
                               "input_tokens": 30_000, "output_tokens": 900})
    assert measured.est_cost_usd == pytest.approx(0.11)
    assert "measured over 5" in measured.source


def test_a_quote_ignores_a_handful_of_samples():
    """Two data points are not a distribution; keep the profile until there are more."""
    from research_fleet.budget import quote

    q = quote("claude-opus-5", observed={"samples": 2, "cost_usd": 0.01})
    assert q.source == "estimated"
    assert q.est_cost_usd > 0.01


def test_the_calibrated_profiles_match_what_agents_actually_cost():
    """Measured agent jobs came in at $0.02 to $0.30. An estimate an order of magnitude
    above that refuses work which would never have breached the budget."""
    from research_fleet.budget import quote

    standard = quote("claude-opus-5", process="agent_standard").est_cost_usd
    assert 0.05 < standard < 0.60, f"opus agent_standard estimated at ${standard:.2f}"

    assert quote("claude-haiku-4-5").est_cost_usd < quote("claude-sonnet-5").est_cost_usd
    assert quote("claude-sonnet-5").est_cost_usd < quote("claude-opus-5").est_cost_usd
    assert quote("claude-opus-5", process="agent_short").est_cost_usd < standard
    assert quote("claude-opus-5", process="agent_long").est_cost_usd > standard
    assert quote("claude-opus-5", effort="low").est_cost_usd < \
        quote("claude-opus-5", effort="max").est_cost_usd


def test_fleet_quotes_from_its_own_history(tmp_path):
    fleet = _fleet(tmp_path)
    try:
        for cost in [0.05, 0.06, 0.07, 0.08]:
            spec = _spec(name="past", kind=JobKind.AGENT, command=[],
                         agent={"task": "t", "backend": "claude-cli"})
            fleet.ledger.upsert_job(
                spec, "succeeded", _result(spec, cost=cost, model="claude-opus-5"))

        q = fleet.quote("claude-opus-5")
        assert "measured" in q.source
        assert q.est_cost_usd == pytest.approx(0.08)
    finally:
        fleet.close()
