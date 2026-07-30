"""The public API surface and the scheduler's control paths.

Covers what a caller touches directly (grid parsing, RunReport, quotes, traces) and the
state transitions that only happen when something goes wrong: approval, denial, failed
dependencies, and cancellation.
"""

from __future__ import annotations

import pytest

from research_fleet import Fleet, JobSpec, Ledger, Resources
from research_fleet.spec import AgentConfig, JobKind, JobState
from research_fleet.sweep import build_sweep, expand_grid, parse_grid_args


def _fleet(tmp_path, **overrides):
    return Fleet(
        root=str(tmp_path / "state"),
        workspace=str(tmp_path),
        executor={"kind": "dry-run"},
        **overrides,
    )


# ---------------------------------------------------------------- grid parsing


def test_grid_args_coerce_to_the_narrowest_type():
    grid = parse_grid_args(["depth=6,12", "lr=1e-3,3e-4", "name=base,tuned"])
    assert grid["depth"] == [6, 12]                    # ints stay ints
    assert grid["lr"] == [1e-3, 3e-4]                  # floats stay floats
    assert grid["name"] == ["base", "tuned"]           # strings stay strings


def test_grid_args_tolerate_whitespace_and_single_values():
    assert parse_grid_args([" lr = 1e-3 "]) == {"lr": [1e-3]}


def test_grid_args_reject_a_missing_equals():
    with pytest.raises(ValueError, match="name=v1,v2"):
        parse_grid_args(["justakey"])


def test_empty_grid_yields_one_point():
    assert expand_grid({}) == [{}]


def test_sweep_from_explicit_points_skips_the_product():
    specs = build_sweep(["train", "--lr", "{lr}"], points=[{"lr": 0.1}, {"lr": 0.2}])
    assert [s.command[-1] for s in specs] == ["0.1", "0.2"]


def test_sweep_names_stay_within_docker_limits():
    specs = build_sweep(["x"], {"averyverylongparametername": list(range(3))})
    assert all(len(s.name) <= 120 for s in specs)


# ------------------------------------------------------------------ RunReport


def test_run_report_summarises_counts_cost_and_failures(tmp_path):
    fleet = _fleet(tmp_path, budget={"max_usd": 5.0})
    try:
        fleet.run_sweep(["true"], {"x": [1, 2]}, gpus=0)
        fleet.submit(JobSpec(name="bad", command=["sh", "-c", "rm -rf /"], resources=Resources(gpus=0)))
        report = fleet.wait(timeout=60)

        assert len(report.succeeded) == 2
        assert len(report.failed) == 1              # the denied job counts as failed
        assert report.total_cost_usd == 0.0         # no agent jobs, so no spend
        assert report.total_tokens == 0

        text = report.summary()
        assert "2 succeeded, 1 failed" in text
        assert report.run_id in text
    finally:
        fleet.close()


def test_run_report_flags_jobs_waiting_on_a_human(tmp_path):
    fleet = _fleet(tmp_path, policy={"network": {"mode": "unrestricted"}})
    try:
        fleet.run_command(["true"], name="gated", gpus=0)
        report = fleet.wait(timeout=30)
        assert len(report.awaiting_approval) == 1
        assert "need approval" in report.summary()
    finally:
        fleet.close()


# ------------------------------------------------------ quotes and traces


def test_quote_and_cost_menu_come_from_config(tmp_path):
    fleet = _fleet(tmp_path, budget={"default_model": "claude-haiku-4-5"})
    try:
        assert fleet.quote().model == "claude-haiku-4-5"
        assert fleet.quote("claude-opus-5").est_cost_usd > fleet.quote().est_cost_usd
        models = {row["model"] for row in fleet.cost_menu()}
        assert "claude-haiku-4-5" in models
    finally:
        fleet.close()


def test_trace_returns_one_jobs_events_in_order(tmp_path):
    fleet = _fleet(tmp_path)
    try:
        rec = fleet.run_command(["true"], name="traced", gpus=0)
        fleet.wait(timeout=30)
        trace = fleet.trace(rec.spec.id)
        assert [e["type"] for e in trace][:2] == ["job.submitted", "job.queued"]
        assert all(e["seq"] < nxt["seq"] for e, nxt in zip(trace, trace[1:]))
    finally:
        fleet.close()


# ------------------------------------------------------------ approval gates


def test_approving_a_gated_job_lets_it_run(tmp_path):
    fleet = _fleet(tmp_path, policy={"network": {"mode": "unrestricted"}})
    try:
        rec = fleet.run_command(["true"], name="gated", gpus=0)
        fleet.wait(timeout=30)
        assert rec.state is JobState.AWAITING_APPROVAL

        assert fleet.approve(rec.spec.id) is True
        report = fleet.wait(timeout=30)
        assert len(report.succeeded) == 1
        assert "job.approved" in {e.type for e in fleet.ledger.events(job_id=rec.spec.id)}
    finally:
        fleet.close()


def test_denying_a_gated_job_records_the_reason(tmp_path):
    fleet = _fleet(tmp_path, policy={"network": {"mode": "unrestricted"}})
    try:
        rec = fleet.run_command(["true"], name="gated", gpus=0)
        fleet.wait(timeout=30)
        assert fleet.deny(rec.spec.id, "not today") is True
        assert rec.state is JobState.DENIED
        assert "not today" in rec.result.error
    finally:
        fleet.close()


def test_approve_and_deny_ignore_jobs_that_are_not_parked(tmp_path):
    fleet = _fleet(tmp_path)
    try:
        rec = fleet.run_command(["true"], name="plain", gpus=0)
        fleet.wait(timeout=30)
        assert fleet.approve(rec.spec.id) is False
        assert fleet.deny(rec.spec.id) is False
        assert fleet.approve("job_does_not_exist") is False
    finally:
        fleet.close()


# -------------------------------------------------------------- dependencies


def test_a_dependent_job_waits_for_its_dependency(tmp_path):
    fleet = _fleet(tmp_path)
    try:
        first = fleet.submit(JobSpec(name="first", command=["true"], resources=Resources(gpus=0)))
        second = fleet.submit(
            JobSpec(name="second", command=["true"], resources=Resources(gpus=0),
                    depends_on=[first.spec.id])
        )
        report = fleet.wait(timeout=60)
        assert len(report.succeeded) == 2
        assert report.results[first.spec.id].ended_at <= report.results[second.spec.id].started_at
    finally:
        fleet.close()


def test_a_dependent_job_is_cancelled_when_its_dependency_is_denied(tmp_path):
    fleet = _fleet(tmp_path)
    try:
        blocked = fleet.submit(
            JobSpec(name="blocked", command=["sh", "-c", "rm -rf /"], resources=Resources(gpus=0))
        )
        assert blocked.state is JobState.DENIED

        follower = fleet.submit(
            JobSpec(name="follower", command=["true"], resources=Resources(gpus=0),
                    depends_on=[blocked.spec.id])
        )
        fleet.wait(timeout=60)
        assert follower.state is JobState.CANCELLED
        reasons = [e.payload.get("reason", "") for e in fleet.ledger.events(job_id=follower.spec.id)]
        assert any("dependency" in r for r in reasons)
    finally:
        fleet.close()


def test_cancel_all_marks_outstanding_jobs_cancelled(tmp_path):
    fleet = _fleet(tmp_path)
    try:
        rec = fleet.submit(
            JobSpec(name="pending", command=["true"], resources=Resources(gpus=0),
                    depends_on=["job_that_never_arrives"])
        )
        fleet.cancel("operator stopped it")
        fleet.wait(timeout=30)
        assert rec.state in {JobState.CANCELLED, JobState.PENDING, JobState.QUEUED}
    finally:
        fleet.close()


# -------------------------------------------------------------- ledger extras


def test_runs_and_jobs_projections_are_queryable(tmp_path):
    fleet = _fleet(tmp_path)
    try:
        fleet.run_sweep(["true"], {"x": [1, 2]}, gpus=0)
        fleet.wait(timeout=60)
        run_id = fleet.run_id
    finally:
        fleet.close()

    ledger = Ledger(tmp_path / "state")
    try:
        runs = {r["run_id"]: r for r in ledger.runs()}
        assert runs[run_id]["jobs"] == 2 and runs[run_id]["succeeded"] == 2
        assert len(ledger.jobs(run_id)) == 2
        assert ledger.jobs("no_such_run") == []
    finally:
        ledger.close()


def test_reindex_rebuilds_the_query_index_from_the_jsonl(tmp_path):
    ledger = Ledger(tmp_path)
    for i in range(5):
        ledger.append("t", {"i": i}, run_id="r", job_id="j")
    assert len(ledger.events(job_id="j")) == 5

    ledger._db.execute("DELETE FROM events")
    ledger._db.commit()
    assert ledger.events(job_id="j") == []

    assert ledger.reindex() == 5
    assert len(ledger.events(job_id="j")) == 5
    ledger.close()


def test_events_can_be_filtered_by_type(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.append("alpha", {}, job_id="j")
    ledger.append("beta", {}, job_id="j")
    ledger.append("alpha", {}, job_id="j")
    assert len(ledger.events(job_id="j", types=["alpha"])) == 2
    assert len(ledger.events(job_id="j", limit=1)) == 1
    ledger.close()


# ------------------------------------------------------------------- config


def test_agent_jobs_require_an_agent_block():
    with pytest.raises(ValueError, match="agent jobs require"):
        JobSpec(kind=JobKind.AGENT, name="a")


def test_command_jobs_require_a_command():
    with pytest.raises(ValueError, match="command jobs require"):
        JobSpec(name="c")


def test_a_spec_fingerprint_ignores_creation_time_but_tracks_content():
    a = JobSpec(name="x", command=["true"], id="job_fixed")
    b = JobSpec(name="x", command=["true"], id="job_fixed")
    assert a.fingerprint() == b.fingerprint()

    c = JobSpec(name="x", command=["false"], id="job_fixed")
    assert c.fingerprint() != a.fingerprint()


def test_a_job_name_defaults_to_its_id():
    spec = JobSpec(command=["true"])
    assert spec.name == spec.id


def test_agent_config_defaults_are_conservative():
    cfg = AgentConfig(task="t")
    assert cfg.backend == "claude-cli"
    assert cfg.allowed_tools is None and cfg.disallowed_tools == []
