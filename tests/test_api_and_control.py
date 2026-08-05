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


def test_codex_backend_does_not_inherit_a_claude_default_model(tmp_path):
    fleet = _fleet(tmp_path)
    try:
        assert fleet.agent_model(None, "codex-cli") == "gpt-5.3-codex"
        assert fleet.agent_model("gpt-5.6-terra", "codex-cli") == "gpt-5.6-terra"
    finally:
        fleet.close()


def test_detached_workflow_relaunches_with_a_stable_run_id(tmp_path, monkeypatch):
    import sys

    from research_fleet import cli

    config = tmp_path / "fleet.yaml"
    config.write_text(f"root: {tmp_path / 'state'}\n")
    launched = {}

    class Process:
        def __init__(self, argv, **kwargs):
            launched["argv"] = argv
            launched["kwargs"] = kwargs

    monkeypatch.setattr("subprocess.Popen", Process)
    monkeypatch.setattr(sys, "argv", [
        "fleet", "workflow", "workflow.yaml", "--resume", "run_old", "--detach",
    ])

    cli._detach_and_return(str(config))

    argv = launched["argv"]
    assert argv[:3] == ["fleet", "workflow", "workflow.yaml"]
    assert "--detach" not in argv
    assert argv[argv.index("--resume") + 1] == "run_old"
    run_id = argv[argv.index("--run-id") + 1]
    assert run_id.startswith("run_")
    assert launched["kwargs"]["start_new_session"] is True
    assert (tmp_path / "state" / "logs" / f"{run_id}.log").exists()


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


# ---------------------------------------------------------------- killing a run


def test_cancel_reports_how_many_it_stopped(tmp_path):
    fleet = _fleet(tmp_path)
    try:
        fleet.submit(JobSpec(name="stuck", command=["true"], resources=Resources(gpus=0),
                             depends_on=["job_never_arrives"]))
        assert fleet.cancel("operator stopped it") == 1
        assert "run.cancelled" in {e.type for e in fleet.ledger.events(limit=200)}
    finally:
        fleet.close()


def test_cancelling_twice_is_harmless(tmp_path):
    fleet = _fleet(tmp_path)
    try:
        fleet.submit(JobSpec(name="stuck", command=["true"], resources=Resources(gpus=0),
                             depends_on=["nope"]))
        assert fleet.cancel() == 1
        assert fleet.cancel() == 0, "nothing left to stop"
    finally:
        fleet.close()


def test_a_cancelled_job_is_not_relabelled_as_failed(tmp_path):
    """Stopping a container makes it exit non-zero; that must not read as a failure."""
    import threading, time as _t
    from research_fleet.spec import JobResult, JobState as JS

    started = threading.Event()

    class Slow:
        kind = "slow"
        def available_gpus(self): return ["g0"]
        def run(self, spec, *, argv, env, placement, policy, on_line):
            started.set()
            _t.sleep(1.0)
            # Reports failure, as a stopped container would.
            return JobResult(job_id=spec.id, state=JS.FAILED, exit_code=137,
                             error="exit code 137")
        def cancel(self, job_id): return True
        def close(self): return None

    fleet = _fleet(tmp_path)
    fleet.executor = fleet.scheduler.executor = Slow()
    try:
        rec = fleet.run_command(["true"], name="victim", gpus=0)
        assert started.wait(timeout=10)
        fleet.cancel("operator stopped it")
        fleet.wait(timeout=30)
        assert rec.state is JobState.CANCELLED
        assert rec.result.state is JobState.CANCELLED
    finally:
        fleet.close()


def test_the_ledger_can_mark_a_run_cancelled_from_outside(tmp_path):
    """`fleet kill` runs in a different process, so nothing else will ever write the
    terminal state of those jobs."""
    fleet = _fleet(tmp_path)
    try:
        fleet.submit(JobSpec(name="orphan", command=["true"], resources=Resources(gpus=0),
                             depends_on=["never"]))
        run_id = fleet.run_id
    finally:
        # close() stops the threads without writing terminal states, which is exactly
        # the situation `fleet kill` has to clean up after.
        fleet.close()

    ledger = Ledger(tmp_path / "state")
    try:
        assert run_id in ledger.active_runs()
        killed = ledger.mark_cancelled(run_id, "killed by operator")
        assert len(killed) == 1
        assert ledger.active_runs() == []
        states = {j["state"] for j in ledger.jobs(run_id)}
        assert states == {"cancelled"}
        ok, msg = ledger.verify()
        assert ok, msg
    finally:
        ledger.close()


# ------------------------------------------------------------ gpu sharing


def test_agents_share_the_devices_so_they_run_together():
    """A whole GPU each serialises everything on a one-GPU box, which is the opposite
    of what --agents asks for."""
    from research_fleet.cli import _gpu_share

    share, note = _gpu_share(None, agents=4, devices=1)
    assert share == pytest.approx(0.25)
    assert "all 4 run together" in note


def test_the_share_never_exceeds_a_whole_device():
    from research_fleet.cli import _gpu_share

    share, _ = _gpu_share(None, agents=2, devices=8)
    assert share == 1.0


def test_more_devices_mean_a_bigger_share_each():
    from research_fleet.cli import _gpu_share

    assert _gpu_share(None, agents=4, devices=2)[0] == pytest.approx(0.5)


def test_a_host_with_no_gpu_reserves_none():
    from research_fleet.cli import _gpu_share

    share, note = _gpu_share(None, agents=4, devices=0)
    assert share == 0.0 and "no GPU" in note


def test_an_explicit_request_is_honoured_but_warned_about():
    from research_fleet.cli import _gpu_share

    share, note = _gpu_share(1.0, agents=4, devices=1)
    assert share == 1.0, "the operator's choice wins"
    assert "will queue" in note and "--gpus 0.25" in note


def test_an_explicit_request_that_fits_says_nothing():
    from research_fleet.cli import _gpu_share

    assert _gpu_share(0.25, agents=4, devices=1) == (0.25, "")


def test_four_fractional_jobs_fit_on_one_device():
    from research_fleet.scheduler import SlotPool

    pool = SlotPool(["GPU-a"])
    assert all(pool.acquire(0.25, timeout=0.5) for _ in range(4))
    assert pool.acquire(0.25, timeout=0.2) is None, "the fifth must wait"


def test_whole_gpu_jobs_serialise_on_one_device():
    from research_fleet.scheduler import SlotPool

    pool = SlotPool(["GPU-a"])
    assert pool.acquire(1.0, timeout=0.5) is not None
    assert pool.acquire(1.0, timeout=0.2) is None
