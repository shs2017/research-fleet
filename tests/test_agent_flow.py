"""Agent-specific behaviour: budget briefing, spool intake, hierarchical grants.

These use a stub executor that impersonates an agent: it emits Claude-CLI
stream-json on stdout and writes child job specs into the mounted spool
directory, exactly as a real agent would. That exercises the whole submit →
policy → budget → ledger path for spawned work without spending money.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from research_fleet import Fleet
from research_fleet.spec import JobKind, JobResult, JobState


class StubAgentExecutor:
    """Pretends to be an agent container: emits usage, optionally spawns children."""

    kind = "stub-agent"

    def __init__(self, children: list[dict] | None = None, tokens: tuple[int, int] = (10_000, 2_000)):
        self.children = children or []
        self.tokens = tokens
        self.seen_argv: list[list[str]] = []
        self.seen_env: list[dict] = []

    def available_gpus(self) -> list[str]:
        return ["GPU-stub-0", "GPU-stub-1"]

    def run(self, spec, *, argv, env, placement, policy, on_line) -> JobResult:
        self.seen_argv.append(argv)
        self.seen_env.append(dict(env))

        if spec.kind is JobKind.AGENT:
            on_line("stdout", json.dumps({"type": "system", "subtype": "init"}))
            on_line(
                "stdout",
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "model": "claude-opus-5",
                            "content": [{"type": "text", "text": "planning the experiment"}],
                            "usage": {
                                "input_tokens": self.tokens[0],
                                "output_tokens": self.tokens[1],
                            },
                        },
                    }
                ),
            )
            # An agent delegating work writes specs into its spool mount.
            spool = next((Path(m.source) for m in spec.mounts if m.target == "/spool"), None)
            if spool is not None:
                for i, child in enumerate(self.children):
                    (spool / f"child-{i}.json").write_text(json.dumps(child), encoding="utf-8")
            on_line("stdout", json.dumps({"type": "result", "subtype": "success", "result": "done"}))

        now = time.time()
        return JobResult(
            job_id=spec.id, state=JobState.SUCCEEDED, exit_code=0,
            started_at=now, ended_at=now, gpu_ids=list(placement.gpu_ids),
        )

    def cancel(self, job_id: str) -> bool:
        return True

    def close(self) -> None:
        return None


class LiveSpoolExecutor(StubAgentExecutor):
    """Keeps the parent alive until its child starts."""

    def __init__(self, child: dict):
        super().__init__(children=[child])
        self.child_started = threading.Event()

    def run(self, spec, *, argv, env, placement, policy, on_line) -> JobResult:
        if spec.kind is JobKind.AGENT:
            spool = next(Path(m.source) for m in spec.mounts if m.target == "/spool")
            (spool / "live-child.json").write_text(json.dumps(self.children[0]), encoding="utf-8")
            assert self.child_started.wait(5), "child was not submitted while parent remained alive"
        else:
            self.child_started.set()
        now = time.time()
        return JobResult(
            job_id=spec.id, state=JobState.SUCCEEDED, exit_code=0,
            started_at=now, ended_at=now, gpu_ids=list(placement.gpu_ids),
        )


def _fleet(tmp_path, executor, **overrides) -> Fleet:
    fleet = Fleet(
        root=str(tmp_path / "state"),
        workspace=str(tmp_path),
        executor={"kind": "dry-run"},
        **overrides,
    )
    # Swap in the stub after construction so config validation still runs.
    fleet.executor = executor
    fleet.scheduler.executor = executor
    return fleet


def test_agent_prompt_carries_budget_and_delegation_brief(tmp_path):
    stub = StubAgentExecutor()
    fleet = _fleet(tmp_path, stub, budget={"max_usd": 30.0})
    try:
        fleet.run_agents("investigate the loss spike", n=1, gpus=1)
        fleet.wait(timeout=30)

        argv = stub.seen_argv[0]
        # The task is passed verbatim; the brief rides in the system prompt so a
        # leading slash command in the task keeps working (`claude -p` only honours
        # one at position 0).
        assert argv[argv.index("-p") + 1] == "investigate the loss spike"
        brief = argv[argv.index("--append-system-prompt") + 1]
        assert "## Budget" in brief
        assert "claude-haiku-4-5" in brief           # cheaper option is offered
        assert "$FLEET_SUBMIT_DIR" in brief          # knows how to delegate

        env = stub.seen_env[0]
        assert env["FLEET_SUBMIT_DIR"] == "/spool"
        assert float(env["FLEET_BUDGET_USD"]) > 0
        # Secrets are forwarded by name only, never a literal value.
        assert env["ANTHROPIC_API_KEY"] == ""
    finally:
        fleet.close()


def test_agent_at_max_depth_is_told_it_cannot_delegate(tmp_path):
    stub = StubAgentExecutor()
    fleet = _fleet(tmp_path, stub, policy={"max_agent_depth": 0})
    try:
        fleet.run_agents("do it yourself", n=1, gpus=1)
        fleet.wait(timeout=30)
        argv = stub.seen_argv[0]
        assert "cannot launch sub-agents" in argv[argv.index("--append-system-prompt") + 1]
    finally:
        fleet.close()


def test_agent_usage_is_billed_and_rolled_up(tmp_path):
    stub = StubAgentExecutor(tokens=(100_000, 20_000))
    fleet = _fleet(tmp_path, stub, budget={"max_usd": 30.0})
    try:
        fleet.run_agents("measure me", n=1, gpus=1)
        report = fleet.wait(timeout=30)

        # 100k input @ $5/Mtok + 20k output @ $25/Mtok = $0.50 + $0.50
        assert report.total_cost_usd == pytest.approx(1.0, rel=0.01)
        assert report.total_tokens == 120_000

        committed = [e for e in fleet.ledger.events(types=["budget.committed"])]
        assert committed, "spend must be recorded in the audit ledger"
        assert committed[0].payload["cost_usd"] == pytest.approx(1.0, rel=0.01)
    finally:
        fleet.close()


def test_agent_spawned_child_runs_and_records_provenance(tmp_path):
    child = {
        "kind": "command",
        "name": "child-job",
        "command": ["python", "train.py", "--lr", "3e-4"],
        "resources": {"gpus": 1},
    }
    stub = StubAgentExecutor(children=[child])
    fleet = _fleet(tmp_path, stub, budget={"max_usd": 30.0})
    try:
        fleet.run_agents("delegate a training run", n=1, gpus=1)
        report = fleet.wait(timeout=60)

        names = {r.spec.name for r in fleet.scheduler._jobs.values()}
        assert "child-job" in names

        child_rec = next(r for r in fleet.scheduler._jobs.values() if r.spec.name == "child-job")
        parent_rec = next(r for r in fleet.scheduler._jobs.values() if r.spec.name == "agent")
        assert child_rec.spec.parent_job_id == parent_rec.spec.id
        assert child_rec.depth == 1

        # The delegation itself is auditable, with the originating file named.
        requested = fleet.ledger.events(types=["agent.job_requested"])
        assert requested and requested[0].payload["parent"] == parent_rec.spec.id
        assert requested[0].payload["source_file"] == "child-0.json"

        assert len(report.succeeded) == 2
        ok, msg = fleet.verify_audit()
        assert ok, msg
    finally:
        fleet.close()


def test_agent_spool_is_drained_while_parent_is_still_running(tmp_path):
    child = {
        "kind": "command", "name": "live-child", "command": ["true"],
        "resources": {"gpus": 1},
    }
    stub = LiveSpoolExecutor(child)
    fleet = _fleet(tmp_path, stub, budget={"max_usd": 30.0})
    try:
        fleet.run_agents("adapt after the experiment", n=1, gpus=1)
        report = fleet.wait(timeout=30)
        assert stub.child_started.is_set()
        succeeded_ids = {result.job_id for result in report.succeeded}
        succeeded_names = {
            record.spec.name for record in fleet.scheduler._jobs.values()
            if record.spec.id in succeeded_ids
        }
        assert succeeded_names >= {"agent", "live-child"}
    finally:
        fleet.close()


def test_child_exceeding_policy_is_rejected_with_a_readable_reason(tmp_path):
    # A child asking for a destructive command must be denied, and the agent
    # must be able to read *why* from its spool directory.
    child = {"kind": "command", "name": "bad-child", "command": ["sh", "-c", "rm -rf /"]}
    stub = StubAgentExecutor(children=[child])
    fleet = _fleet(tmp_path, stub)
    try:
        fleet.run_agents("try something dangerous", n=1, gpus=1)
        fleet.wait(timeout=30)

        denied = [r for r in fleet.scheduler._jobs.values() if r.state is JobState.DENIED]
        assert denied and denied[0].spec.name == "bad-child"

        rejections = fleet.ledger.events(types=["agent.job_rejected"])
        assert rejections
        assert "denied pattern" in rejections[0].payload["reason"]
    finally:
        fleet.close()


def test_child_budget_is_carved_from_the_parent(tmp_path):
    stub = StubAgentExecutor()
    fleet = _fleet(
        tmp_path, stub,
        budget={"max_usd": 20.0, "max_child_grant_fraction": 0.5},
        policy={"max_usd_per_job": 100.0},
    )
    try:
        fleet.run_agents("spend carefully", n=1, gpus=1)
        fleet.wait(timeout=30)

        snapshot = fleet.scheduler.budget.snapshot()
        run_scope = snapshot[fleet.run_id]
        agent_scopes = [v for k, v in snapshot.items() if v["parent"] == fleet.run_id]

        assert agent_scopes, "the agent must get its own budget scope"
        # The grant is bounded by the parent's balance, never larger than it.
        assert agent_scopes[0]["max_usd"] <= run_scope["max_usd"]
    finally:
        fleet.close()


def test_scope_closes_only_after_children_finish(tmp_path):
    """A grant returned early would let a late child outspend its parent's ceiling."""
    child = {
        "kind": "command", "name": "slow-child",
        "command": ["sleep", "0"], "resources": {"gpus": 0},
    }
    stub = StubAgentExecutor(children=[child])
    fleet = _fleet(tmp_path, stub, budget={"max_usd": 20.0})
    try:
        fleet.run_agents("delegate then exit", n=1, gpus=1)
        fleet.wait(timeout=60)

        agent_rec = next(r for r in fleet.scheduler._jobs.values() if r.spec.kind is JobKind.AGENT)
        child_rec = next(r for r in fleet.scheduler._jobs.values() if r.spec.name == "slow-child")

        assert child_rec.state.terminal
        assert not agent_rec.owns_scope, "scope should be closed once the child finished"

        closes = fleet.ledger.events(types=["budget.scope_closed"])
        assert closes, "scope closure must be auditable"

        # After reconciliation the run is charged the real spend, not the grant.
        snap = fleet.scheduler.budget.snapshot()
        run = snap[fleet.run_id]
        assert run["reserved_usd"] == pytest.approx(0.0, abs=1e-6)
        assert run["spent_usd"] == pytest.approx(agent_rec.usage.cost_usd(), rel=0.01)
    finally:
        fleet.close()


def test_fanout_limit_stops_a_delegation_storm(tmp_path):
    children = [
        {"kind": "command", "name": f"c{i}", "command": ["echo", str(i)], "resources": {"gpus": 0}}
        for i in range(6)
    ]
    stub = StubAgentExecutor(children=children)
    fleet = _fleet(tmp_path, stub, policy={"max_children_per_agent": 2})
    try:
        fleet.run_agents("spawn everything", n=1, gpus=1)
        fleet.wait(timeout=60)

        denied = [r for r in fleet.scheduler._jobs.values() if r.state is JobState.DENIED]
        assert len(denied) == 4, "only the first 2 children should be admitted"
        assert "max_children_per_agent" in denied[0].result.error
    finally:
        fleet.close()
