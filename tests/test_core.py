"""Tests for the parts that must not be wrong: audit integrity, budget
arithmetic, policy denials, and the end-to-end dry-run path."""

from __future__ import annotations

import json

import pytest

from research_fleet import (
    Fleet,
    JobSpec,
    Ledger,
    Mount,
    Policy,
    Redactor,
    Resources,
    build_sweep,
    expand_grid,
)
from research_fleet.budget import (
    BudgetExceeded,
    BudgetTracker,
    Usage,
    cost_for,
    quote,
)
from research_fleet.backends import get_backend
from research_fleet.scheduler import SlotPool
from research_fleet.spec import AgentConfig, JobKind


# --------------------------------------------------------------------- ledger


def test_ledger_chain_verifies(tmp_path):
    ledger = Ledger(tmp_path)
    for i in range(20):
        ledger.append("job.output", {"line": f"step {i}"}, run_id="run_1", job_id="job_1")
    ok, msg = ledger.verify()
    assert ok, msg
    assert "20 events" in msg
    ledger.close()


def test_ledger_detects_edited_payload(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.append("job.output", {"line": "clean"}, run_id="r", job_id="j")
    ledger.append("job.output", {"line": "also clean"}, run_id="r", job_id="j")
    ledger.close()

    path = tmp_path / "ledger.jsonl"
    lines = path.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["payload"]["line"] = "tampered"          # edit content, leave hash alone
    lines[0] = json.dumps(rec, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")

    ok, msg = Ledger(tmp_path).verify()
    assert not ok
    assert "payload edited" in msg


def test_ledger_detects_removed_record(tmp_path):
    ledger = Ledger(tmp_path)
    for i in range(5):
        ledger.append("t", {"i": i})
    ledger.close()

    path = tmp_path / "ledger.jsonl"
    lines = path.read_text().splitlines()
    del lines[2]                                  # excise a record entirely
    path.write_text("\n".join(lines) + "\n")

    ok, msg = Ledger(tmp_path).verify()
    assert not ok
    assert "seq gap" in msg


def test_ledger_survives_restart(tmp_path):
    a = Ledger(tmp_path)
    a.append("one", {})
    a.close()
    b = Ledger(tmp_path)          # must resume the chain, not restart it
    b.append("two", {})
    ok, msg = b.verify()
    assert ok, msg
    b.close()


def test_redactor_scrubs_keys_and_values():
    r = Redactor()
    scrubbed = r.scrub(
        {
            "api_key": "hunter2",
            "nested": {"AUTH_TOKEN": "abc"},
            "cmd": "export ANTHROPIC_KEY=sk-ant-api03-AAAABBBBCCCCDDDDEEEE and go",
            "safe": "learning rate 1e-3",
        }
    )
    assert scrubbed["api_key"] == "[REDACTED]"
    assert scrubbed["nested"]["AUTH_TOKEN"] == "[REDACTED]"
    assert "sk-ant" not in scrubbed["cmd"]
    assert scrubbed["safe"] == "learning rate 1e-3"


def test_secrets_never_reach_the_ledger(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.append("job.output", {"line": "token=ghp_ABCDEFGHIJKLMNOPQRST"})
    ledger.close()
    assert "ghp_ABCDEFGHIJKLMNOPQRST" not in (tmp_path / "ledger.jsonl").read_text()


# --------------------------------------------------------------------- budget


def test_cost_computation_matches_price_table():
    mc = cost_for("claude-opus-5")
    # 1M input + 1M output at $5/$25
    assert mc.cost_usd(input_tokens=1_000_000, output_tokens=1_000_000) == pytest.approx(30.0)
    # cache reads bill at 0.1x input
    assert mc.cost_usd(cache_read_tokens=1_000_000) == pytest.approx(0.5)


def test_cheaper_model_quotes_lower():
    opus = quote("claude-opus-5", effort="high")
    haiku = quote("claude-haiku-4-5", effort="high")
    assert haiku.est_cost_usd < opus.est_cost_usd


def test_effort_and_process_scale_the_quote():
    low = quote("claude-sonnet-5", effort="low")
    high = quote("claude-sonnet-5", effort="high")
    assert low.est_cost_usd < high.est_cost_usd

    short = quote("claude-sonnet-5", process="agent_short")
    long_ = quote("claude-sonnet-5", process="agent_long")
    assert long_.est_cost_usd > short.est_cost_usd * 5


def test_budget_rejects_overspend():
    b = BudgetTracker()
    b.open("run", max_usd=1.0, max_tokens=1_000_000)
    b.reserve("run", usd=0.9, tokens=100)
    with pytest.raises(BudgetExceeded):
        b.reserve("run", usd=0.5, tokens=100)


def test_child_cannot_exceed_parent_grant():
    b = BudgetTracker()
    b.open("run", max_usd=10.0, max_tokens=1_000_000)
    b.open("agent", max_usd=4.0, max_tokens=100_000, parent="run")
    with pytest.raises(BudgetExceeded):
        b.reserve("agent", usd=5.0, tokens=10)
    # and a sibling cannot claim more than the run has left
    with pytest.raises(BudgetExceeded):
        b.open("agent2", max_usd=9.0, max_tokens=10_000, parent="run")


def test_spend_rolls_up_to_ancestors():
    b = BudgetTracker()
    b.open("run", max_usd=10.0, max_tokens=10_000_000)
    b.open("parent", max_usd=6.0, max_tokens=1_000_000, parent="run")
    b.open("child", max_usd=3.0, max_tokens=500_000, parent="parent")

    usage = Usage(input_tokens=100_000, output_tokens=20_000, model="claude-opus-5")
    cost = b.commit("child", usage)

    assert b.get("child").spent_usd == pytest.approx(cost)
    assert b.get("parent").spent_usd == pytest.approx(cost)
    assert b.get("run").spent_usd == pytest.approx(cost)


def test_usage_merge_accumulates():
    a = Usage(input_tokens=10, output_tokens=5, model="claude-opus-5", requests=1)
    b = Usage(input_tokens=3, cache_read_tokens=100, requests=1)
    merged = a.merge(b)
    assert merged.input_tokens == 13
    assert merged.cache_read_tokens == 100
    assert merged.requests == 2
    assert merged.model == "claude-opus-5"


# --------------------------------------------------------------------- policy


def _agent_spec(**kw) -> JobSpec:
    base = dict(
        kind=JobKind.AGENT,
        name="a",
        agent=AgentConfig(task="do research"),
    )
    base.update(kw)
    return JobSpec(**base)


def test_policy_blocks_excess_recursion_depth():
    p = Policy(max_agent_depth=1)
    assert p.check(_agent_spec(), depth=2).verdict == "deny"
    assert p.check(_agent_spec(), depth=1).verdict == "allow"


def test_policy_blocks_fanout():
    p = Policy(max_children_per_agent=2)
    spec = _agent_spec(parent_job_id="job_parent")
    assert p.check(spec, sibling_count=2).verdict == "deny"
    assert p.check(spec, sibling_count=1).verdict == "allow"


def test_policy_blocks_dangerous_mounts(tmp_path):
    p = Policy(allowed_mount_roots=[str(tmp_path)])
    spec = JobSpec(name="c", command=["true"], mounts=[Mount(source="/etc", target="/etc")])
    d = p.check(spec)
    assert d.verdict == "deny"
    assert "denied path" in d.reasons[0]


def test_policy_blocks_mount_outside_allowlist(tmp_path):
    outside = tmp_path.parent / "elsewhere"
    outside.mkdir(exist_ok=True)
    p = Policy(allowed_mount_roots=[str(tmp_path)])
    spec = JobSpec(name="c", command=["true"], mounts=[Mount(source=str(outside), target="/x")])
    assert p.check(spec).verdict == "deny"


def test_policy_blocks_destructive_commands():
    p = Policy()
    for cmd in (["sh", "-c", "rm -rf /"], ["sh", "-c", "docker run --privileged x"]):
        assert p.check(JobSpec(name="c", command=cmd)).verdict == "deny"


def test_policy_clamps_timeout_instead_of_denying():
    p = Policy(max_timeout_s=100)
    d = p.check(JobSpec(name="c", command=["true"], timeout_s=10_000))
    assert d.verdict == "allow"
    assert d.mutations["timeout_s"] == 100


def test_policy_denies_over_budget_estimate():
    p = Policy(max_usd_per_job=0.001)
    d = p.check(_agent_spec(), estimate=quote("claude-opus-5", process="agent_long"))
    assert d.verdict == "deny"
    assert "max_usd_per_job" in d.reasons[0]


def test_policy_requires_approval_for_unrestricted_network():
    p = Policy()
    p.network.mode = "unrestricted"
    d = p.check(JobSpec(name="c", command=["true"]))
    assert d.verdict == "require_approval"


def test_container_policy_stays_out_of_the_ship_contract():
    """Fleet must not re-impose isolation research-ship already owns.

    --read-only / --user / --cap-drop ALL would break research-ship's writable
    venv, its non-root dev user, and the NET_ADMIN the firewall needs.
    """
    args = Policy().container.docker_args()
    assert "--pids-limit" in args
    for forbidden in ("--read-only", "--user", "--cap-drop"):
        assert forbidden not in args


# ------------------------------------------------------------------ scheduling


def test_slot_pool_hands_out_whole_devices():
    pool = SlotPool(["GPU-a", "GPU-b"])
    first = pool.acquire(1.0)
    second = pool.acquire(1.0)
    assert first and second and first != second
    assert pool.acquire(1.0, timeout=0.2) is None
    pool.release(first, 1.0)
    assert pool.acquire(1.0, timeout=1.0) is not None


def test_slot_pool_packs_fractional_jobs():
    pool = SlotPool(["GPU-a"])
    a = pool.acquire(0.5)
    b = pool.acquire(0.5)
    assert a == b == ("GPU-a",)          # both packed onto the same device
    assert pool.acquire(0.5, timeout=0.2) is None


# ---------------------------------------------------------------------- sweep


def test_grid_expansion_and_substitution():
    points = expand_grid({"lr": [1e-3, 3e-4], "depth": [6, 12]})
    assert len(points) == 4

    specs = build_sweep(["python", "train.py", "--lr", "{lr}", "--depth", "{depth}"],
                        {"lr": [1e-3], "depth": [6, 12]})
    assert len(specs) == 2
    assert specs[0].command[3] == "0.001"
    assert specs[1].params == {"lr": 1e-3, "depth": 12}


# -------------------------------------------------------------------- backends


def test_claude_backend_parses_usage_and_tools():
    backend = get_backend("claude-cli")
    line = json.dumps(
        {
            "type": "assistant",
            "message": {
                "model": "claude-opus-5",
                "content": [
                    {"type": "text", "text": "checking the data"},
                    {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
                ],
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_read_input_tokens": 900,
                },
            },
        }
    )
    ev = backend.parse_line(line)
    assert ev.type == "tool_use"
    assert ev.tool == "Bash"
    assert ev.usage.input_tokens == 100
    assert ev.usage.cache_read_tokens == 900
    assert ev.usage.model == "claude-opus-5"


def test_claude_backend_keeps_unparseable_lines():
    ev = get_backend("claude-cli").parse_line("warning: something happened")
    assert ev.type == "raw" and "warning" in ev.text


def test_agent_command_includes_budget_brief():
    backend = get_backend("claude-cli")
    argv = backend.build_command(
        AgentConfig(task="find the bug", model="claude-opus-5"), brief="## Budget\n$5 left"
    )
    assert "-p" in argv
    prompt = argv[argv.index("-p") + 1]
    assert "$5 left" in prompt and "find the bug" in prompt
    assert argv[argv.index("--output-format") + 1] == "stream-json"


# ------------------------------------------------------------------ end to end


def test_dry_run_end_to_end(tmp_path):
    fleet = Fleet(
        root=str(tmp_path / "state"),
        workspace=str(tmp_path),
        executor={"kind": "dry-run"},
        budget={"max_usd": 5.0},
    )
    try:
        fleet.run_sweep(["python", "train.py", "--lr", "{lr}"], {"lr": [1e-3, 3e-4]}, gpus=0)
        report = fleet.wait(timeout=60)

        assert len(report.succeeded) == 2
        assert not report.failed

        ok, msg = fleet.verify_audit()
        assert ok, msg
    finally:
        fleet.close()


def test_denied_job_never_runs_and_is_recorded(tmp_path):
    fleet = Fleet(
        root=str(tmp_path / "state"),
        workspace=str(tmp_path),
        executor={"kind": "dry-run"},
    )
    try:
        rec = fleet.submit(JobSpec(name="bad", command=["sh", "-c", "rm -rf /"], resources=Resources(gpus=0)))
        assert rec.state.value == "denied"

        types = {e.type for e in fleet.ledger.events(job_id=rec.spec.id)}
        assert "job.submitted" in types and "job.denied" in types
        ok, _ = fleet.verify_audit()
        assert ok
    finally:
        fleet.close()


# ------------------------------------------------------- config layering


def test_env_vars_override_config_with_double_underscore_nesting(tmp_path, monkeypatch):
    """FLEET_EXECUTOR__KIND=ray must reach executor.kind."""
    monkeypatch.setenv("FLEET_EXECUTOR__KIND", "dry-run")
    monkeypatch.setenv("FLEET_BUDGET__MAX_USD", "7.5")
    monkeypatch.setenv("FLEET_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("FLEET_CONFIG", str(tmp_path / "absent.yaml"))

    from research_fleet.config import load_config

    cfg = load_config()
    assert cfg.executor.kind == "dry-run"
    assert cfg.budget.max_usd == 7.5          # JSON-decoded to a float, not left a string
    assert cfg.workspace == str(tmp_path)


def test_explicit_kwargs_beat_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("FLEET_BUDGET__MAX_USD", "1.0")
    monkeypatch.setenv("FLEET_CONFIG", str(tmp_path / "absent.yaml"))

    from research_fleet.config import load_config

    assert load_config(budget={"max_usd": 42.0}).budget.max_usd == 42.0


def test_a_project_yaml_is_read_and_merged(tmp_path):
    (tmp_path / "fleet.yaml").write_text(
        "image: custom:latest\nbudget:\n  max_usd: 3.0\n  default_effort: low\n"
    )
    from research_fleet.config import load_config

    cfg = load_config(tmp_path / "fleet.yaml")
    assert cfg.image == "custom:latest"
    assert cfg.budget.max_usd == 3.0
    assert cfg.budget.default_effort == "low"
    # Unmentioned keys keep their defaults rather than being cleared.
    assert cfg.budget.default_model == "claude-opus-5"


def test_workspace_becomes_the_default_mount_allowlist(tmp_path):
    from research_fleet.config import FleetConfig

    cfg = FleetConfig(root=str(tmp_path / "root"), workspace=str(tmp_path))
    assert str(tmp_path.resolve()) in cfg.policy.allowed_mount_roots


def test_custom_model_prices_are_registered_from_config(tmp_path):
    from research_fleet.budget import cost_for
    from research_fleet.config import FleetConfig

    FleetConfig(
        root=str(tmp_path),
        budget={"model_costs": {"my-local-model": {"input_per_mtok": 0.0, "output_per_mtok": 0.0}}},
    )
    assert cost_for("my-local-model").cost_usd(input_tokens=10_000_000) == 0.0


# ------------------------------------------------- remaining policy branches


def test_policy_denies_when_the_run_budget_is_already_spent():
    b = BudgetTracker()
    b.open("run", max_usd=0.01, max_tokens=100)
    p = Policy()
    d = p.check(_agent_spec(), budget=b, budget_scope="run",
                estimate=quote("claude-opus-5", process="agent_standard"))
    assert d.verdict == "deny"
    assert "budget exceeded" in d.reasons[0]


def test_policy_reserves_the_estimate_when_it_allows_a_job():
    b = BudgetTracker()
    b.open("run", max_usd=100.0, max_tokens=100_000_000)
    p = Policy()
    est = quote("claude-haiku-4-5", process="agent_short")
    d = p.check(_agent_spec(), budget=b, budget_scope="run", estimate=est)
    assert d.verdict == "allow"
    assert d.mutations["budget_reserved"]["usd"] == est.est_cost_usd
    assert b.get("run").reserved_usd == pytest.approx(est.est_cost_usd)


def test_policy_denies_a_mount_path_it_cannot_resolve(tmp_path):
    """An agent-submitted spec can carry a path that is not resolvable at all."""
    p = Policy(allowed_mount_roots=[str(tmp_path)], fail_closed=True)
    spec = JobSpec(name="c", command=["true"], mounts=[Mount(source="bad\0path", target="/x")])
    d = p.check(spec)
    assert d.verdict == "deny"
    assert "cannot resolve" in d.reasons[0]


def test_policy_allows_a_mount_target_that_does_not_exist_yet(tmp_path):
    """Results directories are created by the job, so they legitimately do not exist yet.
    Path.resolve() is non-strict, which is what makes this work."""
    p = Policy(allowed_mount_roots=[str(tmp_path)], fail_closed=True)
    spec = JobSpec(name="c", command=["true"],
                   mounts=[Mount(source=str(tmp_path / "results" / "run1"), target="/results")])
    assert p.check(spec).verdict == "allow"


def test_policy_merges_its_tool_denylist_into_the_job(tmp_path):
    p = Policy(agent_default_disallowed_tools=["WebFetch"])
    spec = _agent_spec()
    spec.agent.disallowed_tools = ["Bash"]
    d = p.check(spec)
    assert d.verdict == "allow"
    assert d.mutations["agent_disallowed_tools"] == ["Bash", "WebFetch"]


def test_policy_gate_can_be_disabled_by_emptying_require_approval_for():
    p = Policy(require_approval_for=[])
    p.network.mode = "unrestricted"
    assert p.check(JobSpec(name="c", command=["true"])).verdict == "allow"
