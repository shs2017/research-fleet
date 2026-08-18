"""ShipExecutor argv composition.

Uses a fake `ship` script so the contract with research-ship is tested without
needing Docker, a GPU, or a built image. What matters here is the seam: the
flags we pass *to* the ship, and the limits we layer *onto* what it returns.
"""

from __future__ import annotations

import os
import json
import stat
import subprocess

import pytest

from research_fleet.executors import Placement
from research_fleet.executors.ship_exec import ShipExecutor, ShipUnavailable
from research_fleet.policy import Policy
from research_fleet.spec import JobResult, JobSpec, JobState, Mount, Resources

FAKE_SHIP = """#!/usr/bin/env bash
# Records the flags it was called with, then emits a plausible argv.
# Honours --image the way the real ship does, so image selection is tested
# at the seam rather than by rewriting tokens downstream.
printf '%s\\n' "$@" > "${FAKE_SHIP_LOG}"
image=ship-proj:latest
args=("$@")
for i in "${!args[@]}"; do
    [[ "${args[$i]}" == "--image" ]] && image="${args[$((i+1))]}"
done
printf '%s\\n' docker run --rm --gpus all -v /proj:/workspace -w /workspace "${image}"
shift  # drop 'docker-args'
while [[ $# -gt 0 && "$1" != "--" ]]; do shift; done
shift || true
[[ $# -gt 0 ]] && printf '%s\\n' bash -lc "exec $*"
exit 0
"""


@pytest.fixture()
def fake_ship(tmp_path, monkeypatch):
    path = tmp_path / "ship"
    path.write_text(FAKE_SHIP)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "flags.log"
    monkeypatch.setenv("FAKE_SHIP_LOG", str(log))
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    return ShipExecutor(ship_binary=str(path), project_dir=str(tmp_path)), log


def _spec(**kw) -> JobSpec:
    base = dict(name="j", command=["python", "train.py"], resources=Resources(gpus=1, cpus=4, memory_gb=32))
    base.update(kw)
    return JobSpec(**base)


def test_missing_ship_binary_fails_loudly():
    with pytest.raises(ShipUnavailable, match="not found on PATH"):
        ShipExecutor(ship_binary="definitely-not-a-real-binary-xyz")


def test_gpu_uuids_are_passed_through_as_a_device_list(fake_ship):
    executor, log = fake_ship
    executor.build_argv(
        _spec(), argv=["python", "train.py"], env={}, policy=Policy(),
        placement=Placement(gpu_ids=("GPU-aaa", "GPU-bbb")), name="fleet-j",
    )
    flags = log.read_text().splitlines()
    assert '"device=GPU-aaa,GPU-bbb"' in flags
    assert "fleet-j" in flags


def test_secrets_are_requested_by_name_not_by_value(fake_ship):
    executor, log = fake_ship
    executor.build_argv(
        _spec(), argv=["true"],
        env={"ANTHROPIC_API_KEY": "", "FLEET_RUN_ID": "run_1"},
        policy=Policy(), placement=Placement(gpu_ids=()), name="n",
    )
    flags = log.read_text().splitlines()
    # A bare key inherits from the daemon env; a literal secret never appears.
    assert "ANTHROPIC_API_KEY" in flags
    assert not any(f.startswith("ANTHROPIC_API_KEY=") for f in flags)
    assert "FLEET_RUN_ID=run_1" in flags


def test_extra_mounts_are_forwarded(fake_ship):
    executor, log = fake_ship
    executor.build_argv(
        _spec(mounts=[Mount(source="/host/results", target="/results", mode="rw")]),
        argv=["true"], env={}, policy=Policy(),
        placement=Placement(gpu_ids=()), name="n",
    )
    assert "/host/results:/results:rw" in log.read_text().splitlines()


def test_fleet_layers_its_resource_limits_onto_the_ship_argv(fake_ship):
    executor, _ = fake_ship
    argv = executor.build_argv(
        _spec(), argv=["python", "train.py"], env={}, policy=Policy(),
        placement=Placement(gpu_ids=("GPU-aaa",)), name="n",
    )
    assert argv[:2] == ["docker", "run"]
    assert "--cpus" in argv and "4.0" in argv
    assert "--memory" in argv and "32768m" in argv
    assert "--pids-limit" in argv
    # research-ship's own decisions survive intact.
    assert "ship-proj:latest" in argv
    assert "/proj:/workspace" in argv


def test_image_override_is_delegated_to_ship(fake_ship):
    """The fleet passes --image rather than rewriting the ship's output."""
    executor, log = fake_ship
    argv = executor.build_argv(
        _spec(image="my-custom:v2"), argv=["true"], env={}, policy=Policy(),
        placement=Placement(gpu_ids=()), name="n",
    )
    assert "--image" in log.read_text().splitlines()
    assert "my-custom:v2" in argv
    assert "ship-proj:latest" not in argv


def test_isolate_asks_ship_for_a_worktree_on_its_own_branch(fake_ship):
    """An isolated job must not be handed the live working tree."""
    executor, log = fake_ship
    executor.build_argv(
        _spec(isolate=True, name="ablate"), argv=["true"], env={}, policy=Policy(),
        placement=Placement(gpu_ids=()), name="n",
    )
    flags = log.read_text().splitlines()
    assert "--worktree" in flags
    branch = flags[flags.index("--worktree") + 1]
    assert branch.startswith("fleet/") and "ablate" in branch


def test_jobs_are_not_isolated_unless_asked(fake_ship):
    executor, log = fake_ship
    executor.build_argv(
        _spec(), argv=["true"], env={}, policy=Policy(),
        placement=Placement(gpu_ids=()), name="n",
    )
    assert "--worktree" not in log.read_text().splitlines()


def test_parallel_isolated_jobs_get_distinct_branches(fake_ship):
    executor, log = fake_ship
    branches = set()
    for name in ("variant-a", "variant-b"):
        executor.build_argv(
            _spec(isolate=True, name=name), argv=["true"], env={}, policy=Policy(),
            placement=Placement(gpu_ids=()), name="n",
        )
        flags = log.read_text().splitlines()
        branches.add(flags[flags.index("--worktree") + 1])
    assert len(branches) == 2, "each job needs its own branch to be reviewable"


def test_isolated_stage_snapshot_retains_commit_and_patch(tmp_path):
    workspace = tmp_path / "workspace"
    results = tmp_path / "results"
    workspace.mkdir()
    results.mkdir()
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    (workspace / "paper.txt").write_text("before\n")
    subprocess.run(["git", "-C", str(workspace), "add", "-A"], check=True)
    subprocess.run([
        "git", "-C", str(workspace), "-c", "user.name=test",
        "-c", "user.email=test@localhost", "commit", "-qm", "base",
    ], check=True)
    base = subprocess.check_output(
        ["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True,
    ).strip()
    (workspace / "paper.txt").write_text("after\n")
    spec = _spec(
        id="job_snapshot", run_id="run_snapshot", isolate=True,
        mounts=[Mount(source=str(results), target="/results", mode="rw")],
    )
    result = JobResult(
        job_id=spec.id, state=JobState.SUCCEEDED,
        worktree_path=str(workspace), worktree_branch="fleet/stage",
        worktree_base_commit=base,
    )

    executor = object.__new__(ShipExecutor)
    executor._snapshot_worktree(spec, result)

    snapshot = json.loads((results / "snapshot.json").read_text())
    assert snapshot["base_commit"] == base
    assert snapshot["commit"] == result.worktree_commit
    assert snapshot["ref"] == result.worktree_snapshot_ref
    assert "-before" in (results / "snapshot.patch").read_text()
    assert "+after" in (results / "snapshot.patch").read_text()
    assert subprocess.run(
        ["git", "-C", str(workspace), "show-ref", "--verify", snapshot["ref"]],
        check=False,
    ).returncode == 0


def test_openai_api_key_satisfies_the_credential_preflight(fake_ship, monkeypatch):
    executor, _ = fake_ship
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    assert executor.credentials_present(Policy()) is True
