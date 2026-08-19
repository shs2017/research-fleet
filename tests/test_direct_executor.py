from __future__ import annotations

import json
import subprocess
from pathlib import Path

from research_fleet import Fleet
from research_fleet.executors.direct_exec import DirectExecutor
from research_fleet.spec import JobSpec, Mount


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)


def test_direct_codex_uses_workspace_sandbox_and_only_rw_mounts(tmp_path):
    results = tmp_path / "results"
    inputs = tmp_path / "inputs"
    spec = JobSpec(command=["true"], mounts=[
        Mount(source=str(results), target="/results", mode="rw"),
        Mount(source=str(inputs), target="/inputs/first", mode="ro"),
    ])
    command = DirectExecutor._sandbox_codex([
        "codex", "exec", "--dangerously-bypass-approvals-and-sandbox", "task",
    ], spec)

    assert "--dangerously-bypass-approvals-and-sandbox" not in command
    assert command[2:4] == ["--sandbox", "workspace-write"]
    assert command[4:6] == ["--enable", "use_legacy_landlock"]
    assert command[6:8] == ["--add-dir", str(results.resolve())]
    assert str(inputs.resolve()) not in command


def test_direct_non_codex_commands_are_unchanged():
    command = ["python3", "analysis.py"]
    assert DirectExecutor._sandbox_codex(command, JobSpec(command=["true"])) == command


def test_direct_executor_mounts_are_translated_and_read_only_to_codex(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    data = tmp_path / "external-data"
    data.mkdir()
    (data / "value.txt").write_text("visible")
    state = tmp_path / "state"
    fleet = Fleet(
        root=str(state), workspace=str(project),
        executor={
            "kind": "direct", "project_dir": str(project),
            "mounts": [{"source": str(data), "target": "/workspace/data", "mode": "ro"}],
        },
        policy={"allowed_mount_roots": [str(project), str(state), str(data)]},
    )
    try:
        report = fleet.run_workflow({"name": "static-mount", "gpus": 0, "stages": [{
            "name": "read", "kind": "command", "gpus": 0,
            "command": ["python3", "-c",
                        "from pathlib import Path; "
                        "Path('/results/value.txt').write_text("
                        "Path('/workspace/data/value.txt').read_text())"],
        }]})
        assert report.steps["read"].state.value == "succeeded"
        assert (state / "results/static-mount/001/read/value.txt").read_text() == "visible"
    finally:
        fleet.close()


def test_direct_workflow_translates_dependency_and_result_paths(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    fleet = Fleet(
        root=str(state), workspace=str(project),
        executor={"kind": "direct", "project_dir": str(project)},
    )
    try:
        report = fleet.run_workflow({"name": "direct-paths", "gpus": 0, "stages": [
            {"name": "first", "kind": "command", "gpus": 0, "command": [
                "python3", "-c",
                "from pathlib import Path; Path('/results/value.txt').write_text('one')",
            ]},
            {"name": "second", "kind": "command", "needs": ["first"], "gpus": 0,
             "command": ["python3", "-c",
                "from pathlib import Path; value=Path('/inputs/first/value.txt').read_text(); "
                "Path('/results/value.txt').write_text(value + '-two')"]},
        ]})
        assert all(result.state.value == "succeeded" for result in report.run.results.values())
        assert (state / "results/direct-paths/001/second/value.txt").read_text() == "one-two"
    finally:
        fleet.close()


def test_direct_isolated_stages_chain_and_keep_snapshots(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    _git("init", "-q", cwd=project)
    (project / "base.txt").write_text("base")
    _git("add", "-A", cwd=project)
    _git("-c", "user.name=test", "-c", "user.email=test@localhost",
         "commit", "-qm", "base", cwd=project)
    state = tmp_path / "state"
    fleet = Fleet(
        root=str(state), workspace=str(project),
        executor={"kind": "direct", "project_dir": str(project)},
        isolate_agents=True,
    )
    try:
        report = fleet.run_workflow({"name": "direct-isolated", "gpus": 0, "stages": [
            {"name": "first", "kind": "command", "gpus": 0,
             "command": ["python3", "-c",
                         "from pathlib import Path; Path('/workspace/carried.txt').write_text('one')"]},
            {"name": "second", "kind": "command", "needs": ["first"], "gpus": 0,
             "command": ["python3", "-c",
                         "from pathlib import Path; p=Path('/workspace/carried.txt'); "
                         "assert p.read_text() == 'one'; p.write_text('two')"]},
        ]})
        second = report.steps["second"]
        assert second.state.value == "succeeded"
        snapshot = json.loads(
            (state / "results/direct-isolated/001/second/snapshot.json").read_text()
        )
        assert snapshot["base_commit"] != snapshot["commit"]
        shown = _git("show", f"{snapshot['ref']}:carried.txt", cwd=project)
        assert shown.returncode == 0 and shown.stdout == "two"
    finally:
        fleet.close()
