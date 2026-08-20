from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

from research_fleet import Fleet
from research_fleet.executors.direct_exec import DirectExecutor
from research_fleet.policy import Policy
from research_fleet.spec import JobSpec, Mount


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)


def test_direct_codex_uses_workspace_sandbox_and_only_rw_mounts(tmp_path):
    workspace = tmp_path / "worktree"
    results = tmp_path / "results"
    inputs = tmp_path / "inputs"
    spec = JobSpec(command=["true"], mounts=[
        Mount(source=str(results), target="/results", mode="rw"),
        Mount(source=str(inputs), target="/inputs/first", mode="ro"),
    ])
    command = DirectExecutor._sandbox_codex([
        "codex", "exec", "--dangerously-bypass-approvals-and-sandbox", "task",
    ], spec, str(workspace))

    assert "--dangerously-bypass-approvals-and-sandbox" not in command
    assert "--sandbox" not in command and "--add-dir" not in command
    assert 'sandbox_mode="workspace-write"' in command
    roots = next(value for value in command if value.startswith(
        "sandbox_workspace_write.writable_roots="
    ))
    assert str(workspace.resolve()) in roots
    assert str(results.resolve()) in roots
    assert "sandbox_workspace_write.exclude_slash_tmp=true" in command
    assert "sandbox_workspace_write.exclude_tmpdir_env_var=true" in command
    assert "use_legacy_landlock" not in command
    assert str(inputs.resolve()) not in command


def test_direct_sandbox_flags_are_valid_for_persistent_codex_resume(tmp_path):
    workspace = tmp_path / "worktree"
    command = DirectExecutor._sandbox_codex([
        "codex", "exec", "resume", "--json",
        "--dangerously-bypass-approvals-and-sandbox", "session-id", "task",
    ], JobSpec(command=["true"]), str(workspace))

    assert command[:3] == ["codex", "exec", "resume"]
    assert "--sandbox" not in command and "--add-dir" not in command
    assert 'sandbox_mode="workspace-write"' in command


def test_direct_non_codex_commands_are_unchanged():
    command = ["python3", "analysis.py"]
    assert DirectExecutor._sandbox_codex(command, JobSpec(command=["true"])) == command


def test_direct_path_translation_does_not_rewrite_host_path_segments(tmp_path):
    mapping = [("/results", str(tmp_path / "state/results/job"))]
    assert DirectExecutor._translate("read /results/value", mapping).endswith("/value")
    host = str(tmp_path / "old/results/value")
    assert DirectExecutor._translate(host, mapping) == host


def test_nono_maps_workspace_and_mount_permissions(tmp_path):
    nono = tmp_path / "nono"
    nono.write_text("#!/bin/sh\n")
    nono.chmod(0o755)
    workspace = tmp_path / "worktree"
    readonly = tmp_path / "data"
    writable = tmp_path / "results"
    executor = DirectExecutor(str(workspace), nono_binary=str(nono))
    spec = JobSpec(command=["true"], mounts=[
        Mount(source=str(readonly), target="/data", mode="ro"),
        Mount(source=str(writable), target="/results", mode="rw"),
    ])

    command = executor._nono_command(
        ["codex", "exec", "resume", "--dangerously-bypass-approvals-and-sandbox",
         "session", "task"], spec, str(workspace),
    )

    assert command[:3] == [str(nono), "run", "--silent"]
    assert command[command.index("--profile") + 1].endswith("/data/nono-fleet.json")
    assert command[command.index("--sandbox-policy") + 1] == "landlock"
    assert command[command.index("--workdir") + 1] == str(workspace)
    assert "--bypass-protection" in command
    assert command[command.index("--read") + 1] == str(readonly.resolve())
    allow_values = [command[index + 1] for index, value in enumerate(command[:-1])
                    if value == "--allow"]
    assert str(writable.resolve()) in allow_values
    assert command[command.index("--") + 1:] == [
        "codex", "exec", "resume", "--dangerously-bypass-approvals-and-sandbox",
        "session", "task",
    ]


def test_nono_never_exposes_unlisted_host_paths_or_tmp(tmp_path):
    nono = tmp_path / "nono"
    nono.write_text("#!/bin/sh\n")
    nono.chmod(0o755)
    workspace = tmp_path / "worktree"
    results = tmp_path / "results"
    data = tmp_path / "data"
    executor = DirectExecutor(str(workspace), nono_binary=str(nono))
    command = executor._nono_command(
        ["python3", "-c", "pass"],
        JobSpec(command=["true"], mounts=[
            Mount(source=str(results), target="/results", mode="rw"),
            Mount(source=str(data), target="/data", mode="ro"),
        ]),
        str(workspace),
    )
    # The only filesystem capabilities are the worktree and declared mounts.
    assert "/tmp" not in command
    assert str(workspace.resolve()) in command
    assert str(results.resolve()) in command
    assert str(data.resolve()) in command
    assert str(tmp_path.resolve()) not in command


def test_nono_preflight_reports_runtime_failure(tmp_path):
    nono = tmp_path / "nono"
    nono.write_text("#!/bin/sh\necho 'mount denied' >&2\nexit 1\n")
    nono.chmod(0o755)
    project = tmp_path / "project"
    project.mkdir()
    executor = DirectExecutor(str(project), nono_binary=str(nono))
    with pytest.raises(RuntimeError, match="nono could not start"):
        executor.preflight(Policy())


@pytest.mark.skipif(not os.environ.get("NONO_TEST_BINARY"), reason="set NONO_TEST_BINARY")
def test_nono_enforces_workspace_inputs_and_results(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    (data / "input.txt").write_text("input")
    secret = tmp_path / "secret.txt"
    secret.write_text("secret")
    state = tmp_path / "state"
    fleet = Fleet(
        root=str(state), workspace=str(project),
        executor={
            "kind": "nono", "project_dir": str(project),
            "nono_binary": os.environ["NONO_TEST_BINARY"],
            "mounts": [{"source": str(data), "target": "/data", "mode": "ro"}],
        },
        policy={"allowed_mount_roots": [str(project), str(state), str(data)]},
    )
    code = textwrap.dedent(f"""
        from pathlib import Path
        assert Path('/data/input.txt').read_text() == 'input'
        Path('/workspace/local.txt').write_text('local')
        Path('/results/result.txt').write_text('result')
        for path in (Path('/data/input.txt'), Path({str(secret)!r})):
            try:
                path.write_text('bad')
            except PermissionError:
                pass
            else:
                raise AssertionError(f'write allowed: {{path}}')
            try:
                path.unlink()
            except PermissionError:
                pass
            else:
                raise AssertionError(f'delete allowed: {{path}}')
        try:
            Path({str(secret)!r}).read_text()
        except PermissionError:
            pass
        else:
            raise AssertionError('secret readable')
    """)
    try:
        report = fleet.run_workflow({"name": "nono-isolation", "gpus": 0, "stages": [{
            "name": "check", "kind": "command", "gpus": 0,
            "command": ["python3", "-c", code],
        }]})
        assert report.steps["check"].state.value == "succeeded", report.steps["check"].error
        assert (project / "local.txt").read_text() == "local"
        assert (state / "results/nono-isolation/001/check/result.txt").read_text() == "result"
        assert (data / "input.txt").read_text() == "input"
        assert secret.read_text() == "secret"
    finally:
        fleet.close()


@pytest.mark.skipif(not os.environ.get("NONO_TEST_BINARY"), reason="set NONO_TEST_BINARY")
def test_nono_dependencies_are_visible_but_other_runs_are_not(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"

    def make_fleet():
        return Fleet(
            root=str(state), workspace=str(project),
            executor={"kind": "nono", "project_dir": str(project),
                      "nono_binary": os.environ["NONO_TEST_BINARY"]},
        )

    first = make_fleet()
    try:
        report = first.run_workflow({"name": "first-run", "gpus": 0, "stages": [{
            "name": "produce", "kind": "command", "gpus": 0,
            "command": ["python3", "-c",
                        "from pathlib import Path; Path('/results/value').write_text('one')"],
        }]})
        old_result = Path(report.steps["produce"].results_dir) / "value"
    finally:
        first.close()

    second = make_fleet()
    old_prefix, old_suffix = str(old_result).split("/results", 1)
    code = textwrap.dedent(f"""
        from pathlib import Path
        assert Path('/inputs/produce/value').read_text() == 'two'
        try:
            Path({old_prefix!r} + '/res' + 'ults' + {old_suffix!r}).read_text()
        except PermissionError:
            pass
        else:
            raise AssertionError('another run was readable')
        Path('/results/value').write_text('done')
    """)
    try:
        report = second.run_workflow({"name": "second-run", "gpus": 0, "stages": [
            {"name": "produce", "kind": "command", "gpus": 0,
             "command": ["python3", "-c",
                         "from pathlib import Path; Path('/results/value').write_text('two')"]},
            {"name": "consume", "kind": "command", "needs": ["produce"], "gpus": 0,
             "command": ["python3", "-c", code]},
        ]})
        assert report.steps["consume"].state.value == "succeeded", report.steps["consume"].error
        assert Path(report.steps["consume"].results_dir, "value").read_text() == "done"
        assert old_result.read_text() == "one"
    finally:
        second.close()


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


@pytest.mark.skipif(not os.environ.get("NONO_TEST_BINARY"), reason="set NONO_TEST_BINARY")
def test_nono_preserves_isolated_worktree_chain(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    _git("init", "-q", cwd=project)
    (project / "base.txt").write_text("base")
    _git("add", "-A", cwd=project)
    _git("-c", "user.name=test", "-c", "user.email=test@localhost",
         "commit", "-qm", "base", cwd=project)
    state = tmp_path / "state"
    fleet = Fleet(
        root=str(state), workspace=str(project), isolate_agents=True,
        executor={"kind": "nono", "project_dir": str(project),
                  "nono_binary": os.environ["NONO_TEST_BINARY"]},
    )
    try:
        report = fleet.run_workflow({"name": "nono-worktree", "gpus": 0, "stages": [
            {"name": "first", "kind": "command", "gpus": 0,
             "command": ["python3", "-c",
                         "from pathlib import Path; Path('/workspace/carried').write_text('one')"]},
            {"name": "second", "kind": "command", "needs": ["first"], "gpus": 0,
             "command": ["python3", "-c",
                         "from pathlib import Path; p=Path('/workspace/carried'); "
                         "assert p.read_text()=='one'; p.write_text('two')"]},
        ]})
        result = report.steps["second"]
        assert result.state.value == "succeeded"
        snapshot = json.loads(Path(result.results_dir, "snapshot.json").read_text())
        assert _git("show", f"{snapshot['ref']}:carried", cwd=project).stdout == "two"
    finally:
        fleet.close()
