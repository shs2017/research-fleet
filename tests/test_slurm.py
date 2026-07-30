"""Slurm executor.

Uses a fake `srun` and `scancel` so the suite stays hermetic. What matters is the
submission argv, since that is what a cluster will actually honour, plus the same
streaming and cancellation contract the local executor has.
"""

from __future__ import annotations

import stat
import textwrap

import pytest

from research_fleet.config import FleetConfig
from research_fleet.executors import build_executor
from research_fleet.executors.base import Placement
from research_fleet.executors.slurm_exec import SlurmExecutor, SlurmUnavailable
from research_fleet.policy import Policy
from research_fleet.spec import JobSpec, JobState, Mount, Resources


def _script(path, body):
    path.write_text("#!/usr/bin/env bash\n" + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture()
def fake_slurm(tmp_path):
    """A fake srun that records its arguments, plus a scancel that kills it."""
    pidfile = tmp_path / "run.pid"
    srun = _script(tmp_path / "srun", f"""
        printf '%s\\n' "$@" > {tmp_path}/srun.args
        echo $$ > {pidfile}
        trap 'exit 143' TERM
        # Everything after the flags is the payload; run it.
        for arg; do case "$arg" in --*) ;; *) break ;; esac; shift; done
        exec "$@"
    """)
    scancel = _script(tmp_path / "scancel", f"""
        printf '%s\\n' "$@" >> {tmp_path}/scancel.args
        [[ -f {pidfile} ]] && kill "$(cat {pidfile})" 2>/dev/null
        exit 0
    """)
    return srun, scancel, tmp_path


def _executor(fake_slurm, **kw):
    srun, scancel, _ = fake_slurm
    kw.setdefault("workspace", str(srun.parent))
    return SlurmExecutor(srun_binary=str(srun), scancel_binary=str(scancel), **kw)


def _spec(**kw):
    base = dict(name="j", command=["true"],
                resources=Resources(gpus=1, cpus=4, memory_gb=8), timeout_s=120)
    base.update(kw)
    return JobSpec(**base)


def _argv(executor, spec=None, policy=None):
    return executor.build_argv(spec or _spec(), ["echo", "hello"], policy or Policy())


# ------------------------------------------------------------- availability


def test_a_missing_srun_is_reported_clearly():
    with pytest.raises(SlurmUnavailable, match="client tools"):
        SlurmExecutor(srun_binary="definitely-not-srun-xyz")


def test_slots_bound_how_many_submissions_are_held_open(fake_slurm):
    executor = _executor(fake_slurm, slots=3)
    assert len(executor.available_gpus()) == 3


def test_build_executor_dispatches_to_slurm(tmp_path, fake_slurm, monkeypatch):
    srun, _, _ = fake_slurm
    monkeypatch.setenv("PATH", f"{srun.parent}:/usr/bin:/bin")
    cfg = FleetConfig(
        root=str(tmp_path),
        executor={"kind": "slurm", "slurm": {"partition": "gpu", "slots": 2}},
    )
    executor = build_executor(cfg)
    assert executor.kind == "slurm"
    assert executor.partition == "gpu" and executor.slots == 2


# ---------------------------------------------------------------- submission


def test_resources_become_srun_flags(fake_slurm):
    argv = " ".join(_argv(_executor(fake_slurm)))
    assert "--gres=gpu:1" in argv
    assert "--cpus-per-task=4" in argv
    assert "--mem=8192M" in argv
    assert "--time=2" in argv           # 120s rounds up to 2 minutes
    assert argv.endswith("echo hello")


def test_a_fractional_gpu_request_rounds_up(fake_slurm):
    """Slurm's --gres cannot express half a GPU, so asking for one is honest."""
    argv = _argv(_executor(fake_slurm), _spec(resources=Resources(gpus=0.25)))
    assert "--gres=gpu:1" in " ".join(argv)


def test_a_cpu_only_job_requests_no_gpus(fake_slurm):
    argv = " ".join(_argv(_executor(fake_slurm), _spec(resources=Resources(gpus=0))))
    assert "--gres" not in argv


def test_cluster_placement_options_are_passed_through(fake_slurm):
    executor = _executor(fake_slurm, partition="gpu", account="lab", qos="high",
                         extra_args=["--exclusive"])
    argv = " ".join(_argv(executor))
    for expected in ("--partition=gpu", "--account=lab", "--qos=high", "--exclusive"):
        assert expected in argv


def test_the_job_name_lets_scancel_find_it(fake_slurm):
    spec = _spec()
    argv = " ".join(_argv(_executor(fake_slurm), spec))
    assert f"--job-name=fleet-{spec.id}" in argv


# ----------------------------------------------------------------- container


def test_without_an_image_the_command_runs_in_the_allocation(fake_slurm):
    argv = " ".join(_argv(_executor(fake_slurm)))
    assert "apptainer" not in argv


def test_an_image_wraps_the_command_in_apptainer(fake_slurm):
    executor = _executor(fake_slurm, container_image="/images/research.sif")
    argv = _argv(executor)
    joined = " ".join(argv)
    assert "apptainer exec" in joined
    assert "--nv" in argv                       # GPUs need the nvidia bind
    assert "/images/research.sif" in argv
    assert "--pwd" in argv and "/workspace" in joined
    # The payload still comes last, after the container image.
    assert argv.index("/images/research.sif") < argv.index("echo")


def test_extra_mounts_become_apptainer_binds(fake_slurm):
    executor = _executor(fake_slurm, container_image="/images/r.sif")
    argv = " ".join(_argv(
        executor,
        _spec(mounts=[Mount(source="/data", target="/data", mode="ro"),
                      Mount(source="/out", target="/results", mode="rw")]),
    ))
    assert "/data:/data:ro" in argv
    assert "/out:/results" in argv


def test_a_cpu_only_containerised_job_skips_nv(fake_slurm):
    executor = _executor(fake_slurm, container_image="/images/r.sif")
    argv = _argv(executor, _spec(resources=Resources(gpus=0)))
    assert "--nv" not in argv


# ----------------------------------------------------------------------- run


def _run(executor, spec, lines):
    return executor.run(
        spec, argv=["echo", "from-slurm"], env={}, placement=Placement(node="cluster"),
        policy=Policy(), on_line=lambda stream, line: lines.append((stream, line)),
    )


def test_a_submitted_job_streams_output_and_succeeds(fake_slurm):
    executor = _executor(fake_slurm)
    lines = []
    result = _run(executor, _spec(), lines)
    assert result.state is JobState.SUCCEEDED and result.exit_code == 0
    assert ("stdout", "from-slurm") in lines
    assert result.node == "cluster"


def test_a_failing_job_reports_its_exit_code(fake_slurm):
    srun, scancel, work = fake_slurm
    _script(srun, "exit 9")
    executor = _executor(fake_slurm)
    result = _run(executor, _spec(), [])
    assert result.state is JobState.FAILED and result.exit_code == 9


def test_an_unsubmittable_job_fails_rather_than_raising(fake_slurm):
    srun, scancel, _ = fake_slurm
    executor = _executor(fake_slurm)
    executor.srun = str(srun.parent / "vanished")
    result = _run(executor, _spec(), [])
    assert result.state is JobState.FAILED
    assert "failed to submit to slurm" in result.error


def test_cancel_scancels_by_job_name(fake_slurm):
    srun, scancel, work = fake_slurm
    executor = _executor(fake_slurm)
    spec = _spec()
    executor.cancel(spec.id)
    recorded = (work / "scancel.args").read_text()
    assert "--name" in recorded and f"fleet-{spec.id}" in recorded


def test_close_cancels_everything_still_in_flight(fake_slurm):
    executor = _executor(fake_slurm)
    executor.close()      # nothing running, must not raise
    assert executor._procs == {}


def test_secrets_travel_through_the_environment_not_the_argv(fake_slurm):
    """The argv is recorded in the ledger, so a value must never appear in it."""
    srun, scancel, work = fake_slurm
    executor = _executor(fake_slurm)
    lines = []
    executor.run(
        _spec(), argv=["echo", "done"], env={"MY_TOKEN": "s3cret"},
        placement=Placement(node="cluster"), policy=Policy(),
        on_line=lambda s, l: lines.append(l),
    )
    assert "s3cret" not in (work / "srun.args").read_text()
