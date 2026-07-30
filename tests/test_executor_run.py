"""Executor selection, and ShipExecutor's process handling.

`run()` owns streaming, timeouts and cancellation, which is the part most likely to
break silently. A fake `docker` on disk lets all of it be exercised without a daemon.
"""

from __future__ import annotations

import stat
import textwrap
import threading
import time

import pytest

from research_fleet.config import FleetConfig
from research_fleet.executors import DryRunExecutor, build_executor
from research_fleet.executors.base import Placement
from research_fleet.executors.ship_exec import ShipExecutor, ShipUnavailable
from research_fleet.policy import Policy
from research_fleet.spec import JobSpec, JobState, Resources

FAKE_SHIP = """#!/usr/bin/env bash
printf '%s\\n' docker run --rm sandbox:latest
shift
while [[ $# -gt 0 && "$1" != "--" ]]; do shift; done
shift || true
[[ $# -gt 0 ]] && printf '%s\\n' bash -lc "exec $*"
exit 0
"""


def _script(path, body):
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture()
def fake(tmp_path):
    """A fake ship plus a configurable fake docker."""
    ship = _script(tmp_path / "ship", FAKE_SHIP)

    def make_docker(body):
        # PIDFILE lets a fake `docker stop` terminate exactly the fake `docker run`
        # it was asked about, the way the real one does, with nothing else at risk.
        header = f'#!/usr/bin/env bash\nPIDFILE={tmp_path}/run.pid\n'
        return _script(tmp_path / "docker", header + textwrap.dedent(body))

    return ship, make_docker


# A fake `docker run` that blocks until stopped, and a `docker stop` that stops it.
BLOCKING_DOCKER = """
    if [[ "$1" == "stop" ]]; then
        [[ -f "$PIDFILE" ]] && kill "$(cat "$PIDFILE")" 2>/dev/null
        exit 0
    fi
    echo $$ > "$PIDFILE"
    trap 'exit 143' TERM
    for _ in $(seq 600); do sleep 0.1; done
"""


def _spec(**kw):
    base = dict(name="j", command=["true"], resources=Resources(gpus=0), timeout_s=30)
    base.update(kw)
    return JobSpec(**base)


def _run(executor, spec, lines):
    return executor.run(
        spec, argv=["true"], env={}, placement=Placement(gpu_ids=()),
        policy=Policy(), on_line=lambda stream, line: lines.append((stream, line)),
    )


# ------------------------------------------------------------- selection


def test_build_executor_dispatches_on_kind(tmp_path):
    cfg = FleetConfig(root=str(tmp_path), executor={"kind": "dry-run"})
    assert isinstance(build_executor(cfg), DryRunExecutor)


def test_build_executor_rejects_an_unknown_kind(tmp_path):
    cfg = FleetConfig(root=str(tmp_path))
    object.__setattr__(cfg.executor, "kind", "nonsense")
    with pytest.raises(ValueError, match="unknown executor kind"):
        build_executor(cfg)


def test_dry_run_reports_what_it_would_have_done():
    lines = []
    result = _run(DryRunExecutor(), _spec(), lines)
    assert result.state is JobState.SUCCEEDED
    assert any("would execute" in line for _, line in lines)


# --------------------------------------------------------------- run()


def test_streams_stdout_and_stderr_separately(fake):
    ship, make_docker = fake
    docker = make_docker("""
        echo out-one
        echo err-one >&2
        echo out-two
        exit 0
    """)
    executor = ShipExecutor(ship_binary=str(ship), docker_binary=str(docker))
    lines = []
    result = _run(executor, _spec(), lines)

    assert result.state is JobState.SUCCEEDED and result.exit_code == 0
    assert ("stdout", "out-one") in lines and ("stdout", "out-two") in lines
    assert ("stderr", "err-one") in lines
    assert result.duration_s is not None


def test_nonzero_exit_becomes_a_failure_with_the_code(fake):
    ship, make_docker = fake
    docker = make_docker("echo nope >&2\nexit 7")
    executor = ShipExecutor(ship_binary=str(ship), docker_binary=str(docker))
    result = _run(executor, _spec(), [])
    assert result.state is JobState.FAILED
    assert result.exit_code == 7 and "7" in result.error


def test_a_job_that_overruns_its_timeout_is_stopped(fake):
    ship, make_docker = fake
    docker = make_docker(BLOCKING_DOCKER)
    executor = ShipExecutor(ship_binary=str(ship), docker_binary=str(docker))
    started = time.time()
    result = _run(executor, _spec(timeout_s=1), [])
    assert result.state is JobState.FAILED
    assert "timed out" in result.error
    # Stopped promptly, rather than waiting out the sleep or the 30s grace period.
    assert time.time() - started < 15, "timeout did not stop the container promptly"


def test_a_failing_ship_is_reported_not_raised(fake):
    ship, make_docker = fake
    broken = _script(ship.parent / "broken-ship", "#!/usr/bin/env bash\necho bad >&2\nexit 3\n")
    executor = ShipExecutor(ship_binary=str(broken), docker_binary=str(make_docker("exit 0")))
    result = _run(executor, _spec(), [])
    assert result.state is JobState.FAILED
    assert "docker-args` failed" in result.error


def test_unexpected_ship_output_is_rejected(fake):
    ship, make_docker = fake
    odd = _script(ship.parent / "odd-ship", "#!/usr/bin/env bash\necho surprise\n")
    executor = ShipExecutor(ship_binary=str(odd), docker_binary=str(make_docker("exit 0")))
    with pytest.raises(ShipUnavailable, match="unexpected output"):
        executor.build_argv(
            _spec(), argv=["true"], env={}, placement=Placement(gpu_ids=()),
            policy=Policy(), name="n",
        )


def test_cancel_stops_a_running_job(fake):
    ship, make_docker = fake
    docker = make_docker(BLOCKING_DOCKER)
    executor = ShipExecutor(ship_binary=str(ship), docker_binary=str(docker))
    spec = _spec(timeout_s=60)

    result_box = {}
    worker = threading.Thread(target=lambda: result_box.update(r=_run(executor, spec, [])))
    worker.start()
    # Wait for the process to be registered before cancelling it.
    for _ in range(100):
        if executor._procs:
            break
        time.sleep(0.05)
    assert executor.cancel(spec.id) is True
    worker.join(timeout=30)
    assert result_box["r"].state is JobState.FAILED

    assert executor.cancel("job_that_never_ran") is False


def test_close_is_safe_with_nothing_running(fake):
    ship, make_docker = fake
    executor = ShipExecutor(ship_binary=str(ship), docker_binary=str(make_docker("exit 0")))
    executor.close()


def test_available_gpus_degrades_when_nvidia_smi_is_absent(fake, monkeypatch):
    ship, make_docker = fake
    executor = ShipExecutor(ship_binary=str(ship), docker_binary=str(make_docker("exit 0")))
    monkeypatch.setenv("PATH", str(ship.parent))   # no nvidia-smi here
    assert executor.available_gpus() == []
