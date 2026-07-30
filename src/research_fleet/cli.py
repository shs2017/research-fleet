"""`fleet`: the command line entry point.

Also the interface agents use from inside their containers: `fleet submit`
detects `$FLEET_SUBMIT_DIR` and writes a spool file instead of talking to the
scheduler directly, so the same command works on both sides of the container
boundary.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.json import JSON
from rich.table import Table

from .budget import cost_menu
from .config import load_config
from .fleet import Fleet
from .ledger import Ledger
from .spec import JobSpec, new_id
from .sweep import parse_grid_args

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Auditable, containerized, multi-GPU agent fleet for autonomous research.",
)
audit_app = typer.Typer(no_args_is_help=True, help="Inspect and verify the audit ledger.")
app.add_typer(audit_app, name="audit")

console = Console()


def _fleet(config: Optional[str], **overrides) -> Fleet:
    """Build a Fleet, turning setup problems into advice rather than a traceback."""
    from .executors import ShipUnavailable

    try:
        return Fleet(config, **{k: v for k, v in overrides.items() if v is not None})
    except ShipUnavailable as exc:
        console.print(f"[red]cannot start:[/red] {exc}")
        raise typer.Exit(1) from None


def _gpu_share(requested: Optional[float], agents: int, devices: int) -> tuple[float, str]:
    """Choose how much GPU each agent reserves, and explain the choice.

    A whole GPU each serialises everything on a single-GPU box, which is the opposite of
    what `--agents 4` asks for. Sharing the device lets them run together and still see
    it. An explicit value is always honoured, with a warning if it will serialise.
    """
    if requested is not None:
        if requested > 0 and devices:
            concurrent = max(1, int(devices / requested))
            if concurrent < agents:
                return requested, (
                    f"[yellow]note: {requested} GPU(s) each on {devices} device(s) means "
                    f"{concurrent} at a time, so the {agents} agents will queue. "
                    f"Use --gpus {devices / agents:.2f} to run them together.[/yellow]"
                )
        return requested, ""

    if devices == 0:
        return 0.0, "no GPU on this host, so agents run without one"
    share = min(1.0, devices / agents)
    return share, f"{share:.2f} GPU each on {devices} device(s), all {agents} run together"


def _overrides(workspace=None, image=None, executor=None, max_usd=None) -> dict:
    """Map the flags shared by `run` and `sweep` onto config overrides."""
    out: dict = {"workspace": workspace, "image": image}
    if executor:
        out["executor"] = {"kind": executor}
    if max_usd is not None:
        out["budget"] = {"max_usd": max_usd}
    return out


def _cancel_on_interrupt(fleet):
    """Make Ctrl-C stop the run instead of leaving containers behind.

    The default KeyboardInterrupt unwinds through wait(), and close() would then be
    left to clean up while jobs are still going. Cancelling first is both faster and
    honest about what happened.
    """
    import signal

    state = {"cancelling": False}

    def handler(signum, frame):
        if state["cancelling"]:
            console.print("\n[red]forcing exit[/red]")
            raise SystemExit(130)
        state["cancelling"] = True
        console.print("\n[yellow]stopping the run, Ctrl-C again to force[/yellow]")
        stopped = fleet.cancel("interrupted by operator")
        console.print(f"[yellow]cancelled {stopped} job(s)[/yellow]")
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, handler)
    except ValueError:      # not the main thread; leave the default behaviour
        pass


def _live_printer(verbose: bool):
    def on_event(type_: str, payload: dict) -> None:
        if not verbose and type_ in {"agent.raw", "job.output"}:
            return
        job = payload.get("job_id", "")[-6:]
        if type_ == "job.output":
            # Must precede the job.* state branch below, which would otherwise
            # swallow this and print an empty line.
            console.print(f"[dim]{job}[/dim] {payload.get('line', '')}")
        elif type_ == "agent.message":
            console.print(f"[dim]{job}[/dim] {payload.get('text', '')[:400]}")
        elif type_ == "agent.tool_use":
            console.print(f"[dim]{job}[/dim] [cyan]→ {payload.get('tool')}[/cyan]")
        elif type_ == "agent.result":
            console.print(f"[dim]{job}[/dim] [green]✓ done[/green] {payload.get('text', '')[:200]}")
        elif type_.startswith("job."):
            state = type_.split(".", 1)[1]
            colour = {"succeeded": "green", "failed": "red", "denied": "red"}.get(state, "yellow")
            console.print(f"[dim]{job}[/dim] [{colour}]{state}[/{colour}] {payload.get('name', '')}")
        elif verbose:
            console.print(f"[dim]{job} {type_}[/dim] {str(payload)[:300]}")

    return on_event


@app.command()
def run(
    task: str = typer.Argument(..., help="The research task to hand the agents."),
    agents: int = typer.Option(1, "--agents", "-n", help="How many agents to run in parallel."),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    backend: Optional[str] = typer.Option(None, "--backend", help="claude-cli | codex-cli"),
    effort: Optional[str] = typer.Option(None, "--effort", help="low|medium|high|xhigh|max"),
    gpus: Optional[float] = typer.Option(
        None, "--gpus",
        help="GPUs per agent. Default: share the devices so every agent runs at once.",
    ),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    image: Optional[str] = typer.Option(None, "--image"),
    timeout: int = typer.Option(3600, "--timeout", help="Per-agent wall clock, seconds."),
    max_usd: Optional[float] = typer.Option(None, "--max-usd", help="Budget ceiling for the whole run."),
    executor: Optional[str] = typer.Option(None, "--executor", help="ship | slurm | ray | dry-run"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    detach: bool = typer.Option(False, "--detach", "-d", help="Return immediately; run in the background."),
    run_id: Optional[str] = typer.Option(None, "--run-id", hidden=True),
):
    """Launch one or more research agents on a task."""
    if detach:
        _detach_and_return(config)
        return

    fleet = _fleet(
        config,
        **_overrides(workspace, image, executor, max_usd),
        on_event=_live_printer(verbose),
        run_id=run_id,
    )
    _cancel_on_interrupt(fleet)
    try:
        est = fleet.quote(model, effort=effort or fleet.config.budget.default_effort)
        share, note = _gpu_share(gpus, agents, fleet.scheduler.slots.device_count)
        console.print(
            f"[bold]run {fleet.run_id}[/bold]  {agents} agent(s)  "
            f"est. ${est.est_cost_usd * agents:.2f} ({est.source})  "
            f"budget ${fleet.config.budget.max_usd:.2f}"
        )
        if note:
            console.print(f"[dim]{note}[/dim]" if not note.startswith("[") else note)
        fleet.run_agents(
            task, n=agents, model=model, backend=backend, effort=effort,
            gpus=share, timeout_s=timeout,
        )
        report = fleet.wait()
        console.print()
        console.print(report.summary())
        raise typer.Exit(0 if not report.failed else 1)
    finally:
        fleet.close()


@app.command()
def sweep(
    command: list[str] = typer.Argument(..., help="Command to run; use {param} placeholders."),
    grid: list[str] = typer.Option([], "--grid", "-g", help="Repeatable: lr=1e-3,3e-4"),
    gpus: float = typer.Option(1.0, "--gpus"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    image: Optional[str] = typer.Option(None, "--image"),
    timeout: int = typer.Option(3600, "--timeout"),
    executor: Optional[str] = typer.Option(None, "--executor"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run a hyperparameter sweep across the available GPUs. No LLM involved."""
    fleet = _fleet(
        config,
        **_overrides(workspace, image, executor),
        on_event=_live_printer(verbose),
    )
    _cancel_on_interrupt(fleet)
    try:
        parsed = parse_grid_args(grid)
        recs = fleet.run_sweep(command, parsed, gpus=gpus, timeout_s=timeout)
        console.print(f"[bold]run {fleet.run_id}[/bold]  {len(recs)} points")
        report = fleet.wait()
        console.print(report.summary())
        raise typer.Exit(0 if not report.failed else 1)
    finally:
        fleet.close()


@app.command()
def workflow(
    file: str = typer.Argument(..., help="Workflow YAML."),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    max_usd: Optional[float] = typer.Option(None, "--max-usd"),
    executor: Optional[str] = typer.Option(None, "--executor", help="ship | slurm | ray | dry-run"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    plan: bool = typer.Option(False, "--plan", help="Validate and print the stages, run nothing."),
):
    """Run a multi-step pipeline, for example a coder and reviewer loop."""
    from .workflow import Loop, Step, Workflow

    wf = Workflow.from_yaml(file)

    if plan:
        console.print(f"[bold]{wf.name}[/bold]  {wf.description}")
        for note in wf.warnings():
            console.print(f"[yellow]warning:[/yellow] {note}")

        by_name = {st.name: st for st in wf.stages}
        cyclic = {tuple(group) for group in wf.cycles()}
        for depth, wave in enumerate(wf.levels(), start=1):
            if tuple(wave) in cyclic:
                limit = wf.repeat_limit(wave)
                stops = [n for n, _ in wf.stop_conditions(wave)]
                how = f"until {', '.join(stops)} says so, " if stops else ""
                console.print(f"  [dim]wave {depth}[/dim] [yellow]cycle[/yellow] "
                              f"{' -> '.join(wave)} -> {wave[0]}  ({how}max {limit}x)")
                for name in wave:
                    console.print(f"    {name} ({by_name[name].kind.value})"
                                  if isinstance(by_name[name], Step) else f"    {name} (loop)")
                continue
            together = " (in parallel)" if len(wave) > 1 else ""
            console.print(f"  [dim]wave {depth}[/dim]{together}")
            for name in wave:
                stage = by_name[name]
                waits = sorted(wf.graph()[name])
                after = f"  [dim]after {', '.join(waits)}[/dim]" if waits else ""
                if isinstance(stage, Loop):
                    until = (f"until {stage.until.step} contains '{stage.until.output_contains}'"
                             if stage.until and stage.until.output_contains else "until done")
                    console.print(f"    loop {name}  max {stage.max_iterations}x, {until}{after}")
                    for inner in stage.steps:
                        console.print(f"      - {inner.name} ({inner.kind.value})")
                else:
                    fan = ""
                    if stage.for_each:
                        fan = f" x{len(stage.for_each)}"
                    elif stage.copies > 1:
                        fan = f" x{stage.copies}"
                    console.print(f"    {name} ({stage.kind.value}){fan}{after}")
        return

    fleet = _fleet(
        config,
        **_overrides(workspace, None, executor, max_usd),
        on_event=_live_printer(verbose),
    )
    _cancel_on_interrupt(fleet)
    try:
        console.print(f"[bold]run {fleet.run_id}[/bold]  workflow {wf.name}")
        report = fleet.run_workflow(wf)
        console.print()
        console.print(report.summary())
        raise typer.Exit(0 if not report.run.failed else 1)
    finally:
        fleet.close()


@app.command()
def submit(
    file: Optional[str] = typer.Option(None, "--file", "-f", help="Path to a JSON job spec."),
    name: Optional[str] = typer.Option(None, "--name"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """Submit a job spec.

    Inside an agent container this writes to $FLEET_SUBMIT_DIR, where the
    scheduler picks it up and applies policy and budget checks. Outside, it
    submits directly.
    """
    raw = Path(file).read_text(encoding="utf-8") if file else sys.stdin.read()
    payload = json.loads(raw)

    spool = os.environ.get("FLEET_SUBMIT_DIR")
    if spool:
        target = Path(spool) / f"{name or payload.get('name') or new_id('req')}.json"
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"queued for scheduler review: {target.name}")
        console.print(
            "[dim]Policy and budget are applied by the scheduler; if it is rejected "
            "you will find <name>.rejected.json alongside it.[/dim]"
        )
        return

    fleet = _fleet(config)
    try:
        rec = fleet.submit(JobSpec(**payload))
        console.print(f"{rec.spec.id}  {rec.state.value}")
        report = fleet.wait()
        console.print(report.summary())
    finally:
        fleet.close()


@app.command()
def uninstall(
    prefix: str = typer.Argument("", help="Prefix it was installed under. Default ~/.local."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask."),
):
    """Remove the `fleet` command. Ledgers and results are kept."""
    import os
    import shutil
    import subprocess

    root = Path(prefix).expanduser() if prefix else Path.home() / ".local"
    if shutil.which("uv") is None:
        console.print("[red]uv not found[/red]; remove it however you installed it.")
        raise typer.Exit(1)
    if not yes and not typer.confirm(f"Remove fleet from {root}?"):
        raise typer.Exit(0)

    env = dict(
        os.environ,
        UV_TOOL_BIN_DIR=str(root / "bin"),
        UV_TOOL_DIR=str(root / "share" / "research-fleet" / "tools"),
    )
    done = subprocess.run(
        ["uv", "tool", "uninstall", "research-fleet"],
        capture_output=True, text=True, env=env, check=False,
    )

    # Past this point the environment this process is running from has been deleted,
    # so anything that imports lazily (rich markup, typer's error rendering) will fail.
    # Plain print and SystemExit need nothing that is not already loaded.
    if done.returncode == 0:
        print(f"Removed {root / 'bin' / 'fleet'}")
        print("Ledgers and results under ~/.research-fleet were kept.")
        raise SystemExit(0)
    print(f"fleet was not installed under {root}")
    raise SystemExit(1)


@app.command()
def cost(
    process: str = typer.Option("agent_standard", "--process",
                                help="single_call|workflow|agent_short|agent_standard|agent_long"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """Show what work costs on each model, before you run it."""
    cfg = load_config(config)
    table = Table(title=f"Estimated cost per {process.replace('_', ' ')}")
    for col in ("model", "effort", "est. cost", "est. tokens", "$/Mtok in", "$/Mtok out"):
        table.add_column(col)
    for row in cost_menu(cfg.budget.delegation_models, process=process):
        table.add_row(
            row["model"], row["effort"], f"${row['est_cost_usd']:.3f}",
            f"{row['est_total_tokens']:,}",
            f"${row['input_per_mtok']:.2f}", f"${row['output_per_mtok']:.2f}",
        )
    console.print(table)
    console.print(f"[dim]Run budget ceiling: ${cfg.budget.max_usd:.2f} / {cfg.budget.max_tokens:,} tokens[/dim]")


def _detach_and_return(config: Optional[str]) -> None:
    """Re-run this command in the background, logging to a file.

    The scheduler has to stay alive to stream output and settle the budget, so detaching
    means handing the work to a child in its own session rather than exiting early.
    """
    import os
    import subprocess
    import sys

    from .spec import new_id

    cfg = load_config(config)
    run_id = new_id("run")
    log_dir = cfg.root_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_id}.log"

    argv = [a for a in sys.argv if a not in ("--detach", "-d")]
    argv += ["--run-id", run_id]

    with log_path.open("w", encoding="utf-8") as log:
        subprocess.Popen(
            argv, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,        # survives this shell closing
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    console.print(f"[bold]run {run_id}[/bold] started in the background")
    console.print(f"  fleet watch {run_id}      follow it")
    console.print(f"  fleet ls {run_id}         job states")
    console.print(f"  fleet kill {run_id}       stop it")
    console.print(f"[dim]  log: {log_path}[/dim]")


@app.command()
def watch(
    run_id: str = typer.Argument(..., help="Run to follow."),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """Follow a detached run's output."""
    cfg = load_config(config)
    log_path = cfg.root_path / "logs" / f"{run_id}.log"
    if not log_path.exists():
        console.print(f"[red]no log for {run_id}[/red] at {log_path}")
        raise typer.Exit(1)
    import subprocess

    try:
        subprocess.run(["tail", "-n", "+1", "-f", str(log_path)], check=False)
    except KeyboardInterrupt:
        pass


@app.command()
def login(
    import_host: bool = typer.Option(
        False, "--import",
        help="Copy the credentials already on this host instead of signing in.",
    ),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """Give this project's agents credentials, by delegating to research-ship.

    Credentials belong to the environment, not the scheduler, so this is a thin wrapper
    over `ship login`. It runs against the project the fleet is configured for, which is
    what makes it easy to get wrong by hand.
    """
    import os
    import subprocess

    cfg = load_config(config, workspace=workspace)
    project = str(Path(cfg.executor.project_dir or cfg.workspace).expanduser().resolve())
    argv = [cfg.executor.ship_binary, "login"] + (["--import"] if import_host else [])
    console.print(f"[dim]project: {project}[/dim]")
    done = subprocess.run(argv, env={**os.environ, "SHIP_PROJECT_DIR": project})
    raise typer.Exit(done.returncode)


@app.command()
def kill(
    run_id: Optional[str] = typer.Argument(None, help="Which run. Default: every active run."),
    root: Optional[str] = typer.Option(None, "--root", help="State directory the run used."),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """Stop a run's containers and cluster jobs, from any shell.

    Works across processes: containers carry a `fleet.run` label and Slurm jobs are
    named after the job id, so this does not need the scheduler that started them.
    """
    import shutil
    import subprocess

    cfg = load_config(config, root=root)
    ledger = Ledger(cfg.root_path)
    try:
        # Runs are grouped per state directory, so name it. Killing nothing usually
        # means the run lives under a different root.
        console.print(f"[dim]state: {cfg.root_path}[/dim]")
        targets = [run_id] if run_id else ledger.active_runs()
        if not targets:
            console.print("No active runs here.")
            return

        for target in targets:
            containers = subprocess.run(
                ["docker", "ps", "-q", "-f", f"label=fleet.run={target}"],
                capture_output=True, text=True, check=False,
            ).stdout.split()
            if containers:
                subprocess.run(["docker", "stop", "-t", "10", *containers],
                               capture_output=True, text=True, check=False)

            # Slurm jobs are not containers. `scancel` exits 0 even when the name
            # matches nothing, so this is reported as signalled rather than confirmed.
            unfinished = [
                j["job_id"] for j in ledger.jobs(target)
                if j["state"] not in {"succeeded", "failed", "cancelled", "denied"}
            ]
            signalled = 0
            if shutil.which("scancel") and unfinished:
                for job_id in unfinished:
                    subprocess.run(["scancel", "--name", f"fleet-{job_id}"],
                                   capture_output=True, text=True, check=False)
                signalled = len(unfinished)

            marked = ledger.mark_cancelled(target, "killed by operator")
            parts = [f"{len(containers)} container(s) stopped"]
            if signalled:
                parts.append(f"{signalled} slurm name(s) signalled")
            parts.append(f"{len(marked)} job(s) marked cancelled")
            console.print(f"[yellow]{target}[/yellow]: " + ", ".join(parts))
            if not containers and marked:
                console.print(
                    "[dim]  nothing was running; those jobs were stale entries whose "
                    "scheduler had already gone[/dim]"
                )
    finally:
        ledger.close()


@app.command()
def usage(
    by: str = typer.Option("run", "--by", "-b",
                           help="run, model, stage, attempt, name, kind, backend, day, job. "
                                "Comma separated for a finer grain, e.g. run,stage,attempt."),
    run_id: Optional[str] = typer.Option(None, "--run", help="Only this run."),
    days: Optional[int] = typer.Option(None, "--days", help="Only the last N days."),
    kind: Optional[str] = typer.Option(None, "--kind", help="agent or command."),
    jobs: bool = typer.Option(False, "--jobs", help="List every job instead of totals."),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """Compute and spend across runs, from the audit ledger."""
    cfg = load_config(config)
    ledger = Ledger(cfg.root_path)
    since = time.time() - days * 86400 if days else None
    try:
        if jobs:
            rows = ledger.usage_rows(run_id=run_id, since=since, kind=kind)
            table = Table(title="jobs")
            for col in ("run", "stage", "try", "model", "tokens", "cost",
                        "agent_s", "wall_s", "gpu_s", "state"):
                table.add_column(col)
            for r in rows:
                table.add_row(
                    (r["run_id"] or "")[-8:], r["stage"] or r["name"], str(r["attempt"]),
                    (r["model"] or "-").replace("claude-", ""),
                    f"{r['total_tokens']:,}", f"${r['cost_usd']:.4f}",
                    f"{(r['agent_seconds'] or 0):.0f}", f"{(r['duration_s'] or 0):.0f}",
                    f"{r['gpu_seconds']:.0f}", r["state"],
                )
            console.print(table)
        else:
            grouped = ledger.usage_by(by, run_id=run_id, since=since, kind=kind)
            keys = [k.strip() for k in by.split(",") if k.strip()]
            table = Table(title=f"usage by {by}")
            for key in keys:
                table.add_column(key)
            for col in ("jobs", "tokens", "cost", "agent_s", "wall_s", "gpu_s"):
                table.add_column(col, justify="right")
            for r in grouped:
                cells = [str(r[k] if r[k] is not None else "-")[-24:] for k in keys]
                table.add_row(
                    *cells, str(r["jobs"]), f"{r['total_tokens']:,}",
                    f"${r['cost_usd']:.4f}", f"{r['agent_seconds']:.0f}",
                    f"{r['duration_s']:.0f}", f"{r['gpu_seconds']:.0f}",
                )
            console.print(table)

        total = ledger.usage_totals(run_id=run_id, since=since, kind=kind)
        console.print(
            f"[bold]{total['jobs']} job(s)[/bold]  ${total['cost_usd']:.4f}  "
            f"{total['total_tokens']:,} tokens  "
            f"{total['agent_seconds']:.0f}s agent / {total['duration_s']:.0f}s wall  "
            f"{total['gpu_seconds']:.0f} GPU-seconds  "
            f"{total['requests']} request(s)"
        )
        if total.get("unpriced_jobs"):
            console.print(
                f"[yellow]{total['unpriced_jobs']} job(s) used a model with no price, "
                "so their cost reads as zero.[/yellow]"
            )
    finally:
        ledger.close()


@app.command()
def runs(config: Optional[str] = typer.Option(None, "--config", "-c")):
    """List past runs."""
    cfg = load_config(config)
    ledger = Ledger(cfg.root_path)
    table = Table(title="runs")
    for col in ("run_id", "jobs", "ok", "failed", "started"):
        table.add_column(col)
    for r in ledger.runs():
        table.add_row(
            r["run_id"], str(r["jobs"]), str(r["succeeded"]), str(r["failed"]),
            time.strftime("%Y-%m-%d %H:%M", time.localtime(r["started_at"] or 0)),
        )
    console.print(table)
    ledger.close()


@app.command("ls")
def list_jobs(
    run_id: Optional[str] = typer.Argument(None),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """List jobs, optionally filtered to one run."""
    cfg = load_config(config)
    ledger = Ledger(cfg.root_path)
    table = Table(title=f"jobs{f' in {run_id}' if run_id else ''}")
    for col in ("job_id", "name", "kind", "state", "parent"):
        table.add_column(col)
    for j in ledger.jobs(run_id):
        colour = {"succeeded": "green", "failed": "red", "denied": "red"}.get(j["state"], "yellow")
        table.add_row(j["job_id"], j["name"], j["kind"],
                      f"[{colour}]{j['state']}[/{colour}]", j["parent"] or "-")
    console.print(table)
    ledger.close()


@app.command()
def trace(
    job_id: str = typer.Argument(..., help="Job to replay."),
    types: Optional[str] = typer.Option(None, "--types", help="Comma-separated event types to include."),
    limit: int = typer.Option(500, "--limit"),
    raw: bool = typer.Option(False, "--raw", help="Emit JSONL instead of prose."),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """Replay one job's full reasoning and tool trace from the ledger."""
    cfg = load_config(config)
    ledger = Ledger(cfg.root_path)
    wanted = types.split(",") if types else None
    for ev in ledger.events(job_id=job_id, types=wanted, limit=limit):
        if raw:
            print(ev.to_json())
            continue
        ts = time.strftime("%H:%M:%S", time.localtime(ev.ts))
        if ev.type == "job.output":
            stream = ev.payload.get("stream", "stdout")
            style = "red" if stream == "stderr" else "white"
            console.print(f"[dim]{ts}[/dim] [{style}]{ev.payload.get('line', '')}[/{style}]")
        elif ev.type == "agent.message":
            console.print(f"[dim]{ts}[/dim] [bold]agent[/bold]: {ev.payload.get('text', '')}")
        elif ev.type == "agent.tool_use":
            console.print(f"[dim]{ts}[/dim] [cyan]tool[/cyan] {ev.payload.get('tool')}")
            console.print(JSON.from_data(ev.payload.get("payload", {})), style="dim")
        elif ev.type == "agent.tool_result":
            console.print(f"[dim]{ts}[/dim] [magenta]result[/magenta]")
            console.print(JSON.from_data(ev.payload.get("payload", {})), style="dim")
        elif ev.type == "budget.committed":
            u = ev.payload.get("usage", {})
            console.print(
                f"[dim]{ts}[/dim] [yellow]budget[/yellow] ${ev.payload.get('cost_usd', 0):.4f} "
                f"({u.get('total_tokens', 0):,} tokens)"
            )
        else:
            console.print(f"[dim]{ts} {ev.type}[/dim] {str(ev.payload)[:300]}")
    ledger.close()


@audit_app.command("verify")
def audit_verify(config: Optional[str] = typer.Option(None, "--config", "-c")):
    """Verify the hash chain: proves the log has not been edited."""
    cfg = load_config(config)
    ledger = Ledger(cfg.root_path)
    ok, msg = ledger.verify()
    ledger.close()
    if ok:
        console.print(f"[green]✓ audit chain intact[/green]: {msg}")
        raise typer.Exit(0)
    console.print(f"[red]✗ audit chain broken[/red]: {msg}")
    raise typer.Exit(2)


@audit_app.command("export")
def audit_export(
    run_id: Optional[str] = typer.Option(None, "--run"),
    out: Optional[str] = typer.Option(None, "--out", "-o"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """Export the ledger (optionally one run) as JSONL for external review."""
    cfg = load_config(config)
    ledger = Ledger(cfg.root_path)
    sink = open(out, "w", encoding="utf-8") if out else sys.stdout
    try:
        for ev in ledger.read_all():
            if run_id and ev.run_id != run_id:
                continue
            sink.write(ev.to_json() + "\n")
    finally:
        if out:
            sink.close()
        ledger.close()


@audit_app.command("reindex")
def audit_reindex(config: Optional[str] = typer.Option(None, "--config", "-c")):
    """Rebuild the query index from the append-only JSONL."""
    cfg = load_config(config)
    ledger = Ledger(cfg.root_path)
    n = ledger.reindex()
    ledger.close()
    console.print(f"reindexed {n} events")


@app.command()
def pending(config: Optional[str] = typer.Option(None, "--config", "-c")):
    """List jobs parked waiting for operator approval, and why.

    Approval itself is in-process: a parked job belongs to a live scheduler, so
    grant it with `fleet.approve(job_id)` from the session that submitted it (or
    via the MCP server). There is deliberately no cross-process approve command:
    a job whose scheduler has exited cannot be resumed, only resubmitted.
    """
    cfg = load_config(config)
    ledger = Ledger(cfg.root_path)
    rows = [j for j in ledger.jobs() if j["state"] == "awaiting_approval"]
    if not rows:
        console.print("nothing awaiting approval")
    for j in rows:
        reasons = [
            e.payload.get("reasons", [])
            for e in ledger.events(job_id=j["job_id"], types=["job.awaiting_approval"])
        ]
        flat = [r for group in reasons for r in group]
        console.print(f"[yellow]{j['job_id']}[/yellow]  {j['name']}  {'; '.join(flat)}")
    ledger.close()


@app.command()
def policy(config: Optional[str] = typer.Option(None, "--config", "-c")):
    """Print the effective safeguard policy."""
    cfg = load_config(config)
    console.print(JSON.from_data(cfg.policy.model_dump(mode="json")))


@app.command()
def mcp(config: Optional[str] = typer.Option(None, "--config", "-c")):
    """Serve the fleet as an MCP server so Claude Code / Codex can drive it."""
    from .mcp_server import serve

    serve(config)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
