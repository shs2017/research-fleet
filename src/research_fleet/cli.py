"""`fleet`: the command line entry point.

It intentionally exposes workflows and direct runs rather than the scheduler's
internal job-spec API.
"""

from __future__ import annotations

import os
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table

from .budget import codex_credits, cost_menu
from .config import load_config
from .fleet import CredentialsUnavailable, Fleet
from .ledger import Ledger
from . import sharedprompt

app = typer.Typer(
    add_completion=True,
    no_args_is_help=True,
    help="Configurable, reproducible agent workflows for autonomous research.",
)
audit_app = typer.Typer(no_args_is_help=True, help="Inspect and verify the audit ledger.")
app.add_typer(audit_app, name="audit")

console = Console()


def _complete_runs(ctx: typer.Context, incomplete: str) -> list[str]:
    """Complete run ids from the configured ledger and detached-run logs."""
    config = (ctx.params or {}).get("config") if ctx is not None else None
    root = (ctx.params or {}).get("root") if ctx is not None else None
    try:
        cfg = load_config(config, root=root)
        with Ledger(cfg.root_path) as ledger:
            found = {r["run_id"] for r in ledger.runs()}
            found.update(ev.run_id for ev in ledger.events(limit=100000) if ev.run_id)
        log_dir = cfg.root_path / "logs"
        if log_dir.exists():
            found.update(p.stem for p in log_dir.glob("run_*.log"))
        return sorted(run_id for run_id in found if run_id.startswith(incomplete))
    except Exception:
        # Completion must never turn a typo or unavailable state directory into a
        # traceback in the user's interactive shell.
        return []


def _complete_jobs(ctx: typer.Context, incomplete: str) -> list[str]:
    """Complete job ids from the configured ledger."""
    config = (ctx.params or {}).get("config") if ctx is not None else None
    try:
        cfg = load_config(config)
        with Ledger(cfg.root_path) as ledger:
            return sorted(
                job["job_id"] for job in ledger.jobs()
                if job["job_id"].startswith(incomplete)
            )
    except Exception:
        return []


def _choices(*values: str):
    def complete(incomplete: str) -> list[str]:
        return [value for value in values if value.startswith(incomplete)]

    return complete


_BACKENDS = _choices("claude-cli", "codex-cli")
_EFFORTS = _choices("low", "medium", "high", "xhigh", "max")
_EXECUTORS = _choices("ship", "nono", "direct", "dry-run")


def _fleet(config: Optional[str], **overrides) -> Fleet:
    """Build a Fleet, turning setup problems into advice rather than a traceback."""
    from .executors import ShipUnavailable

    try:
        return Fleet(config, **{k: v for k, v in overrides.items() if v is not None})
    except ShipUnavailable as exc:
        console.print(f"[red]cannot start:[/red] {exc}")
        raise typer.Exit(1) from None


def _credentials_error(exc: CredentialsUnavailable) -> None:
    """Render expected login setup as a CLI message instead of a traceback."""
    console.print(f"[red]cannot run agents:[/red] {exc}")
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


def _cpu_share(requested: Optional[float], agents: int) -> tuple[float, str]:
    """Choose how many CPUs each agent reserves, and explain the choice.

    Docker rejects a container outright if `--cpus` exceeds the host's core count
    (`range of CPUs is from 0.01 to N`), so an unset request is derived from what
    the host actually has rather than a fixed guess that breaks on small hosts.
    """
    total = os.cpu_count() or 1
    if requested is not None:
        if requested > total:
            return requested, (
                f"[yellow]note: --cpus {requested} exceeds the {total} core(s) on this host; "
                f"docker will refuse it. Try --cpus {total / agents:.2f}.[/yellow]"
            )
        return requested, ""
    share = max(0.01, total / agents)
    return share, f"{share:.2f} CPU(s) each on {total} core(s)"


def _overrides(workspace=None, image=None, executor=None, max_usd=None) -> dict:
    """Map run flags onto config overrides."""
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
    backend: Optional[str] = typer.Option(
        None, "--backend", help="claude-cli | codex-cli", autocompletion=_BACKENDS
    ),
    effort: Optional[str] = typer.Option(
        None, "--effort", help="low|medium|high|xhigh|max", autocompletion=_EFFORTS
    ),
    gpus: Optional[float] = typer.Option(
        None, "--gpus",
        help="GPUs per agent. Default: share the devices so every agent runs at once.",
    ),
    cpus: Optional[float] = typer.Option(
        None, "--cpus",
        help="CPUs per agent. Default: share the host's cores so every agent runs at once.",
    ),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    image: Optional[str] = typer.Option(None, "--image"),
    timeout: int = typer.Option(3600, "--timeout", help="Per-agent wall clock, seconds."),
    max_usd: Optional[float] = typer.Option(None, "--max-usd", help="Budget ceiling for the whole run."),
    executor: Optional[str] = typer.Option(
        None, "--executor", help="ship | nono | direct | dry-run", autocompletion=_EXECUTORS
    ),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    detach: bool = typer.Option(False, "--detach", "-d", help="Return immediately; run in the background."),
    resume_from: Optional[str] = typer.Option(
        None, "--resume", help="Continue this run in its existing workspace/results directory.",
        autocompletion=_complete_runs,
    ),
    run_id: Optional[str] = typer.Option(None, "--run-id", hidden=True),
):
    """Launch one or more research agents on a task.

    ``--resume`` reuses the prior run's results directory and workspace context while
    submitting the requested task as a new continuation attempt. It does not resume
    the provider's interactive conversation/session.
    """
    if detach:
        _detach_and_return(config)
        return

    with _fleet(
        config, **_overrides(workspace, image, executor, max_usd),
        on_event=_live_printer(verbose), run_id=run_id,
    ) as fleet:
        _cancel_on_interrupt(fleet)
        chosen_model = fleet.agent_model(model, backend)
        est = fleet.quote(chosen_model, effort=effort or fleet.config.budget.default_effort)
        share, note = _gpu_share(gpus, agents, fleet.scheduler.slots.device_count)
        cpu_share, cpu_note = _cpu_share(cpus, agents)
        console.print(
            f"[bold]run {fleet.run_id}[/bold]  {agents} agent(s)  "
            f"est. ${est.est_cost_usd * agents:.2f} ({est.source})  "
            f"budget ${fleet.config.budget.max_usd:.2f}"
        )
        if note:
            console.print(f"[dim]{note}[/dim]" if not note.startswith("[") else note)
        if cpu_note:
            console.print(f"[dim]{cpu_note}[/dim]" if not cpu_note.startswith("[") else cpu_note)
        try:
            fleet.run_agents(
                task, n=agents, model=model, backend=backend, effort=effort,
                gpus=share, cpus=cpu_share, timeout_s=timeout,
                resume_from=resume_from,
            )
        except CredentialsUnavailable as exc:
            _credentials_error(exc)
        report = fleet.wait()
        console.print()
        console.print(report.summary())
        raise typer.Exit(0 if not report.failed else 1)


@app.command()
def workflow(
    file: Path = typer.Argument(..., help="Workflow YAML.", exists=True, dir_okay=False),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    max_usd: Optional[float] = typer.Option(None, "--max-usd"),
    executor: Optional[str] = typer.Option(
        None, "--executor", help="ship | nono | direct | dry-run", autocompletion=_EXECUTORS
    ),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    plan: bool = typer.Option(False, "--plan", help="Validate and print the stages, run nothing."),
    resume_from: Optional[str] = typer.Option(
        None, "--resume", help="Continue from the last compatible checkpoint in this run.",
        autocompletion=_complete_runs,
    ),
    base_run: Optional[str] = typer.Option(
        None, "--from-run", help="Reuse a run's outputs/files but execute all stages again.",
        autocompletion=_complete_runs,
    ),
    parameter: list[str] = typer.Option(
        [], "--set", help="Set a workflow parameter (key=value); repeat for ablations/seeds."
    ),
    ablation: Optional[str] = typer.Option(None, "--ablation", help="Select a structural or prompt variant."),
    detach: bool = typer.Option(
        False, "--detach", "-d", help="Return immediately; run the workflow in the background."
    ),
    run_id: Optional[str] = typer.Option(None, "--run-id", hidden=True),
):
    """Run a multi-step pipeline, for example a coder and reviewer loop."""
    from .workflow import Loop, Step, Workflow

    wf = Workflow.from_yaml(file, ablation=ablation)
    for assignment in parameter:
        if "=" not in assignment:
            raise typer.BadParameter("workflow parameters must use key=value", param_hint="--set")
        key, value = assignment.split("=", 1)
        key = key.strip()
        if not key:
            raise typer.BadParameter("workflow parameter name cannot be empty", param_hint="--set")
        try:
            wf.parameters[key] = yaml.safe_load(value)
        except yaml.YAMLError as exc:
            raise typer.BadParameter(f"invalid value for {key}: {exc}", param_hint="--set") from exc

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

    if detach:
        _detach_and_return(config)
        return

    with _fleet(
        config, **_overrides(workspace, None, executor, max_usd),
        on_event=_live_printer(verbose),
        run_id=run_id,
    ) as fleet:
        _cancel_on_interrupt(fleet)
        console.print(f"[bold]run {fleet.run_id}[/bold]  workflow {wf.name}")
        try:
            report = fleet.run_workflow(wf, resume_from=resume_from, base_run=base_run)
        except CredentialsUnavailable as exc:
            _credentials_error(exc)
        console.print()
        console.print(report.summary())
        raise typer.Exit(0 if not report.run.failed else 1)


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
    models = list(dict.fromkeys([cfg.budget.default_model, *cfg.budget.delegation_models]))
    for row in cost_menu(models, process=process):
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
        process = subprocess.Popen(
            argv, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,        # survives this shell closing
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    (log_dir / f"{run_id}.pid").write_text(f"{process.pid}\n", encoding="utf-8")
    console.print(f"[bold]run {run_id}[/bold] started in the background")
    console.print(f"  fleet log {run_id} -f     follow it")
    console.print(f"  fleet jobs {run_id}       job states")
    console.print(f"  fleet kill {run_id}       stop it")
    console.print(f"[dim]  log: {log_path}[/dim]")


def _follow_run_log(
    log_path: Path,
    root_path: Path,
    run_id: str,
    *,
    pid_path: Path | None = None,
    poll_interval: float = 0.2,
) -> None:
    """Stream a detached log and stop once its ledger run is terminal.

    ``tail -f`` never exits when the scheduler closes and its output can be block
    buffered when followed by another process. Reading here lets us
    flush every append, survive a truncated/replaced log, and use the ledger as the
    authoritative completion signal.
    """
    offset = 0
    saw_jobs = False
    terminal_polls = 0
    terminal_states = {"succeeded", "failed", "cancelled", "denied"}

    while True:
        try:
            size = log_path.stat().st_size
            if size < offset:
                offset = 0
            with log_path.open("r", encoding="utf-8", errors="replace") as log:
                log.seek(offset)
                chunk = log.read()
                offset = log.tell()
        except FileNotFoundError:
            chunk = ""
            offset = 0

        if chunk:
            sys.stdout.write(chunk)
            sys.stdout.flush()

        with Ledger(root_path) as ledger:
            jobs = ledger.jobs(run_id=run_id)
        saw_jobs = saw_jobs or bool(jobs)
        terminal = saw_jobs and jobs and all(job["state"] in terminal_states for job in jobs)
        if not saw_jobs and pid_path is not None and pid_path.exists():
            try:
                pid = int(pid_path.read_text(encoding="utf-8").strip())
                stat = Path(f"/proc/{pid}/stat")
                process_alive = stat.exists() and stat.read_text().split()[2] != "Z"
            except (OSError, ValueError, IndexError):
                process_alive = False
            terminal = not process_alive
        terminal_polls = terminal_polls + 1 if terminal else 0

        # One extra poll drains text written just after the terminal ledger event.
        if terminal_polls >= 2:
            return
        time.sleep(poll_interval)


def _render_log_event(ev, *, raw: bool = False, show_job: bool = False) -> None:
    """Render one ledger event for both run- and job-level logs."""
    if raw:
        # Keep the original epoch ``ts`` for machines, and add a readable
        # timezone-local timestamp for people inspecting JSONL logs.
        record = vars(ev).copy()
        record["timestamp"] = time.strftime(
            "%Y-%m-%dT%H:%M:%S%z", time.localtime(ev.ts)
        )
        import json
        print(json.dumps(record, separators=(",", ":"), default=str))
        return
    ts = time.strftime("%H:%M:%S", time.localtime(ev.ts))
    job = f" [dim]{(ev.job_id or '')[-6:]}[/dim]" if show_job and ev.job_id else ""
    if ev.type == "job.submitted":
        spec = ev.payload.get("spec", {})
        agent = spec.get("agent") or {}
        if agent:
            session = agent.get("session_id")
            detail = "continued session" if session else "new session"
            title = f"[dim]{ts}[/dim]{job} prompt · {spec.get('name', 'agent')} · {detail}"
            parts = []
            if agent.get("system_prompt"):
                parts.append("[bold]System[/bold]\n" + agent["system_prompt"])
            parts.append("[bold]Task[/bold]\n" + agent.get("task", ""))
            console.print(Panel(
                "\n\n".join(parts), title=title, title_align="left",
                border_style="green", padding=(0, 1),
            ))
        else:
            console.print(
                f"[dim]{ts}[/dim]{job} [green]command[/green] "
                f"[bold]{spec.get('name', '')}[/bold] "
                f"{shlex.join(spec.get('command') or [])}"
            )
    elif ev.type == "job.output":
        stream = ev.payload.get("stream", "stdout")
        style = "red" if stream == "stderr" else "white"
        console.print(f"[dim]{ts}[/dim]{job} [{style}]{ev.payload.get('line', '')}[/{style}]")
    elif ev.type == "agent.message":
        console.print(Panel(
            ev.payload.get("text", ""), title=f"[dim]{ts}[/dim]{job} agent",
            title_align="left", border_style="blue", padding=(0, 1),
        ))
    elif ev.type == "agent.tool_use":
        console.print(
            f"[dim]{ts}[/dim]{job} [cyan]◆ tool[/cyan] [bold]{ev.payload.get('tool')}[/bold]"
        )
        console.print(JSON.from_data(ev.payload.get("payload", {})), style="dim", soft_wrap=True)
    elif ev.type == "agent.tool_result":
        console.print(f"[dim]{ts}[/dim]{job} [magenta]◇ result[/magenta]")
        console.print(JSON.from_data(ev.payload.get("payload", {})), style="dim", soft_wrap=True)
    elif ev.type == "budget.committed":
        usage = ev.payload.get("usage", {})
        console.print(
            f"[dim]{ts}[/dim]{job} [yellow]budget[/yellow] "
            f"${ev.payload.get('cost_usd', 0):.4f} "
            f"({usage.get('total_tokens', 0):,} tokens)"
        )
    else:
        state_style = {
            "job.succeeded": "green", "job.failed": "red",
            "job.denied": "red", "job.running": "cyan",
        }.get(ev.type, "dim")
        console.print(
            f"[dim]{ts}[/dim]{job} [{state_style}]● {ev.type}[/{state_style}] "
            f"[dim]{str(ev.payload)[:300]}[/dim]"
        )


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
    run_id: Optional[str] = typer.Argument(
        None, help="Which run. Default: every active run.", autocompletion=_complete_runs
    ),
    root: Optional[str] = typer.Option(None, "--root", help="State directory the run used."),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """Stop a run's containers from any shell.

    This works across processes because containers carry a `fleet.run` label.
    """
    import os
    import signal
    import subprocess
    import time

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

            # Host/nonо jobs run outside the scheduler process. The direct
            # executor records one PID per job; terminate the whole process group
            # so nono and the agent/compiler children cannot survive the kill.
            pid_dir = cfg.root_path / "nono" / "pids" / target
            host_pids: list[int] = []
            if pid_dir.is_dir():
                for path in pid_dir.glob("*.pid"):
                    try:
                        pid = int(path.read_text(encoding="ascii").strip())
                    except (OSError, ValueError):
                        continue
                    if pid > 1:
                        try:
                            os.killpg(pid, signal.SIGTERM)
                            host_pids.append(pid)
                        except ProcessLookupError:
                            pass
                        except PermissionError:
                            console.print(f"[red]permission denied stopping process group {pid}[/red]")
            # Older nono jobs may predate PID-file support. Their worktree path
            # still carries the run id, so use /proc cwd as a conservative fallback.
            if not host_pids:
                marker = f"fleet-{target}"
                for cwd_link in Path("/proc").glob("[0-9]*/cwd"):
                    try:
                        pid = int(cwd_link.parent.name)
                        cwd = os.readlink(cwd_link)
                    except (OSError, ValueError):
                        continue
                    if marker not in cwd or pid <= 1:
                        continue
                    try:
                        os.killpg(pid, signal.SIGTERM)
                        host_pids.append(pid)
                    except (ProcessLookupError, PermissionError):
                        pass
                if host_pids:
                    time.sleep(1)
                    for pid in host_pids:
                        try:
                            os.killpg(pid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            pass

            marked = ledger.mark_cancelled(target, "killed by operator")
            parts = [f"{len(containers)} container(s) stopped",
                     f"{len(host_pids)} host job(s) stopped"]
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
    run_id: Optional[str] = typer.Option(
        None, "--run", help="Only this run.", autocompletion=_complete_runs
    ),
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
def init(
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    force: bool = typer.Option(
        False, "--force", help="Overwrite an edited FLEET.md with the current default."
    ),
):
    """Write the default agent instructions (FLEET.md) into a project."""
    cfg = load_config(config, **_overrides(workspace=workspace))
    target = Path(cfg.workspace).expanduser().resolve()
    path, outcome = sharedprompt.write_default(target, force=force)

    if outcome == "written":
        console.print(f"[green]Wrote[/] {path}")
        console.print(
            "Every agent job in this project now gets these instructions. They are "
            "general on purpose;\nedit the file to add anything specific to this "
            "project, and fleet will not overwrite it."
        )
    elif outcome == "unchanged":
        console.print(f"{path} already matches the default; nothing to do.")
    else:
        console.print(f"[yellow]Kept[/] your edited {path} (use --force to replace it).")


@app.command()
def runs(config: Optional[str] = typer.Option(None, "--config", "-c")):
    """List past runs."""
    cfg = load_config(config)
    with Ledger(cfg.root_path) as ledger:
        table = Table(title="runs")
        for col in ("run_id", "jobs", "ok", "failed", "started"):
            table.add_column(col)
        for r in ledger.runs():
            table.add_row(
                r["run_id"], str(r["jobs"]), str(r["succeeded"]), str(r["failed"]),
                time.strftime("%Y-%m-%d %H:%M", time.localtime(r["started_at"] or 0)),
            )
        console.print(table)


@app.command()
def jobs(
    run_id: Optional[str] = typer.Argument(None, autocompletion=_complete_runs),
    state: str = typer.Option(
        "all", "--state", "-s",
        help="all | current | failed | succeeded | pending",
        autocompletion=_choices("all", "current", "failed", "succeeded", "pending"),
    ),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """List old or current jobs, optionally filtered by run and state."""
    groups = {
        "all": None,
        "current": {"pending", "queued", "awaiting_approval", "running"},
        "failed": {"failed", "denied", "cancelled"},
        "succeeded": {"succeeded"},
        "pending": {"awaiting_approval"},
    }
    if state not in groups:
        raise typer.BadParameter("use all, current, failed, succeeded, or pending", param_hint="--state")
    cfg = load_config(config)
    with Ledger(cfg.root_path) as ledger:
        rows = ledger.jobs(run_id)
        if groups[state] is not None:
            rows = [row for row in rows if row["state"] in groups[state]]
        usage = {row["job_id"]: row for row in ledger.usage_rows(run_id=run_id)}

        # Show the run definition separately when one run was selected. This makes
        # seed/ablation/workflow identity explicit instead of burying it in prompts.
        if run_id:
            starts = ledger.events(run_id=run_id, types=["workflow.started"], limit=1)
            if starts:
                payload = starts[0].payload
                parameters = payload.get("parameters") or {}
                meta = Table(title="run", show_header=False, box=None)
                meta.add_column("field", style="dim")
                meta.add_column("value")
                meta.add_row("run", run_id)
                meta.add_row("workflow", str(payload.get("name", "-")))
                meta.add_row("seed", str(parameters.get("seed", "-")))
                meta.add_row("ablation", str(parameters.get("ablation", "-")))
                meta.add_row("variant", str(parameters.get("variant", "-")))
                console.print(meta)

        table = Table(title=f"{state} jobs{f' in {run_id}' if run_id else ''}")
        columns = (("job", "stage", "try", "seed/ablation", "model/effort", "state",
                    "usage (input / cached / cache-write / output / total / cost)")
                   if run_id else
                   ("job", "run", "stage", "try", "seed/ablation", "model/effort", "state",
                    "usage (input / cached / cache-write / output / total / cost)"))
        for col in columns:
            table.add_column(col)
        for j in rows:
            colour = {"succeeded": "green", "failed": "red", "denied": "red"}.get(
                j["state"], "yellow"
            )
            try:
                spec = json.loads(j.get("spec") or "{}")
            except (TypeError, ValueError):
                spec = {}
            labels = spec.get("labels") or {}
            agent = spec.get("agent") or {}
            u = usage.get(j["job_id"], {})
            model = u.get("model") or agent.get("model") or "-"
            effort = agent.get("effort") or "-"
            credits = codex_credits(
                model,
                input_tokens=u.get("input_tokens", 0),
                cache_read_tokens=u.get("cache_read_tokens", 0),
                output_tokens=u.get("output_tokens", 0),
            ) if u else None
            identity = f"{labels.get('seed', '-')} / {labels.get('ablation', '-')}"
            values = [
                j["job_id"],
                labels.get("stage") or j["name"],
                str(labels.get("attempt") or "1"),
                identity,
            f"{model} / {effort} / {labels.get('execution_mode', 'standard')}",
                f"[{colour}]{j['state']}[/{colour}]",
                (f"{u.get('input_tokens', 0):,} / {u.get('cache_read_tokens', 0):,} / "
                 f"{u.get('cache_write_tokens', 0):,} / {u.get('output_tokens', 0):,} / "
                 f"{u.get('total_tokens', 0):,} / "
                 f"${u.get('cost_usd', 0.0):.4f} / "
                 f"{credits:.2f} credits" if u and not u.get("unpriced") else
                 ("unpriced" if u else "-")),
            ]
            if not run_id:
                values.insert(1, (j.get("run_id") or "-")[-8:])
            table.add_row(*values)
        console.print(table)
        total = ledger.usage_totals(run_id=run_id)
        total_credits = 0.0
        credit_jobs = 0
        for u in usage.values():
            c = codex_credits(u.get("model"), input_tokens=u.get("input_tokens", 0),
                              cache_read_tokens=u.get("cache_read_tokens", 0),
                              output_tokens=u.get("output_tokens", 0))
            if c is not None:
                total_credits += c
                credit_jobs += 1
        console.print(
            f"[bold]usage:[/bold] {total['total_tokens']:,} tokens  "
            f"${total['cost_usd']:.4f}  "
            f"(input {total['input_tokens']:,}, cached {total['cache_read_tokens']:,}, "
            f"output {total['output_tokens']:,})"
        )
        if credit_jobs:
            console.print(
                f"[bold]Codex subscription:[/bold] {total_credits:.2f} credits "
                "(remaining allowance/percentage is not exposed by the CLI)"
            )


@app.command()
def log(
    target: str = typer.Argument(..., help="Run or job ID to show."),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow until it finishes."),
    types: Optional[str] = typer.Option(None, "--types", help="Comma-separated event types to include."),
    limit: int = typer.Option(500, "--limit"),
    raw: bool = typer.Option(False, "--raw", help="Emit JSONL instead of prose."),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """Show a run log or a job's agent/tool history."""
    cfg = load_config(config)
    if target.startswith("run_"):
        log_path = cfg.root_path / "logs" / f"{target}.log"
        wanted = types.split(",") if types else None
        terminal_states = {"succeeded", "failed", "denied", "cancelled"}
        seen: set[int] = set()
        while True:
            with Ledger(cfg.root_path) as ledger:
                events = ledger.events(run_id=target, types=wanted, limit=limit)
                rows = ledger.jobs(run_id=target)
            if not events and not rows:
                console.print(f"[yellow]No events found for {target}[/yellow]")
                raise typer.Exit(1)
            if not raw and not seen:
                console.print(Panel.fit(
                    f"[bold]{target}[/bold]\n[dim]{len(rows)} job(s)[/dim]",
                    title="Fleet log", border_style="cyan",
                ))
            for ev in events:
                if ev.seq not in seen:
                    _render_log_event(ev, raw=raw, show_job=True)
                    seen.add(ev.seq)
            if not follow or (rows and all(row["state"] in terminal_states for row in rows)):
                return
            try:
                time.sleep(0.2)
            except KeyboardInterrupt:
                return

    job_id = target
    wanted = types.split(",") if types else None
    with Ledger(cfg.root_path) as ledger:
        events = ledger.events(job_id=job_id, types=wanted, limit=limit)
        jobs = {job["job_id"]: job for job in ledger.jobs()}
    if not events:
        console.print(f"[yellow]No events found for {job_id}[/yellow]")
        raise typer.Exit(1)
    if not raw:
        job = jobs.get(job_id, {})
        subtitle = " · ".join(filter(None, [job.get("name"), job.get("state")]))
        console.print(Panel.fit(
            f"[bold]{job_id}[/bold]\n[dim]{subtitle or 'event history'}[/dim]",
            title="Fleet log", border_style="cyan",
        ))
    seen: set[int] = set()
    try:
        while True:
            for ev in events:
                if ev.seq not in seen:
                    _render_log_event(ev, raw=raw)
                    seen.add(ev.seq)
            if not follow:
                return
            with Ledger(cfg.root_path) as ledger:
                rows = {row["job_id"]: row for row in ledger.jobs()}
                events = ledger.events(job_id=job_id, types=wanted, limit=limit)
            if rows.get(job_id, {}).get("state") in {"succeeded", "failed", "denied", "cancelled"}:
                for ev in events:
                    if ev.seq not in seen:
                        _render_log_event(ev, raw=raw)
                return
            time.sleep(0.2)
    except KeyboardInterrupt:
        return


@audit_app.command("verify")
def audit_verify(config: Optional[str] = typer.Option(None, "--config", "-c")):
    """Verify the hash chain: proves the log has not been edited."""
    cfg = load_config(config)
    with Ledger(cfg.root_path) as ledger:
        ok, msg = ledger.verify()
    if ok:
        console.print(f"[green]✓ audit chain intact[/green]: {msg}")
        raise typer.Exit(0)
    console.print(f"[red]✗ audit chain broken[/red]: {msg}")
    raise typer.Exit(2)


@audit_app.command("export")
def audit_export(
    run_id: Optional[str] = typer.Option(None, "--run", autocompletion=_complete_runs),
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
    with Ledger(cfg.root_path) as ledger:
        n = ledger.reindex()
    console.print(f"reindexed {n} events")


@app.command()
def policy(config: Optional[str] = typer.Option(None, "--config", "-c")):
    """Print the effective safeguard policy."""
    cfg = load_config(config)
    console.print(JSON.from_data(cfg.policy.model_dump(mode="json")))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
