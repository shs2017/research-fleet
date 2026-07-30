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

from .budget import cost_menu, quote
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
    return Fleet(config, **{k: v for k, v in overrides.items() if v is not None})


def _overrides(workspace=None, image=None, executor=None, max_usd=None) -> dict:
    """Map the flags shared by `run` and `sweep` onto config overrides."""
    out: dict = {"workspace": workspace, "image": image}
    if executor:
        out["executor"] = {"kind": executor}
    if max_usd is not None:
        out["budget"] = {"max_usd": max_usd}
    return out


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
    gpus: float = typer.Option(1.0, "--gpus", help="GPUs per agent; <1 packs several per device."),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w"),
    image: Optional[str] = typer.Option(None, "--image"),
    timeout: int = typer.Option(3600, "--timeout", help="Per-agent wall clock, seconds."),
    max_usd: Optional[float] = typer.Option(None, "--max-usd", help="Budget ceiling for the whole run."),
    executor: Optional[str] = typer.Option(None, "--executor", help="ship | slurm | ray | dry-run"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Launch one or more research agents on a task."""
    fleet = _fleet(
        config,
        **_overrides(workspace, image, executor, max_usd),
        on_event=_live_printer(verbose),
    )
    try:
        est = quote(model or fleet.config.budget.default_model,
                    effort=effort or fleet.config.budget.default_effort)
        console.print(
            f"[bold]run {fleet.run_id}[/bold]  {agents} agent(s)  "
            f"est. ${est.est_cost_usd * agents:.2f}  "
            f"budget ${fleet.config.budget.max_usd:.2f}"
        )
        fleet.run_agents(
            task, n=agents, model=model, backend=backend, effort=effort,
            gpus=gpus, timeout_s=timeout,
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
    from .workflow import Loop, Workflow

    wf = Workflow.from_yaml(file)

    if plan:
        console.print(f"[bold]{wf.name}[/bold]  {wf.description}")
        for stage in wf.stages:
            if isinstance(stage, Loop):
                until = f"until {stage.until.step} contains '{stage.until.output_contains}'" \
                    if stage.until and stage.until.output_contains else "until done"
                console.print(f"  loop {stage.name}  max {stage.max_iterations}x, {until}")
                for inner in stage.steps:
                    console.print(f"    - {inner.name} ({inner.kind.value})")
            else:
                fan = ""
                if stage.for_each:
                    fan = f" x{len(stage.for_each)}"
                elif stage.copies > 1:
                    fan = f" x{stage.copies}"
                console.print(f"  {stage.name} ({stage.kind.value}){fan}")
        return

    fleet = _fleet(
        config,
        **_overrides(workspace, None, executor, max_usd),
        on_event=_live_printer(verbose),
    )
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
