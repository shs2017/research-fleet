"""Expose fleet launches, status, traces, and budgets through MCP."""

from __future__ import annotations

import json
import threading
from typing import Any, Optional

from .budget import cost_menu, quote
from .config import load_config
from .fleet import Fleet
from .ledger import Ledger

_ACTIVE: dict[str, Fleet] = {}
_LOCK = threading.Lock()


def _remember(fleet: Fleet) -> None:
    with _LOCK:
        _ACTIVE[fleet.run_id] = fleet


def _get(run_id: str) -> Fleet | None:
    with _LOCK:
        return _ACTIVE.get(run_id)


def serve(config_path: Optional[str] = None) -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - optional extra
        raise SystemExit(
            "MCP server needs the extra: pip install 'research-fleet[mcp]'"
        ) from exc

    cfg = load_config(config_path)
    mcp = FastMCP("research-fleet")

    @mcp.tool()
    def fleet_quote(model: str = "", effort: str = "high", process: str = "agent_standard") -> str:
        """Estimate the token and dollar cost of a unit of work before running it.

        process: single_call | workflow | agent_short | agent_standard | agent_long
        """
        q = quote(model or cfg.budget.default_model, effort=effort, process=process)
        return json.dumps(
            {"quote": q.to_dict(), "menu": cost_menu(cfg.budget.delegation_models, process=process)},
            indent=2,
        )

    @mcp.tool()
    def fleet_run_agents(
        task: str,
        agents: int = 1,
        model: str = "",
        effort: str = "high",
        gpus: float = 1.0,
        timeout_s: int = 3600,
        max_usd: float = 0.0,
    ) -> str:
        """Launch research agents in containers across GPUs. Returns a run id immediately.

        Each agent gets its own container, GPU slot and budget scope. Poll
        fleet_status(run_id) for progress and fleet_trace(job_id) for reasoning.
        """
        overrides: dict[str, Any] = {}
        if max_usd:
            overrides["budget"] = {"max_usd": max_usd}
        fleet = Fleet(config_path, **overrides)
        _remember(fleet)
        recs = fleet.run_agents(
            task, n=agents, model=model or None, effort=effort, gpus=gpus, timeout_s=timeout_s
        )
        est = quote(model or cfg.budget.default_model, effort=effort)
        return json.dumps(
            {
                "run_id": fleet.run_id,
                "job_ids": [r.spec.id for r in recs],
                "estimated_cost_usd": round(est.est_cost_usd * agents, 4),
                "budget_remaining_usd": round(fleet.scheduler.budget.get(fleet.run_id).remaining_usd, 4),
                "next": "poll fleet_status(run_id)",
            },
            indent=2,
        )

    @mcp.tool()
    def fleet_run_sweep(
        command: list[str],
        grid: dict[str, list] | None = None,
        gpus: float = 1.0,
        timeout_s: int = 3600,
    ) -> str:
        """Run a hyperparameter sweep across GPUs. No LLM cost; use {param} in the command."""
        fleet = Fleet(config_path)
        _remember(fleet)
        recs = fleet.run_sweep(command, grid or {}, gpus=gpus, timeout_s=timeout_s)
        return json.dumps(
            {"run_id": fleet.run_id, "points": len(recs),
             "job_ids": [r.spec.id for r in recs]},
            indent=2,
        )

    @mcp.tool()
    def fleet_status(run_id: str) -> str:
        """Current job states and budget consumption for a run."""
        fleet = _get(run_id)
        if fleet is not None:
            return json.dumps(fleet.status(), indent=2, default=str)
        with Ledger(cfg.root_path) as ledger:
            return json.dumps({"run_id": run_id, "jobs": ledger.jobs(run_id)}, indent=2, default=str)

    @mcp.tool()
    def fleet_trace(job_id: str, limit: int = 200) -> str:
        """Read a job's reasoning and tool-use trace from the audit ledger."""
        with Ledger(cfg.root_path) as ledger:
            events = ledger.events(job_id=job_id, limit=limit)
            return json.dumps(
                [{"seq": e.seq, "type": e.type, **e.payload} for e in events],
                indent=2, default=str,
            )

    @mcp.tool()
    def fleet_audit_verify() -> str:
        """Verify the audit chain has not been tampered with."""
        with Ledger(cfg.root_path) as ledger:
            ok, msg = ledger.verify()
            return json.dumps({"intact": ok, "detail": msg}, indent=2)

    @mcp.tool()
    def fleet_cancel(run_id: str, reason: str = "cancelled via MCP") -> str:
        """Stop every running job in a run."""
        fleet = _get(run_id)
        if fleet is None:
            return json.dumps({"cancelled": False, "reason": "run not active in this server"})
        fleet.cancel(reason)
        return json.dumps({"cancelled": True, "run_id": run_id})

    mcp.run()
