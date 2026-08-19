"""Layered configuration.

Precedence, lowest to highest:

    built-in defaults
      < ~/.config/research-fleet/config.yaml
      < ./fleet.yaml (or $FLEET_CONFIG)
      < FLEET_* environment variables
      < explicit kwargs / CLI flags

Everything is a plain pydantic model, so the same object is what you get from
`fleet.yaml` and from `FleetConfig(...)` in Python.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from .budget import ModelCost, register_model_cost
from .policy import Policy
from .spec import Mount

USER_CONFIG = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser() / "research-fleet" / "config.yaml"
PROJECT_CONFIG = Path("fleet.yaml")


class BudgetConfig(BaseModel):
    max_usd: float = Field(50.0, gt=0, description="Ceiling for an entire run, sub-agents included.")
    max_tokens: int = Field(100_000_000, gt=0)
    default_model: str = "claude-opus-5"
    # Models a spawning agent is allowed to choose from, cheapest-first in the brief.
    delegation_models: list[str] = Field(
        default_factory=lambda: ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"]
    )
    default_effort: str = "high"
    model_costs: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Override or add price-table entries; keys match ModelCost fields.",
    )


class ExecutorConfig(BaseModel):
    kind: Literal["ship", "direct", "dry-run"] = "ship"
    ship_binary: str = Field("ship", description="The research-ship launcher, on PATH.")
    project_dir: str | None = Field(
        None,
        description="Project whose .ship.conf and image to use. Defaults to `workspace`.",
    )
    docker_binary: str = "docker"
    mounts: list[Mount] = Field(
        default_factory=list,
        description="Files or directories exposed to every job at a stable path.",
    )


class AgentBackendConfig(BaseModel):
    name: str = "claude-cli"
    binary: str | None = Field(None, description="Override the executable; defaults per backend.")
    passthrough_env: list[str] = Field(
        default_factory=lambda: ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"],
        description="Host env vars forwarded into agent containers. Values are never logged.",
    )
    extra_args: list[str] = Field(default_factory=list)
    shared_prompt: str = Field(
        "",
        description="Standing instructions given to every agent job, on top of the "
                    "briefs fleet generates. A path relative to the workspace; empty "
                    "means look for the conventional locations (see SHARED_PROMPT_NAMES).",
    )


class FleetConfig(BaseModel):
    root: str = Field("~/.research-fleet", description="Where the ledger, results and state live.")
    image: str = Field(
        "", description="Override the image; empty means whatever research-ship resolves for the project."
    )
    workspace: str = Field(".", description="Host directory mounted at /workspace.")
    isolate_agents: bool = Field(
        False,
        description="Give every agent job its own git worktree and branch. Strongly "
                    "recommended for unattended runs; requires the workspace to be a git repo.",
    )
    results_dir: str = Field("", description="Defaults to <root>/results.")

    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    agent: AgentBackendConfig = Field(default_factory=AgentBackendConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    policy: Policy = Field(default_factory=Policy)

    env: dict[str, str] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)

    def model_post_init(self, _ctx: Any) -> None:
        if not self.results_dir:
            object.__setattr__(self, "results_dir", str(Path(self.root).expanduser() / "results"))
        # Workspace is the implicit mount allowlist root unless one was given.
        if not self.policy.allowed_mount_roots:
            self.policy.allowed_mount_roots = [
                str(Path(self.workspace).expanduser().resolve()),
                str(Path(self.results_dir).expanduser()),
            ]
        for name, fields in self.budget.model_costs.items():
            register_model_cost(ModelCost(model=name, **fields))

    # Both are resolved to absolute paths: everything under `root` becomes a
    # becomes a Docker bind-mount source, and the daemon rejects a relative one.
    # A relative `root:` in a project's fleet.yaml is otherwise only caught when the
    # first container fails to start.
    @property
    def root_path(self) -> Path:
        return Path(self.root).expanduser().resolve()

    @property
    def results_path(self) -> Path:
        return Path(self.results_dir).expanduser().resolve()


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _env_overrides() -> dict:
    """FLEET_EXECUTOR__KIND=dry-run -> nested executor configuration."""
    out: dict = {}
    for key, raw in os.environ.items():
        if not key.startswith("FLEET_") or key == "FLEET_CONFIG":
            continue
        path = key[len("FLEET_"):].lower().split("__")
        try:
            value: Any = json.loads(raw)
        except (ValueError, TypeError):
            value = raw
        cursor = out
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = value
    return out


def load_config(path: str | Path | None = None, **overrides: Any) -> FleetConfig:
    layers: list[dict] = [{}]
    layers.append(_load_yaml(USER_CONFIG))

    explicit = Path(path) if path else Path(os.environ.get("FLEET_CONFIG", PROJECT_CONFIG))
    layers.append(_load_yaml(explicit))
    layers.append(_env_overrides())
    layers.append({k: v for k, v in overrides.items() if v is not None})

    merged: dict = {}
    for layer in layers:
        merged = _deep_merge(merged, layer)
    return FleetConfig(**merged)
