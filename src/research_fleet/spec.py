"""Job specifications: the data model everything else operates on.

A fleet run is a DAG of `JobSpec`s. Two kinds exist:

  * `command`: run a process in a container (a training run, an eval, a sweep point)
  * `agent`  : run an LLM coding agent in a container; it may submit further jobs

Both share the same resource, isolation and audit surface, which is what makes
"agent that launches its own sweep" work without a second scheduler.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class JobKind(str, Enum):
    COMMAND = "command"
    AGENT = "agent"


class JobState(str, Enum):
    PENDING = "pending"          # accepted, waiting on deps
    QUEUED = "queued"            # deps met, waiting on a GPU slot
    AWAITING_APPROVAL = "awaiting_approval"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DENIED = "denied"            # rejected by policy before it ever ran

    @property
    def terminal(self) -> bool:
        return self in {
            JobState.SUCCEEDED,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.DENIED,
        }


class Resources(BaseModel):
    """What one job needs from the cluster."""

    gpus: float = Field(1.0, ge=0.0, description="Fractional GPUs are allowed; 0.5 packs two jobs per device.")
    cpus: float = Field(2.0, gt=0.0)
    memory_gb: float = Field(16.0, gt=0.0)
    shm_size_gb: float = Field(2.0, gt=0.0, description="PyTorch DataLoader workers need this above Docker's 64MB default.")


class Mount(BaseModel):
    source: str
    target: str
    mode: Literal["ro", "rw"] = "ro"

    def to_docker_arg(self) -> str:
        return f"{self.source}:{self.target}:{self.mode}"


class AgentConfig(BaseModel):
    """How to drive the agent for an `agent` job."""

    backend: str = Field("claude-cli", description="Key into research_fleet.backends.BACKENDS")
    model: str | None = Field(None, description="Backend-specific model id. None = backend default.")
    task: str = Field(..., description="The prompt / research question given to the agent.")
    system_prompt: str | None = None
    # Validated here rather than left to the harness: `claude --effort <bad>` only warns
    # and silently falls back to the default, so a typo would quietly cost you the
    # reasoning depth you asked for. Not every backend supports it (codex-cli ignores it).
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    max_turns: int | None = Field(None, ge=1)
    allowed_tools: list[str] | None = Field(
        None, description="Tool allowlist passed through to the agent CLI. None = backend default."
    )
    disallowed_tools: list[str] = Field(default_factory=list)
    extra_args: list[str] = Field(default_factory=list)


class JobSpec(BaseModel):
    id: str = Field(default_factory=lambda: new_id("job"))
    run_id: str = ""
    kind: JobKind = JobKind.COMMAND
    name: str = ""

    # command jobs
    command: list[str] = Field(default_factory=list)

    # agent jobs
    agent: AgentConfig | None = None

    image: str = Field(
        "", description="Empty = the image research-ship resolves for the project. Set to override."
    )
    env: dict[str, str] = Field(default_factory=dict)
    mounts: list[Mount] = Field(default_factory=list)
    resources: Resources = Field(default_factory=Resources)

    isolate: bool = Field(
        False,
        description="Run against a git worktree on its own branch instead of the live tree.",
    )
    worktree_base: str | None = Field(
        None,
        description="Name of another isolated job in this run to branch from instead of the "
                    "live tree's HEAD. That job's pending changes are committed first, so this "
                    "one picks up where it left off while still getting its own reviewable branch.",
    )
    depends_on: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict, description="Sweep coordinates; recorded verbatim in the ledger.")
    labels: dict[str, str] = Field(default_factory=dict)

    timeout_s: int = Field(3600, gt=0)
    parent_job_id: str | None = Field(None, description="Set when an agent job submits this one.")

    created_at: float = Field(default_factory=time.time)

    @model_validator(mode="after")
    def _check_kind(self) -> JobSpec:
        if self.kind is JobKind.COMMAND and not self.command:
            raise ValueError("command jobs require a non-empty `command`")
        if self.kind is JobKind.AGENT and self.agent is None:
            raise ValueError("agent jobs require an `agent` config")
        if not self.name:
            self.name = self.id
        return self

    def fingerprint(self) -> str:
        """Stable content hash: lets the ledger prove a job wasn't edited after the fact."""
        payload = self.model_dump(mode="json", exclude={"created_at"})
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


class JobResult(BaseModel):
    job_id: str
    state: JobState
    exit_code: int | None = None
    started_at: float | None = None
    ended_at: float | None = None
    error: str | None = None
    node: str | None = None
    gpu_ids: list[str] = Field(default_factory=list)
    container_id: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    worktree_path: str | None = Field(
        None, description="Where an isolated job's files actually landed, if it isolated."
    )
    worktree_branch: str | None = Field(
        None, description="The branch that path is checked out on."
    )
    usage: dict[str, Any] = Field(default_factory=dict, description="Token/cost accounting for agent jobs.")
    output: str = Field(
        "", description="The job's answer: an agent's final message, or a command's last output."
    )
    agent_seconds: float | None = Field(
        None,
        description="Time the harness reported working, excluding container start and "
                    "teardown. None for command jobs.",
    )

    @property
    def duration_s(self) -> float | None:
        if self.started_at is None or self.ended_at is None:
            return None
        return self.ended_at - self.started_at
