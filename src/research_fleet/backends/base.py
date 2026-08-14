"""Agent backend protocol.

A backend knows three things and nothing else:

  * how to turn an `AgentConfig` into an argv to run inside the container,
  * which host env vars it needs forwarded,
  * how to parse one line of the agent's stdout into a normalised `AgentEvent`.

That keeps the scheduler, ledger and budget code provider-agnostic: adding a
backend never touches them. Parsing is line-oriented so reasoning is streamed
into the ledger as it happens rather than buffered until exit, and a job that dies
mid-run still leaves a complete trace up to the failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..budget import Usage
from ..spec import AgentConfig

LEDGER_TEXT_LIMIT = 8000
"""How much of one event's text the ledger and live display carry."""


@dataclass
class AgentEvent:
    """One normalised step of agent activity."""

    type: str                       # message | thinking | tool_use | tool_result | result | system | raw
    text: str = ""
    tool: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    usage: Usage | None = None
    is_error: bool = False
    full_text: str = ""
    session_id: str | None = None
    """The untruncated `text`, where a backend caps what it puts in `text`.

    `text` is what the ledger and the live display carry, so backends clip it to keep
    one event from dominating the trace. A final answer is also the job's deliverable
    and is handed to later workflow stages verbatim, so it must survive whole: set this
    to the full string and the scheduler will prefer it for the job's output.
    """

    def to_ledger(self) -> dict[str, Any]:
        d: dict[str, Any] = {"text": self.text} if self.text else {}
        if self.tool:
            d["tool"] = self.tool
        if self.payload:
            d["payload"] = self.payload
        if self.usage:
            d["usage"] = self.usage.to_dict()
        if self.is_error:
            d["is_error"] = True
        if self.session_id:
            d["session_id"] = self.session_id
        return d


@runtime_checkable
class AgentBackend(Protocol):
    name: str

    def build_command(self, agent: AgentConfig, *, brief: str = "") -> list[str]:
        """argv executed as the container entrypoint."""
        ...

    def required_env(self) -> list[str]:
        """Host env var names to forward. Values are never written to the ledger."""
        ...

    def parse_line(self, line: str) -> AgentEvent | None:
        """Map one stdout line to an event, or None to ignore it."""
        ...

    def default_model(self) -> str:
        ...


BACKENDS: dict[str, AgentBackend] = {}


def register(backend: AgentBackend) -> AgentBackend:
    BACKENDS[backend.name] = backend
    return backend


def get_backend(name: str) -> AgentBackend:
    if name not in BACKENDS:
        raise KeyError(f"unknown agent backend {name!r}; available: {', '.join(sorted(BACKENDS))}")
    return BACKENDS[name]
