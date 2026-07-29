"""Pluggable agent backends. Importing this package registers the built-ins."""

from .base import BACKENDS, AgentBackend, AgentEvent, get_backend, register  # noqa: F401
from . import claude_cli, codex_cli  # noqa: F401  (import for side-effect registration)

__all__ = ["BACKENDS", "AgentBackend", "AgentEvent", "get_backend", "register"]
