"""Claude Code CLI backend.

Drives `claude -p ... --output-format stream-json`, which emits one JSON object
per line. Usage is reported per assistant message and again in a final `result`
record; we accumulate the per-message counts and prefer the final record's
totals when present, since it is the authoritative figure the API billed.
"""

from __future__ import annotations

import json
import shlex

from ..budget import Usage
from ..spec import AgentConfig
from .base import AgentEvent, register


class ClaudeCLIBackend:
    name = "claude-cli"

    def __init__(self, binary: str = "claude"):
        self.binary = binary

    def default_model(self) -> str:
        return "claude-opus-5"

    def build_command(self, agent: AgentConfig, *, brief: str = "") -> list[str]:
        prompt = agent.task if not brief else f"{brief}\n\n---\n\n{agent.task}"
        argv = [
            self.binary,
            "-p", prompt,
            "--output-format", "stream-json",
            "--verbose",
        ]
        argv += ["--model", agent.model or self.default_model()]
        if agent.system_prompt:
            argv += ["--append-system-prompt", agent.system_prompt]
        if agent.max_turns:
            argv += ["--max-turns", str(agent.max_turns)]
        if agent.allowed_tools:
            argv += ["--allowedTools", ",".join(agent.allowed_tools)]
        if agent.disallowed_tools:
            argv += ["--disallowedTools", ",".join(agent.disallowed_tools)]
        argv += list(agent.extra_args)
        return argv

    def required_env(self) -> list[str]:
        return ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL"]

    # ------------------------------------------------------------------ parse

    @staticmethod
    def _usage_from(obj: dict, model: str = "") -> Usage | None:
        u = obj.get("usage")
        if not isinstance(u, dict):
            return None
        return Usage(
            input_tokens=int(u.get("input_tokens") or 0),
            output_tokens=int(u.get("output_tokens") or 0),
            cache_read_tokens=int(u.get("cache_read_input_tokens") or 0),
            cache_write_tokens=int(u.get("cache_creation_input_tokens") or 0),
            model=model or obj.get("model") or "",
            requests=1,
        )

    def parse_line(self, line: str) -> AgentEvent | None:
        line = line.strip()
        if not line:
            return None
        if not line.startswith("{"):
            # Non-JSON chatter (warnings, progress bars) still belongs in the trace.
            return AgentEvent(type="raw", text=line)
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return AgentEvent(type="raw", text=line)

        kind = obj.get("type")

        if kind == "system":
            return AgentEvent(type="system", payload={k: v for k, v in obj.items() if k != "type"})

        if kind == "assistant":
            msg = obj.get("message") or {}
            model = msg.get("model", "")
            usage = self._usage_from(msg, model)
            texts, tools = [], []
            for block in msg.get("content") or []:
                btype = block.get("type")
                if btype == "text":
                    texts.append(block.get("text", ""))
                elif btype == "thinking":
                    thought = block.get("thinking") or ""
                    if thought:
                        texts.append(f"[thinking] {thought}")
                elif btype == "tool_use":
                    tools.append({"name": block.get("name"), "input": block.get("input")})
            if tools:
                first = tools[0]
                return AgentEvent(
                    type="tool_use",
                    text="\n".join(texts),
                    tool=first.get("name"),
                    payload={"calls": tools},
                    usage=usage,
                )
            return AgentEvent(type="message", text="\n".join(texts), usage=usage)

        if kind == "user":
            msg = obj.get("message") or {}
            results = []
            for block in msg.get("content") or []:
                if block.get("type") == "tool_result":
                    content = block.get("content")
                    if isinstance(content, list):
                        content = " ".join(
                            c.get("text", "") for c in content if isinstance(c, dict)
                        )
                    results.append(
                        {
                            "tool_use_id": block.get("tool_use_id"),
                            "is_error": bool(block.get("is_error")),
                            "content": str(content)[:4000],
                        }
                    )
            if results:
                return AgentEvent(
                    type="tool_result",
                    payload={"results": results},
                    is_error=any(r["is_error"] for r in results),
                )
            return None

        if kind == "result":
            usage = self._usage_from(obj, obj.get("model", ""))
            return AgentEvent(
                type="result",
                text=str(obj.get("result") or "")[:8000],
                usage=usage,
                is_error=bool(obj.get("is_error")),
                payload={
                    "subtype": obj.get("subtype"),
                    "num_turns": obj.get("num_turns"),
                    "duration_ms": obj.get("duration_ms"),
                    # The CLI's own cost figure; we recompute from tokens but keep
                    # this to cross-check the price table hasn't drifted.
                    "reported_cost_usd": obj.get("total_cost_usd"),
                },
            )

        return AgentEvent(type="raw", payload=obj)

    def __repr__(self) -> str:
        return f"<ClaudeCLIBackend {shlex.quote(self.binary)}>"


register(ClaudeCLIBackend())
