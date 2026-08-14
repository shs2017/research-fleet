"""Claude Code CLI backend.

Drives `claude -p ... --output-format stream-json`, which emits one JSON object
per line. Usage is reported per assistant message and again in a final `result`
record; we accumulate the per-message counts and prefer the final record's
totals when present, since it is the authoritative figure the API billed.
"""

from __future__ import annotations

import json
import shlex
import warnings

from ..budget import Usage
from ..spec import AgentConfig
from .base import LEDGER_TEXT_LIMIT, AgentEvent, register

GOAL_PREFIX = "/goal "
GOAL_LIMIT = 4000
"""`claude -p "/goal ..."` treats the whole task as a goal condition, which the CLI
caps at this many characters and refuses beyond it -- while still exiting 0."""


class ClaudeCLIBackend:
    name = "claude-cli"

    def __init__(self, binary: str = "claude"):
        self.binary = binary

    def default_model(self) -> str:
        return "claude-opus-5"

    def build_command(self, agent: AgentConfig, *, brief: str = "") -> list[str]:
        # The budget brief goes into the system prompt rather than in front of the
        # task. `claude -p` interprets a leading slash command (`/goal ...`), and it
        # only does so at position 0 -- prepending anything turns the command into
        # ordinary prose, silently. Keeping the task verbatim preserves that, and the
        # brief is operator context anyway, which is what a system prompt is for.
        prompt = agent.task
        if prompt.startswith(GOAL_PREFIX) and len(prompt) - len(GOAL_PREFIX) > GOAL_LIMIT:
            # `/goal` makes the whole task a goal condition, which the CLI caps. It
            # rejects the prompt while still exiting 0, so without this the first sign
            # of trouble is a stage that ran for 17ms and produced a one-line "answer".
            warnings.warn(
                f"the task begins with `{GOAL_PREFIX.strip()}` and is "
                f"{len(prompt) - len(GOAL_PREFIX)} characters; the CLI caps a goal "
                f"condition at {GOAL_LIMIT} and will refuse it. Shorten the prompt, or "
                f"drop the leading `{GOAL_PREFIX.strip()}`.",
                stacklevel=2,
            )
        system = "\n\n".join(p for p in (brief, agent.system_prompt) if p)
        argv = [
            self.binary,
            "-p", prompt,
            "--output-format", "stream-json",
            "--verbose",
            # Without this the agent cannot edit files, which makes it useless for
            # research. It is safe here for the same reason it is safe in
            # research-ship: the container, not the prompt, is the isolation boundary,
            # and `policy` plus `--worktree` bound what that container can reach.
            "--dangerously-skip-permissions",
        ]
        if agent.session_id:
            argv += ["--resume", agent.session_id]
        argv += ["--model", agent.model or self.default_model()]
        if system:
            argv += ["--append-system-prompt", system]
        if agent.effort:
            argv += ["--effort", agent.effort]
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

    # The harness reports this for messages it generated locally, which are not billed.
    SYNTHETIC_MODEL = "<synthetic>"

    @classmethod
    def _usage_from(cls, obj: dict, model: str = "") -> Usage | None:
        u = obj.get("usage")
        if not isinstance(u, dict):
            return None
        name = model or obj.get("model") or ""
        if name == cls.SYNTHETIC_MODEL:
            name = ""
        return Usage(
            input_tokens=int(u.get("input_tokens") or 0),
            output_tokens=int(u.get("output_tokens") or 0),
            cache_read_tokens=int(u.get("cache_read_input_tokens") or 0),
            cache_write_tokens=int(u.get("cache_creation_input_tokens") or 0),
            model=name,
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
            session_id = obj.get("session_id")
            return AgentEvent(
                type="system",
                session_id=str(session_id) if session_id else None,
                payload={k: v for k, v in obj.items() if k != "type"},
            )

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
            result = str(obj.get("result") or "")
            # The CLI reports `subtype: success, is_error: false, exit 0` even when it
            # refused the prompt outright -- an over-long `/goal`, for instance -- and
            # the refusal arrives as the result text. Zero turns with zero tokens means
            # the model was never called, which no genuine run does, so treat it as the
            # failure it is rather than letting a stage "succeed" having done nothing.
            # Explicitly zero, not merely absent: a harness that reports no turn count
            # gives no evidence either way, and inferring failure from silence would
            # fail every run that omits the field.
            never_ran = obj.get("num_turns") == 0 and not (usage and usage.total_tokens)
            return AgentEvent(
                type="result",
                text=result[:LEDGER_TEXT_LIMIT],
                full_text=result,
                usage=usage,
                is_error=bool(obj.get("is_error")) or never_ran,
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
