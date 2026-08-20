"""Codex CLI backend.

`codex exec --json` emits JSONL. Its schema is less stable than Claude's, so
this parser is deliberately tolerant: anything it doesn't recognise still lands
in the ledger as a `raw` event rather than being dropped. Auditability beats
tidiness: an unparsed line you can read later is better than a silent gap.

Token accounting: Codex reports OpenAI-shaped usage keys. Built-in OpenAI models
have price-table entries; custom/provider models remain explicitly unpriced unless
registered through `budget.model_costs`.
"""

from __future__ import annotations

import json

from ..budget import MODEL_COSTS, Usage
from ..spec import AgentConfig
from .base import LEDGER_TEXT_LIMIT, AgentEvent, register


class CodexCLIBackend:
    name = "codex-cli"

    def __init__(self, binary: str = "codex"):
        self.binary = binary

    def default_model(self) -> str:
        return "gpt-5.3-codex"

    def build_command(self, agent: AgentConfig, *, brief: str = "") -> list[str]:
        context = "\n\n".join(p for p in (brief, agent.system_prompt) if p)
        prompt = agent.task if not context else f"{context}\n\n---\n\n{agent.task}"
        argv = [self.binary, "exec"]
        if agent.session_id:
            argv += ["resume"]
        argv += ["--json"]
        if agent.model:
            argv += ["--model", agent.model]
        if agent.effort:
            # Codex CLI exposes config keys through `-c`; keep this separate from
            # the prompt so the requested reasoning level actually reaches the model.
            argv += ["-c", f'model_reasoning_effort="{agent.effort}"']
        if agent.execution_mode == "ultra":
            # Codex CLI's Ultra-style execution is exposed through its stable
            # multi-agent feature; this is intentionally independent of effort.
            argv += ["--enable", "multi_agent"]
        # Sandboxing is handled by the container; the CLI's own sandbox would be
        # redundant and blocks legitimate workspace writes.
        argv += ["--dangerously-bypass-approvals-and-sandbox"]
        argv += list(agent.extra_args)
        if agent.session_id:
            argv += [agent.session_id]
        argv += [prompt]
        return argv

    def required_env(self) -> list[str]:
        return ["OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL"]

    @staticmethod
    def _usage_from(obj: dict) -> Usage | None:
        u = obj.get("usage") or obj.get("token_usage")
        if not isinstance(u, dict):
            return None
        cached = int(
            (u.get("cached_input_tokens") or (u.get("input_tokens_details") or {}).get("cached_tokens") or 0)
        )
        total_in = int(u.get("input_tokens") or u.get("prompt_tokens") or 0)
        model = obj.get("model") or ""
        return Usage(
            input_tokens=max(0, total_in - cached),
            output_tokens=int(u.get("output_tokens") or u.get("completion_tokens") or 0),
            cache_read_tokens=cached,
            model=model if model in MODEL_COSTS else "",
            requests=1,
        )

    def parse_line(self, line: str) -> AgentEvent | None:
        line = line.strip()
        if not line:
            return None
        if not line.startswith("{"):
            return AgentEvent(type="raw", text=line)
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return AgentEvent(type="raw", text=line)

        # Current Codex CLI wraps messages/tool calls in `item.*` events. Older
        # releases used `msg`, so retain both shapes for long-lived installations.
        item = obj.get("item") if isinstance(obj.get("item"), dict) else None
        msg = item or (obj.get("msg") if isinstance(obj.get("msg"), dict) else obj)
        kind = msg.get("type") or obj.get("type") or ""
        usage = self._usage_from(msg) or self._usage_from(obj)

        if obj.get("type") in {"thread.started", "session.started"}:
            session_id = obj.get("thread_id") or obj.get("session_id") or msg.get("id")
            return AgentEvent(type="system", session_id=str(session_id) if session_id else None,
                              payload=obj)

        if kind in {"agent_message", "assistant_message", "message"}:
            return AgentEvent(type="message", text=str(msg.get("message") or msg.get("text") or ""), usage=usage)
        if kind in {"agent_reasoning", "reasoning"}:
            return AgentEvent(type="thinking", text=str(msg.get("text") or msg.get("summary") or ""), usage=usage)
        if kind in {"exec_command_begin", "tool_call", "function_call", "patch_apply_begin",
                    "command_execution", "mcp_tool_call", "file_change"}:
            completed = obj.get("type") == "item.completed" or msg.get("status") == "completed"
            return AgentEvent(
                type="tool_result" if completed else "tool_use",
                tool=str(msg.get("name") or kind),
                payload={k: v for k, v in msg.items() if k != "type"},
                is_error=bool(msg.get("exit_code")) or msg.get("status") == "failed",
                usage=usage,
            )
        if kind in {"exec_command_end", "tool_result", "patch_apply_end"}:
            return AgentEvent(
                type="tool_result",
                tool=str(msg.get("name") or kind),
                payload={k: v for k, v in msg.items() if k != "type"},
                is_error=bool(msg.get("exit_code")) or msg.get("success") is False,
                usage=usage,
            )
        if kind in {"task_complete", "turn_complete", "turn.completed", "result"} or \
                obj.get("type") == "turn.completed":
            answer = str(msg.get("last_agent_message") or msg.get("text") or "")
            return AgentEvent(
                type="result",
                text=answer[:LEDGER_TEXT_LIMIT],
                full_text=answer,
                usage=usage,
                payload={k: v for k, v in msg.items() if k != "type"},
            )
        if kind in {"error", "stream_error"}:
            return AgentEvent(type="result", text=str(msg.get("message") or ""), is_error=True, payload=msg)

        return AgentEvent(type="raw", payload=obj, usage=usage)


register(CodexCLIBackend())
