"""Backend parsing.

Both backends turn a line of harness output into a normalised event. These tests pin
the shapes that matter: usage accounting, tool calls, errors, and the guarantee that an
unrecognised line is still surfaced rather than dropped.
"""

from __future__ import annotations

import json

import pytest

from research_fleet.backends import get_backend
from research_fleet.spec import AgentConfig


@pytest.fixture()
def codex():
    return get_backend("codex-cli")


@pytest.fixture()
def claude():
    return get_backend("claude-cli")


# ------------------------------------------------------------------ registry


def test_unknown_backend_names_the_alternatives():
    with pytest.raises(KeyError, match="claude-cli"):
        get_backend("no-such-backend")


# ------------------------------------------------------------------- codex


def test_codex_builds_a_command_with_the_brief_prepended(codex):
    argv = codex.build_command(AgentConfig(
        task="find the bug", model="gpt-5.3-codex", effort="xhigh",
        system_prompt="PROJECT RULES",
    ), brief="BUDGET")
    assert argv[0] == "codex" and argv[1] == "exec"
    assert argv[-1].startswith("BUDGET") and "find the bug" in argv[-1]
    assert "--model" in argv
    assert argv[argv.index("-c") + 1] == 'model_reasoning_effort="xhigh"'
    assert "PROJECT RULES" in argv[-1]


def test_codex_has_a_priced_backend_default(codex):
    from research_fleet.budget import cost_for

    assert codex.default_model() == "gpt-5.3-codex"
    assert cost_for(codex.default_model()).output_per_mtok == pytest.approx(14.0)


def test_codex_parses_agent_message(codex):
    ev = codex.parse_line(json.dumps({"msg": {"type": "agent_message", "message": "hello"}}))
    assert ev.type == "message" and ev.text == "hello"


def test_codex_parses_reasoning_as_thinking(codex):
    ev = codex.parse_line(json.dumps({"msg": {"type": "agent_reasoning", "text": "hmm"}}))
    assert ev.type == "thinking" and ev.text == "hmm"


def test_codex_parses_current_item_and_turn_events(codex):
    message = codex.parse_line(json.dumps({
        "type": "item.completed",
        "item": {"id": "item_1", "type": "agent_message", "text": "finished"},
    }))
    assert message.type == "message" and message.text == "finished"

    command = codex.parse_line(json.dumps({
        "type": "item.completed",
        "item": {"id": "item_2", "type": "command_execution", "exit_code": 0},
    }))
    assert command.type == "tool_result" and not command.is_error

    turn = codex.parse_line(json.dumps({
        "type": "turn.completed",
        "usage": {"input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 20},
    }))
    assert turn.type == "result"
    assert turn.usage.input_tokens == 60
    assert turn.usage.cache_read_tokens == 40


def test_codex_parses_tool_call_and_result(codex):
    begin = codex.parse_line(json.dumps({"msg": {"type": "exec_command_begin", "name": "bash"}}))
    assert begin.type == "tool_use" and begin.tool == "bash"

    ok = codex.parse_line(json.dumps({"msg": {"type": "exec_command_end", "exit_code": 0}}))
    assert ok.type == "tool_result" and not ok.is_error

    bad = codex.parse_line(json.dumps({"msg": {"type": "exec_command_end", "exit_code": 1}}))
    assert bad.is_error


def test_codex_flags_errors(codex):
    ev = codex.parse_line(json.dumps({"msg": {"type": "error", "message": "boom"}}))
    assert ev.type == "result" and ev.is_error and "boom" in ev.text


def test_codex_maps_openai_shaped_usage(codex):
    ev = codex.parse_line(
        json.dumps(
            {
                "msg": {
                    "type": "agent_message",
                    "message": "done",
                    "usage": {"input_tokens": 1000, "cached_input_tokens": 400, "output_tokens": 50},
                }
            }
        )
    )
    # Cached tokens are billed differently, so they must not stay in input_tokens.
    assert ev.usage.input_tokens == 600
    assert ev.usage.cache_read_tokens == 400
    assert ev.usage.output_tokens == 50


def test_codex_leaves_unknown_models_uncosted(codex):
    """A model absent from the price table must not be silently billed as something else."""
    ev = codex.parse_line(
        json.dumps({"msg": {"type": "agent_message", "message": "x", "model": "private-model",
                            "usage": {"input_tokens": 10}}})
    )
    assert ev.usage.model == ""


def test_both_backends_keep_unparseable_lines(claude, codex):
    for backend in (claude, codex):
        assert backend.parse_line("not json at all").type == "raw"
        assert backend.parse_line("{bad json").type == "raw"
        assert backend.parse_line("   ") is None


# ------------------------------------------------------------------ claude


def test_claude_prefers_the_final_result_usage(claude):
    ev = claude.parse_line(
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "result": "all done",
                "total_cost_usd": 0.5,
                "usage": {"input_tokens": 10, "output_tokens": 2},
            }
        )
    )
    assert ev.type == "result"
    assert ev.usage.input_tokens == 10
    # The harness's own figure is retained so drift from our table is detectable.
    assert ev.payload["reported_cost_usd"] == 0.5


def test_claude_surfaces_thinking_blocks(claude):
    ev = claude.parse_line(
        json.dumps({"type": "assistant", "message": {"content": [
            {"type": "thinking", "thinking": "let me check"}]}})
    )
    assert "let me check" in ev.text


def test_claude_marks_failed_tool_results(claude):
    ev = claude.parse_line(
        json.dumps({"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "is_error": True, "content": "nope"}]}})
    )
    assert ev.type == "tool_result" and ev.is_error


def test_claude_passes_tool_restrictions_through(claude):
    argv = claude.build_command(
        AgentConfig(task="t", allowed_tools=["Read", "Bash"], disallowed_tools=["Write"], max_turns=3)
    )
    assert argv[argv.index("--allowedTools") + 1] == "Read,Bash"
    assert argv[argv.index("--disallowedTools") + 1] == "Write"
    assert argv[argv.index("--max-turns") + 1] == "3"


def test_agent_event_to_ledger_omits_empty_fields(claude):
    ev = claude.parse_line(json.dumps({"type": "assistant", "message": {"content": []}}))
    assert ev.to_ledger() == {}


# ------------------------------------------------- unpriced and synthetic models


def test_synthetic_messages_are_not_attributed_to_a_model(claude):
    """The harness reports <synthetic> for messages it generated locally, which are
    not billed. Recording that as a model made cost lookup fail and took down runs."""
    ev = claude.parse_line(
        json.dumps({"type": "assistant", "message": {
            "model": "<synthetic>", "content": [{"type": "text", "text": "note"}],
            "usage": {"input_tokens": 5, "output_tokens": 1}}})
    )
    assert ev.usage.model == ""
    assert ev.usage.cost_usd() == 0.0


def test_an_unpriced_model_costs_zero_rather_than_raising():
    from research_fleet.budget import Usage

    usage = Usage(input_tokens=1_000_000, model="some-model-we-do-not-price")
    assert usage.cost_usd() == 0.0
    assert usage.priced is False
    assert usage.to_dict()["unpriced_model"] is True


def test_a_priced_model_is_marked_as_such():
    from research_fleet.budget import Usage

    usage = Usage(input_tokens=1_000_000, model="claude-opus-5")
    assert usage.priced is True
    assert usage.to_dict()["unpriced_model"] is False
    assert usage.cost_usd() == pytest.approx(5.0)


def test_quotes_still_reject_an_unknown_model():
    """Tolerance is for reported usage only; a typo in a quote must still be loud."""
    from research_fleet.budget import quote

    with pytest.raises(KeyError, match="unknown model"):
        quote("claude-opus-6-typo")


def test_agents_can_actually_edit_files(claude):
    """The permission bypass is required, not optional: without it an agent can read
    but not write, so every research task fails in a way that looks like refusal.
    Isolation comes from the container and from --worktree, not from this prompt."""
    argv = claude.build_command(AgentConfig(task="fix the bug"))
    assert "--dangerously-skip-permissions" in argv


def test_codex_agents_can_also_edit_files(codex):
    argv = codex.build_command(AgentConfig(task="fix the bug"))
    assert any("dangerously-bypass" in a for a in argv)
