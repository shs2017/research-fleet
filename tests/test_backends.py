"""Backend parsing.

Both backends turn a line of harness output into a normalised event. These tests pin
the shapes that matter: usage accounting, tool calls, errors, and the guarantee that an
unrecognised line is still surfaced rather than dropped.
"""

from __future__ import annotations

import json

import pytest

from research_fleet.backends import get_backend
from research_fleet.backends.base import LEDGER_TEXT_LIMIT
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


def test_codex_resumes_an_explicit_session(codex):
    argv = codex.build_command(AgentConfig(task="next", session_id="thread-123"))
    assert argv[:3] == ["codex", "exec", "resume"]
    assert argv[-2:] == ["thread-123", "next"]


def test_codex_captures_started_thread(codex):
    event = codex.parse_line(json.dumps({"type": "thread.started", "thread_id": "thread-123"}))
    assert event.type == "system" and event.session_id == "thread-123"


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


def test_claude_resumes_an_explicit_session(claude):
    argv = claude.build_command(AgentConfig(task="next", session_id="session-123"))
    assert argv[argv.index("--resume") + 1] == "session-123"


def test_claude_captures_started_session(claude):
    event = claude.parse_line(json.dumps({"type": "system", "session_id": "session-123"}))
    assert event.type == "system" and event.session_id == "session-123"


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


# ------------------------------------------------------- final answer length


def test_claude_keeps_the_whole_final_answer(claude):
    """A long final message is the job's deliverable and is handed to the next
    workflow stage verbatim, so clipping it silently truncates the handoff. The
    ledger copy stays bounded; `full_text` does not."""
    answer = "F1: " + ("x" * 20000)
    ev = claude.parse_line(json.dumps({"type": "result", "subtype": "success", "result": answer}))
    assert ev.full_text == answer
    assert len(ev.text) == LEDGER_TEXT_LIMIT
    assert "text" in ev.to_ledger() and len(ev.to_ledger()["text"]) == LEDGER_TEXT_LIMIT


def test_codex_keeps_the_whole_final_answer(codex):
    answer = "F1: " + ("x" * 20000)
    ev = codex.parse_line(json.dumps({"msg": {"type": "task_complete", "last_agent_message": answer}}))
    assert ev.full_text == answer
    assert len(ev.text) == LEDGER_TEXT_LIMIT


def test_a_short_answer_needs_no_clipping(claude):
    ev = claude.parse_line(json.dumps({"type": "result", "result": "done"}))
    assert ev.text == ev.full_text == "done"


# ------------------------------------------------- a harness that exits 0 having failed


def test_claude_a_refused_prompt_is_an_error_despite_exit_zero(claude):
    """The CLI answers `subtype: success, is_error: false` even when it rejected the
    prompt outright, putting the refusal in the result text. Zero turns and zero tokens
    means the model was never called -- which is a failure, not a short answer."""
    ev = claude.parse_line(json.dumps({
        "type": "result", "subtype": "success", "is_error": False,
        "num_turns": 0, "duration_ms": 17,
        "result": "Goal condition is limited to 4000 characters (got 4184)",
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }))
    assert ev.is_error, "a stage that never called the model was reported as success"
    assert "4000 characters" in ev.full_text


def test_claude_a_real_answer_is_not_flagged(claude):
    ev = claude.parse_line(json.dumps({
        "type": "result", "subtype": "success", "is_error": False,
        "num_turns": 12, "result": "F1: ...",
        "usage": {"input_tokens": 900, "output_tokens": 400},
    }))
    assert not ev.is_error


def test_claude_a_terse_but_genuine_run_is_not_flagged(claude):
    """One turn and few tokens is a short answer, not a refusal."""
    ev = claude.parse_line(json.dumps({
        "type": "result", "subtype": "success", "num_turns": 1, "result": "OK",
        "usage": {"input_tokens": 12, "output_tokens": 1},
    }))
    assert not ev.is_error


def test_claude_an_explicit_error_is_still_an_error(claude):
    ev = claude.parse_line(json.dumps({
        "type": "result", "subtype": "error", "is_error": True, "num_turns": 3,
        "result": "boom", "usage": {"input_tokens": 5, "output_tokens": 5},
    }))
    assert ev.is_error


def test_claude_warns_before_running_an_over_long_goal(claude):
    """Cheaper to catch here than after a container start and a 17ms 'success'."""
    task = "/goal " + ("x" * 4100)
    with pytest.warns(UserWarning, match="goal condition"):
        claude.build_command(AgentConfig(task=task))


def test_claude_does_not_warn_about_a_long_ordinary_prompt(claude):
    """The cap applies to goal conditions, not to prompts generally."""
    import warnings as w
    with w.catch_warnings():
        w.simplefilter("error")
        claude.build_command(AgentConfig(task="x" * 40000))
        claude.build_command(AgentConfig(task="/goal " + "x" * 100))
