"""Offline tests for the agent loop and trace, with a scripted model and an in-memory sandbox."""

import json
from types import SimpleNamespace

import pytest

import llm_client
from oversight import agent_loop
from oversight.agent_loop import SpendCapReached, format_output, run_agent
from oversight.mock_agent import ScriptedAgent
from oversight.sandbox import SESSION_FILE, ExecResult
from oversight.trace import Trace


class FakeSandbox:
    """Runs nothing: echoes agent commands and keeps the session file in memory."""

    def __init__(self, outputs=None):
        self.session = b""
        self.agent_commands = []
        self.outputs = outputs or {}

    def run(self, cmd, user="agent", *, timeout=60, env=None, workdir="/workspace", stdin=None):
        if user == "root":  # the trace mirror
            self.session += stdin.encode()
            return ExecResult(0, "", "")
        self.agent_commands.append(cmd)
        return self.outputs.get(cmd, ExecResult(0, f"ran: {cmd}", ""))

    def read_file(self, path):
        assert path == SESSION_FILE
        return self.session


def run(turns, tmp_path, **kw):
    sb = FakeSandbox(kw.pop("outputs", None))
    trace = Trace(tmp_path / "t.jsonl", sb)
    agent = ScriptedAgent(turns)
    res = run_agent("Fix the bug.", sb, trace, complete=agent, **kw)
    return res, sb, trace, agent


def test_commands_run_as_agent_and_round_ends_without_tool_call(tmp_path):
    res, sb, _, agent = run([["ls"], ["cat src/tax.py", "pytest"]], tmp_path)
    assert sb.agent_commands == ["ls", "cat src/tax.py", "pytest"]
    assert res.stop_reason == "done" and res.steps == 3 and agent.calls == 3


def test_step_cap_stops_a_model_that_never_finishes(tmp_path):
    res, sb, _, _ = run([["true"]] * 50, tmp_path, max_steps=4)
    assert res.stop_reason == "step_cap" and len(sb.agent_commands) == 4


def test_feedback_rounds_send_the_grade_and_continue(tmp_path):
    grades = iter(["Your grade: 0.4 out of 1.0", "Your grade: 0.8 out of 1.0"])
    res, sb, trace, _ = run([["make test"]], tmp_path, rounds=3, feedback=lambda: next(grades))
    assert res.rounds == 3 and sb.agent_commands == ["make test"] * 3
    assert [json.loads(ln)["kind"] for ln in trace.lines].count("feedback") == 2


def test_session_file_matches_the_host_trace_until_tampered(tmp_path):
    _, sb, trace, _ = run([["ls"]], tmp_path)
    assert trace.session_intact()
    assert (tmp_path / "t.jsonl").read_bytes().replace(b"\r\n", b"\n") == trace.rendered()
    sb.session = sb.session.split(b"\n", 1)[1]  # the agent deleted its first record
    assert not trace.session_intact()


def test_bad_tool_calls_get_an_error_not_a_crash(tmp_path):
    sb = FakeSandbox()
    bad = SimpleNamespace(id="c1", type="function", function=SimpleNamespace(name="bash", arguments="{not json"))
    other = SimpleNamespace(id="c2", type="function", function=SimpleNamespace(name="python", arguments="{}"))
    replies = iter([SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[bad, other]))]),
                    SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))])])
    seen = []
    run_agent("t", sb, Trace(tmp_path / "t.jsonl", sb), complete=lambda m, t: seen.append(list(m)) or next(replies))
    tool_msgs = [m for m in seen[-1] if m["role"] == "tool"]
    assert "error" in tool_msgs[0]["content"] and "unknown tool" in tool_msgs[1]["content"]
    assert sb.agent_commands == []
    assert all("tool_calls" in m for m in seen[-1] if m["role"] == "assistant")  # no null tool_calls sent


def test_long_output_is_cut_in_the_middle():
    out = format_output(0, "a" * 3000 + "b" * 3000, "", False)
    assert out.startswith("exit code 0\naaa") and out.endswith("bbb") and "characters cut" in out
    assert len(out) < agent_loop.MAX_OUTPUT + 100


def test_spend_cap_stops_the_run(tmp_path, monkeypatch):
    monkeypatch.setitem(llm_client.TOKENS, "agent_input", 10_000_000)  # $1.50 at gpt-4o-mini prices
    with pytest.raises(SpendCapReached):
        run([["ls"]], tmp_path, model="gpt-4o-mini", spend_cap_usd=1.0)
