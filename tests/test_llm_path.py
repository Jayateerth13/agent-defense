"""Exercises the real (non-mock) LLM code path - AgentLLM + policy generator + llm_client - against a
fake OpenAI-compatible backend, so message/tool conversion bugs show up without an API key."""

import json

from agentdojo.task_suite.load_suites import get_suite
from openai.types.chat import ChatCompletion

import config
import llm_client
from agent import baseline_runner, progent_runner
from agent.common import run_one

SUITE = get_suite(config.BENCHMARK_VERSION, config.SUITE_NAME)


def completion(content=None, tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [
            {"id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
            for i, (n, a) in enumerate(tool_calls)
        ]
    return ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 0, "model": "fake",
        "choices": [{"index": 0, "finish_reason": "stop", "message": msg}],
    })


class FakeBackend:
    """Agent: calls get_most_recent_transactions once, then answers.  Policy LLM: allows exactly that."""

    def __init__(self):
        self.calls = []

    def __call__(self, messages, *, tools=None, json_mode=False, temperature=0.0, tag="agent", model=None, seed=None):
        self.calls.append((tag, [m["role"] for m in messages]))
        if tag == "policy" or tools is None:
            sys_prompt = messages[0]["content"]
            if sys_prompt.startswith("Your task is to set permissions"):
                return completion(json.dumps([{"name": "get_most_recent_transactions",
                                               "args": {"n": {"type": "integer", "maximum": 500}}}]))
            return completion("No")  # update step 1: no update needed
        if not any(m["role"] == "tool" for m in messages):
            return completion(None, [("get_most_recent_transactions", {"n": 100})])
        return completion("Your total spending in March 2022 was 1050.")


def test_agent_and_policy_llm_paths(monkeypatch):
    fake = FakeBackend()
    monkeypatch.setattr(llm_client, "chat_completion", fake)
    ut = SUITE.get_user_task_by_id("user_task_1")

    base = run_one(SUITE, baseline_runner.build_pipeline(), ut, None, {})
    assert base.error is None, base.extra
    assert base.n_tool_calls == 1
    assert fake.calls[0][1][0] == "system"  # 'developer' role was mapped to 'system'

    fake.calls.clear()
    pipe = progent_runner.build_pipeline()
    prog = run_one(SUITE, pipe, ut, None, {})
    assert prog.error is None, prog.extra
    s = pipe.progent_init.session
    assert s.n_blocked == 0 and s.n_policy_llm_calls == 2  # 1 initial policy + 1 update after the tool call
    assert [e["result"] for e in s.events if e["type"] == "update"] == ["NO_CHANGE"]
    assert sum(1 for t, _ in fake.calls if t == "policy") == 2

