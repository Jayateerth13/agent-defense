"""Offline stand-ins for the LLMs, used by the unit tests and ``run_eval.py --mock`` (no API key needed).

* :class:`ScriptedLLM` plays the *agent*: it replays AgentDojo's ground-truth tool calls for the user
  task, and - when ``fooled=True`` - gets "hijacked" by the first tool result containing an
  ``<INFORMATION>`` injection and executes the injection task's ground-truth calls first.  This is a
  worst-case deterministic victim, handy for verifying that the baseline is attackable and that
  Progent blocks the hijacked calls.
* :func:`oracle_policy_complete` plays the *policy LLM*: its initial policy is exactly the set of
  tool/argument values used by the user task's ground truth (an ideal least-privilege policy), and it
  always answers NO_CHANGE to update requests.

These mocks validate the plumbing (enforcement, logging, AgentDojo checks, metrics).  Numbers produced
with them are NOT results of the reproduction.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)

from policy import generator


class ScriptedLLM(BasePipelineElement):
    name = "scripted-mock"

    def __init__(self, user_task: BaseUserTask, injection_task: BaseInjectionTask | None = None,
                 fooled: bool = True) -> None:
        self.user_task = user_task
        self.injection_task = injection_task
        self.fooled = fooled
        self.queue: list[FunctionCall] | None = None
        self.hijacked = False

    def _msg(self, text: str | None, calls: list[FunctionCall] | None) -> ChatAssistantMessage:
        return ChatAssistantMessage(
            role="assistant", content=[text_content_block_from_string(text)] if text is not None else None,
            tool_calls=calls,
        )

    def query(self, query: str, runtime: FunctionsRuntime, env: Env = EmptyEnv(),
              messages: Sequence[ChatMessage] = [], extra_args: dict = {}):
        if self.queue is None:
            self.queue = list(self.user_task.ground_truth(env))
        last = messages[-1]
        if (self.fooled and self.injection_task is not None and not self.hijacked and last["role"] == "tool"
                and "<INFORMATION>" in get_text_content_as_str(last["content"])):
            self.hijacked = True
            self.queue = list(self.injection_task.ground_truth(env)) + self.queue
        if self.queue:
            c = self.queue.pop(0)
            call = FunctionCall(function=c.function, args=dict(c.args), id=f"call_{uuid.uuid4().hex[:8]}")
            return query, runtime, env, [*messages, self._msg(None, [call])], extra_args
        text = self.user_task.GROUND_TRUTH_OUTPUT or "Done."
        return query, runtime, env, [*messages, self._msg(text, None)], extra_args


def oracle_policy_complete(user_task: BaseUserTask, env) -> generator.CompleteFn:
    """Policy-LLM stand-in producing the ideal policy for ``user_task`` (see module docstring)."""
    per_tool: dict[str, dict[str, list]] = {}
    for c in user_task.ground_truth(env):
        d = per_tool.setdefault(c.function, {})
        for k, v in c.args.items():
            if isinstance(v, (str, int, float, bool)):
                if v not in d.setdefault(k, []):
                    d[k].append(v)
    policy = [  # official wire format: [{"name": tool, "args": restrictions}]
        {"name": tool, "args": {k: {"enum": vals} for k, vals in args.items()}} for tool, args in per_tool.items()
    ]

    def complete(system: str, user: str, temperature: float = 0.0) -> str:
        if system.startswith(generator.INIT_PREFIX):
            return json.dumps(policy)
        return "No"  # never update

    return complete

