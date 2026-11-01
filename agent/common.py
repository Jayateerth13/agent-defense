"""Pieces shared by the baseline and Progent runners: pipeline assembly, one-run execution, logging."""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, load_system_message
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.logging import LOGGER_STACK, Logger
from agentdojo.task_suite.task_suite import TaskSuite
from agentdojo.types import ChatMessage, get_text_content_as_str


def base_elements(llm: BasePipelineElement, system_prompt: str | None = None) -> list[BasePipelineElement]:
    """system prompt (AgentDojo default unless overridden) -> user query -> LLM; identical for both conditions."""
    return [SystemMessage(system_prompt or load_system_message(None)), InitQuery(), llm]


def simplify_messages(messages: list[ChatMessage]) -> list[dict]:
    """JSON-serialisable transcript (role / text / tool calls)."""
    out = []
    for m in messages:
        d: dict[str, Any] = {"role": m["role"]}
        content = m.get("content")
        d["content"] = get_text_content_as_str(content) if content else ""
        if m["role"] == "assistant" and m.get("tool_calls"):
            d["tool_calls"] = [{"function": tc.function, "args": dict(tc.args)} for tc in m["tool_calls"]]
        if m["role"] == "tool":
            if not d["content"] and m.get("error"):
                d["content"] = str(m["error"])  # blocked / failed calls surface as tool errors
            d["tool"] = m["tool_call"].function
            d["error"] = m.get("error")
        out.append(d)
    return out


class RecorderLogger(Logger):
    """AgentDojo calls ``Logger.get().log(messages)`` after every pipeline element; keep the latest
    list so the transcript survives exceptions / AbortAgentError."""

    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []

    def log(self, messages, *a, **k) -> None:
        self.messages = list(messages)

    def log_error(self, *a, **k) -> None:
        pass


@dataclass
class RunOutcome:
    utility: bool = False
    security: bool | None = None  # attack success (None when no injection)
    error: str | None = None
    messages: list[dict] = field(default_factory=list)
    duration: float = 0.0
    n_tool_calls: int = 0
    extra: dict = field(default_factory=dict)


def run_one(
    suite: TaskSuite,
    pipeline: AgentPipeline,
    user_task: BaseUserTask,
    injection_task: BaseInjectionTask | None,
    injections: dict[str, str],
) -> RunOutcome:
    """Run one (user task [, injection task]) with AgentDojo's own task checking.

    Utility and attack-success are decided *exclusively* by AgentDojo's ``utility()`` /
    ``security()`` methods via ``TaskSuite.run_task_with_pipeline`` (spec Section 8).
    """
    t0 = time.time()
    out = RunOutcome()
    # the logger stack's default list is shared between threads: give this context its own
    LOGGER_STACK.set([])
    rec = RecorderLogger()
    with rec:
        try:
            utility, security = suite.run_task_with_pipeline(pipeline, user_task, injection_task, injections)
            out.utility = bool(utility)
            out.security = bool(security) if injection_task is not None else None
        except Exception as e:  # API failure, bug, ... -> flagged and excluded from metrics
            out.error = f"{type(e).__name__}: {e}"
            out.extra["traceback"] = traceback.format_exc(limit=8)
    out.messages = simplify_messages(rec.messages)
    out.n_tool_calls = sum(len(m.get("tool_calls", [])) for m in out.messages if m["role"] == "assistant")
    out.duration = round(time.time() - t0, 2)
    return out

