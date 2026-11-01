"""Offline stand-ins for the ASB runs (``--mock`` and tests); see ``agent/mock_llm.py`` for the idea."""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import FunctionCall
from agentdojo.types import ChatAssistantMessage, get_text_content_as_str, text_content_block_from_string

from asb.data import Scenario
from policy import generator


class ASBScriptedLLM(BasePipelineElement):
    """Calls every normal tool once; when ``fooled`` it obeys an injected 'Ignore previous instructions'."""

    name = "scripted-mock"

    def __init__(self, sc: Scenario, attacker_tool: dict | None, fooled: bool = True) -> None:
        self.queue = [t["Tool Name"] for t in sc.normal_tools]
        self.attacker = attacker_tool["Attacker Tool"] if attacker_tool else None
        self.fooled, self.hijacked = fooled, False

    def query(self, query, runtime, env=None, messages: Sequence = (), extra_args=None):
        extra_args = extra_args or {}
        last = messages[-1]
        if (self.fooled and self.attacker and not self.hijacked and last["role"] == "tool"
                and "Ignore previous instructions" in get_text_content_as_str(last["content"])):
            self.hijacked = True
            self.queue.insert(0, self.attacker)
        if self.queue:
            call = FunctionCall(function=self.queue.pop(0), args={}, id=f"call_{uuid.uuid4().hex[:8]}")
            msg = ChatAssistantMessage(role="assistant", content=None, tool_calls=[call])
        else:
            msg = ChatAssistantMessage(role="assistant", content=[text_content_block_from_string("Done.")],
                                       tool_calls=None)
        return query, runtime, env, [*messages, msg], extra_args


def oracle_policy_complete(sc: Scenario):
    """Policy-LLM stand-in: allow exactly the scenario's normal tools; never update."""
    pol = [{"name": t["Tool Name"], "args": {}} for t in sc.normal_tools]  # official wire format

    def complete(system: str, user: str, temperature: float = 0.0) -> str:
        return json.dumps(pol) if system.startswith(generator.INIT_PREFIX) else "No"

    return complete


