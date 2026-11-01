"""Agent LLM pipeline element (AgentDojo ``BasePipelineElement``) for OpenAI-compatible APIs.

AgentDojo's built-in ``OpenAILLM`` hard-codes the OpenAI client and sends the system prompt with
the ``developer`` role, which Gemini's OpenAI-compatible endpoint does not accept.  This subclass
reuses AgentDojo's message/tool conversion helpers but routes through ``llm_client`` (one shared,
retrying client for both OpenAI and Gemini) and maps ``developer`` -> ``system``.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from agentdojo.agent_pipeline.llms.openai_llm import (
    OpenAILLM,
    _function_to_openai,
    _message_to_openai,
    _openai_to_assistant_message,
)
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage

import config
import llm_client


#: tool_call id -> provider "extra_content" (Gemini thought signatures), process-wide, ids are unique.
_THOUGHT_SIGNATURES: dict[str, dict] = {}


class AgentLLM(OpenAILLM):
    def __init__(self, model: str | None = None, temperature: float = 0.0) -> None:
        # client is unused (calls go through llm_client) but the parent expects the attribute
        super().__init__(client=None, model=model or config.MODEL, temperature=temperature)  # type: ignore[arg-type]
        self.name = self.model

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        oai_messages = []
        for m in messages:
            om = dict(_message_to_openai(m, self.model))
            if om["role"] == "developer":
                om["role"] = "system"
            if om["role"] == "assistant" and om.get("tool_calls"):
                # Gemini 3.x "thinking" models require the opaque thought_signature of each tool call
                # to be echoed back in the history (OpenAI-compat: tool_call["extra_content"]).
                om["tool_calls"] = [
                    {**tc, "extra_content": _THOUGHT_SIGNATURES[tc["id"]]} if tc.get("id") in _THOUGHT_SIGNATURES
                    else tc
                    for tc in om["tool_calls"]
                ]
            oai_messages.append(om)
        tools = [_function_to_openai(t) for t in runtime.functions.values()]
        completion = llm_client.chat_completion(oai_messages, tools=tools, temperature=self.temperature, tag="agent",
                                                model=self.model)
        output = _openai_to_assistant_message(completion.choices[0].message)
        raw_calls = completion.choices[0].message.tool_calls or []
        for i, tc in enumerate(output["tool_calls"] or []):
            if not tc.id:  # some OpenAI-compatible backends omit tool-call ids
                tc.id = f"call_{uuid.uuid4().hex[:12]}"
            extra = (getattr(raw_calls[i], "model_extra", None) or {}).get("extra_content") if i < len(raw_calls) else None
            if extra:
                _THOUGHT_SIGNATURES[tc.id] = extra
        return query, runtime, env, [*messages, output], extra_args
