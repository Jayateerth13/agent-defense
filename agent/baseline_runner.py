"""Undefended baseline: AgentDojo's standard tool-calling agent loop with NO policy layer."""

from __future__ import annotations

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor, tool_result_to_str

from agent.common import base_elements
from agent.llm import AgentLLM


def build_pipeline(llm: BasePipelineElement | None = None, system_prompt: str | None = None) -> AgentPipeline:
    """system msg -> user query -> LLM -> loop(ToolsExecutor, LLM)  (== AgentDojo ``defense=None``)."""
    llm = llm or AgentLLM()
    loop = ToolsExecutionLoop([ToolsExecutor(tool_result_to_str), llm])
    pipeline = AgentPipeline([*base_elements(llm, system_prompt), loop])
    pipeline.name = getattr(llm, "name", None) or "baseline"
    return pipeline
