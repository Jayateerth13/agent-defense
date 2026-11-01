"""Run one ASB (task [, attacker tool]) case through either pipeline and score it with ASB's checks."""

from __future__ import annotations

import time
import traceback

from agentdojo.agent_pipeline.errors import AbortAgentError
from agentdojo.logging import LOGGER_STACK

from agent.common import RecorderLogger, RunOutcome, simplify_messages
from asb import env as asb_env
from asb.data import Scenario


def run_asb_case(pipeline, sc: Scenario, task_idx: int, attacker_tool: dict | None) -> RunOutcome:
    t0 = time.time()
    out = RunOutcome()
    runtime, environment = asb_env.build_runtime(sc, attacker_tool)
    LOGGER_STACK.set([])
    rec = RecorderLogger()
    with rec:
        try:
            try:
                _, _, _, messages, _ = pipeline.query(sc.tasks[task_idx], runtime, environment)
                rec.messages = list(messages)
            except AbortAgentError as e:  # Progent `terminate_execution` fallback
                rec.messages = list(e.messages)
        except Exception as e:
            out.error = f"{type(e).__name__}: {e}"
            out.extra["traceback"] = traceback.format_exc(limit=8)
    out.messages = simplify_messages(rec.messages)
    out.n_tool_calls = sum(len(m.get("tool_calls", [])) for m in out.messages if m["role"] == "assistant")
    if not out.error:
        out.utility = asb_env.original_task_succeeded(out.messages, sc)
        out.security = asb_env.attack_succeeded(out.messages, attacker_tool) if attacker_tool else None
    out.duration = round(time.time() - t0, 2)
    return out
