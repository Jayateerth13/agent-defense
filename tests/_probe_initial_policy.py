"""Manual probe (NOT a test, not collected): print the raw initial-policy LLM answer for a Banking task.
One real API call.  Usage: python tests/_probe_initial_policy.py [user_task_id]"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentdojo.functions_runtime import FunctionsRuntime  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402

import config  # noqa: E402
from policy import generator  # noqa: E402

s = get_suite(config.BENCHMARK_VERSION, "banking")
tools = generator.tools_for_llm(FunctionsRuntime(s.tools).functions)
print(json.dumps(tools)[:600], "...\n")
q = s.user_tasks[sys.argv[1] if len(sys.argv) > 1 else "user_task_0"].PROMPT
raw = generator._default_complete(generator.initial_system_prompt(), "TOOLS: " + json.dumps(tools) + "\nUSER_QUERY: " + q, 0.0)
print("RAW:", raw)
