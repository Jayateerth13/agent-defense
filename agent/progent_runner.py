"""Progent condition: AgentDojo's agent loop with the Progent policy layer around tool execution.

Architecture follows the official implementation (sunblaze-ucb/progent, ``secagent``):

    system msg -> user query -> ProgentInit -> LLM -> loop( ProgentToolsExecutor , LLM )

* ``ProgentInit``          generates the initial least-privilege policy from the (trusted) user task
                           and the tool descriptions (official prompt, ``policy/generator.py``).
* ``ProgentToolsExecutor`` replaces AgentDojo's ``ToolsExecutor``.  Every tool call is checked against
                           the current policy *before* execution.  A blocked call yields the error text
                           ``ValidationError: <reason>. Please try other tools or arguments and continue to
                           finish the user task: <task>.`` as the tool error (the agent adapts).  After each
                           executor step (all tool calls of one assistant turn, blocked ones included, as in
                           the official ``tool_execution.py``) the two-step policy update runs.
* Human-written always-allowed tools (priority 1, ``config.ALWAYS_ALLOWED``) survive every update.

Additions over the official code (kept because the task owner asked for logging): every policy update is
classified by Z3 as narrowing/expansion (``policy/smt_check.py``), expansions are auto-approved (worst-case
configuration) and everything is stored in the per-run ``progent`` log field.

Per-run state lives in a :class:`ProgentSession` that travels in ``extra_args`` (and is also kept on
``pipeline.progent_init.session`` so the evaluation can log it).
"""

from __future__ import annotations

import time
from ast import literal_eval
from collections.abc import Sequence
from typing import Any

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.errors import AbortAgentError
from agentdojo.agent_pipeline.llms.google_llm import EMPTY_FUNCTION_NAME
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, is_string_list, tool_result_to_str
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage, ChatToolResultMessage, text_content_block_from_string

import config
import llm_client
from agent.common import base_elements
from agent.llm import AgentLLM
from policy import enforcer, generator
from policy.convert import always_allowed_tools, with_always_allowed
from policy.schema import Policy, tool_arg_names, tool_arg_types, tool_specs
from policy.smt_check import compare_policies

BLOCK_SUFFIX = "Please try other tools or arguments and continue to finish the user task: {task}."


class ProgentSession:
    """Mutable per-run Progent state: current policy + an audit log of everything it did."""

    def __init__(self, user_task: str, runtime_functions: dict, complete: generator.CompleteFn | None = None,
                 task_type: str = "general", always_allowed: tuple[list[str], bool] | None = None,
                 update_enabled: bool = True) -> None:
        self.user_task = user_task
        self.task_type = task_type
        #: False => the policy is generated once and never updated (what the official ASB integration does)
        self.update_enabled = update_enabled
        self.tools = generator.tools_for_llm(runtime_functions)  # what the policy LLM sees (official format)
        specs = tool_specs(runtime_functions)  # our own view, for the Z3 argument-type map
        self.arg_types = tool_arg_types(specs)
        self.tool_args = tool_arg_names(specs)
        names, all_no_arg = always_allowed or ([], False)
        self.always = always_allowed_tools(self.tools, names, all_no_arg)
        self._raw_complete = complete
        self.policy = with_always_allowed(Policy.empty(), self.always)
        self.initial_policy: Policy | None = None
        self.events: list[dict] = []
        self.n_policy_llm_calls = 0
        self.n_blocked = 0
        self.n_expansions = 0
        self.n_updates = 0

    # -- LLM plumbing --------------------------------------------------------------
    def _complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        self.n_policy_llm_calls += 1
        fn = self._raw_complete or generator._default_complete
        return fn(system, user, temperature)

    # -- initial policy ----------------------------------------------------------------
    def init_policy(self) -> None:
        generated = generator.generate_initial_policy(
            self.user_task, self.tools, self.tool_args, self._complete, self.task_type)
        self.policy = with_always_allowed(generated, self.always)
        self.initial_policy = self.policy
        self.events.append({"type": "initial_policy", "policy": self.policy.to_json()})

    # -- enforcement -----------------------------------------------------------------------
    def check(self, tool: str, args: dict[str, Any]) -> enforcer.Decision:
        d = enforcer.check_call(self.policy, tool, args)
        if not d.allowed:
            self.n_blocked += 1
        self.events.append({"type": "tool_call", "tool": tool, "args": _jsonable(args), **d.to_json()})
        return d

    def blocked_text(self, decision: enforcer.Decision) -> str:
        """Text the agent receives for a blocked call (official wording; AgentDojo prefixes the exception name)."""
        return f"ValidationError: {decision.message}. " + BLOCK_SUFFIX.format(task=self.user_task)

    # -- policy update (two-step, official) + Z3 audit ------------------------------------------
    def after_batch(self, calls: list[dict], results: list[str]) -> None:
        """Called once per executor step with all valid calls of the step and their (formatted) results."""
        if not calls or not self.update_enabled:
            return
        tool, args = calls[-1]["name"], calls[-1]["args"]
        try:
            generated = generator.propose_update(
                self.user_task, self.tools, self.tool_args, self.policy, calls, str(results), self._complete)
        except llm_client.QuotaExhaustedError:
            raise  # daily quota gone: abort the run (the driver stops and does not log it)
        except Exception as e:  # official SECAGENT_IGNORE_UPDATE_ERROR=True: keep the current policy
            self.events.append({"type": "update_error", "tool": tool, "reason": f"{type(e).__name__}: {e}"})
            return
        if generated is None:
            self.events.append({"type": "update", "tool": tool, "result": "NO_CHANGE"})
            return
        candidate = with_always_allowed(generated, self.always)
        if candidate == self.policy:
            self.events.append({"type": "update", "tool": tool, "result": "NO_CHANGE"})
            return
        diff = compare_policies(self.policy, candidate, self.arg_types, current_call=(tool, args))
        ev = {"type": "update", "tool": tool, "result": diff.kind, "diff": diff.to_json(),
              "new_policy": candidate.to_json()}
        if diff.is_expansion:
            if config.ONLY_ALLOW_NARROW:  # official SECAGENT_ONLY_ALLOW_NARROW
                ev["approval"] = "discarded (only narrowing allowed)"
                self.events.append(ev)
                return
            # Auto-approver (spec Section 2): expansion "requires" approval and always receives it.
            ev["approval"] = "auto-approved"
            self.n_expansions += 1
        else:
            ev["approval"] = "not required (narrowing)"
        self.events.append(ev)
        self.policy = candidate
        self.n_updates += 1

    def summary(self) -> dict:
        return {
            "initial_policy": self.initial_policy.to_json() if self.initial_policy else None,
            "final_policy": self.policy.to_json(),
            "events": self.events,
            "update_enabled": self.update_enabled,
            "n_policy_llm_calls": self.n_policy_llm_calls,
            "n_blocked": self.n_blocked,
            "n_updates": self.n_updates,
            "n_expansions": self.n_expansions,
        }


def _jsonable(x: Any) -> Any:
    import json

    try:
        json.dumps(x)
        return x
    except TypeError:
        return repr(x)


class ProgentInit(BasePipelineElement):
    """Creates a fresh :class:`ProgentSession` and its initial policy for each run."""

    def __init__(self, complete: generator.CompleteFn | None = None, task_type: str = "general",
                 always_allowed: tuple[list[str], bool] | None = None, update_enabled: bool = True) -> None:
        self.complete = complete
        self.task_type = task_type
        self.always_allowed = always_allowed
        self.update_enabled = update_enabled
        self.session: ProgentSession | None = None  # last session (pipelines are built per run)

    def query(self, query: str, runtime: FunctionsRuntime, env: Env = EmptyEnv(),
              messages: Sequence[ChatMessage] = [], extra_args: dict = {}):
        session = ProgentSession(query, runtime.functions, self.complete, self.task_type, self.always_allowed,
                                 self.update_enabled)
        self.session = session
        session.init_policy()
        # NOTE: return a *new* dict; AgentDojo's default ``extra_args={}`` is shared between runs.
        return query, runtime, env, messages, {**extra_args, "progent": session}


class ProgentToolsExecutor(BasePipelineElement):
    """Drop-in replacement for AgentDojo's ``ToolsExecutor`` with policy enforcement."""

    def __init__(self, tool_output_formatter=tool_result_to_str) -> None:
        self.output_formatter = tool_output_formatter

    @staticmethod
    def _result(tc, text: str, error: str | None = None) -> ChatToolResultMessage:
        return ChatToolResultMessage(
            role="tool", content=[text_content_block_from_string(text)], tool_call_id=tc.id, tool_call=tc, error=error
        )

    def query(self, query: str, runtime: FunctionsRuntime, env: Env = EmptyEnv(),
              messages: Sequence[ChatMessage] = [], extra_args: dict = {}):
        if not messages or messages[-1]["role"] != "assistant" or not messages[-1]["tool_calls"]:
            return query, runtime, env, messages, extra_args
        session: ProgentSession = extra_args["progent"]
        results = []
        upd_calls: list[dict] = []  # calls (and results) fed to the policy update, as in the official executor
        upd_results: list[str] = []
        for tc in messages[-1]["tool_calls"]:
            # --- same validity handling as AgentDojo's ToolsExecutor ---
            if tc.function == EMPTY_FUNCTION_NAME:
                results.append(self._result(tc, "", "Empty function name provided. Provide a valid function name."))
                continue
            if tc.function not in (t.name for t in runtime.functions.values()):
                results.append(self._result(tc, "", f"Invalid tool {tc.function} provided."))
                continue
            for k, v in tc.args.items():
                if isinstance(v, str) and is_string_list(v):
                    tc.args[k] = literal_eval(v)

            # --- Progent: enforce policy before executing ---
            decision = session.check(tc.function, _typed_args(runtime, tc.function, tc.args))
            if not decision.allowed:
                text = session.blocked_text(decision)
                if decision.fallback == "terminate_execution":
                    session.events.append({"type": "terminated", "tool": tc.function})
                    raise AbortAgentError(text, list(messages) + results, env)
                if decision.fallback == "request_user_inspection":
                    # no human in the loop in this evaluation: the (simulated) user does not approve
                    text += " The call requires user inspection, which was not granted."
                # official: the blocked call surfaces as a tool *error* (empty result)
                results.append(self._result(tc, "", text))
                upd_calls.append({"name": tc.function, "args": dict(tc.args)})
                upd_results.append("")
                continue

            result, error = runtime.run_function(env, tc.function, tc.args)
            formatted = self.output_formatter(result)
            results.append(self._result(tc, formatted, error))
            upd_calls.append({"name": tc.function, "args": dict(tc.args)})
            upd_results.append(formatted)
        session.after_batch(upd_calls, upd_results)
        return query, runtime, env, [*messages, *results], extra_args


def _typed_args(runtime: FunctionsRuntime, tool: str, args: dict) -> dict:
    """The arguments the policy is checked against: coerced with the tool's own pydantic schema (as the
    runtime will) and with defaults applied.  The official wrapper checks ``bound_args.apply_defaults()``,
    so an omitted optional argument is validated as its default value (e.g. ``None``).  Falls back to the
    raw args when validation fails (the runtime will then report the error itself)."""
    try:
        return runtime.functions[tool].parameters.model_validate(dict(args)).model_dump()
    except Exception:
        return dict(args)


def build_pipeline(
    llm: BasePipelineElement | None = None,
    complete: generator.CompleteFn | None = None,
    system_prompt: str | None = None,
    task_type: str = "general",
    always_allowed: tuple[list[str], bool] | None = None,
    update_enabled: bool = True,
) -> AgentPipeline:
    llm = llm or AgentLLM()
    init = ProgentInit(complete, task_type, always_allowed, update_enabled)
    loop = ToolsExecutionLoop([ProgentToolsExecutor(tool_result_to_str), llm])
    elements = base_elements(llm, system_prompt)
    # insert ProgentInit right after the user query is added, before the first LLM call
    pipeline = AgentPipeline([*elements[:2], init, elements[2], loop])
    pipeline.name = getattr(llm, "name", None) or "progent"
    pipeline.progent_init = init  # type: ignore[attr-defined]
    return pipeline


