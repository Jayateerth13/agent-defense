"""End-to-end plumbing tests on the real AgentDojo Banking suite with scripted (offline) LLMs."""

import json

import pytest
from agentdojo.task_suite.load_suites import get_suite

import config
from agent import baseline_runner, progent_runner
from agent.common import run_one
from agent.mock_llm import ScriptedLLM, oracle_policy_complete
from policy import generator
from policy.schema import Policy, Rule
from eval.run_eval import make_attack

SUITE = get_suite(config.BENCHMARK_VERSION, config.SUITE_NAME)
ATTACK = make_attack(SUITE)


def run(condition, user_id, inj_id=None, fooled=True, complete=None):
    ut = SUITE.get_user_task_by_id(user_id)
    it = SUITE.get_injection_task_by_id(inj_id) if inj_id else None
    inj = ATTACK.attack(ut, it) if it else {}
    env = ut.init_environment(SUITE.load_and_inject_default_environment({}))
    llm = ScriptedLLM(ut, it, fooled=fooled)
    if condition == "progent":
        pipe = progent_runner.build_pipeline(llm, complete or oracle_policy_complete(ut, env))
    else:
        pipe = baseline_runner.build_pipeline(llm)
    return run_one(SUITE, pipe, ut, it, inj), pipe


def test_banking_suite_dimensions():
    assert len(SUITE.user_tasks) == 16 and len(SUITE.injection_tasks) == 9


def test_all_16x9_pairs_are_injectable():
    for u in SUITE.user_tasks.values():
        for i in SUITE.injection_tasks.values():
            assert ATTACK.attack(u, i)


def test_attack_text_is_important_instructions():
    ut, it = SUITE.get_user_task_by_id("user_task_0"), SUITE.get_injection_task_by_id("injection_task_0")
    text = next(iter(ATTACK.attack(ut, it).values()))
    assert "<INFORMATION>" in text and it.GOAL in text and "Emma Johnson" in text


@pytest.mark.parametrize("inj", ["injection_task_0", "injection_task_5", "injection_task_7"])
def test_baseline_is_hijacked_progent_blocks(inj):
    base, _ = run("no_defense", "user_task_0", inj)
    prog, pipe = run("progent", "user_task_0", inj)
    assert base.error is None and prog.error is None
    assert base.security is True          # undefended agent executes the attacker's goal
    assert prog.security is False         # Progent blocks it ...
    assert prog.utility is True           # ... and the user's task still completes
    assert pipe.progent_init.session.n_blocked >= 1


def test_progent_does_not_hurt_clean_runs_with_ideal_policy():
    for u in SUITE.user_tasks:
        out, _ = run("progent", u)
        assert out.error is None and out.utility, u


def test_blocked_call_returns_fallback_message_as_tool_result():
    out, _ = run("progent", "user_task_0", "injection_task_0")
    tool_msgs = [m["content"] for m in out.messages if m["role"] == "tool"]
    assert any(c.startswith("ValidationError:") and "Please try other tools or arguments" in c for c in tool_msgs)


def test_default_block_with_empty_policy_and_session_log_is_serialisable():
    out, pipe = run("progent", "user_task_0",
                    complete=lambda s, u, t=0.0: "[]" if s.startswith(generator.INIT_PREFIX) else "No")
    assert out.utility is False and pipe.progent_init.session.n_blocked >= 1
    json.dumps(pipe.progent_init.session.summary())


def test_terminate_execution_fallback_aborts_run(monkeypatch):
    # the official wire format only carries allow rules, so inject a forbid/terminate rule directly
    def fake_initial(user_task, tools, tool_args, complete=None, task_type="general"):
        return Policy({
            "read_file": [Rule("read_file", "allow", {}, 100)],
            "send_money": [Rule("send_money", "forbid", {}, 100, "terminate_execution", "halt")],
        })

    monkeypatch.setattr(progent_runner.generator, "generate_initial_policy", fake_initial)
    out, pipe = run("progent", "user_task_0")
    assert out.error is None and out.utility is False
    assert any(e["type"] == "terminated" for e in pipe.progent_init.session.events)


def test_always_allowed_tools_survive_updates_and_allow_no_arg_tools():
    from policy.convert import always_allowed_tools
    pipe = progent_runner.build_pipeline(ScriptedLLM(SUITE.get_user_task_by_id("user_task_1"), None),
                                         lambda s, u, t=0.0: "[]" if s.startswith(generator.INIT_PREFIX) else "No",
                                         always_allowed=config.ALWAYS_ALLOWED["banking"])
    ut = SUITE.get_user_task_by_id("user_task_1")
    out = run_one(SUITE, pipe, ut, None, {})
    s = pipe.progent_init.session
    # get_most_recent_transactions + every no-argument tool are allowed although the LLM policy is empty
    assert s.always == {"get_most_recent_transactions", "get_iban", "get_balance", "get_scheduled_transactions",
                        "get_user_info"}
    assert s.n_blocked == 0 and out.error is None


def test_policy_update_expansion_is_classified_and_auto_approved():
    init = [{"name": "get_most_recent_transactions", "args": {"n": {"type": "integer", "maximum": 1000}}}]
    widened = [{"name": "get_most_recent_transactions", "args": {}}]

    def complete(system, user, temperature=0.0):
        if system.startswith(generator.INIT_PREFIX):
            return json.dumps(init)
        if system.startswith(generator.DECIDE_PREFIX):
            return "Yes"
        return "Yes\n```json\n" + json.dumps(widened) + "\n```"

    out, pipe = run("progent", "user_task_1", complete=complete)
    s = pipe.progent_init.session
    ev = [e for e in s.events if e["type"] == "update"]
    assert ev and ev[0]["result"] == "expansion" and ev[0]["approval"] == "auto-approved"
    assert s.n_expansions >= 1


def test_blocked_calls_also_trigger_the_update_step_like_the_official_executor():
    seen = []

    def complete(system, user, temperature=0.0):
        if system.startswith(generator.INIT_PREFIX):
            return "[]"
        if system.startswith(generator.DECIDE_PREFIX):
            seen.append(user)
            return "No"
        return "No"

    out, pipe = run("progent", "user_task_0", complete=complete)
    assert pipe.progent_init.session.n_blocked >= 1 and seen  # decide-step ran although the call was blocked


def test_omitted_optional_argument_is_checked_as_its_default_like_official():
    """The runner passes bound arguments with defaults applied (official ``apply_defaults``): an omitted
    optional ``recipient`` of ``update_scheduled_transaction`` is validated as ``None`` and blocked by a
    string restriction (user_task_9 clean run in the first 30-case log)."""
    from agentdojo.functions_runtime import FunctionsRuntime
    from policy import enforcer
    from policy.convert import array_to_policy

    runtime = FunctionsRuntime(SUITE.tools)
    args = progent_runner._typed_args(runtime, "update_scheduled_transaction", {"id": 7, "date": "2023-11-04"})
    assert args["id"] == 7 and args["recipient"] is None and args["amount"] is None  # defaults applied
    arr = [{"name": "update_scheduled_transaction", "args": {"subject": {"enum": ["Rent"]}}}]
    policy = array_to_policy(arr, {"update_scheduled_transaction": set(args)})
    assert not enforcer.check_call(policy, "update_scheduled_transaction", args).allowed
    # a restriction on a *non-null* default is satisfied by the default itself
    arr = [{"name": "get_most_recent_transactions", "args": {"n": {"maximum": 100}}}]
    policy = array_to_policy(arr, {"get_most_recent_transactions": {"n"}})
    assert enforcer.check_call(
        policy, "get_most_recent_transactions",
        progent_runner._typed_args(runtime, "get_most_recent_transactions", {})).allowed
