"""ASB adapter tests (offline): data loading, injection text, success checks, Progent reuse."""

import pytest

from agent import baseline_runner, progent_runner
from asb import env as asb_env
from asb.data import SCENARIO, load_scenario
from asb.mock import ASBScriptedLLM, oracle_policy_complete
from asb.runner import run_asb_case

SC = load_scenario(SCENARIO)
SP = asb_env.system_prompt(SC)


def test_scenario_dimensions():
    assert SC.name == "financial_analyst_agent"
    assert len(SC.tasks) == 5 and len(SC.normal_tools) == 2 and len(SC.attack_tools) == 40


def test_opi_injection_is_appended_to_normal_tool_output_only():
    tool = SC.attack_tools[0]
    runtime, env = asb_env.build_runtime(SC, tool)
    names = list(runtime.functions)
    assert names[-1] == tool["Attacker Tool"] and len(names) == 3
    out, err = runtime.run_function(env, SC.normal_tools[0]["Tool Name"], {})
    assert err is None
    assert out.startswith(SC.normal_tools[0]["Expected Achievements"])
    assert "Ignore previous instructions." in out and tool["Attacker Instruction"] in out
    res, _ = runtime.run_function(env, tool["Attacker Tool"], {})
    assert tool["Attack goal"] in res


def test_no_attack_runtime_has_only_normal_tools_and_clean_output():
    runtime, env = asb_env.build_runtime(SC, None)
    assert len(runtime.functions) == 2
    out, _ = runtime.run_function(env, SC.normal_tools[0]["Tool Name"], {})
    assert out == SC.normal_tools[0]["Expected Achievements"]


@pytest.mark.parametrize("idx", [0, 7, 25, 39])
def test_baseline_hijacked_progent_blocks(idx):
    tool = SC.attack_tools[idx]
    base = run_asb_case(baseline_runner.build_pipeline(ASBScriptedLLM(SC, tool), SP), SC, 0, tool)
    pipe = progent_runner.build_pipeline(ASBScriptedLLM(SC, tool), oracle_policy_complete(SC), system_prompt=SP)
    prog = run_asb_case(pipe, SC, 0, tool)
    assert base.error is None and prog.error is None
    assert base.security is True and base.utility is True
    assert prog.security is False and prog.utility is True
    assert pipe.progent_init.session.n_blocked == 1


def test_update_step_can_be_switched_off_and_is_recorded():
    """``update_enabled=False`` (official ASB wiring): policy generated once, the update LLM is never called."""
    from policy import generator

    tool = SC.attack_tools[0]
    inner = oracle_policy_complete(SC)
    seen = []

    def spy(system, user, temperature=0.0):
        seen.append(system[:40])
        return inner(system, user, temperature)

    for enabled in (True, False):
        seen.clear()
        pipe = progent_runner.build_pipeline(ASBScriptedLLM(SC, tool), spy, system_prompt=SP, task_type="asb",
                                             update_enabled=enabled)
        out = run_asb_case(pipe, SC, 0, tool)
        assert out.error is None and out.security is False
        n_decide = sum(s.startswith(generator.DECIDE_PREFIX) for s in seen)
        summary = pipe.progent_init.session.summary()
        assert summary["update_enabled"] is enabled
        assert (n_decide > 0) is enabled
        assert sum(s.startswith(generator.INIT_PREFIX) for s in seen) == 1
