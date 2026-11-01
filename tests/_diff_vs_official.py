"""Manual differential test (NOT collected by pytest): our enforcement vs the OFFICIAL Progent checker.

Loads ``secagent/tool.py`` from a clone of sunblaze-ucb/progent (path in ``PROGENT_REPO`` / default below),
and compares allow/block decisions of
  (a) every tool call recorded in results/raw_logs.jsonl (replayed with the policy active at that moment), and
  (b) random policies/calls generated over the real Banking tool schemas (fuzz).
No API calls.  Usage:  python tests/_diff_vs_official.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPO = os.environ.get("PROGENT_REPO", str(Path(os.environ["TEMP"]) / "progent_ref"))
sys.path.insert(0, REPO)

from agentdojo.functions_runtime import FunctionsRuntime  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402

import config  # noqa: E402
import secagent.tool as OFF  # noqa: E402  (official implementation)
from policy import enforcer  # noqa: E402
from policy.convert import array_to_policy, with_always_allowed, always_allowed_tools  # noqa: E402
from policy.schema import Policy, Rule, tool_arg_names, tool_specs  # noqa: E402
from policy import generator  # noqa: E402

SUITE = get_suite(config.BENCHMARK_VERSION, config.SUITE_NAME)
RT = FunctionsRuntime(SUITE.tools)
FUNCS = RT.functions
SPECS = tool_specs(FUNCS)
TOOL_ARGS = tool_arg_names(SPECS)
TOOLS_LLM = generator.tools_for_llm(FUNCS)
ALWAYS = always_allowed_tools(TOOLS_LLM, *config.ALWAYS_ALLOWED["banking"])


def official_decision(policy_json: dict, tool: str, kwargs: dict) -> bool:
    """policy_json = our Policy.to_json(); converted to the official tuple store."""
    OFF.security_policy = {
        t: [(r["priority"], 0 if r["effect"] == "allow" else 1, r["conditions"], 0) for r in rs]
        for t, rs in policy_json.items()
    }
    OFF.init_user_query = "q"
    OFF.sort_policy()
    try:
        OFF.check_tool_call(tool, kwargs)
        return True
    except Exception:
        return False


def our_decision(policy_json: dict, tool: str, args: dict) -> bool:
    rules = {
        t: [Rule(t, r["effect"], r["conditions"], r["priority"], r.get("fallback", "return_msg")) for r in rs]
        for t, rs in policy_json.items()
    }
    return enforcer.check_call(Policy(rules), tool, args).allowed


def with_defaults(tool: str, args: dict) -> dict:
    """What the official wrapper validates: bound arguments with defaults applied."""
    return FUNCS[tool].parameters.model_validate(dict(args)).model_dump()


def ours_args(tool: str, args: dict) -> dict:
    """What our runner passes to the enforcer (``agent.progent_runner._typed_args``): defaults applied."""
    return with_defaults(tool, args)


# ------------------------------------------------------------------ (a) replay of real runs
def replay() -> None:
    path = ROOT / "results" / "raw_logs.jsonl"
    stats, diffs = Counter(), []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["condition"] != "progent" or not r.get("progent"):
            continue
        pol = None
        for e in r["progent"]["events"]:
            if e["type"] == "initial_policy":
                pol = e["policy"]
            elif e["type"] == "update" and e.get("new_policy") and not str(e.get("approval", "")).startswith("discarded"):
                pol = e["new_policy"]
            elif e["type"] == "tool_call" and pol is not None:
                t, a = e["tool"], e["args"]
                o = official_decision(pol, t, with_defaults(t, a))
                n = our_decision(pol, t, ours_args(t, a))  # current enforcer on the same (policy, call)
                stats["calls"] += 1
                stats["agree_logged"] += (o == e["allowed"])
                stats["agree_now"] += (o == n)
                stats["flipped_vs_log"] += (n != e["allowed"])
                if o != n:
                    diffs.append((r["user_task"], r["injection_task"], t, a, "ours=" + str(n), "official=" + str(o)))
    print(f"(a) REPLAY of logged Progent runs: {stats['calls']} tool-call decisions; logged decision agrees with "
          f"official: {stats['agree_logged']}; CURRENT enforcer agrees with official: {stats['agree_now']}; "
          f"current enforcer differs from the logged decision: {stats['flipped_vs_log']}")
    for d in diffs:
        print("   DIFF", d)


# ------------------------------------------------------------------ (b) fuzz
POOL = {
    "string": ["", "a", "UK12345678901234567890", "DE89370400440532013000", "US133000000121212121212", "Rent", "2022-03-01",
               "bill-december-2023.txt", "landlord-notices.txt", "pizza"],
    "number": [0, 1, 10, 98.7, 100, 1200.0, 1e9],
    "integer": [0, 1, 5, 7, 100, 1000],
    "boolean": [True, False],
}


def arg_type(tool, arg):
    p = next(s for s in SPECS if s["name"] == tool)["arguments"][arg]
    return p.get("type", "string")


def rand_restriction(rng, typ):
    kind = rng.choice(["enum", "const", "max", "min", "pattern_dict", "pattern_str", "type_only", "empty"])
    pool = POOL.get(typ, POOL["string"])
    if kind == "enum":
        return {"type": typ, "enum": rng.sample(pool, k=min(len(pool), rng.randint(1, 3)))}
    if kind == "const":
        return {"const": rng.choice(pool)}
    if kind == "max" and typ in ("number", "integer"):
        return {"type": typ, "maximum": rng.choice([1, 10, 100, 1000])}
    if kind == "min" and typ in ("number", "integer"):
        return {"type": typ, "minimum": rng.choice([0, 1, 10])}
    if kind == "pattern_dict" and typ == "string":
        return {"type": "string", "pattern": rng.choice(["^UK", "Rent", "^2022-03", r"\.txt$"])}
    if kind == "pattern_str" and typ == "string":
        return rng.choice(["^UK", "Rent", "bill"])
    if kind == "type_only":
        return {"type": typ}
    return {}


def fuzz(n_policies=400, calls_per_policy=25, seed=0) -> None:
    rng = random.Random(seed)
    names = [s["name"] for s in SPECS]
    cats, examples = Counter(), {}
    total = 0
    for _ in range(n_policies):
        arr = []
        for t in rng.sample(names, k=rng.randint(1, 4)):
            for _dup in range(rng.choice([1, 1, 1, 2])):  # sometimes two rules for one tool
                args = {a: rand_restriction(rng, arg_type(t, a)) for a in TOOL_ARGS[t] if rng.random() < 0.6}
                arr.append({"name": t, "args": args})
        try:
            ours = with_always_allowed(array_to_policy(arr, TOOL_ARGS), ALWAYS).to_json()
        except Exception:
            continue
        for _c in range(calls_per_policy):
            t = rng.choice(names)
            spec = next(s for s in SPECS if s["name"] == t)
            call = {}
            for a, p in spec["arguments"].items():
                required = a in spec["required"]
                if required or rng.random() < 0.5:
                    call[a] = rng.choice(POOL.get(p.get("type", "string"), POOL["string"]))
            try:
                full = with_defaults(t, call)
            except Exception:
                continue
            a_ours = our_decision(ours, t, ours_args(t, call))
            # official policy store built from the *official-format* array (+ always allowed), not from ours
            off_pol = {}
            for it in arr:
                off_pol.setdefault(it["name"], []).append((100, 0, it["args"], 0))
            for tn in sorted(ALWAYS):
                off_pol.setdefault(tn, []).append((1, 0, {}, 0))
            OFF.security_policy = off_pol
            OFF.init_user_query = "q"
            OFF.sort_policy()
            try:
                OFF.check_tool_call(t, full)
                a_off = True
            except Exception:
                a_off = False
            total += 1
            if a_ours == a_off:
                cats["agree"] += 1
                continue
            # classify the disagreement
            n_rules = len([x for x in arr if x["name"] == t])
            none_defaulted = any(full[k] is None and k not in call for k in full)
            has_str_restr = any(isinstance(v, str) for x in arr if x["name"] == t for v in x["args"].values())
            if n_rules > 1:
                cat = "multiple generated rules for the tool"
            elif none_defaulted:
                cat = "omitted optional arg (default None) is constrained"
            elif has_str_restr:
                cat = "string (regex) restriction"
            else:
                cat = "other"
            cats[f"{cat}: ours={'allow' if a_ours else 'block'} official={'allow' if a_off else 'block'}"] += 1
            examples.setdefault(cat, (t, call, [x for x in arr if x["name"] == t]))
    print(f"(b) FUZZ: {total} random (policy, call) pairs over the real Banking tools")
    for k, v in cats.most_common():
        print(f"   {v:6d}  {k}")
    for k, v in examples.items():
        print("   example for", k, "->", json.dumps(v)[:300])


if __name__ == "__main__":
    replay()
    fuzz()
