"""Manual follow-up to _diff_vs_official.py: (1) block-message equality on logged blocked calls,
(2) how often the two known semantic differences could even occur in real logged policies/calls."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _diff_vs_official as D  # noqa: E402  (also loads the official module)

OFF, ROOT = D.OFF, D.ROOT
from policy import enforcer  # noqa: E402
from policy.schema import Policy, Rule  # noqa: E402

msg_same = msg_total = 0
dup_policies = policies = 0
optional_calls = calls = 0
examples = []
for line in (ROOT / "results" / "raw_logs.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    if r["condition"] != "progent" or not r.get("progent"):
        continue
    task = r["messages"][1]["content"]
    pol = None
    for e in r["progent"]["events"]:
        if e["type"] == "initial_policy" or (e["type"] == "update" and e.get("new_policy")):
            pol = e.get("policy") or e["new_policy"]
            policies += 1
            if any(sum(1 for x in rs if x["priority"] == 100) > 1 for rs in pol.values()):
                dup_policies += 1
        elif e["type"] == "tool_call" and pol is not None:
            t, a = e["tool"], e["args"]
            full = D.with_defaults(t, a)
            calls += 1
            if any(v is None and k not in a for k, v in full.items()):
                optional_calls += 1
            if not e["allowed"]:
                OFF.security_policy = {tt: [(x["priority"], 0, x["conditions"], 0) for x in rs] for tt, rs in pol.items()}
                OFF.init_user_query = task
                OFF.sort_policy()
                try:
                    OFF.check_tool_call(t, full)
                    off_msg = None
                except Exception as ex:
                    off_msg = f"{type(ex).__name__}: {ex}"
                rules = {tt: [Rule(tt, x["effect"], x["conditions"], x["priority"]) for x in rs] for tt, rs in pol.items()}
                dec = enforcer.check_call(Policy(rules), t, e["args"])
                ours = f"ValidationError: {dec.message}. Please try other tools or arguments and continue to finish the user task: {task}."
                msg_total += 1
                msg_same += (off_msg == ours)
                if off_msg != ours and "is not allowed" not in off_msg and len(examples) < 2:
                    examples.append((off_msg[:300], ours[:300]))
print(f"block messages identical to the official text: {msg_same}/{msg_total}")
for o, u in examples:
    print("  official:", o, "\n  ours:    ", u)
print(f"policies (initial+updates) with >1 generated rule for the same tool: {dup_policies}/{policies}")
print(f"tool calls that omit an optional (None-default) argument: {optional_calls}/{calls}")

